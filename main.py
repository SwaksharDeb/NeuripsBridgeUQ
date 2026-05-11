"""BridgeUQ — unified entry point.

Switch between brain and cardiac with ``--dataset {brain,cardiac}``. The
shared per-batch flow (predict σ → sample → backprop → save s^K → run UQ
→ aggregate) is identical to the legacy ``uncertainty_brain_sde_v3/main.py``
and ``uncertainty_sde_combined_acdc_v3/main.py`` modulo the dataset-specific
seams (data ingest, output dir, metrics) which are encapsulated by the
adapter in :mod:`BridgeUQ.adapters`.

Examples:

    python -m BridgeUQ.main --dataset brain   --backbone tlrn --batch_size 3
    python -m BridgeUQ.main --dataset cardiac --backbone tlrn --batch_size 8
"""

import os
import sys
import copy
import argparse
import gc

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path (mirrors legacy main.py behavior)
sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from BridgeUQ.networks         import ScalingFactorNetwork
from BridgeUQ.brownian_bridge  import BrownianBridgeLearnedScaling
from BridgeUQ.trainer          import ScalingFactorTrainer
from BridgeUQ.losses           import LTMALoss
from BridgeUQ.adapters         import get_adapter


def _parse_args():
    """Two-stage argparse: first peek at --dataset (with add_help=False so
    `--help` doesn't fire before dataset-specific args register), then let
    the adapter register its own args, then real parse with help."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', choices=['brain', 'cardiac'], required=True)
    pre, _ = pre_parser.parse_known_args()
    adapter = get_adapter(pre.dataset)

    parser = argparse.ArgumentParser(description='BridgeUQ — unified UQ training')
    parser.add_argument('--dataset', choices=['brain', 'cardiac'], required=True,
                        help='Which dataset / pipeline to run.')

    # Shared training args
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Loader batch size (default: brain=3, cardiac=8).')
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='N posterior reverse-SDE trajectories per inner iter for the '
                             'mean-of-loss MC estimator (default: brain=100, cardiac=500).')
    parser.add_argument('--inner_iterations', type=int, default=None,
                        help='K test-time training iterations per minibatch (default: brain=1000, cardiac=500).')
    parser.add_argument('--num_diffusion_steps', type=int, default=None,
                        help='SDE discretization steps (default: brain=7, cardiac=14).')
    parser.add_argument('--num_runs', type=int, default=None,
                        help='Posterior samples drawn per minibatch for UQ (default: 100).')

    parser.add_argument('--loss',
                        choices=['mse', 'l1', 'ncc', 'gncc', 'ssim', 'msssim',
                                 'mi', 'nmi', 'ngf', 'mind', 'deepsim', 'deepsim_mse'],
                        default='mse', help='Similarity loss type (default: mse)')
    parser.add_argument('--ncc_win', type=int, default=9)
    parser.add_argument('--ssim_win', type=int, default=11)
    parser.add_argument('--mi_bins', type=int, default=32)

    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Enable fp16 autocast + GradScaler (SDE coeffs stay fp32).')
    parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false',
                        help='Disable mixed precision (full fp32).')

    parser.add_argument('--out_subdir', type=str, default=None,
                        help="Override the trailing visualization subdir name "
                             "(default: legacy 'uncertainty_brain_sde_v3' or "
                             "'uncertainty_sde_combined_acdc_v3'). "
                             "Pass 'BridgeUQ' for a clean new path.")

    # Now register dataset-specific args (using the adapter we discovered above)
    adapter.add_dataset_args(parser)
    args = parser.parse_args()

    # Apply per-dataset defaults for the shared args we left at None
    hp = adapter.default_hyperparameters()
    if args.batch_size is None:
        args.batch_size = 3 if pre.dataset == 'brain' else 8
    if args.num_train_samples is None:
        args.num_train_samples = 100 if pre.dataset == 'brain' else 500
    if args.inner_iterations is None:
        args.inner_iterations = hp['INNER_ITERATIONS']
    if args.num_diffusion_steps is None:
        args.num_diffusion_steps = hp['NUM_DIFFUSION_STEPS']
    if args.num_runs is None:
        args.num_runs = hp['NUM_RUNS']

    return args, adapter, hp


def main():
    args, adapter, hp = _parse_args()

    # ------------------------------------------------------------------
    # Build loader (and any adapter-private context, e.g. live model)
    # ------------------------------------------------------------------
    print(f"BridgeUQ — dataset = {adapter.name.upper()}, backbone = {args.backbone.upper()}")
    loader, ctx = adapter.build_loader(args)

    # Resolve output dir (cardiac needs ctx['config'] which build_loader produced)
    if adapter.name == 'cardiac':
        output_dir = adapter.resolve_output_dir(ctx, args)
    else:
        output_dir = adapter.get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # ------------------------------------------------------------------
    # Probe data dimensions
    # ------------------------------------------------------------------
    num_time_steps, img_size = adapter.get_dimensions(loader, ctx)
    print(f"Image size: {img_size}    Target time steps: {num_time_steps}    Batches: {len(loader)}")

    # ------------------------------------------------------------------
    # Build network / loss / bridge / trainer  (shared core, no branching)
    # ------------------------------------------------------------------
    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps, img_size=img_size, num_heads=8,
    )
    print(f"  Network parameters: {sum(p.numel() for p in scaling_network.parameters()):,}")

    loss_fn = LTMALoss(
        lambda_sim=hp['LAMBDA_SIM'], lambda_reg=hp['LAMBDA_REG'],
        lambda_scale=hp['LAMBDA_SCALE'], alpha_scale=hp['ALPHA_SCALE'],
        lambda_low_structure=hp['LAMBDA_LOW_STRUCTURE'],
        sim_type=args.loss, ncc_win=args.ncc_win,
        ssim_win=args.ssim_win, mi_bins=args.mi_bins,
    )
    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=args.num_diffusion_steps,
        img_size=img_size, reg_weight=hp['REG_WEIGHT'],
    )
    trainer = ScalingFactorTrainer(
        scaling_network=scaling_network, brownian_bridge=brownian_bridge, loss_fn=loss_fn,
        lr=hp['LEARNING_RATE'], device='cuda', img_size=img_size,
        gamma=hp['GAMMA'], guidance_scale=hp['GUIDANCE_SCALE'],
        use_amp=args.mixed_precision,
    )

    # ------------------------------------------------------------------
    # Per-batch reinit snapshot (test-time training)
    # ------------------------------------------------------------------
    init_network_state   = copy.deepcopy(scaling_network.state_dict())
    init_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())

    print(f"\n{'='*60}\nTRAINING (single pass): {args.inner_iterations} inner iters per minibatch")
    print(f"  Posterior SDE: gamma={hp['GAMMA']}, guidance_scale={hp['GUIDANCE_SCALE']}")
    print(f"  Loss weights:  sim={hp['LAMBDA_SIM']}, scale={hp['LAMBDA_SCALE']}, alpha={hp['ALPHA_SCALE']}")
    print(f"  Train trajectories per inner iter (N): {args.num_train_samples}")
    print(f"{'='*60}")

    losses_log                 = []
    all_subject_ids            = []
    all_sparsification_results = []
    all_rc_results             = []
    all_reg_metrics            = []

    global_subj_idx = 0
    pbar = tqdm(loader, desc="Training (single pass)")

    for batch_idx, batch in enumerate(pbar):
        # --- Reinit network + optimizer ---
        scaling_network.load_state_dict(init_network_state)
        trainer.optimizer.load_state_dict(init_optimizer_state)

        # --- Adapter: extract canonical inputs (and live registration if cardiac) ---
        bd = adapter.process_batch(batch, args, ctx)

        # --- Train K inner iterations ---
        vis_save_path = f"{output_dir}/training_vis/batch_{batch_idx:03d}"
        loss_dict, scaling_factors_bb = trainer.train_step(
            bd['source_img'], bd['target_imgs'], bd['v_series_bb'],
            inner_iterations=args.inner_iterations,
            num_train_samples=args.num_train_samples,
            vis_save_path=vis_save_path,
            vis_subject_idx=None, vis_interval=50,
        )
        losses_log.append(loss_dict)

        # --- Persist s^K ---
        s_K_dir = f"{output_dir}/s_K"
        os.makedirs(s_K_dir, exist_ok=True)
        s_K_path = os.path.join(s_K_dir, f"batch_{batch_idx:03d}.pt")
        scaling_factors_raw = scaling_factors_bb[:, 1].detach().cpu()  # [B, 2, H, W]

        file_paths = bd['file_paths']
        subject_paths_for_save = []
        B = bd['series'].shape[0]
        for i in range(B):
            fp = file_paths[i] if isinstance(file_paths, list) else file_paths
            subject_paths_for_save.append(
                str(fp) if fp is not None else f"subj_{global_subj_idx + i:05d}"
            )
        torch.save({
            'batch_idx':           batch_idx,
            'inner_iterations':    args.inner_iterations,
            'subject_paths':       subject_paths_for_save,
            'scaling_factors_raw': scaling_factors_raw,
            'gamma':               hp['GAMMA'],
            'guidance_scale':      hp['GUIDANCE_SCALE'],
            'num_diffusion_steps': args.num_diffusion_steps,
            'num_cardiac_frames':  len(bd['v_series_bb']),
            'loss_dict':           loss_dict,
        }, s_K_path)
        print(f"  [batch {batch_idx}] s^K saved: {s_K_path}")

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'sim':  f"{loss_dict['similarity']:.4f}",
            'bn':   f"{loss_dict.get('bridge_norm', 0):.4f}",
            'inv_gamma': f"{loss_dict['inv_gamma_prior']:.4f}",
            'lr':   f"{trainer.get_lr():.2e}",
        })

        # --- Run UQ on this minibatch using s^K ---
        scaling_network.eval()
        num_frames_bb = len(bd['v_series_bb'])
        sampled_velocities = {i: [] for i in range(num_frames_bb)}
        for run in range(args.num_runs):
            v_t_list, _ = trainer.compute_brownian_bridge_velocities(
                bd['v_series_bb'], scaling_factors_bb,
                source_img=bd['source_img'], target_imgs=bd['target_imgs'],
            )
            for fi in range(num_frames_bb):
                sampled_velocities[fi].append(v_t_list[fi].detach().clone())
            if (run + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # --- Per-subject metrics ---
        batch_subject_ids            = []
        batch_sparsification_results = []
        batch_rc_results             = []
        batch_reg_metrics            = []
        for subj_idx in range(B):
            sid, spars, rc, reg_metrics = adapter.per_subject_metrics(
                batch_idx, subj_idx, bd, sampled_velocities, img_size,
            )
            batch_subject_ids.append(sid)
            batch_sparsification_results.append(spars)
            batch_rc_results.append(rc)
            batch_reg_metrics.append(reg_metrics)

        all_subject_ids.extend(batch_subject_ids)
        all_sparsification_results.extend(batch_sparsification_results)
        all_rc_results.extend(batch_rc_results)
        all_reg_metrics.extend(batch_reg_metrics)
        global_subj_idx += B
        scaling_network.train()

        # --- Per-batch aggregate save (adapter writes the right format) ---
        adapter.write_per_batch_summary(
            batch_idx, output_dir,
            batch_subject_ids, batch_sparsification_results,
            batch_rc_results, batch_reg_metrics,
        )

        del bd, scaling_factors_bb, sampled_velocities
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Run-level aggregate
    # ------------------------------------------------------------------
    avg_loss = {k: np.mean([l[k] for l in losses_log]) for k in losses_log[0].keys()}
    print(f"\nRun complete: "
          f"Loss={avg_loss['total']:.6f}, "
          f"Sim={avg_loss['similarity']:.6f}, "
          f"BridgeNorm={avg_loss.get('bridge_norm', 0):.6f}, "
          f"InvScale={avg_loss['inv_gamma_prior']:.4f}, "
          f"LR={trainer.get_lr():.2e}")

    adapter.write_run_summary(
        output_dir, all_subject_ids,
        all_sparsification_results, all_rc_results, all_reg_metrics,
    )


if __name__ == "__main__":
    main()
