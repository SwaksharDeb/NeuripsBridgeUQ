"""
Brain Uncertainty Quantification — single-pass K-iteration test-time training.

Mirrors the cardiac uncertainty_sde_combined_acdc_v2 algorithm:
  1. ScalingFactorNetwork takes (image sequence, v_reg, v_current) — 3-branch fusion.
  2. Per minibatch: K=2000 inner iterations of test-time training (backprop every step).
  3. Per minibatch UQ: NUM_RUNS calls to compute_brownian_bridge_velocities.
  4. Per-batch and run-level aggregate (sparsification, RC, registration metrics).

Backbone selection picks the precomputed velocity directory (no live registration).

Run with:
    python -m uncertainty_brain_sde_v3.main --backbone tlrn
    python -m uncertainty_brain_sde_v3.main --backbone ltma --batch_size 4
    python -m uncertainty_brain_sde_v3.main --backbone tm
    python -m uncertainty_brain_sde_v3.main --backbone voxelmorph
"""

import os
import sys
import argparse

# Add project root to path
sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

# Add the parent directory to allow imports when running directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import copy
import torch
import numpy as np
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

# v2 components (3-branch network, K-loop trainer, posterior reverse SDE BB, MLE loss)
from uncertainty_brain_sde_v3.networks         import ScalingFactorNetwork
from uncertainty_brain_sde_v3.brownian_bridge  import BrownianBridgeLearnedScaling
from uncertainty_brain_sde_v3.trainer          import ScalingFactorTrainer
from uncertainty_brain_sde_v3.losses           import LTMALoss
from uncertainty_brain_sde_v3.warping          import warp_image_with_velocity

# Aggregate UQ helpers (sparsification, risk-coverage, registration metrics)
from uncertainty_brain_sde_v3.test_fast import (
    _compute_sparsification_curves,
    _compute_risk_coverage_curves,
    _plot_avg_sparsification,
    _plot_avg_risk_coverage,
    compute_registration_metrics,
)

# v1 — preserved (data + ncc helper)
from uncertainty_brain_sde.data_utils import BrainDataset, compute_ncc_vx_calibration


# Each backbone's precomputed brain .pt directory (BrainDataset-compatible).
# Provenance configs (not loaded here):
#   tlrn:       LightingTemplate_2/2026Experiments/BRAIN/settings/TLRN/basic_brain.yaml
#   ltma:       LightingTemplate_LTMA/2026Experiments/BRAIN/settings/TLMA_TGrad/basic_brain.yaml
#   tm:         TM/2026Experiments/BRAIN/settings/TransMorph/basic_brain.yaml (3D, axial slice z=64)
#   voxelmorph: load_brain_registration_model() → DIGIT_reg
#               (/scratch/swd9tc/4D_brain_v2_multigpu/log_results/models/regae_23_05_2025.pt)
_BACKBONE_DATA_PATHS = {
    'tlrn':       '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/dataset/precomputed_brain_z64_tlrn',
    'ltma':       '/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/dataset/precomputed_brain_z64_ltma',
    'tm':         '/scratch/swd9tc/Uncertanity_quantification/TM/dataset/precomputed_brain_z64_tm',
    'voxelmorph': '/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty/dataset/precomputed_brain_z64',
}

# Folder tag for the centralized output path under LightingTemplate_2/2026Experiments/BRAIN/outputs/<TAG>
_BACKBONE_TAGS = {
    'tlrn':       'TLRN',
    'ltma':       'TLMA_TGrad',
    'tm':         'TransMorph',
    'voxelmorph': 'VoxelMorph',
}


def collate_brain_batch(batch):
    """Custom collate function to handle list of velocity tensors."""
    series = torch.stack([b['series'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch])
    filenames = [b['filename'] for b in batch]

    num_frames = len(batch[0]['v_series'])
    v_series = []
    for t in range(num_frames):
        v_t = torch.stack([b['v_series'][t] for b in batch])
        v_series.append(v_t)

    return {
        'series': series,
        'v_series': v_series,
        'label': labels,
        'filename': filenames,
    }


def main():
    parser = argparse.ArgumentParser(description='Brain UQ v2 (K-iteration test-time training)')
    parser.add_argument('--backbone',
                        choices=list(_BACKBONE_DATA_PATHS.keys()),
                        default='tlrn',
                        help='Pretrained registration backbone (selects precomputed data path)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override the auto-resolved precomputed data path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override the auto-resolved output directory')
    parser.add_argument('--use_train', action='store_true', default=True,
                        help='Iterate over train+val+test (ConcatDataset). Default: test only.')
    parser.add_argument('--batch_size', type=int, default=3,   #57
                        help='Override the loader batch size (default: 4)')
    parser.add_argument('--loss',
                        choices=['mse', 'l1', 'ncc', 'gncc', 'ssim', 'msssim',
                                 'mi', 'nmi', 'ngf', 'mind', 'deepsim', 'deepsim_mse'],
                        default='mse',
                        help='Similarity loss type (default: mse)')
    parser.add_argument('--ncc_win', type=int, default=9)
    parser.add_argument('--ssim_win', type=int, default=11)
    parser.add_argument('--mi_bins', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='mni88',
                        choices=['mni88', 'full'],
                        help="Dataset selection: 'mni88' for MNI_88.csv only, "
                             "'full' for MNI_88.csv + MNI_data_DX_4f.csv (default: mni88)")
    parser.add_argument('--num_train_samples', type=int, default=100,
                        help='N posterior reverse-SDE trajectories per inner iter; '
                             'their mean velocity feeds the loss and v_current^{k+1}. '
                             'Set to 1 to recover the v2 single-sample baseline.')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Enable fp16 autocast + GradScaler for the inner loop; '
                             'SDE coefficients stay fp32. Roughly halves activation '
                             'memory of the per-sample SDE graph (default: True).')
    parser.add_argument('--no_mixed_precision', dest='mixed_precision',
                        action='store_false',
                        help='Disable mixed precision (full fp32).')
    args = parser.parse_args()

    # ==========================================================================
    # Parameters (brain-specific, matching brain pipeline)
    # ==========================================================================
    NUM_DIFFUSION_STEPS = 7
    NUM_RUNS = 100
    LEARNING_RATE = 1e-4
    INNER_ITERATIONS = 1000 #2000     # K iterations per minibatch (test-time training)

    # Loss weights
    LAMBDA_SIM = 1.0
    LAMBDA_REG = 0.0
    LAMBDA_SCALE = 3
    ALPHA_SCALE = 0.0001
    LAMBDA_LOW_STRUCTURE = 0.0

    # Posterior SDE parameters
    GAMMA = 0.0001 / 2
    GUIDANCE_SCALE = 1.0
    REG_WEIGHT = 0.1

    # ==========================================================================
    # Resolve backbone-specific data path and output directory
    # ==========================================================================
    data_path = args.data_path or _BACKBONE_DATA_PATHS[args.backbone]
    btag = _BACKBONE_TAGS[args.backbone]
    base = "basic_brain" if args.loss == 'mse' else f"basic_brain_{args.loss}"
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = (
            "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/"
            f"2026Experiments/BRAIN/outputs/{btag}/{base}/visualization/uncertainty_brain_sde_v3"
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using backbone: {args.backbone.upper()}")
    print(f"Data path:      {data_path}")
    print(f"Output dir:     {OUTPUT_DIR}")
    if args.loss != 'mse':
        print(f"Loss:           {args.loss.upper()}")

    # ==========================================================================
    # Load data (BrainDataset reads precomputed .pt files generated offline)
    # ==========================================================================
    if not os.path.isdir(data_path) or not any(
        f.endswith('.pt') for d, _, fs in os.walk(data_path) for f in fs
    ):
        raise FileNotFoundError(
            f"Precomputed brain data not found at: {data_path}\n"
            f"Run that backbone's precompute_velocities.py first."
        )

    print(f"\nLoading precomputed brain data (dataset={args.dataset})...")
    train_dataset = BrainDataset(data_path, split='all', mode='train', dataset=args.dataset)
    val_dataset   = BrainDataset(data_path, split='all', mode='val',   dataset=args.dataset)
    test_dataset  = BrainDataset(data_path, split='all', mode='test',  dataset=args.dataset)

    if args.use_train:
        active_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
        print(f"Iterating over train+val+test = {len(active_dataset)} samples")
    else:
        active_dataset = test_dataset
        print(f"Iterating over test = {len(active_dataset)} samples")

    data_loader = DataLoader(
        active_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_brain_batch,
        num_workers=2,
        pin_memory=True,
    )

    # Determine dimensions
    print("\nDetermining data dimensions from first batch...")
    first_batch = next(iter(data_loader))
    first_series   = first_batch['series']
    first_v_series = first_batch['v_series']

    num_time_steps = len(first_v_series) - 1   # v_series has zero endpoints
    img_size = first_series.shape[-1]
    print(f"  Image size:           {img_size}")
    print(f"  Frames (cyclic):      {len(first_v_series)}")
    print(f"  Target time steps:    {num_time_steps}")
    print(f"  Number of batches:    {len(data_loader)}")
    del first_series, first_v_series

    # ==========================================================================
    # Initialize components
    # ==========================================================================
    print("\nInitializing 3-branch ScalingFactorNetwork...")
    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps,
        img_size=img_size,
        num_heads=8,
    )
    print(f"  Parameters: {sum(p.numel() for p in scaling_network.parameters()):,}")

    loss_fn = LTMALoss(
        lambda_sim=LAMBDA_SIM,
        lambda_reg=LAMBDA_REG,
        lambda_scale=LAMBDA_SCALE,
        alpha_scale=ALPHA_SCALE,
        lambda_low_structure=LAMBDA_LOW_STRUCTURE,
        sim_type=args.loss,
        ncc_win=args.ncc_win,
        ssim_win=args.ssim_win,
        mi_bins=args.mi_bins,
    )

    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        img_size=img_size,
        reg_weight=REG_WEIGHT,
    )

    trainer = ScalingFactorTrainer(
        scaling_network=scaling_network,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        lr=LEARNING_RATE,
        device='cuda',
        img_size=img_size,
        gamma=GAMMA,
        guidance_scale=GUIDANCE_SCALE,
        use_amp=args.mixed_precision,
    )

    # ==========================================================================
    # TRAINING (single pass — K inner iterations per minibatch)
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"BRAIN UQ v2 — single pass, {INNER_ITERATIONS} inner iters per minibatch")
    print(f"  Posterior SDE: gamma={GAMMA}, guidance_scale={GUIDANCE_SCALE}")
    print(f"  Loss weights:  sim={LAMBDA_SIM}, scale={LAMBDA_SCALE}, alpha={ALPHA_SCALE}")
    print(f"  Network input: COMBINED (images + v_reg + v_current)")
    print(f"{'='*60}")

    # Per-batch reinit snapshot (test-time training: no weight transfer between minibatches)
    init_network_state   = copy.deepcopy(scaling_network.state_dict())
    init_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())

    # Run-level accumulators
    losses_log                 = []
    all_subject_ids            = []
    all_sparsification_results = []
    all_rc_results             = []
    all_reg_metrics            = []

    # Running global subject index used as the s^K key when filenames are not
    # available. shuffle=False loader → deterministic regardless of batch_size,
    # so test_fast.py / test.py can iterate the same dataset and match keys.
    global_subj_idx = 0

    pbar = tqdm(data_loader, desc="Training (single pass)")
    for batch_idx, batch in enumerate(pbar):
        # Reinitialize scaling network + optimizer for this minibatch
        scaling_network.load_state_dict(init_network_state)
        trainer.optimizer.load_state_dict(init_optimizer_state)

        series   = batch['series'].cuda()
        v_series = [v.cuda() for v in batch['v_series']]   # already has zero endpoints
        filenames = batch['filename']

        source_img  = series[:, 0:1]
        target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]
        v_series_bb = v_series   # zero endpoint at t=0 already present

        # ----- TRAINING: K inner iterations -----
        vis_save_path = f"{OUTPUT_DIR}/training_vis/batch_{batch_idx:03d}"
        loss_dict, scaling_factors_bb = trainer.train_step(
            source_img, target_imgs, v_series_bb,
            inner_iterations=INNER_ITERATIONS,
            num_train_samples=args.num_train_samples,
            vis_save_path=vis_save_path,
            vis_subject_idx=None,
            vis_interval=50,
        )
        losses_log.append(loss_dict)

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'sim':  f"{loss_dict['similarity']:.4f}",
            'bn':   f"{loss_dict.get('bridge_norm', 0):.4f}",
            'inv_gamma': f"{loss_dict['inv_gamma_prior']:.4f}",
            'lr':   f"{trainer.get_lr():.2e}",
        })

        # Persist s^K (the K-th-iteration scaling factors used for UQ sampling
        # below) so test_fast.py / test.py can reproduce main.py's posterior
        # reverse SDE samples without redoing test-time training.
        # scaling_factors_bb has shape [B, T+1, 2, H, W]; index 0 is a zero
        # frame, indices 1..T are scaling_factors_raw broadcast across time
        # (see trainer.py:147-149). So scaling_factors_bb[:, 1] is the raw
        # [B, 2, H, W] tensor we need.
        s_K_dir = f"{OUTPUT_DIR}/s_K"
        os.makedirs(s_K_dir, exist_ok=True)
        s_K_path = os.path.join(s_K_dir, f"batch_{batch_idx:03d}.pt")

        scaling_factors_raw = scaling_factors_bb[:, 1].detach().cpu()  # [B, 2, H, W]

        subject_paths_for_save = []
        for i in range(series.shape[0]):
            fp = filenames[i] if i < len(filenames) else None
            subject_paths_for_save.append(
                str(fp) if fp is not None else f"subj_{global_subj_idx + i:05d}"
            )

        torch.save({
            'batch_idx': batch_idx,
            'inner_iterations': INNER_ITERATIONS,
            'subject_paths': subject_paths_for_save,
            'scaling_factors_raw': scaling_factors_raw,
            'gamma': GAMMA,
            'guidance_scale': GUIDANCE_SCALE,
            'num_diffusion_steps': NUM_DIFFUSION_STEPS,
            'num_cardiac_frames': len(v_series_bb),
            'loss_dict': loss_dict,
        }, s_K_path)
        print(f"  [batch {batch_idx}] s^K saved: {s_K_path}")

        # ----- UQ on this same minibatch using s^K -----
        # NOTE: cannot wrap in torch.no_grad() — run_posterior_reverse_process
        # internally calls torch.autograd.grad. All inputs are detached, so no
        # autograd graph accumulates.
        scaling_network.eval()
        num_frames_bb = len(v_series_bb)
        sampled_velocities = {i: [] for i in range(num_frames_bb)}
        for run in range(NUM_RUNS):
            v_t_list, _ = trainer.compute_brownian_bridge_velocities(
                v_series_bb, scaling_factors_bb,
                source_img=source_img, target_imgs=target_imgs,
            )
            for fi in range(num_frames_bb):
                sampled_velocities[fi].append(v_t_list[fi].detach().clone())
            if (run + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # ----- Per-subject curve collection -----
        batch_subject_ids            = []
        batch_sparsification_results = []
        batch_rc_results             = []
        batch_reg_metrics            = []
        for subj_idx in range(series.shape[0]):
            fname = filenames[subj_idx] if subj_idx < len(filenames) else f"subj{subj_idx}"
            subject_id = f"batch{batch_idx}_{fname}"

            spars = _compute_sparsification_curves(
                sampled_velocities, v_series_bb, series,
                source_img, subj_idx, img_size,
            )
            rc = _compute_risk_coverage_curves(
                sampled_velocities, v_series_bb, series,
                source_img, subj_idx, img_size,
            )
            reg_metrics = compute_registration_metrics(
                source_img, target_imgs, v_series_bb,
                sampled_velocities, subj_idx,
                inshape=(img_size, img_size),
                filename=fname,
            )

            batch_subject_ids.append(subject_id)
            batch_sparsification_results.append(spars)
            batch_rc_results.append(rc)
            batch_reg_metrics.append(reg_metrics)

        all_subject_ids.extend(batch_subject_ids)
        all_sparsification_results.extend(batch_sparsification_results)
        all_rc_results.extend(batch_rc_results)
        all_reg_metrics.extend(batch_reg_metrics)
        global_subj_idx += series.shape[0]
        scaling_network.train()

        # ----- Per-batch aggregate save -----
        batch_summary_folder = f"{OUTPUT_DIR}/aggregate/batch_{batch_idx:03d}"
        os.makedirs(batch_summary_folder, exist_ok=True)

        if batch_sparsification_results:
            _plot_avg_sparsification(batch_sparsification_results, batch_subject_ids, batch_summary_folder)
        if batch_rc_results:
            _plot_avg_risk_coverage(batch_rc_results, batch_subject_ids, batch_summary_folder)

        if batch_sparsification_results and batch_rc_results:
            batch_frame_indices = sorted(
                {fi for sr in batch_sparsification_results for fi in sr.keys()}
                | {fi for rr in batch_rc_results for fi in rr.keys()}
            )
            blines = ["=" * 70,
                      f"nAURC & nAUSE PER TIME FRAME (batch {batch_idx})".center(70),
                      "=" * 70,
                      f"  {'Frame':>5}  |  {'nAURC mean':>10}  {'nAURC std':>10}  |  "
                      f"{'nAUSE mean':>10}  {'nAUSE std':>10}  |  {'N':>4}",
                      "  " + "-" * 65]
            for fi in batch_frame_indices:
                naurc_vals = [rr[fi]['aurc_norm'] for rr in batch_rc_results if fi in rr]
                nause_vals = [sr[fi]['ause_norm'] for sr in batch_sparsification_results if fi in sr]
                n = len(naurc_vals)
                blines.append(
                    f"  {fi:>5}  |  {np.mean(naurc_vals):>10.4f}  {np.std(naurc_vals):>10.4f}  |  "
                    f"{np.mean(nause_vals):>10.4f}  {np.std(nause_vals):>10.4f}  |  {n:>4}"
                )
            bsubj_mean_naurc = [np.mean([rr[fi]['aurc_norm'] for fi in rr]) for rr in batch_rc_results if rr]
            bsubj_mean_nause = [np.mean([sr[fi]['ause_norm'] for fi in sr]) for sr in batch_sparsification_results if sr]
            blines += ["  " + "-" * 65,
                       f"  {'All':>5}  |  {np.mean(bsubj_mean_naurc):>10.4f}  {np.std(bsubj_mean_naurc):>10.4f}  |  "
                       f"{np.mean(bsubj_mean_nause):>10.4f}  {np.std(bsubj_mean_nause):>10.4f}  |  {len(batch_rc_results):>4}",
                       "=" * 70]
            with open(os.path.join(batch_summary_folder, "per_frame_nAURC_nAUSE_summary.txt"), "w") as f:
                f.write("\n".join(blines) + "\n")

        if batch_reg_metrics:
            b_reg_rmse   = [m['registration']['avg_rmse']         for m in batch_reg_metrics]
            b_reg_dice   = [m['registration']['avg_dice']         for m in batch_reg_metrics if m['registration']['avg_dice']  is not None]
            b_reg_negjac = [m['registration']['avg_neg_jac_pct']  for m in batch_reg_metrics]
            b_mv_rmse    = [m['mean_velocity']['avg_rmse']        for m in batch_reg_metrics]
            b_mv_dice    = [m['mean_velocity']['avg_dice']        for m in batch_reg_metrics if m['mean_velocity']['avg_dice'] is not None]
            b_mv_negjac  = [m['mean_velocity']['avg_neg_jac_pct'] for m in batch_reg_metrics]

            b_save_data = {
                'reg_rmse':        np.array(b_reg_rmse),
                'reg_neg_jac_pct': np.array(b_reg_negjac),
                'mv_rmse':         np.array(b_mv_rmse),
                'mv_neg_jac_pct':  np.array(b_mv_negjac),
            }
            if b_reg_dice: b_save_data['reg_dice'] = np.array(b_reg_dice)
            if b_mv_dice:  b_save_data['mv_dice']  = np.array(b_mv_dice)
            np.savez(os.path.join(batch_summary_folder, 'registration_metrics.npz'), **b_save_data)

            with open(os.path.join(batch_summary_folder, 'registration_metrics_summary.txt'), "w") as f:
                f.write(f"Registration metrics — batch {batch_idx} ({len(batch_reg_metrics)} subjects)\n")
                f.write(f"Pretrained Registration:\n")
                f.write(f"  RMSE: {np.mean(b_reg_rmse):.6f} +/- {np.std(b_reg_rmse):.6f}\n")
                if b_reg_dice: f.write(f"  Dice: {np.mean(b_reg_dice):.6f} +/- {np.std(b_reg_dice):.6f}\n")
                f.write(f"  Neg Jac %: {np.mean(b_reg_negjac):.4f} +/- {np.std(b_reg_negjac):.4f}\n")
                f.write(f"BridgeUQ Mean Velocity:\n")
                f.write(f"  RMSE: {np.mean(b_mv_rmse):.6f} +/- {np.std(b_mv_rmse):.6f}\n")
                if b_mv_dice: f.write(f"  Dice: {np.mean(b_mv_dice):.6f} +/- {np.std(b_mv_dice):.6f}\n")
                f.write(f"  Neg Jac %: {np.mean(b_mv_negjac):.4f} +/- {np.std(b_mv_negjac):.4f}\n")

        print(f"  [batch {batch_idx}] per-batch results saved to: {batch_summary_folder}")

        del series, v_series, source_img, target_imgs, v_series_bb
        del scaling_factors_bb, sampled_velocities
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================================================
    # End-of-run aggregate save
    # ==========================================================================
    avg_loss = {key: np.mean([l[key] for l in losses_log]) for key in losses_log[0].keys()}
    print(f"\nRun complete: "
          f"Loss={avg_loss['total']:.6f}, "
          f"Sim={avg_loss['similarity']:.6f}, "
          f"BridgeNorm={avg_loss.get('bridge_norm', 0):.6f}, "
          f"InvScale={avg_loss['inv_gamma_prior']:.4f}, "
          f"LR={trainer.get_lr():.2e}")

    summary_folder = f"{OUTPUT_DIR}/aggregate"
    os.makedirs(summary_folder, exist_ok=True)

    if all_sparsification_results:
        _plot_avg_sparsification(all_sparsification_results, all_subject_ids, summary_folder)
    if all_rc_results:
        _plot_avg_risk_coverage(all_rc_results, all_subject_ids, summary_folder)

    if all_sparsification_results and all_rc_results:
        all_frame_indices = sorted(
            {fi for sr in all_sparsification_results for fi in sr.keys()}
            | {fi for rr in all_rc_results for fi in rr.keys()}
        )
        lines = ["=" * 70,
                 "nAURC & nAUSE PER TIME FRAME (averaged across subjects)".center(70),
                 "=" * 70,
                 f"  {'Frame':>5}  |  {'nAURC mean':>10}  {'nAURC std':>10}  |  "
                 f"{'nAUSE mean':>10}  {'nAUSE std':>10}  |  {'N':>4}",
                 "  " + "-" * 65]
        for fi in all_frame_indices:
            naurc_vals = [rr[fi]['aurc_norm'] for rr in all_rc_results if fi in rr]
            nause_vals = [sr[fi]['ause_norm'] for sr in all_sparsification_results if fi in sr]
            n = len(naurc_vals)
            lines.append(
                f"  {fi:>5}  |  {np.mean(naurc_vals):>10.4f}  {np.std(naurc_vals):>10.4f}  |  "
                f"{np.mean(nause_vals):>10.4f}  {np.std(nause_vals):>10.4f}  |  {n:>4}"
            )
        subj_mean_naurc = [np.mean([rr[fi]['aurc_norm'] for fi in rr]) for rr in all_rc_results if rr]
        subj_mean_nause = [np.mean([sr[fi]['ause_norm'] for fi in sr]) for sr in all_sparsification_results if sr]
        lines += ["  " + "-" * 65,
                  f"  {'All':>5}  |  {np.mean(subj_mean_naurc):>10.4f}  {np.std(subj_mean_naurc):>10.4f}  |  "
                  f"{np.mean(subj_mean_nause):>10.4f}  {np.std(subj_mean_nause):>10.4f}  |  {len(all_rc_results):>4}",
                  "=" * 70]
        with open(os.path.join(summary_folder, "per_frame_nAURC_nAUSE_summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    if all_reg_metrics:
        reg_rmse_all   = [m['registration']['avg_rmse']         for m in all_reg_metrics]
        reg_dice_all   = [m['registration']['avg_dice']         for m in all_reg_metrics if m['registration']['avg_dice']  is not None]
        reg_negjac_all = [m['registration']['avg_neg_jac_pct']  for m in all_reg_metrics]
        mv_rmse_all    = [m['mean_velocity']['avg_rmse']        for m in all_reg_metrics]
        mv_dice_all    = [m['mean_velocity']['avg_dice']        for m in all_reg_metrics if m['mean_velocity']['avg_dice'] is not None]
        mv_negjac_all  = [m['mean_velocity']['avg_neg_jac_pct'] for m in all_reg_metrics]

        save_data = {
            'reg_rmse':        np.array(reg_rmse_all),
            'reg_neg_jac_pct': np.array(reg_negjac_all),
            'mv_rmse':         np.array(mv_rmse_all),
            'mv_neg_jac_pct':  np.array(mv_negjac_all),
        }
        if reg_dice_all: save_data['reg_dice'] = np.array(reg_dice_all)
        if mv_dice_all:  save_data['mv_dice']  = np.array(mv_dice_all)
        np.savez(os.path.join(summary_folder, 'registration_metrics.npz'), **save_data)

        with open(os.path.join(summary_folder, 'registration_metrics_summary.txt'), "w") as f:
            f.write(f"Registration metrics averaged over {len(all_reg_metrics)} subjects\n")
            f.write(f"Pretrained Registration:\n")
            f.write(f"  RMSE: {np.mean(reg_rmse_all):.6f} +/- {np.std(reg_rmse_all):.6f}\n")
            if reg_dice_all: f.write(f"  Dice: {np.mean(reg_dice_all):.6f} +/- {np.std(reg_dice_all):.6f}\n")
            f.write(f"  Neg Jac %: {np.mean(reg_negjac_all):.4f} +/- {np.std(reg_negjac_all):.4f}\n")
            f.write(f"BridgeUQ Mean Velocity:\n")
            f.write(f"  RMSE: {np.mean(mv_rmse_all):.6f} +/- {np.std(mv_rmse_all):.6f}\n")
            if mv_dice_all: f.write(f"  Dice: {np.mean(mv_dice_all):.6f} +/- {np.std(mv_dice_all):.6f}\n")
            f.write(f"  Neg Jac %: {np.mean(mv_negjac_all):.4f} +/- {np.std(mv_negjac_all):.4f}\n")

    # No final scaling_network.pth save: test_fast.py / test.py read s^K
    # directly from s_K/batch_*.pt and never need the network weights.


if __name__ == "__main__":
    main()
