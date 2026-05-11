"""
Fast aggregate test for Brain SDE UQ -- PULPo-style regime switch.

Mirrors uncertainty_brain_sde/test_fast.py but ports the AUSE/AURC routing
from pulbo_brain_v2/test.py:

  * Position-space (velocity) uncertainty applies a gradient mask
    (|grad(source)| > threshold) on the source image and switches to
    regime B (no inversion) when the mask is active. Falls back to
    regime A (inverted ranking, equivalent to 1/(u+eps)) when the mask
    is disabled or empty.
  * Image-space uncertainty (var(m o phi)) is always regime B with no mask.

Run with:
    python -m uncertainty_brain_sde.test_fast_pulpo
    python -m uncertainty_brain_sde.test_fast_pulpo --uncertainty_space velocity
    python -m uncertainty_brain_sde.test_fast_pulpo --uncertainty_space velocity --no_mask
    python -m uncertainty_brain_sde.test_fast_pulpo --gradient_threshold 0.01
"""

import os
import sys

sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import glob as globmod
import re
import torch
import numpy as np
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

from uncertainty_brain_sde.networks import ScalingFactorNetwork
from uncertainty_brain_sde.losses import LTMALoss
from uncertainty_brain_sde.brownian_bridge import BrownianBridgeLearnedScaling
from uncertainty_brain_sde.trainer import ScalingFactorTrainer
from uncertainty_brain_sde.warping import warp_image_with_velocity
from uncertainty_brain_sde.test_fast import (
    FRAME_LABELS,
    _plot_avg_sparsification,
    _plot_avg_risk_coverage,
    compute_registration_metrics,
)


# =========================================================================
# Gradient mask -- ports pulbo_brain_v2/test.py::_image_gradient_mask
# =========================================================================

def _image_gradient_mask(img_np, threshold=0.0):
    """Boolean mask of pixels with |grad(img_np)| > threshold.

    Excludes flat-tissue pixels from position-space uncertainty scoring:
    in those pixels the aperture problem gives var(v) without an intensity
    consequence, decoupling the ranking from pixel_error.
    """
    gy, gx = np.gradient(img_np.astype(np.float32))
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return grad_mag > threshold


# =========================================================================
# Sparsification curves with invert + mask (PULPo signature)
# =========================================================================

def _compute_sparsification_curves(sampled_velocities, v_reg_list, series,
                                   source_img, subject_idx, img_size,
                                   uncertainty_space='velocity',
                                   gradient_threshold=0.0,
                                   use_mask=True):
    """Per-subject sparsification curves with PULPo-style regime switch.

    For uncertainty_space='velocity':
        - if use_mask and mask has >= 2 pixels: regime B (invert=False, masked)
        - else:                                  regime A (invert=True,  unmasked)

    For uncertainty_space='image':
        - always regime B (invert=False, unmasked).
    """
    num_frames = len(v_reg_list)
    fractions = np.linspace(0, 0.9, 19)
    inshape = (img_size, img_size)

    frame_results = {}

    # Source-image gradient mask is shared across frames (depends only on source).
    mask_bool = None
    if uncertainty_space == 'velocity' and use_mask:
        source_np = source_img[subject_idx, 0].cpu().numpy()
        candidate = _image_gradient_mask(source_np, threshold=gradient_threshold)
        if candidate.sum() >= 2:
            mask_bool = candidate

    if uncertainty_space == 'image':
        invert = False
    elif mask_bool is not None:
        invert = False  # regime B
    else:
        invert = True   # regime A

    for frame_idx in range(1, num_frames - 1):
        if frame_idx not in sampled_velocities or len(sampled_velocities[frame_idx]) == 0:
            continue

        if uncertainty_space == 'image':
            warped_samples = []
            for v in sampled_velocities[frame_idx]:
                v_sub = v[subject_idx:subject_idx+1]
                warped = warp_image_with_velocity(
                    source_img[subject_idx:subject_idx+1], v_sub, inshape=inshape
                )
                warped_samples.append(warped[0, 0].cpu().numpy())
            warped_samples = np.stack(warped_samples, axis=0)
            uncertainty = np.var(warped_samples, axis=0)
        else:
            velocities_at_t = np.stack(
                [v[subject_idx].cpu().numpy() for v in sampled_velocities[frame_idx]], axis=0
            )
            variance = np.var(velocities_at_t, axis=0)
            uncertainty = np.sqrt(variance[0] + variance[1])

        if np.max(uncertainty) < 1e-8:
            continue

        target = series[subject_idx, frame_idx].cpu().numpy()
        v_reg = v_reg_list[frame_idx][subject_idx:subject_idx+1]
        deformed = warp_image_with_velocity(
            source_img[subject_idx:subject_idx+1], v_reg, inshape=inshape
        )
        deformed = deformed[0, 0].cpu().numpy()
        pixel_error = (deformed - target) ** 2

        if mask_bool is not None:
            m = mask_bool.ravel()
            unc_flat = uncertainty.flatten()[m]
            err_flat = pixel_error.flatten()[m]
        else:
            unc_flat = uncertainty.flatten()
            err_flat = pixel_error.flatten()
        n_pixels = len(unc_flat)
        if n_pixels < 2:
            continue

        unc_order = np.argsort(unc_flat) if invert else np.argsort(-unc_flat)
        oracle_order = np.argsort(-err_flat)

        unc_curve = np.zeros(len(fractions))
        oracle_curve = np.zeros(len(fractions))
        random_curve = np.zeros(len(fractions))
        overall_mean = float(np.mean(err_flat))

        for j, frac in enumerate(fractions):
            n_remove = int(frac * n_pixels)
            n_remain = n_pixels - n_remove
            if n_remain == 0:
                unc_curve[j] = oracle_curve[j] = random_curve[j] = 0.0
            else:
                unc_curve[j] = np.mean(err_flat[unc_order[n_remove:]])
                oracle_curve[j] = np.mean(err_flat[oracle_order[n_remove:]])
                random_curve[j] = overall_mean

        random_curve[-1] = 0.0

        ause = np.trapz(unc_curve - oracle_curve, fractions)
        random_area = np.trapz(random_curve - oracle_curve, fractions)
        ause_norm = ause / random_area if random_area > 1e-12 else 0.0

        frame_results[frame_idx] = {
            'fractions': fractions,
            'uncertainty_curve': unc_curve,
            'oracle_curve': oracle_curve,
            'random_curve': random_curve,
            'ause': ause,
            'ause_norm': ause_norm,
        }

    return frame_results


# =========================================================================
# Risk-coverage curves with invert + mask (PULPo signature)
# =========================================================================

def _compute_risk_coverage_curves(sampled_velocities, v_reg_list, series,
                                  source_img, subject_idx, img_size,
                                  uncertainty_space='velocity',
                                  gradient_threshold=0.0,
                                  use_mask=True):
    """Per-subject risk-coverage curves with PULPo-style regime switch."""
    num_frames = len(v_reg_list)
    coverages = np.linspace(0.15, 1, 86)
    inshape = (img_size, img_size)

    frame_results = {}

    mask_bool = None
    if uncertainty_space == 'velocity' and use_mask:
        source_np = source_img[subject_idx, 0].cpu().numpy()
        candidate = _image_gradient_mask(source_np, threshold=gradient_threshold)
        if candidate.sum() >= 2:
            mask_bool = candidate

    if uncertainty_space == 'image':
        invert = False
    elif mask_bool is not None:
        invert = False
    else:
        invert = True

    for frame_idx in range(1, num_frames - 1):
        if frame_idx not in sampled_velocities or len(sampled_velocities[frame_idx]) == 0:
            continue

        if uncertainty_space == 'image':
            warped_samples = []
            for v in sampled_velocities[frame_idx]:
                v_sub = v[subject_idx:subject_idx+1]
                warped = warp_image_with_velocity(
                    source_img[subject_idx:subject_idx+1], v_sub, inshape=inshape
                )
                warped_samples.append(warped[0, 0].cpu().numpy())
            warped_samples = np.stack(warped_samples, axis=0)
            uncertainty = np.var(warped_samples, axis=0)
        else:
            velocities_at_t = np.stack(
                [v[subject_idx].cpu().numpy() for v in sampled_velocities[frame_idx]], axis=0
            )
            variance = np.var(velocities_at_t, axis=0)
            uncertainty = np.sqrt(variance[0] + variance[1])

        if np.max(uncertainty) < 1e-8:
            continue

        target = series[subject_idx, frame_idx].cpu().numpy()
        v_reg = v_reg_list[frame_idx][subject_idx:subject_idx+1]
        deformed = warp_image_with_velocity(
            source_img[subject_idx:subject_idx+1], v_reg, inshape=inshape
        )
        deformed = deformed[0, 0].cpu().numpy()
        pixel_error = (deformed - target) ** 2

        if mask_bool is not None:
            m = mask_bool.ravel()
            unc_flat = uncertainty.flatten()[m]
            err_flat = pixel_error.flatten()[m]
        else:
            unc_flat = uncertainty.flatten()
            err_flat = pixel_error.flatten()
        n_pixels = len(unc_flat)
        if n_pixels < 2:
            continue

        unc_asc_order = np.argsort(-unc_flat) if invert else np.argsort(unc_flat)
        oracle_asc_order = np.argsort(err_flat)

        unc_rc_curve = np.zeros(len(coverages))
        oracle_rc_curve = np.zeros(len(coverages))
        random_rc_curve = np.zeros(len(coverages))
        overall_mean = float(np.mean(err_flat))

        for j, cov in enumerate(coverages):
            n_retain = max(int(cov * n_pixels), 1)
            unc_rc_curve[j] = np.mean(err_flat[unc_asc_order[:n_retain]])
            oracle_rc_curve[j] = np.mean(err_flat[oracle_asc_order[:n_retain]])
            random_rc_curve[j] = overall_mean

        aurc = np.trapz(unc_rc_curve, coverages)
        oracle_aurc = np.trapz(oracle_rc_curve, coverages)
        random_aurc = np.trapz(random_rc_curve, coverages)
        aurc_norm = (aurc - oracle_aurc) / (random_aurc - oracle_aurc) \
            if (random_aurc - oracle_aurc) > 1e-12 else 0.0

        frame_results[frame_idx] = {
            'coverages': coverages,
            'unc_rc_curve': unc_rc_curve,
            'oracle_rc_curve': oracle_rc_curve,
            'random_rc_curve': random_rc_curve,
            'aurc': aurc,
            'aurc_norm': aurc_norm,
        }

    return frame_results


# =========================================================================
# Main
# =========================================================================

def test_fast_pulpo():
    parser = argparse.ArgumentParser(
        description="Fast Brain SDE Test (TLRN backbone) -- PULPo-style regime switch"
    )
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of UQ runs (default: 100)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str,
                        default='2026Experiments/BRAIN/outputs/TLRN/basic_brain_mse/visualization/uncertainty_brain_sde/checkpoints/checkpoint_epoch_500.pth',
                        help="Path to checkpoint file. If None, auto-detect latest.")
    parser.add_argument("--use_train", action="store_true",
                        help="Use training data instead of test data")
    parser.add_argument("--compute_reg_metrics", action="store_true",
                        help="Also compute registration metrics (RMSE, Neg Jac %%)")
    parser.add_argument("--data_path", type=str,
                        default='/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/dataset/precomputed_brain_z64_tlrn',
                        help="Path to precomputed brain data")
    parser.add_argument("--dataset", type=str, default="mni88",
                        choices=["mni88", "full"],
                        help="Dataset selection: 'mni88' for MNI_88.csv only, "
                             "'full' for MNI_88.csv + MNI_data_DX_4f.csv (default: mni88)")
    parser.add_argument("--uncertainty_space", type=str, default="image",
                        choices=["velocity", "image"],
                        help="'velocity' = sqrt(var(v_x)+var(v_y)) over samples; "
                             "'image' = var(m o phi) over sampled warps.")
    parser.add_argument("--gradient_threshold", type=float, default=0.0,
                        help="Source-image gradient threshold for the PULPo mask "
                             "(only applies to uncertainty_space='velocity'). "
                             "Pixels with |grad(source)| > threshold are kept; "
                             "flat-tissue pixels are excluded. Default 0.0.")
    parser.add_argument("--no_mask", action="store_true",
                        help="Disable the gradient mask entirely. With this flag, "
                             "velocity-space uses regime A (inverted ranking, no mask) "
                             "-- the test_fast.py default.")
    args = parser.parse_args()

    NUM_DIFFUSION_STEPS = 7
    BATCH_SIZE = 4
    NUM_RUNS = args.num_runs
    use_mask = not args.no_mask

    LAMBDA_SCALE = 3
    ALPHA_SCALE = 0.0001
    GAMMA = 0.0001 / 2
    GUIDANCE_SCALE = 1.0

    OUTPUT_DIR = '2026Experiments/BRAIN/outputs/TLRN/basic_brain_mse/visualization/uncertainty_brain_sde/test_results'

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading precomputed brain data (dataset={args.dataset})...")
    from uncertainty_brain_sde.data_utils import BrainDataset

    def collate_brain_batch(batch):
        series = torch.stack([b['series'] for b in batch])
        labels = torch.tensor([b['label'] for b in batch])
        filenames = [b['filename'] for b in batch]
        ptids = [b['ptid'] for b in batch]
        age_lists = [b['age_list'] for b in batch]
        num_frames = len(batch[0]['v_series'])
        v_series = [torch.stack([b['v_series'][t] for b in batch]) for t in range(num_frames)]
        return {'series': series, 'v_series': v_series, 'label': labels,
                'filename': filenames, 'ptid': ptids, 'age_list': age_lists}

    train_dataset = BrainDataset(args.data_path, split='all', mode='train', dataset=args.dataset)
    val_dataset = BrainDataset(args.data_path, split='all', mode='val', dataset=args.dataset)
    test_dataset = BrainDataset(args.data_path, split='all', mode='test', dataset=args.dataset)

    eval_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    data_split = "ALL (train+val+test)"

    data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_brain_batch, num_workers=2, pin_memory=True)
    print(f"Using {data_split} set ({len(eval_dataset)} samples)")

    first_batch = next(iter(data_loader))
    num_time_steps = len(first_batch['v_series']) - 1
    img_size = first_batch['series'].shape[-1]
    print(f"Image size: {img_size}, time steps: {num_time_steps}")
    del first_batch

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps, img_size=img_size, num_heads=8
    )

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = f"{OUTPUT_DIR}/checkpoints"
        ckpt_files = globmod.glob(f"{ckpt_dir}/checkpoint_epoch_*.pth")
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        def _epoch_num(p):
            m = re.search(r'checkpoint_epoch_(\d+)\.pth', p)
            return int(m.group(1)) if m else 0
        checkpoint_path = max(ckpt_files, key=_epoch_num)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    scaling_network.load_state_dict(checkpoint['model_state_dict'])
    scaling_network.cuda().eval()

    ckpt_config = checkpoint.get('config', {})
    trained_epochs = checkpoint.get('epoch', 0)
    print(f"  Trained for {trained_epochs} epochs, best loss: {checkpoint.get('best_loss', 0):.6f}")

    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=NUM_DIFFUSION_STEPS, img_size=img_size
    )

    loss_fn = LTMALoss(
        lambda_sim=1.0,
        lambda_reg=0.0,
        lambda_scale=ckpt_config.get('lambda_scale', LAMBDA_SCALE),
        alpha_scale=ckpt_config.get('alpha_scale', ALPHA_SCALE),
        use_ncc=False,
    )

    trainer = ScalingFactorTrainer(
        scaling_network=scaling_network,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        lr=1e-4,
        device='cuda',
        img_size=img_size,
        gamma=ckpt_config.get('gamma', GAMMA),
        guidance_scale=ckpt_config.get('guidance_scale', GUIDANCE_SCALE),
    )

    # ------------------------------------------------------------------
    # Test loop
    # ------------------------------------------------------------------
    all_subject_ids = []
    all_reg_metrics = []
    all_sparsification_results = []
    all_rc_results = []

    if args.uncertainty_space == 'image':
        regime_desc = "image-space (regime B, no mask)"
    elif use_mask:
        regime_desc = (f"velocity-space + gradient mask "
                       f"(regime B, threshold={args.gradient_threshold})")
    else:
        regime_desc = "velocity-space, no mask (regime A, inverted ranking)"

    total_batches = len(data_loader)
    print(f"\n{'='*60}")
    print(f"FAST TEST PULPo-style ({data_split}): {total_batches} batches, {NUM_RUNS} UQ runs")
    print(f"  Routing: {regime_desc}")
    print(f"{'='*60}")

    for batch_idx, batch in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}/{total_batches}")

        series = batch['series'].cuda()
        v_series = [v.cuda() for v in batch['v_series']]
        ptids_batch = batch['ptid']
        filenames_batch = batch['filename']

        source_img = series[:, 0:1]
        target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]
        batch_size = series.shape[0]

        v_series_bb = v_series

        print("  Computing scaling factors...")
        _, scaling_factors_bb = trainer.validate(source_img, target_imgs, v_series_bb)

        print(f"  Running UQ ({NUM_RUNS} runs)...")
        num_frames_bb = len(v_series_bb)
        sampled_velocities = {i: [] for i in range(num_frames_bb)}

        for run in tqdm(range(NUM_RUNS), desc=f"  Batch {batch_idx + 1} UQ runs"):
            sampled = brownian_bridge.sample_bridge_transition(
                v_series_bb, scaling_factors_bb, device='cuda'
            )
            for frame_idx in range(num_frames_bb):
                sampled_velocities[frame_idx].append(sampled[frame_idx].clone())
            del sampled
            if (run + 1) % 10 == 0:
                torch.cuda.empty_cache()

        for subj_idx in range(batch_size):
            ptid = ptids_batch[subj_idx]
            subject_id = f"batch{batch_idx}_{ptid}"
            all_subject_ids.append(subject_id)

            spars_results = _compute_sparsification_curves(
                sampled_velocities, v_series_bb, series, source_img, subj_idx, img_size,
                uncertainty_space=args.uncertainty_space,
                gradient_threshold=args.gradient_threshold,
                use_mask=use_mask,
            )
            all_sparsification_results.append(spars_results)

            rc_results = _compute_risk_coverage_curves(
                sampled_velocities, v_series_bb, series, source_img, subj_idx, img_size,
                uncertainty_space=args.uncertainty_space,
                gradient_threshold=args.gradient_threshold,
                use_mask=use_mask,
            )
            all_rc_results.append(rc_results)

            if args.compute_reg_metrics:
                reg_metrics = compute_registration_metrics(
                    source_img=source_img,
                    target_imgs=target_imgs,
                    v_reg_list=v_series_bb,
                    sampled_velocities=sampled_velocities,
                    subject_idx=subj_idx,
                    inshape=(img_size, img_size),
                    filename=filenames_batch[subj_idx],
                )
                all_reg_metrics.append(reg_metrics)

            avg_nause = np.mean([r['ause_norm'] for r in spars_results.values()]) \
                if spars_results else float('nan')
            avg_naurc = np.mean([r['aurc_norm'] for r in rc_results.values()]) \
                if rc_results else float('nan')
            print(f"    {subject_id}: nAUSE={avg_nause:.4f}, nAURC={avg_naurc:.4f}")

        del series, v_series, v_series_bb, source_img, target_imgs
        del scaling_factors_bb, sampled_velocities
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------
    tag = args.uncertainty_space
    if args.uncertainty_space == 'velocity':
        tag += ('_masked' if use_mask else '_nomask')
        if use_mask:
            tag += f"_gt{args.gradient_threshold}"
    summary_folder = f"{OUTPUT_DIR}/test_results_pulpo_{tag}"
    os.makedirs(summary_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({len(all_subject_ids)} subjects)")
    print(f"{'='*60}")

    if all_sparsification_results:
        _plot_avg_sparsification(all_sparsification_results, all_subject_ids, summary_folder)
    if all_rc_results:
        _plot_avg_risk_coverage(all_rc_results, all_subject_ids, summary_folder)

    # Per-frame summary table
    if all_sparsification_results and all_rc_results:
        all_frame_indices = set()
        for sr in all_sparsification_results:
            all_frame_indices.update(sr.keys())
        for rr in all_rc_results:
            all_frame_indices.update(rr.keys())
        all_frame_indices = sorted(all_frame_indices)

        summary_lines = []
        summary_lines.append(f"{'='*70}")
        summary_lines.append(f"{'nAURC & nAUSE PER TIME FRAME (averaged across subjects)':^70}")
        summary_lines.append(f"  Routing: {regime_desc}")
        summary_lines.append(f"{'='*70}")
        summary_lines.append(f"  {'Frame':>5}  |  {'nAURC mean':>10}  {'nAURC std':>10}  |  "
                             f"{'nAUSE mean':>10}  {'nAUSE std':>10}  |  {'N':>4}")
        summary_lines.append(f"  {'-'*65}")

        for fi in all_frame_indices:
            naurc_vals = [rr[fi]['aurc_norm'] for rr in all_rc_results if fi in rr]
            nause_vals = [sr[fi]['ause_norm'] for sr in all_sparsification_results if fi in sr]
            n = len(naurc_vals)
            naurc_mean = np.mean(naurc_vals) if naurc_vals else float('nan')
            naurc_std = np.std(naurc_vals) if naurc_vals else float('nan')
            nause_mean = np.mean(nause_vals) if nause_vals else float('nan')
            nause_std = np.std(nause_vals) if nause_vals else float('nan')
            summary_lines.append(
                f"  {fi:>5}  |  {naurc_mean:>10.4f}  {naurc_std:>10.4f}  |  "
                f"{nause_mean:>10.4f}  {nause_std:>10.4f}  |  {n:>4}"
            )

        summary_lines.append(f"  {'-'*65}")
        n_subj_uq = len(all_rc_results)
        subj_mean_naurc = []
        subj_mean_nause = []
        for i in range(n_subj_uq):
            rr = all_rc_results[i]
            sr = all_sparsification_results[i]
            if rr:
                subj_mean_naurc.append(np.mean([rr[fi]['aurc_norm'] for fi in rr]))
            if sr:
                subj_mean_nause.append(np.mean([sr[fi]['ause_norm'] for fi in sr]))
        overall_naurc_mean = np.mean(subj_mean_naurc) if subj_mean_naurc else float('nan')
        overall_naurc_std = np.std(subj_mean_naurc) if subj_mean_naurc else float('nan')
        overall_nause_mean = np.mean(subj_mean_nause) if subj_mean_nause else float('nan')
        overall_nause_std = np.std(subj_mean_nause) if subj_mean_nause else float('nan')
        summary_lines.append(
            f"  {'All':>5}  |  {overall_naurc_mean:>10.4f}  {overall_naurc_std:>10.4f}  |  "
            f"{overall_nause_mean:>10.4f}  {overall_nause_std:>10.4f}  |  {n_subj_uq:>4}"
        )
        summary_lines.append(f"{'='*70}")

        print("\n" + "\n".join(summary_lines))
        summary_txt_path = os.path.join(summary_folder, "per_frame_nAURC_nAUSE_summary.txt")
        with open(summary_txt_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"Saved per-frame summary to: {summary_txt_path}")

    # Per-subject summary text
    summary_file = os.path.join(summary_folder, 'fast_test_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Brain SDE Fast Test (PULPo-style routing)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of subjects: {len(all_subject_ids)}\n")
        f.write(f"NUM_DIFFUSION_STEPS: {NUM_DIFFUSION_STEPS}\n")
        f.write(f"NUM_RUNS: {NUM_RUNS}\n")
        f.write(f"Data split: {data_split}\n")
        f.write(f"Routing: {regime_desc}\n\n")

        f.write(f"{'Subject ID':<40} {'nAUSE':>10} {'nAURC':>10}\n")
        f.write("-" * 60 + "\n")
        for i, subject_id in enumerate(all_subject_ids):
            sr = all_sparsification_results[i]
            rr = all_rc_results[i]
            nause = np.mean([r['ause_norm'] for r in sr.values()]) if sr else float('nan')
            naurc = np.mean([r['aurc_norm'] for r in rr.values()]) if rr else float('nan')
            f.write(f"{subject_id:<40} {nause:>10.4f} {naurc:>10.4f}\n")

    print(f"\nResults saved to: {summary_folder}")
    print("Fast test (PULPo-style) complete!")


if __name__ == "__main__":
    test_fast_pulpo()
