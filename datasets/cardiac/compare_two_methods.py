#!/usr/bin/env python3
"""
Combined comparison of TLRN and VoxelMorph uncertainty_sde_combined methods.

Runs both pipelines on the same test subject and generates a single figure
showing the image sequence and uncertainty maps side-by-side with a shared
colorbar scale.

Layout (rows x cardiac_frames columns):
  Row 0: Source image (shared)
  Row 1: Target image (shared)
  Row 2: TLRN — Deformed (registration network)
  Row 3: VxM  — Deformed (registration network)
  Row 4: TLRN — Deformed (mean UQ velocity)
  Row 5: VxM  — Deformed (mean UQ velocity)
  Row 6: TLRN — Uncertainty (velocity variance magnitude)
  Row 7: VxM  — Uncertainty (velocity variance magnitude)

Usage:
    python -m uncertainty_sde_combined.compare_two_methods \
        --subject_idx 5 --batch_idx 12 --num_runs 100
"""

import os
import sys

# Add project roots to path
sys.path.insert(0, "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

import argparse
import torch
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========================================================================
# TLRN imports (from LightingTemplate_2)
# ========================================================================
from uncertainty_sde_combined_acdc.networks import ScalingFactorNetwork as TLRN_ScalingNet
from uncertainty_sde_combined_acdc.losses import LTMALoss as TLRN_LTMALoss
from uncertainty_sde_combined_acdc.brownian_bridge import BrownianBridgeLearnedScaling as TLRN_BB
from uncertainty_sde_combined_acdc.trainer import ScalingFactorTrainer as TLRN_Trainer
from uncertainty_sde_combined_acdc.data_utils import load_model_and_data as tlrn_load_model_and_data
from uncertainty_sde_combined_acdc.warping import warp_image_with_velocity

# ========================================================================
# VoxelMorph imports are deferred to avoid sys.path conflicts.
# The VxM model code needs its own project root on sys.path for internal
# imports (e.g. ``from utils import LpLoss``).  We add it lazily right
# before loading the VxM modules so that it does not shadow the TLRN
# ``utils`` package which is already importable.
# ========================================================================
import importlib.util

VXM_ROOT = "/scratch/swd9tc/Uncertanity_quantification/voxel_and_R2R/voxelmorph"

def _load_vxm_modules():
    """Import all VxM uncertainty_sde_combined modules (call once, after TLRN is done).

    The VxM codebase has its own top-level ``utils`` module (a single file)
    that conflicts with the TLRN ``utils`` package already cached in
    ``sys.modules``.  We temporarily evict the TLRN utils, put VXM_ROOT
    first on ``sys.path``, import everything, then restore the original
    utils module.
    """
    # Save and evict any cached 'utils' and sub-modules
    saved_utils = {}
    for key in list(sys.modules.keys()):
        if key == 'utils' or key.startswith('utils.'):
            saved_utils[key] = sys.modules.pop(key)

    # Put VXM_ROOT at the front so its ``utils.py`` is found first
    if VXM_ROOT not in sys.path:
        sys.path.insert(0, VXM_ROOT)

    specs = {
        "data_utils": "uncertainty_sde_combined/data_utils.py",
        "networks":   "uncertainty_sde_combined/networks.py",
        "losses":     "uncertainty_sde_combined/losses.py",
        "bb":         "uncertainty_sde_combined/brownian_bridge.py",
        "trainer":    "uncertainty_sde_combined/trainer.py",
        "warping":    "uncertainty_sde_combined/warping.py",
    }
    mods = {}
    for name, rel_path in specs.items():
        full = os.path.join(VXM_ROOT, rel_path)
        spec = importlib.util.spec_from_file_location(f"vxm_{name}", full)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod

    # Restore original utils modules so TLRN code keeps working
    for key in list(sys.modules.keys()):
        if key == 'utils' or key.startswith('utils.'):
            sys.modules.pop(key, None)
    sys.modules.update(saved_utils)

    return (
        mods["data_utils"].load_model_and_data,
        mods["networks"].ScalingFactorNetwork,
        mods["losses"].LTMALoss,
        mods["bb"].BrownianBridgeLearnedScaling,
        mods["trainer"].ScalingFactorTrainer,
        mods["warping"].warp_image_with_velocity,
    )

# Placeholders — filled in main() after TLRN work is done
vxm_load_model_and_data = None
VXM_ScalingNet = None
VXM_LTMALoss = None
VXM_BB = None
VXM_Trainer = None
vxm_warp_image = None


# ========================================================================
# Helper: run UQ sampling for one method
# ========================================================================

def run_uq_pipeline(model, data_loader, scaling_network, brownian_bridge, trainer,
                    batch_idx, subject_idx, num_runs, sampling_method, method_name):
    """
    Run UQ pipeline for a given method.

    Returns:
        series         : [B, T, H, W] image series tensor (cuda)
        Sdef_series    : list of [B, 1, H, W] deformed images
        v_series_bb    : list of velocity fields (including zero at t=0)
        sampled_velocities : dict {frame_idx -> list of velocity tensors}
        scaling_factors_bb : [B, T, 2, H, W]
    """
    print(f"\n{'='*60}")
    print(f"Running UQ for {method_name}")
    print(f"{'='*60}")

    # Navigate to the requested batch
    print(f"  Getting batch {batch_idx}...")
    test_batch = None
    for bi, batch in enumerate(data_loader):
        if bi == batch_idx:
            test_batch = batch
            break
    if test_batch is None:
        raise ValueError(f"batch_idx={batch_idx} exceeds number of batches ({bi + 1})")

    series = test_batch['series'].cuda()
    lv_segs = test_batch.get('lv_segs')
    if lv_segs is not None:
        lv_segs = lv_segs.cuda()
    else:
        lv_segs = torch.zeros_like(series[:, 0:1]).cuda()

    # Extract velocity fields
    print(f"  Extracting velocity fields...")
    with torch.no_grad():
        Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = \
            model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)

    source_img = series[:, 0:1]
    target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]

    # Create v_series with zero velocity at t=0
    zero_velocity = torch.zeros_like(v_series[0])
    v_series_bb = [zero_velocity] + v_series

    # Get scaling factors
    print(f"  Computing scaling factors...")
    _, scaling_factors_bb = trainer.validate(source_img, target_imgs, v_series_bb)

    # Run UQ sampling
    num_time_steps_bb = len(v_series_bb)
    sampled_velocities = {i: [] for i in range(num_time_steps_bb)}

    print(f"  Running {num_runs} UQ samples (method: {sampling_method})...")
    for run in tqdm(range(num_runs), desc=f"{method_name} UQ"):
        if sampling_method == "bridge_transition":
            sampled = brownian_bridge.sample_bridge_transition(
                v_series_bb, scaling_factors_bb, device='cuda'
            )
        else:  # posterior_reverse_sde
            v_0 = v_series_bb[0].clone()
            v_T = v_series_bb[-1].clone()
            trajectory = brownian_bridge.run_posterior_reverse_process(
                v_series_bb, scaling_factors_bb, v_0, v_T,
                source_img, target_imgs,
                gamma=0.0001 / 2, guidance_scale=1.0, device='cuda'
            )
            sampled = {}
            T_diff = brownian_bridge.T
            for fi in range(num_time_steps_bb):
                traj_idx = T_diff - int(fi * T_diff / (num_time_steps_bb - 1)) if num_time_steps_bb > 1 else 0
                traj_idx = min(max(traj_idx, 0), len(trajectory) - 1)
                sampled[fi] = trajectory[traj_idx]

        for frame_idx in range(num_time_steps_bb):
            sampled_velocities[frame_idx].append(sampled[frame_idx].clone())
        del sampled
        if (run + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return series, Sdef_series, v_series_bb, sampled_velocities, scaling_factors_bb


def extract_frame_data(series, Sdef_series, v_series_bb, sampled_velocities,
                       subject_idx, warp_fn):
    """
    Extract per-frame numpy arrays for visualization.

    Returns:
        source_img_np : [H, W]
        target_imgs_np: list of [H, W]
        deformed_reg_np: list of [H, W]  (from registration network)
        deformed_uq_np : list of [H, W]  (warped with mean sampled velocity)
        var_mag_np     : list of [H, W]  (velocity variance magnitude)
    """
    num_frames = len(v_series_bb)
    source_img_np = series[subject_idx, 0].cpu().numpy()
    source_tensor = series[subject_idx:subject_idx + 1, 0:1].clone()  # [1,1,H,W]
    img_size = source_tensor.shape[-1]

    target_imgs_np = []
    deformed_reg_np = []
    deformed_uq_np = []
    var_mag_np = []

    for i in range(num_frames):
        # Target
        if i == 0:
            target_imgs_np.append(source_img_np.copy())
            deformed_reg_np.append(source_img_np.copy())
        else:
            target_imgs_np.append(series[subject_idx, i].cpu().numpy())
            deformed_reg_np.append(Sdef_series[i - 1][subject_idx, 0].cpu().numpy())

        # Mean velocity and variance from samples
        if i in sampled_velocities and len(sampled_velocities[i]) > 0:
            vels = np.stack([v[subject_idx].cpu().numpy() for v in sampled_velocities[i]], axis=0)
            mean_v = np.mean(vels, axis=0)
            variance = np.var(vels, axis=0)
            var_mag = np.sqrt(variance[0] + variance[1])
            mean_v_tensor = torch.from_numpy(mean_v).unsqueeze(0).cuda()
        else:
            var_mag = np.zeros_like(source_img_np)
            mean_v_tensor = v_series_bb[i][subject_idx:subject_idx + 1].clone()

        var_mag_np.append(var_mag)

        # Warp source with mean sampled velocity
        with torch.no_grad():
            warped = warp_fn(source_tensor, mean_v_tensor, inshape=(img_size, img_size))
            deformed_uq_np.append(warped[0, 0].cpu().numpy())

    return source_img_np, target_imgs_np, deformed_reg_np, deformed_uq_np, var_mag_np


# ========================================================================
# Initialisation helpers
# ========================================================================

def init_tlrn_components(num_time_steps, img_size):
    """Initialise TLRN scaling network, brownian bridge, trainer from checkpoint."""
    NUM_DIFFUSION_STEPS = 14
    CHECKPOINT_PATH = '2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03/visualization/uncertainty_sde_combined/checkpoints/checkpoint_epoch_500.pth'

    scaling_net = TLRN_ScalingNet(num_time_steps=num_time_steps, img_size=img_size, num_heads=8)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    scaling_net.load_state_dict(checkpoint['model_state_dict'])
    scaling_net.cuda().eval()

    ckpt_config = checkpoint.get('config', {})
    bb = TLRN_BB(num_diffusion_steps=NUM_DIFFUSION_STEPS, img_size=img_size)
    loss_fn = TLRN_LTMALoss(
        lambda_sim=ckpt_config.get('lambda_sim', 1.0),
        lambda_reg=ckpt_config.get('lambda_reg', 0.0),
        lambda_scale=ckpt_config.get('lambda_scale', 0.001),
        lambda_low_structure=ckpt_config.get('lambda_low_structure', 0.0),
        use_ncc=False
    )
    trainer = TLRN_Trainer(scaling_network=scaling_net, brownian_bridge=bb,
                           loss_fn=loss_fn, lr=1e-4, device='cuda', img_size=img_size)
    return scaling_net, bb, trainer


def init_vxm_components(num_time_steps, img_size,
                        ScalingNetCls, BBCls, LossCls, TrainerCls):
    """Initialise VoxelMorph scaling network, brownian bridge, trainer from checkpoint."""
    NUM_DIFFUSION_STEPS = 14
    CHECKPOINT_PATH = 'visualization_voxelmorph/uncertainty_sde_combined/checkpoints/checkpoint_epoch_500.pth'

    # The VxM checkpoint lives relative to the voxelmorph project dir
    ckpt_full = os.path.join(VXM_ROOT, CHECKPOINT_PATH)

    scaling_net = ScalingNetCls(num_time_steps=num_time_steps, img_size=img_size, num_heads=8)
    checkpoint = torch.load(ckpt_full, map_location='cpu')
    scaling_net.load_state_dict(checkpoint['model_state_dict'])
    scaling_net.cuda().eval()

    ckpt_config = checkpoint.get('config', {})
    bb = BBCls(num_diffusion_steps=NUM_DIFFUSION_STEPS, img_size=img_size)
    loss_fn = LossCls(
        lambda_sim=ckpt_config.get('lambda_sim', 1.0),
        lambda_reg=ckpt_config.get('lambda_reg', 0.0),
        lambda_scale=ckpt_config.get('lambda_scale', 0.001),
        lambda_low_structure=ckpt_config.get('lambda_low_structure', 0.0),
        use_ncc=False
    )
    trainer = TrainerCls(scaling_network=scaling_net, brownian_bridge=bb,
                         loss_fn=loss_fn, lr=1e-4, device='cuda', img_size=img_size)
    return scaling_net, bb, trainer


# ========================================================================
# Visualisation
# ========================================================================

def make_combined_figure(tlrn_data, vxm_data, save_path):
    """
    Create combined comparison figure.

    Each *_data is a tuple:
        (source_img_np, target_imgs_np, deformed_reg_np, deformed_uq_np, var_mag_np)

    Layout:
        Row 0 : Image sequence (frame 0 = source, frames 1..T = targets)
        Row 1 : TLRN Uncertainty
        Row 2 : VxM  Uncertainty
    """
    src_t, tgt_t, dreg_t, duq_t, var_t = tlrn_data
    src_v, tgt_v, dreg_v, duq_v, var_v = vxm_data

    num_frames = len(tgt_t)

    # ---- Diagnostic: print per-frame statistics for both methods ----
    print("\n" + "=" * 80)
    print("Uncertainty map statistics (velocity variance magnitude)")
    print("=" * 80)
    print(f"{'Frame':>6} | {'TLRN mean':>12} {'TLRN std':>12} {'TLRN max':>12} | "
          f"{'VxM mean':>12} {'VxM std':>12} {'VxM max':>12} | "
          f"{'ratio(VxM/TLRN)':>16}")
    print("-" * 120)
    ratios = []
    for i in range(num_frames):
        t_mean, t_std, t_max = np.mean(var_t[i]), np.std(var_t[i]), np.max(var_t[i])
        v_mean, v_std, v_max = np.mean(var_v[i]), np.std(var_v[i]), np.max(var_v[i])
        ratio = v_mean / t_mean if t_mean > 1e-10 else float('inf')
        ratios.append(ratio)
        print(f"{i:>6} | {t_mean:>12.6f} {t_std:>12.6f} {t_max:>12.6f} | "
              f"{v_mean:>12.6f} {v_std:>12.6f} {v_max:>12.6f} | "
              f"{ratio:>16.4f}")

    # Overall statistics
    all_t = np.concatenate([v.ravel() for v in var_t])
    all_v = np.concatenate([v.ravel() for v in var_v])
    overall_ratio = np.mean(all_v) / np.mean(all_t) if np.mean(all_t) > 1e-10 else float('inf')
    print("-" * 120)
    print(f"{'ALL':>6} | {np.mean(all_t):>12.6f} {np.std(all_t):>12.6f} {np.max(all_t):>12.6f} | "
          f"{np.mean(all_v):>12.6f} {np.std(all_v):>12.6f} {np.max(all_v):>12.6f} | "
          f"{overall_ratio:>16.4f}")

    # Compute a scalar multiplier for the smaller method so both span similar ranges
    # Scale the smaller one UP to match the larger one's mean
    if overall_ratio > 1.0:
        # VxM is larger — scale TLRN up
        scale_tlrn = overall_ratio
        scale_vxm = 1.0
        print(f"\nVxM uncertainty is {overall_ratio:.2f}x larger on average.")
        print(f"Scaling TLRN maps by {scale_tlrn:.4f} to match VxM range.")
    else:
        # TLRN is larger — scale VxM up
        scale_tlrn = 1.0
        scale_vxm = 1.0 / overall_ratio
        print(f"\nTLRN uncertainty is {1.0/overall_ratio:.2f}x larger on average.")
        print(f"Scaling VxM maps by {scale_vxm:.4f} to match TLRN range.")

    # Apply scaling
    var_t_scaled = [v * scale_tlrn for v in var_t]
    var_v_scaled = [v * scale_vxm for v in var_v]

    # Compute shared colorbar range from scaled maps (excluding endpoints)
    all_var_interior = []
    for i in range(num_frames):
        if 0 < i < num_frames - 1:
            all_var_interior.append(var_t_scaled[i])
            all_var_interior.append(var_v_scaled[i])
    if all_var_interior:
        var_vmax = max(np.max(v) for v in all_var_interior)
    else:
        var_vmax = max(np.max(var_t_scaled[0]), np.max(var_v_scaled[0]))
    var_vmax = max(var_vmax, 0.1)

    nrows = 3
    ncols = num_frames + 1  # +1 for colorbar column
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2 * num_frames + 1.5, 2 * nrows + 2),
                             gridspec_kw={'width_ratios': [1] * num_frames + [0.08]})

    im_var = None

    for i in range(num_frames):
        # Row 0: Image sequence (source at frame 0, targets for the rest)
        axes[0, i].imshow(tgt_t[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Frame {i}', fontsize=9)
        axes[0, i].axis('off')

        # Row 1: TLRN Uncertainty (scaled, shared range)
        im_var = axes[1, i].imshow(var_t_scaled[i], cmap='jet', vmin=0, vmax=var_vmax)
        axes[1, i].axis('off')

        # Row 2: VxM Uncertainty (scaled, shared range)
        axes[2, i].imshow(var_v_scaled[i], cmap='jet', vmin=0, vmax=var_vmax)
        axes[2, i].axis('off')

    # Colorbar column
    axes[0, -1].axis('off')

    if im_var is not None:
        cbar = fig.colorbar(im_var, cax=axes[1, -1], orientation='vertical')
        label_suffix = ""
        if scale_tlrn != 1.0:
            label_suffix = f"\n(TLRN x{scale_tlrn:.1f})"
        elif scale_vxm != 1.0:
            label_suffix = f"\n(VxM x{scale_vxm:.1f})"
        cbar.set_label(f'Vel. Var. Mag.{label_suffix}', fontsize=8)
    axes[2, -1].axis('off')

    # Row labels
    if scale_tlrn != 1.0:
        tlrn_label = f'TLRN Unc. (x{scale_tlrn:.1f})'
    else:
        tlrn_label = 'TLRN Uncertainty'
    if scale_vxm != 1.0:
        vxm_label = f'VxM Unc. (x{scale_vxm:.1f})'
    else:
        vxm_label = 'VxM Uncertainty'
    labels = ['Image Sequence', tlrn_label, vxm_label]
    for r, lab in enumerate(labels):
        axes[r, 0].set_ylabel(lab, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined figure: {save_path}")
    plt.close()


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare TLRN and VxM uncertainty_sde_combined")
    parser.add_argument("--subject_idx", type=int, default=5)
    parser.add_argument("--batch_idx", type=int, default=12)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--sampling_method", type=str, default="posterior_reverse_sde",
                        choices=["bridge_transition", "posterior_reverse_sde"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load TLRN model + data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading TLRN model and data")
    print("=" * 60)
    tlrn_model, tlrn_train_loader, tlrn_test_loader, tlrn_config = tlrn_load_model_and_data()

    # Determine num_time_steps from first batch
    first_batch = next(iter(tlrn_test_loader))
    first_series = first_batch['series'].cuda()
    first_lv = first_batch.get('lv_segs')
    if first_lv is not None:
        first_lv = first_lv.cuda()
    else:
        first_lv = torch.zeros_like(first_series[:, 0:1]).cuda()
    with torch.no_grad():
        _, first_v, _, _, _ = tlrn_model.sequence_register_no_avg_lowf_addlatentf(first_series, first_lv)
    num_time_steps = len(first_v)
    img_size = first_series.shape[-1]
    del first_series, first_lv, first_v
    torch.cuda.empty_cache()

    # Initialise TLRN UQ components
    tlrn_scaling, tlrn_bb, tlrn_trainer = init_tlrn_components(num_time_steps, img_size)

    # Run TLRN pipeline
    tlrn_series, tlrn_Sdef, tlrn_v_bb, tlrn_sampled, tlrn_sf = run_uq_pipeline(
        model=tlrn_model, data_loader=tlrn_test_loader,
        scaling_network=tlrn_scaling, brownian_bridge=tlrn_bb, trainer=tlrn_trainer,
        batch_idx=args.batch_idx, subject_idx=args.subject_idx,
        num_runs=args.num_runs, sampling_method=args.sampling_method,
        method_name="TLRN"
    )

    # Extract TLRN frame data
    tlrn_data = extract_frame_data(
        tlrn_series, tlrn_Sdef, tlrn_v_bb, tlrn_sampled,
        args.subject_idx, warp_image_with_velocity
    )

    # Free TLRN model (keep tlrn_config for output path)
    del tlrn_model, tlrn_scaling, tlrn_bb, tlrn_trainer
    del tlrn_series, tlrn_Sdef, tlrn_v_bb, tlrn_sampled, tlrn_sf
    torch.cuda.empty_cache()
    gc.collect()
    print("Released TLRN resources")

    # ------------------------------------------------------------------
    # 2. Load VoxelMorph model + data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading VoxelMorph model and data")
    print("=" * 60)

    # Lazy-import VxM modules now that TLRN is done
    (vxm_load_fn, VXM_ScalingNet, VXM_LTMALoss,
     VXM_BBCls, VXM_TrainerCls, vxm_warp_image) = _load_vxm_modules()

    vxm_model, vxm_train_loader, vxm_test_loader, vxm_config = vxm_load_fn()

    # Initialise VxM UQ components (same num_time_steps and img_size — shared dataset)
    vxm_scaling, vxm_bb, vxm_trainer = init_vxm_components(
        num_time_steps, img_size,
        VXM_ScalingNet, VXM_BBCls, VXM_LTMALoss, VXM_TrainerCls
    )

    # Run VxM pipeline
    vxm_series, vxm_Sdef, vxm_v_bb, vxm_sampled, vxm_sf = run_uq_pipeline(
        model=vxm_model, data_loader=vxm_test_loader,
        scaling_network=vxm_scaling, brownian_bridge=vxm_bb, trainer=vxm_trainer,
        batch_idx=args.batch_idx, subject_idx=args.subject_idx,
        num_runs=args.num_runs, sampling_method=args.sampling_method,
        method_name="VoxelMorph"
    )

    # Extract VxM frame data
    vxm_data = extract_frame_data(
        vxm_series, vxm_Sdef, vxm_v_bb, vxm_sampled,
        args.subject_idx, vxm_warp_image
    )

    # Free VxM model
    del vxm_model, vxm_scaling, vxm_bb, vxm_trainer
    del vxm_series, vxm_Sdef, vxm_v_bb, vxm_sampled, vxm_sf
    torch.cuda.empty_cache()
    gc.collect()
    print("Released VoxelMorph resources")

    # ------------------------------------------------------------------
    # 3. Generate combined figure
    # ------------------------------------------------------------------
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(
            tlrn_config.model.params.workdir,
            "visualization/uncertainty_sde_combined/comparison_tlrn_vxm"
        )
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(
        out_dir,
        f"comparison_batch{args.batch_idx}_subj{args.subject_idx}.png"
    )
    make_combined_figure(tlrn_data, vxm_data, save_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
