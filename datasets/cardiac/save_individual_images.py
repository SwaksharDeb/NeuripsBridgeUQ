"""
Save individual visualization images for a given subject.

Produces the same composite plot as visualize_uncertainty_cardiac_frames() and also
saves each component as a separate high-resolution image:
  - Source image
  - Target images (per frame)
  - Transformation field (deformation grid overlay, per frame)
  - Warped image (per frame)
  - Difference map (|target - warped|, per frame)
  - Uncertainty map (variance overlay, per frame)

Usage:
    python -m uncertainty_inv_gamma_prior.save_individual_images --subject_idx 0
    python -m uncertainty_inv_gamma_prior.save_individual_images --batch_idx 1 --subject_idx 2
    python -m uncertainty_inv_gamma_prior.save_individual_images --subject_idx 0 --num_runs 100
"""

import os
import sys

# Add project root to path
sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import glob as globmod
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm

from uncertainty_sde_combined_acdc.networks import ScalingFactorNetwork
from uncertainty_sde_combined_acdc.losses import LTMALoss
from uncertainty_sde_combined_acdc.brownian_bridge import BrownianBridgeLearnedScaling
from uncertainty_sde_combined_acdc.trainer import ScalingFactorTrainer
from uncertainty_sde_combined_acdc.data_utils import load_model_and_data
from uncertainty_sde_combined_acdc.visualization import visualize_uncertainty_cardiac_frames
from uncertainty_sde_combined_acdc.warping import warp_image_with_velocity
from utils.Int import VecInt


def save_individual_frame_images(sampled_velocities, v_reg_list, Sdef_series, series,
                                  scaling_factors, save_folder, subject_idx=0, batch_idx=0):
    """
    Save individual images for each frame: source, target, transformation field,
    warped image, difference map, and uncertainty map.

    Args:
        sampled_velocities: Dict mapping frame_idx -> list of velocity tensors
        v_reg_list: List of registered velocity fields (including zero at t=0)
        Sdef_series: List of deformed images from registration
        series: Original image series [B, T, H, W]
        scaling_factors: [B, T, 2, H, W] learned scaling factors
        save_folder: Where to save images
        subject_idx: Which subject to visualize
        batch_idx: Which batch (used for filename suffix)
    """
    suffix = f"_b{batch_idx}s{subject_idx}"
    indiv_dir = os.path.join(save_folder, "individual_frames")
    os.makedirs(indiv_dir, exist_ok=True)

    num_cardiac_frames = len(v_reg_list)
    source_img = series[subject_idx, 0].cpu().numpy().T  # (W,H) -> (H,W) for display

    # Prepare source tensor for warping
    source_img_tensor = series[subject_idx:subject_idx+1, 0:1].clone()  # [1, 1, H, W]
    img_size = source_img_tensor.shape[-1]
    inshape = (img_size, img_size)

    # Initialize velocity integrator
    vec_int = VecInt(inshape, TSteps=7).cuda()

    # First pass: compute mean velocities and variance for all frames
    all_mean_v = []
    all_var_mag = []

    for i in range(num_cardiac_frames):
        velocities_at_t = np.stack(
            [v[subject_idx].cpu().numpy() for v in sampled_velocities[i]], axis=0
        )
        mean_v = np.mean(velocities_at_t, axis=0)
        variance = np.var(velocities_at_t, axis=0)
        var_mag = np.sqrt(variance[0] + variance[1]).T
        all_mean_v.append(torch.from_numpy(mean_v).unsqueeze(0).cuda())
        all_var_mag.append(var_mag)

    # Compute global variance max (excluding endpoints)
    var_vmax = 0.0
    for i in range(num_cardiac_frames):
        if 0 < i < num_cardiac_frames - 1:
            var_vmax = max(var_vmax, np.max(all_var_mag[i]))
    var_vmax = max(var_vmax, 0.1)

    # Track global difference map max for consistent colorbar
    diff_vmax = 0.0

    # Save source image (same for all frames)
    fig_s, ax_s = plt.subplots(figsize=(4, 4))
    ax_s.imshow(source_img, cmap='gray', vmin=0, vmax=1)
    ax_s.axis('off')
    fig_s.savefig(os.path.join(indiv_dir, f"source{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_s)

    for i in range(num_cardiac_frames):
        print(f"  Saving frame {i}/{num_cardiac_frames - 1}...")

        # --- Target image ---
        if i == 0:
            target_img = source_img
        else:
            target_img = series[subject_idx, i].cpu().numpy().T

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(target_img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        fig.savefig(os.path.join(indiv_dir, f"target_frame_{i}{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # --- Warped image (using mean sampled velocity) ---
        with torch.no_grad():
            mean_v_tensor = all_mean_v[i]
            warped_mean = warp_image_with_velocity(
                source_img_tensor, mean_v_tensor, inshape=inshape
            )
            warped_mean_np = warped_mean[0, 0].cpu().numpy().T

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(warped_mean_np, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        fig.savefig(os.path.join(indiv_dir, f"warped_frame_{i}{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # --- Transformation field (deformation grid overlay) ---
        with torch.no_grad():
            disp_list = vec_int(mean_v_tensor)
            displacement = disp_list[-1]  # [1, 2, H, W]

            disp_x = displacement[0, 0]
            disp_y = displacement[0, 1]

            grid_spacing = 4
            y_coords = torch.arange(0, img_size)
            x_coords = torch.arange(0, img_size)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

            deformed_x = grid_x.cuda() + disp_x
            deformed_y = grid_y.cuda() + disp_y

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(source_img, cmap='gray', vmin=0, vmax=1)
        for j in range(0, img_size, grid_spacing):
            ax.plot(deformed_y[j, :].detach().cpu().numpy(),
                    deformed_x[j, :].detach().cpu().numpy(),
                    'r-', linewidth=0.9, alpha=0.9)
        for j in range(0, img_size, grid_spacing):
            ax.plot(deformed_y[:, j].detach().cpu().numpy(),
                    deformed_x[:, j].detach().cpu().numpy(),
                    'r-', linewidth=0.9, alpha=0.9)
        ax.set_xlim(0, img_size)
        ax.set_ylim(img_size, 0)
        ax.axis('off')
        fig.savefig(os.path.join(indiv_dir, f"transformation_field_frame_{i}{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # --- Difference map (|target - warped|) ---
        diff_map = np.abs(target_img - warped_mean_np)
        diff_map_max = diff_map.max() if diff_map.max() > 0 else 1.0
        if 0 < i < num_cardiac_frames - 1:
            diff_vmax = max(diff_vmax, diff_map_max)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(diff_map, cmap='hot', vmin=0, vmax=diff_map_max)
        ax.axis('off')
        fig.savefig(os.path.join(indiv_dir, f"difference_map_frame_{i}{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # --- Uncertainty map (variance overlay on source) ---
        var_mag = all_var_mag[i]
        var_mag_norm = var_mag / var_vmax if var_vmax > 0 else var_mag

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(source_img, cmap='gray')
        ax.imshow(var_mag_norm, cmap='jet_r', vmin=0.2, vmax=0.7, alpha=0.8)
        ax.axis('off')
        fig.savefig(os.path.join(indiv_dir, f"uncertainty_map_frame_{i}{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Save standalone colorbar for uncertainty
    fig_cb, ax_cb = plt.subplots(figsize=(0.3, 4))
    norm = plt.Normalize(vmin=0.2, vmax=0.7)
    cb = fig_cb.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet_r'),
                         cax=ax_cb, orientation='vertical')
    cb.set_ticks([])
    fig_cb.savefig(os.path.join(indiv_dir, f"uncertainty_colorbar{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_cb)

    # Save standalone colorbar for difference map
    diff_vmax = max(diff_vmax, 0.01)
    fig_cb2, ax_cb2 = plt.subplots(figsize=(0.3, 4))
    norm2 = plt.Normalize(vmin=0, vmax=diff_vmax)
    cb2 = fig_cb2.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap='hot'),
                            cax=ax_cb2, orientation='vertical')
    cb2.set_ticks([])
    fig_cb2.savefig(os.path.join(indiv_dir, f"difference_colorbar{suffix}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_cb2)

    total_saved = 1 + num_cardiac_frames * 5 + 2  # source + 5 per frame + 2 colorbars
    print(f"Saved {total_saved} individual images to: {indiv_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Save individual visualization images for a subject"
    )
    parser.add_argument("--subject_idx", type=int, required=True,
                        help="Subject index within the test batch")
    parser.add_argument("--batch_idx", type=int, default=0,
                        help="Batch index in test loader (default: 0)")
    parser.add_argument("--num_runs", type=int, default=3000,
                        help="Number of UQ runs (default: 3000)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    SUBJECT_IDX = args.subject_idx
    BATCH_IDX = args.batch_idx
    NUM_DIFFUSION_STEPS = 14
    NUM_RUNS = args.num_runs

    CHECKPOINT_PATH = '2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03/visualization/uncertainty_learned_sde/checkpoints/checkpoint_epoch_1800.pth'

    # ======================================================================
    # Load model and data
    # ======================================================================
    model, train_loader, test_loader, config = load_model_and_data()

    # Get first batch to determine dimensions
    print("\nDetermining data dimensions from first batch...")
    first_batch = next(iter(test_loader))
    first_series = first_batch['series'].cuda()
    first_lv_segs = first_batch.get('lv_segs')
    if first_lv_segs is not None:
        first_lv_segs = first_lv_segs.cuda()
    else:
        first_lv_segs = torch.zeros_like(first_series[:, 0:1]).cuda()

    with torch.no_grad():
        _, first_v_series, _, _, _ = model.sequence_register_no_avg_lowf_addlatentf(
            first_series, first_lv_segs
        )

    num_time_steps = len(first_v_series)
    img_size = first_series.shape[-1]

    print(f"  Image size: {img_size}")
    print(f"  Number of time steps: {num_time_steps}")

    del first_series, first_lv_segs, first_v_series
    torch.cuda.empty_cache()

    # ======================================================================
    # Initialize components
    # ======================================================================
    print("\nInitializing Scaling Factor Network...")
    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps,
        img_size=img_size,
        num_heads=8
    )

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    scaling_network.load_state_dict(checkpoint['model_state_dict'])
    scaling_network.cuda()
    scaling_network.eval()

    ckpt_config = checkpoint.get('config', {})
    training_history = checkpoint.get('training_history', [])
    trained_epochs = checkpoint.get('epoch', 0)
    print(f"  Trained for {trained_epochs} epochs")

    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        img_size=img_size
    )

    loss_fn = LTMALoss(
        lambda_sim=ckpt_config.get('lambda_sim', 1.0),
        lambda_reg=ckpt_config.get('lambda_reg', 0.0),
        lambda_scale=ckpt_config.get('lambda_scale', 0.001),
        lambda_low_structure=ckpt_config.get('lambda_low_structure', 0.0),
        use_ncc=False
    )

    trainer = ScalingFactorTrainer(
        scaling_network=scaling_network,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        lr=1e-4,
        device='cuda',
        img_size=img_size
    )

    # ======================================================================
    # Load test batch and extract registration outputs
    # ======================================================================
    print(f"\nGetting test batch {BATCH_IDX}...")
    test_iter = iter(test_loader)
    for skip_i in range(BATCH_IDX + 1):
        try:
            test_batch = next(test_iter)
        except StopIteration:
            raise ValueError(
                f"batch_idx={BATCH_IDX} is out of range. "
                f"Test loader only has {skip_i} batch(es)."
            )
    print(f"  Loaded batch {BATCH_IDX}")

    series = test_batch['series'].cuda()
    lv_segs = test_batch.get('lv_segs')
    file_paths = test_batch.get('path', [None] * series.shape[0])
    if lv_segs is not None:
        lv_segs = lv_segs.cuda()
    else:
        lv_segs = torch.zeros_like(series[:, 0:1]).cuda()

    batch_size = series.shape[0]
    if SUBJECT_IDX >= batch_size:
        print(f"Warning: subject_idx={SUBJECT_IDX} >= batch_size={batch_size}. Using 0.")
        SUBJECT_IDX = 0

    print(f"Extracting velocity fields (batch_size={batch_size})...")
    with torch.no_grad():
        Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = \
            model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("Released registration model from GPU memory")

    source_img = series[:, 0:1]
    target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]

    # Create v_series with zero velocity at start for Brownian Bridge
    zero_velocity = torch.zeros_like(v_series[0])
    v_series_bb = [zero_velocity] + v_series

    # Compute scaling factors
    print("Computing scaling factors...")
    _, scaling_factors_bb = trainer.validate(source_img, target_imgs, v_series_bb)

    # ======================================================================
    # Run uncertainty quantification
    # ======================================================================
    print(f"Running uncertainty quantification with {NUM_RUNS} runs...")
    v_0 = zero_velocity.clone()
    v_T = v_series[-1].clone()

    num_time_steps_bb = len(v_series_bb)
    sampled_velocities = {i: [] for i in range(num_time_steps_bb)}

    for run in tqdm(range(NUM_RUNS), desc="UQ runs"):
        trajectory = brownian_bridge.run_reverse_process(
            v_series_bb, scaling_factors_bb, v_0, v_T, device='cuda'
        )

        for frame_idx in range(num_time_steps_bb):
            traj_idx = NUM_DIFFUSION_STEPS - int(
                frame_idx * NUM_DIFFUSION_STEPS / (num_time_steps_bb - 1)
            ) if num_time_steps_bb > 1 else 0
            traj_idx = min(max(traj_idx, 0), len(trajectory) - 1)
            sampled_velocities[frame_idx].append(trajectory[traj_idx].clone())

        del trajectory
        if (run + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # ======================================================================
    # Determine save folder
    # ======================================================================
    filename = file_paths[SUBJECT_IDX] if isinstance(file_paths, list) else file_paths
    if filename is not None:
        base_filename = os.path.basename(filename).replace('.npy', '').replace('.npz', '')
        subject_id = f"batch{BATCH_IDX}_{base_filename}"
    else:
        subject_id = f"batch{BATCH_IDX}_subject_{SUBJECT_IDX}"

    if args.output_dir is not None:
        save_folder = args.output_dir
    else:
        save_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "saved_images",
            subject_id
        )
    os.makedirs(save_folder, exist_ok=True)

    # ======================================================================
    # 1) Composite plot (same as visualize_uncertainty_cardiac_frames)
    # ======================================================================
    print(f"\nGenerating composite plot for subject {SUBJECT_IDX} ({subject_id})...")
    visualize_uncertainty_cardiac_frames(
        sampled_velocities, v_series_bb, Sdef_series, series,
        scaling_factors_bb, brownian_bridge, save_folder,
        subject_idx=SUBJECT_IDX, save_individual=False
    )

    # ======================================================================
    # 2) Save individual images
    # ======================================================================
    print("Saving individual frame images...")
    save_individual_frame_images(
        sampled_velocities, v_series_bb, Sdef_series, series,
        scaling_factors_bb, save_folder, subject_idx=SUBJECT_IDX, batch_idx=BATCH_IDX
    )

    print(f"\nAll outputs saved to: {save_folder}")


if __name__ == "__main__":
    main()
