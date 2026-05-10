"""
Visualization functions for uncertainty quantification results.

Contains:
- visualize_scaling_factors: Visualize learned scaling factors
- visualize_uncertainty_cardiac_frames: Comprehensive uncertainty visualization
- visualize_scaling_factor_analysis: Detailed scaling factor analysis
- plot_training_history: Plot training loss history
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from uncertainty_brain_sde.warping import warp_image_with_velocity
from utils.Int import jacobian_det_for_one, VecInt


def visualize_scaling_factors(scaling_factors, v_reg_list, save_folder, subject_idx=0):
    """
    Visualize the learned scaling factors.

    Args:
        scaling_factors: [B, T, 2, H, W] learned scaling factors (T frames including t=0)
        v_reg_list: List of velocity fields (T frames including zero velocity at t=0)
        save_folder: Where to save visualizations
        subject_idx: Which subject to visualize
    """
    os.makedirs(save_folder, exist_ok=True)

    T = scaling_factors.shape[1]  # Number of frames (including t=0)

    fig, axes = plt.subplots(4, T, figsize=(2.5 * T, 10))

    for t in range(T):
        # Scaling factor for x component
        #s_x = scaling_factors[subject_idx, t, 0].cpu().numpy().T
        s_x = scaling_factors[subject_idx, t, 0].cpu().numpy()
        # Scaling factor for y component
        #s_y = scaling_factors[subject_idx, t, 1].cpu().numpy().T
        s_y = scaling_factors[subject_idx, t, 1].cpu().numpy()
        # Scaling factor magnitude
        s_mag = np.sqrt(s_x**2 + s_y**2)

        # Velocity magnitude
        v = v_reg_list[t][subject_idx].cpu().numpy()
        #v_mag = np.sqrt(v[0]**2 + v[1]**2).T
        v_mag = np.sqrt(v[0]**2 + v[1]**2)

        im0 = axes[0, t].imshow(s_x, cmap='viridis')
        axes[0, t].set_title(f't={t}', fontsize=10)  # t=0 to t=T-1
        axes[0, t].axis('off')
        plt.colorbar(im0, ax=axes[0, t], fraction=0.046)

        im1 = axes[1, t].imshow(s_y, cmap='viridis')
        axes[1, t].axis('off')
        plt.colorbar(im1, ax=axes[1, t], fraction=0.046)

        im2 = axes[2, t].imshow(s_mag, cmap='hot')
        axes[2, t].axis('off')
        plt.colorbar(im2, ax=axes[2, t], fraction=0.046)

        im3 = axes[3, t].imshow(v_mag, cmap='jet')
        axes[3, t].axis('off')
        plt.colorbar(im3, ax=axes[3, t], fraction=0.046)

    axes[0, 0].set_ylabel('s_x', fontsize=12)
    axes[1, 0].set_ylabel('s_y', fontsize=12)
    axes[2, 0].set_ylabel('|s|', fontsize=12)
    axes[3, 0].set_ylabel('|v|', fontsize=12)

    plt.tight_layout()
    save_path = f"{save_folder}/learned_scaling_factors.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_uncertainty_cardiac_frames(sampled_velocities, v_reg_list, Sdef_series, series,
                                          scaling_factors, brownian_bridge, save_folder,
                                          subject_idx=0, save_individual=False,
                                          uncertainty_space='velocity'):
    """
    Comprehensive visualization of uncertainty at cardiac frame timesteps.

    Args:
        sampled_velocities: Dict mapping frame_idx -> list of velocity tensors
        v_reg_list: List of registered velocity fields
        Sdef_series: List of deformed images from registration
        series: Original image series
        scaling_factors: [B, T, 2, H, W] learned scaling factors
        brownian_bridge: BrownianBridgeLearnedScaling instance
        save_folder: Where to save visualizations
        subject_idx: Which subject to visualize
        uncertainty_space: 'velocity' -> row 6 shows sqrt(var(v_x)+var(v_y))
                           'image'    -> row 6 shows var(m o phi) over sampled warps
    """
    os.makedirs(save_folder, exist_ok=True)

    num_cardiac_frames = len(v_reg_list)

    fig, axes = plt.subplots(10, num_cardiac_frames + 1, figsize=(2 * num_cardiac_frames + 1.5, 22),
                              gridspec_kw={'width_ratios': [1]*num_cardiac_frames + [0.08]})

    #source_img = series[subject_idx, 0].cpu().numpy().T  # Transpose (W,H)->(H,W) for display
    source_img = series[subject_idx, 0].cpu().numpy()

    # First pass: compute global min/max and store variance for each frame
    all_v_reg_mag = []
    all_mean_mag = []
    all_var_mag = []
    all_scale_mag = []
    all_mean_v = []  # Store mean velocity tensors for warping

    # Get source image tensor for warping
    source_img_tensor = series[subject_idx:subject_idx+1, 0:1].clone()  # [1, 1, H, W]
    img_size = source_img_tensor.shape[-1]
    inshape = (img_size, img_size)

    # Initialize velocity integrator for Jacobian computation
    vec_int = VecInt(inshape, TSteps=7).cuda()

    for i in range(num_cardiac_frames):
        v_reg_np = v_reg_list[i][subject_idx].cpu().numpy()
        #v_reg_mag = np.sqrt(v_reg_np[0]**2 + v_reg_np[1]**2).T
        v_reg_mag = np.sqrt(v_reg_np[0]**2 + v_reg_np[1]**2)
        all_v_reg_mag.append(v_reg_mag)

        # Compute variance from sampled trajectories
        if i in sampled_velocities and len(sampled_velocities[i]) > 0:
            velocities_at_t = np.stack([v[subject_idx].cpu().numpy() for v in sampled_velocities[i]], axis=0)
            mean_v = np.mean(velocities_at_t, axis=0)
            mean_mag = np.sqrt(mean_v[0]**2 + mean_v[1]**2)
            all_mean_v.append(torch.from_numpy(mean_v).unsqueeze(0).cuda())  # [1, 2, H, W]

            if uncertainty_space == 'image':
                # var(m o phi) over warped source samples
                with torch.no_grad():
                    warped_samples = []
                    for v in sampled_velocities[i]:
                        v_sub = v[subject_idx:subject_idx+1].cuda() if not v.is_cuda else v[subject_idx:subject_idx+1]
                        warped = warp_image_with_velocity(
                            source_img_tensor.cuda(), v_sub, inshape=inshape
                        )
                        warped_samples.append(warped[0, 0].cpu().numpy())
                warped_samples = np.stack(warped_samples, axis=0)
                var_mag = np.var(warped_samples, axis=0)
            else:
                variance = np.var(velocities_at_t, axis=0)
                var_mag = np.sqrt(variance[0] + variance[1])
        else:
            var_mag = np.zeros_like(v_reg_mag)
            mean_mag = v_reg_mag.copy()
            all_mean_v.append(v_reg_list[i][subject_idx:subject_idx+1].clone())  # Use reg velocity as fallback

        all_mean_mag.append(mean_mag)
        all_var_mag.append(var_mag)

        # Scaling factor magnitude
        s = scaling_factors[subject_idx, i].cpu().numpy()
        #s_mag = np.sqrt(s[0]**2 + s[1]**2).T
        s_mag = np.sqrt(s[0]**2 + s[1]**2)
        all_scale_mag.append(s_mag)

        all_var_vals = np.concatenate([v.ravel() for v in all_var_mag])
        var_vmin = np.percentile(all_var_vals, 2)
        var_vmax = np.percentile(all_var_vals, 70)
        var_vmax = max(var_vmax, 1e-6)  # Ensure non-zero range

    vel_vmin, vel_vmax = 0, max(np.max(all_v_reg_mag), np.max(all_mean_mag))
    # Note: scaling factors will use per-frame min/max instead of global

    # Compute global variance max (excluding endpoints which have near-zero variance)
    var_vmax = 0.0
    for i in range(num_cardiac_frames):
        if i > 0 and i < num_cardiac_frames - 1:  # Skip endpoints
            var_vmax = max(var_vmax, np.max(all_var_mag[i]))
    # Floor depends on uncertainty space: velocity magnitudes ~O(0.1-1),
    # image variances ~O(1e-4 - 1e-2). Use a small floor to avoid div-by-zero
    # without artificially compressing the image-space dynamic range.
    var_vmax = max(var_vmax, 1e-8)

    im_vel = None
    im_var = None

    for i in range(num_cardiac_frames):
        # At t=0, target = source (identity transformation)
        if i == 0:
            target_img = source_img  # t=0: target is the source itself
            deformed_img_reg = source_img  # t=0: no deformation
        else:
            #target_img = series[subject_idx, i].cpu().numpy().T  # Transpose (W,H)->(H,W)
            target_img = series[subject_idx, i].cpu().numpy()
            #deformed_img_reg = Sdef_series[i-1][subject_idx, 0].cpu().numpy().T  # Transpose (W,H)->(H,W)
            deformed_img_reg = Sdef_series[i-1][subject_idx, 0].cpu().numpy()

        v_reg_mag = all_v_reg_mag[i]
        mean_mag = all_mean_mag[i]
        var_mag = all_var_mag[i]
        s_mag = all_scale_mag[i]

        # Row 0: Source image
        axes[0, i].imshow(source_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Frame {i}', fontsize=10)  # t=0 to t=T-1
        axes[0, i].axis('off')

        # Row 1: Target image
        axes[1, i].imshow(target_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

        # Row 2: Deformed image (registration)
        axes[2, i].imshow(deformed_img_reg, cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')

        # Row 3: Warped image using mean sampled velocity (from Brownian Bridge)
        with torch.no_grad():
            mean_v_tensor = all_mean_v[i]
            warped_mean = warp_image_with_velocity(
                source_img_tensor, mean_v_tensor, inshape=(img_size, img_size)
            )
            #warped_mean_np = warped_mean[0, 0].cpu().numpy().T  # Transpose (W,H)->(H,W)
            warped_mean_np = warped_mean[0, 0].cpu().numpy()
        axes[3, i].imshow(warped_mean_np, cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')

        # Row 4: Registered velocity magnitude
        im_vel = axes[4, i].imshow(v_reg_mag, cmap='jet', vmin=vel_vmin, vmax=vel_vmax)
        axes[4, i].axis('off')

        # Row 5: Mean sampled velocity magnitude
        axes[5, i].imshow(mean_mag, cmap='jet', vmin=vel_vmin, vmax=vel_vmax)
        axes[5, i].axis('off')

        # Row 6: Variance map (globally normalized)
        # im_var = axes[6, i].imshow(var_mag, cmap='hot', vmin=0, vmax=var_vmax)
        # axes[6, i].axis('off')
        # if i == num_cardiac_frames - 1:
        #     plt.colorbar(im_var, ax=axes[6, i], fraction=0.046)

        # from matplotlib.ticker import MultipleLocator
        # from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

        # # Create custom colormap: viridis for 0-0.25, yellow for 0.25-1
        # # Get viridis colors for the lower range
        # viridis = plt.cm.viridis
        # n_viridis = 77  # Number of colors for 0-0.25 range
        # n_yellow = 179  # Number of colors for 0.25-1 range (3x more since it covers 3x the range)

        # # Build color list: viridis gradient for lower, yellow for upper
        # colors_lower = [viridis(i / n_viridis) for i in range(n_viridis)]
        # colors_upper = [(1.0, 1.0, 0.0, 1.0)] * n_yellow  # Yellow (RGBA)
        # all_colors = colors_lower + colors_upper

        # custom_cmap = LinearSegmentedColormap.from_list('viridis_yellow', all_colors, N=256)

        # # Normalize variance to 0-1 range
        var_mag_norm = (var_mag / var_vmax) if var_vmax > 0 else var_mag
        # Mask background: set uncertainty to NaN where brain is absent
        #brain_mask = source_img > 0.01
        #var_mag_masked = np.where(brain_mask, var_mag_norm, np.nan)
        # #var_mag_norm = var_mag
        #axes[6, i].imshow(var_mag, cmap='jet', vmin=var_vmin, vmax=var_vmax)
        axes[6, i].set_facecolor('black')
        var_vmin_plot = 0.0 if uncertainty_space == 'image' else 0.3
        axes[6, i].imshow(var_mag_norm, cmap='jet', vmin=var_vmin_plot, vmax=1)
        axes[6, i].axis('off')
        # var_mag_norm = (var_mag / var_vmax) if var_vmax > 0 else var_mag
        # #var_mag_norm = var_mag
        # axes[6, i].imshow(source_img, cmap='gray')
        # #im3 = axes[6, i].imshow(var_mag_norm, cmap='jet', vmin=0, vmax=1, alpha=0.8)  # jet_r
        # im3 = axes[6, i].imshow(var_mag_norm, cmap='jet', vmin=0.5, vmax=1, alpha=1.0)
        # #im3 = axes[6, i].imshow(var_mag_norm, cmap='jet', vmin=var_mag_norm.min(), vmax=var_vmax, alpha=1.0)
        # axes[6, i].axis('off')
        # if i == num_cardiac_frames - 1:
        #     plt.colorbar(im3, ax=axes[6, i], fraction=0.046)

        # Row 7: Learned scaling factor magnitude (per-frame colorbar)
        # im_scale = axes[7, i].imshow(s_mag, cmap='viridis')
        # axes[7, i].axis('off')
        # plt.colorbar(im_scale, ax=axes[7, i], fraction=0.046)

        axes[7, i].imshow(source_img, cmap='gray')
        im3 = axes[7, i].imshow(s_mag, cmap='viridis', vmin=0, vmax=s_mag.max(), alpha=0.7)
        axes[7, i].axis('off')
        if i == num_cardiac_frames - 1:
            plt.colorbar(im3, ax=axes[7, i], fraction=0.046)

        # Row 8: Transformation field (deformation grid overlay)
        with torch.no_grad():
            mean_v_tensor = all_mean_v[i]  # [1, 2, H, W]
            # Integrate velocity to get displacement field
            disp_list = vec_int(mean_v_tensor)
            displacement = disp_list[-1]  # [1, 2, H, W]

            # Get displacement field
            disp_x = displacement[0, 0]  # [H, W] - x displacement
            disp_y = displacement[0, 1]  # [H, W] - y displacement

            # Create base grid (original coordinates in pixels)
            grid_spacing = 4
            y_coords = torch.arange(0, img_size)
            x_coords = torch.arange(0, img_size)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Calculate deformed positions (displacement already in pixel-relative coords)
            deformed_x = grid_x.cuda() + disp_x 
            deformed_y = grid_y.cuda() + disp_y

        axes[8, i].imshow(source_img, cmap='gray', vmin=0, vmax=1)

        # # Plot horizontal grid lines (swap x/y for transposed image)
        # for j in range(0, img_size, grid_spacing):
        #     axes[8, i].plot(deformed_y[j, :].detach().cpu().numpy(),
        #                    deformed_x[j, :].detach().cpu().numpy(),
        #                    'r-', linewidth=0.8, alpha=0.9)

        # # Plot vertical grid lines (swap x/y for transposed image)
        # for j in range(0, img_size, grid_spacing):
        #     axes[8, i].plot(deformed_y[:, j].detach().cpu().numpy(),
        #                    deformed_x[:, j].detach().cpu().numpy(),
        #                    'r-', linewidth=0.8, alpha=0.9)

        # Plot horizontal grid lines
        for j in range(0, img_size, grid_spacing):
            axes[8, i].plot(deformed_x[j, :].detach().cpu().numpy(),
                           deformed_y[j, :].detach().cpu().numpy(),
                           'r-', linewidth=0.8, alpha=0.9)

        # Plot vertical grid lines
        for j in range(0, img_size, grid_spacing):
            axes[8, i].plot(deformed_x[:, j].detach().cpu().numpy(),
                           deformed_y[:, j].detach().cpu().numpy(),
                           'r-', linewidth=0.8, alpha=0.9)

        axes[8, i].set_xlim(0, img_size)
        axes[8, i].set_ylim(img_size, 0)  # Flip y-axis to match image coordinates
        axes[8, i].axis('off')

        # Row 9: Transformation field from registration model (v_reg_list)
        with torch.no_grad():
            v_reg_tensor = v_reg_list[i][subject_idx:subject_idx+1]  # [1, 2, W, H]
            # Integrate velocity to get displacement field
            disp_list_reg = vec_int(v_reg_tensor)
            displacement_reg = disp_list_reg[-1]  # [1, 2, W, H]

            # Get displacement field
            disp_x_reg = displacement_reg[0, 0]  # [W, H] - displacement along W
            disp_y_reg = displacement_reg[0, 1]  # [W, H] - displacement along H

            # Calculate deformed positions
            deformed_x_reg = grid_x.cuda() + disp_x_reg
            deformed_y_reg = grid_y.cuda() + disp_y_reg

        axes[9, i].imshow(source_img, cmap='gray', vmin=0, vmax=1)

        # # Plot horizontal grid lines (swap x/y for transposed image)
        # for j in range(0, img_size, grid_spacing):
        #     axes[9, i].plot(deformed_y_reg[j, :].detach().cpu().numpy(),
        #                    deformed_x_reg[j, :].detach().cpu().numpy(),
        #                    'r-', linewidth=0.8, alpha=0.9)

        # # Plot vertical grid lines (swap x/y for transposed image)
        # for j in range(0, img_size, grid_spacing):
        #     axes[9, i].plot(deformed_y_reg[:, j].detach().cpu().numpy(),
        #                    deformed_x_reg[:, j].detach().cpu().numpy(),
        #                    'r-', linewidth=0.8, alpha=0.9)

        # Plot horizontal grid lines
        for j in range(0, img_size, grid_spacing):
            axes[9, i].plot(deformed_x_reg[j, :].detach().cpu().numpy(),
                           deformed_y_reg[j, :].detach().cpu().numpy(),
                           'r-', linewidth=0.8, alpha=0.9)

        # Plot vertical grid lines
        for j in range(0, img_size, grid_spacing):
            axes[9, i].plot(deformed_x_reg[:, j].detach().cpu().numpy(),
                           deformed_y_reg[:, j].detach().cpu().numpy(),
                           'r-', linewidth=0.8, alpha=0.9)

        axes[9, i].set_xlim(0, img_size)
        axes[9, i].set_ylim(img_size, 0)  # Flip y-axis to match image coordinates
        axes[9, i].axis('off')

        # # Row 10: Negative Jacobian determinant (folding regions)
        # # Skip first and last frames (boundary frames)
        # if i == 0 or i == num_cardiac_frames - 1:
        #     axes[8, i].imshow(source_img, cmap='gray')
        #     axes[8, i].axis('off')
        #     axes[8, i].text(0.5, 0.5, 'N/A', transform=axes[8, i].transAxes,
        #                     fontsize=10, color='white', ha='center', va='center',
        #                     bbox=dict(boxstyle='round', facecolor='gray', alpha=0.5))
        # else:
        #     # Compute Jacobian determinant from displacement field (integrated velocity)
        #     mean_v_tensor = all_mean_v[i]  # [1, 2, H, W]
        #     with torch.no_grad():
        #         disp_list = vec_int(mean_v_tensor)
        #         displacement = disp_list[-1]  # [1, 2, H, W]
        #         disp_np = displacement.detach().cpu().numpy()

        #     # Use the same function as in test.py for consistency
        #     # Note: jacobian_det_for_one excludes boundary pixels (default 2)
        #     neg_pct, det_J_cropped = jacobian_det_for_one(disp_np, inshape=inshape)

        #     # Pad det_J back to full size for visualization overlay
        #     # Compute actual boundary based on size difference
        #     cropped_h, cropped_w = det_J_cropped.shape
        #     boundary_h = (img_size - cropped_h) // 2
        #     boundary_w = (img_size - cropped_w) // 2
        #     det_J_full = np.ones((img_size, img_size))  # Initialize with 1 (no folding)
        #     det_J_full[boundary_h:boundary_h+cropped_h, boundary_w:boundary_w+cropped_w] = det_J_cropped

        #     # Visualize: show source image with negative det regions highlighted
        #     axes[8, i].imshow(source_img, cmap='gray')
        #     # Show negative regions in red, with intensity proportional to how negative
        #     neg_det_values = np.where(det_J_full < 0, -det_J_full, 0)  # Magnitude of negative values
        #     if neg_det_values.max() > 0:
        #         im_neg_det = axes[8, i].imshow(neg_det_values, cmap='Reds',
        #                                         vmin=0, vmax=max(neg_det_values.max(), 0.1),
        #                                         alpha=0.8)
        #     else:
        #         # No negative values - show empty overlay
        #         im_neg_det = axes[8, i].imshow(neg_det_values, cmap='Reds',
        #                                         vmin=0, vmax=0.1, alpha=0.8)
        #     axes[8, i].axis('off')

        #     # Add count of negative pixels and percentage as text
        #     num_neg_pixels = np.sum(det_J_cropped < 0)
        #     axes[8, i].text(0.02, 0.98, f'n={num_neg_pixels}\n{neg_pct*100:.2f}%',
        #                     transform=axes[8, i].transAxes,
        #                     fontsize=8, color='white', verticalalignment='top',
        #                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        #     if i == num_cardiac_frames - 2:  # Add colorbar to second-to-last frame
        #         plt.colorbar(im_neg_det, ax=axes[8, i], fraction=0.046)

    # Colorbars in last column
    for row in range(4):
        axes[row, -1].axis('off')

    if im_vel is not None:
        cbar_vel = fig.colorbar(im_vel, cax=axes[4, -1], orientation='vertical')
        cbar_vel.set_label('|v|', fontsize=10)
        axes[5, -1].axis('off')

    # Turn off the last column for rows 6, 7, 8, and 9 (colorbars are inline now)
    axes[6, -1].axis('off')
    axes[7, -1].axis('off')
    axes[8, -1].axis('off')
    axes[9, -1].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Source', fontsize=12)
    axes[1, 0].set_ylabel('Target', fontsize=12)
    axes[2, 0].set_ylabel('Def (Reg)', fontsize=12)
    axes[3, 0].set_ylabel('Def (UQ)', fontsize=12)  # Deformed using mean sampled velocity
    axes[4, 0].set_ylabel('Reg |v|', fontsize=12)
    axes[5, 0].set_ylabel('Mean |v|', fontsize=12)
    var_label = 'Var (image)' if uncertainty_space == 'image' else 'Var (velocity)'
    axes[6, 0].set_ylabel(var_label, fontsize=12)
    axes[7, 0].set_ylabel('Scale |s|', fontsize=12)
    axes[8, 0].set_ylabel('Grid (UQ)', fontsize=12)
    axes[9, 0].set_ylabel('Grid (Reg)', fontsize=12)

    plt.tight_layout()
    save_path = f"{save_folder}/uncertainty_cardiac_frames_learned_scaling.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Save individual images
    if save_individual:
        indiv_dir = os.path.join(save_folder, "individual_images")
        os.makedirs(indiv_dir, exist_ok=True)

        # Source image (same for all frames)
        fig_s, ax_s = plt.subplots(figsize=(4, 4), facecolor='black')
        ax_s.imshow(source_img, cmap='gray', vmin=0, vmax=1)
        ax_s.axis('off')
        fig_s.savefig(os.path.join(indiv_dir, "source.png"), dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig_s)

        # var_mag_max = 0.0
        # for t in range(num_cardiac_frames):
        #     var = all_var_mag[t]
        #     var_mag_max = max(var_mag_max, var.max())
        # # Ensure vmax is not zero to avoid issues
        # var_mag_max = max(var_mag_max, 1e-6)

        for i in range(num_cardiac_frames):
            if i == 0:
                target_img = source_img
                deformed_img_reg = source_img
            else:
                #target_img = series[subject_idx, i].cpu().numpy().T
                target_img = series[subject_idx, i].cpu().numpy()
                #deformed_img_reg = Sdef_series[i-1][subject_idx, 0].cpu().numpy().T
                deformed_img_reg = Sdef_series[i-1][subject_idx, 0].cpu().numpy()

            var_mag = all_var_mag[i]

            # Warped using mean sampled velocity
            with torch.no_grad():
                mean_v_tensor = all_mean_v[i]
                warped_mean = warp_image_with_velocity(
                    source_img_tensor, mean_v_tensor, inshape=(img_size, img_size)
                )
                #warped_mean_np = warped_mean[0, 0].cpu().numpy().T
                warped_mean_np = warped_mean[0, 0].cpu().numpy()

            # Target
            fig_t, ax_t = plt.subplots(figsize=(4, 4), facecolor='black')
            ax_t.imshow(target_img, cmap='gray', vmin=0, vmax=1)
            ax_t.axis('off')
            fig_t.savefig(os.path.join(indiv_dir, f"target_frame_{i}.png"), dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_t)

            # Deformed (registration)
            fig_d, ax_d = plt.subplots(figsize=(4, 4), facecolor='black')
            ax_d.imshow(deformed_img_reg, cmap='gray', vmin=0, vmax=1)
            ax_d.axis('off')
            fig_d.savefig(os.path.join(indiv_dir, f"deformed_reg_frame_{i}.png"), dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_d)

            # Deformed (UQ mean velocity)
            fig_w, ax_w = plt.subplots(figsize=(4, 4), facecolor='black')
            ax_w.imshow(warped_mean_np, cmap='gray', vmin=0, vmax=1)
            ax_w.axis('off')
            fig_w.savefig(os.path.join(indiv_dir, f"deformed_uq_frame_{i}.png"), dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_w)

            # Uncertainty (variance) map — mask background to black
            var_mag_norm_indiv = (var_mag / var_vmax) if var_vmax > 0 else var_mag
            #brain_mask_indiv = source_img > 0.01
            #var_mag_masked_indiv = np.where(brain_mask_indiv, var_mag_norm_indiv, np.nan)
            fig_u, ax_u = plt.subplots(figsize=(4, 4), facecolor='black')
            ax_u.set_facecolor('black')
            # ax_u.imshow(var_mag, cmap='jet', vmin=var_vmin, vmax=var_vmax)
            var_vmin_indiv = 0.0 if uncertainty_space == 'image' else 0.3
            ax_u.imshow(var_mag_norm_indiv, cmap='jet', vmin=var_vmin_indiv, vmax=1)
            ax_u.axis('off')
            fig_u.savefig(os.path.join(indiv_dir, f"uncertainty_frame_{i}.png"), dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_u)
            # var_mag_norm = var_mag / var_vmax if var_vmax > 0 else var_mag
            # fig_u, ax_u = plt.subplots(figsize=(4, 4))
            # ax_u.imshow(source_img, cmap='gray')
            # #im_u = ax_u.imshow(var_mag_norm, cmap='hot', vmin=0, vmax=1, alpha=0.8)
            # im_u = ax_u.imshow(var_mag_norm, cmap='jet', vmin=0.5, vmax=1, alpha=1.0)   # 0, var_vmax 
            # ax_u.axis('off')
            # fig_u.savefig(os.path.join(indiv_dir, f"uncertainty_frame_{i}.png"), dpi=150, bbox_inches='tight', pad_inches=0)
            # plt.close(fig_u)

        # Save standalone colorbar (no tick labels) once
        fig_cb, ax_cb = plt.subplots(figsize=(0.3, 4))
        norm = plt.Normalize(vmin=0.2, vmax=0.7)
        cb = fig_cb.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet_r'),
                             cax=ax_cb, orientation='vertical')
        cb.set_ticks([])
        fig_cb.savefig(os.path.join(indiv_dir, "uncertainty_colorbar.png"), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig_cb)

        print(f"Saved {num_cardiac_frames * 4 + 1} individual images to: {indiv_dir}")


def visualize_scaling_factor_analysis(scaling_factors, v_reg_list, sampled_velocities,
                                       save_folder, subject_idx=0,
                                       uncertainty_space='velocity', source_img=None):
    """
    Detailed analysis of the learned scaling factors.

    Args:
        scaling_factors: [B, T, 2, H, W] learned scaling factors
        v_reg_list: List of velocity fields
        sampled_velocities: Dict mapping frame_idx -> list of velocity tensors
        save_folder: Where to save visualizations
        subject_idx: Which subject to analyze
        uncertainty_space: 'velocity' (default) -> panel 3 uses velocity-sample variance.
                           'image' -> panel 3 uses var(m o phi) over warped source samples.
        source_img: [B, 1, H, W] source image tensor; required when uncertainty_space='image'.
    """
    os.makedirs(save_folder, exist_ok=True)

    T = scaling_factors.shape[1]
    s = scaling_factors[subject_idx].cpu().numpy()  # [T, 2, H, W]

    # Create a comprehensive analysis figure
    fig = plt.figure(figsize=(16, 12))

    # 1. Scaling factor time evolution (spatial average)
    ax1 = fig.add_subplot(2, 3, 1)
    s_mean_per_t = [np.mean(np.sqrt(s[t, 0]**2 + s[t, 1]**2)) for t in range(T)]
    s_std_per_t = [np.std(np.sqrt(s[t, 0]**2 + s[t, 1]**2)) for t in range(T)]
    ax1.errorbar(range(T), s_mean_per_t, yerr=s_std_per_t, marker='o', capsize=3)  # t=0 to t=T-1
    ax1.set_xlabel('Cardiac Frame (t)')
    ax1.set_ylabel('Mean Scaling Factor |s|')
    ax1.set_title('Scaling Factor vs. Time')
    ax1.grid(True, alpha=0.3)

    # 2. Velocity magnitude vs scaling factor correlation
    ax2 = fig.add_subplot(2, 3, 2)
    all_v_mag = []
    all_s_mag = []
    for t in range(T):
        v = v_reg_list[t][subject_idx].cpu().numpy()
        v_mag = np.sqrt(v[0]**2 + v[1]**2).flatten()
        s_mag = np.sqrt(s[t, 0]**2 + s[t, 1]**2).flatten()
        all_v_mag.extend(v_mag)
        all_s_mag.extend(s_mag)

    # Subsample for visualization
    idx = np.random.choice(len(all_v_mag), min(5000, len(all_v_mag)), replace=False)
    ax2.scatter(np.array(all_v_mag)[idx], np.array(all_s_mag)[idx], alpha=0.1, s=1)
    ax2.set_xlabel('Velocity Magnitude |v|')
    ax2.set_ylabel('Scaling Factor |s|')
    ax2.set_title('Scaling vs. Velocity Correlation')

    # 3. Variance vs scaling factor correlation
    ax3 = fig.add_subplot(2, 3, 3)
    all_var = []
    all_s = []
    img_size = scaling_factors.shape[-1]
    inshape = (img_size, img_size)
    if uncertainty_space == 'image' and source_img is not None:
        src_sub = source_img[subject_idx:subject_idx+1].cuda() \
            if not source_img.is_cuda else source_img[subject_idx:subject_idx+1]
    else:
        src_sub = None

    for t in range(T):
        if t in sampled_velocities and len(sampled_velocities[t]) > 0:
            if uncertainty_space == 'image' and src_sub is not None:
                with torch.no_grad():
                    warped_samples = []
                    for vel in sampled_velocities[t]:
                        v_sub = vel[subject_idx:subject_idx+1].cuda() \
                            if not vel.is_cuda else vel[subject_idx:subject_idx+1]
                        warped = warp_image_with_velocity(src_sub, v_sub, inshape=inshape)
                        warped_samples.append(warped[0, 0].cpu().numpy())
                warped_samples = np.stack(warped_samples, axis=0)
                var_mag = np.var(warped_samples, axis=0).flatten()
            else:
                velocities_at_t = np.stack([vel[subject_idx].cpu().numpy() for vel in sampled_velocities[t]], axis=0)
                var = np.var(velocities_at_t, axis=0)
                var_mag = np.sqrt(var[0]**2 + var[1]**2).flatten()
            s_mag = np.sqrt(s[t, 0]**2 + s[t, 1]**2).flatten()
            all_var.extend(var_mag)
            all_s.extend(s_mag)

    if all_var:
        idx = np.random.choice(len(all_var), min(5000, len(all_var)), replace=False)
        ax3.scatter(np.array(all_var)[idx], np.array(all_s)[idx], alpha=0.1, s=1)
    var_xlabel = 'Variance (image)' if uncertainty_space == 'image' else 'Variance (velocity)'
    ax3.set_xlabel(var_xlabel)
    ax3.set_ylabel('Scaling Factor |s|')
    ax3.set_title('Scaling vs. Variance Correlation')

    # 4. Scaling factor histogram
    ax4 = fig.add_subplot(2, 3, 4)
    s_all = np.sqrt(s[:, 0]**2 + s[:, 1]**2).flatten()
    ax4.hist(s_all, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(s_all), color='red', linestyle='--', label=f'Mean: {np.mean(s_all):.3f}')
    ax4.set_xlabel('Scaling Factor |s|')
    ax4.set_ylabel('Count')
    ax4.set_title('Scaling Factor Distribution')
    ax4.legend()

    # 5. Scaling factor x component spatial pattern (middle frame)
    ax5 = fig.add_subplot(2, 3, 5)
    mid_t = T // 2
    #im5 = ax5.imshow(s[mid_t, 0].T, cmap='RdBu_r')
    im5 = ax5.imshow(s[mid_t, 0], cmap='RdBu_r')
    ax5.set_title(f's_x at Frame {mid_t}')  # t=0 to t=T-1
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # 6. Scaling factor y component spatial pattern (middle frame)
    ax6 = fig.add_subplot(2, 3, 6)
    #im6 = ax6.imshow(s[mid_t, 1].T, cmap='RdBu_r')
    im6 = ax6.imshow(s[mid_t, 1], cmap='RdBu_r')
    ax6.set_title(f's_y at Frame {mid_t}')  # t=0 to t=T-1
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    plt.tight_layout()
    save_path = f"{save_folder}/scaling_factor_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_history(training_history, save_folder):
    """
    Plot training loss history.

    Args:
        training_history: List of loss dictionaries
        save_folder: Where to save the plot
    """
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(training_history) + 1)
    plt.plot(epochs, [h['total'] for h in training_history], label='Total Loss')
    plt.plot(epochs, [h['similarity'] for h in training_history], label='Similarity')
    plt.plot(epochs, [h['regularity'] for h in training_history], label='Regularity')
    plt.plot(epochs, [h['inverse_scale'] for h in training_history], label='Inverse Scale')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig(f"{save_folder}/training_history.png", dpi=150)
    plt.close()
    print(f"Saved: {save_folder}/training_history.png")


def save_statistics(save_folder, subject_idx, num_diffusion_steps, num_runs, num_time_steps,
                    num_epochs, lambda_sim, lambda_reg, lambda_scale,
                    scaling_factors, ncc_vx_per_frame, overall_ncc_vx, overall_std):
    """
    Save uncertainty quantification statistics to a text file.

    Args:
        save_folder: Where to save the file
        subject_idx: Subject index
        num_diffusion_steps: Number of diffusion steps (T)
        num_runs: Number of independent runs (N)
        num_time_steps: Number of cardiac frames
        num_epochs: Number of training epochs
        lambda_sim: Similarity loss weight
        lambda_reg: Regularization loss weight
        lambda_scale: Scale loss weight
        scaling_factors: [B, T, 2, H, W] learned scaling factors
        ncc_vx_per_frame: Dict with per-frame NCC_VX statistics
        overall_ncc_vx: Overall NCC_VX value
        overall_std: Standard deviation across frames
    """
    os.makedirs(save_folder, exist_ok=True)

    stats_file = f"{save_folder}/uncertainty_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Uncertainty Quantification with Learned Scaling Factors\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Subject index: {subject_idx}\n")
        f.write(f"Number of diffusion steps (T): {num_diffusion_steps}\n")
        f.write(f"Number of independent runs (N): {num_runs}\n")
        f.write(f"Number of cardiac frames: {num_time_steps}\n")
        f.write(f"Training epochs: {num_epochs}\n\n")

        f.write("Loss Weights (Equation 2):\n")
        f.write(f"  lambda_sim: {lambda_sim}\n")
        f.write(f"  lambda_reg: {lambda_reg}\n")
        f.write(f"  lambda_scale: {lambda_scale}\n\n")

        f.write("Learned Scaling Factor Statistics:\n")
        f.write(f"  Mean: {scaling_factors.mean().item():.6f}\n")
        f.write(f"  Std: {scaling_factors.std().item():.6f}\n")
        f.write(f"  Min: {scaling_factors.min().item():.6f}\n")
        f.write(f"  Max: {scaling_factors.max().item():.6f}\n\n")

        f.write("=" * 60 + "\n")
        f.write("NCC_VX Calibration Metric\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall NCC_VX: {overall_ncc_vx:.6f} +/- {overall_std:.6f}\n\n")

        f.write("Per-frame NCC_VX:\n")
        if isinstance(ncc_vx_per_frame, dict):
            f.write("Frame (t)\tNCC_VX\tMean_Variance\tMean_MSE\n")
            for frame_idx in sorted(ncc_vx_per_frame.keys()):
                stats = ncc_vx_per_frame[frame_idx]
                f.write(f"{frame_idx}\t{stats['ncc_vx']:.6f}\t"
                       f"{stats['mean_variance']:.6f}\t{stats['mean_mse']:.6f}\n")
        else:
            f.write("Frame (t)\tNCC_VX\n")
            for frame_idx, val in enumerate(ncc_vx_per_frame):
                f.write(f"{frame_idx}\t{val:.6f}\n")

    print(f"Saved: {stats_file}")
