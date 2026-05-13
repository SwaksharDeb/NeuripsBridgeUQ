"""
Trainer for the scaling factor prediction network with posterior reverse SDE.

Key differences from uncertainty_sde_velInput:
  1. The scaling network receives BOTH images AND velocity fields as input.
  2. Uses MLE optimization (num_em_samples=1) instead of EM.
"""

import os
import sys

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from uncertainty_brain_sde_v3.warping import warp_image_with_velocity


class _NullContext:
    """No-op context manager — used in place of autocast when AMP is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class ScalingFactorTrainer:
    """
    Trainer for the scaling factor prediction network.
    Uses posterior reverse SDE and combined image+velocity input scaling network.
    Uses MLE optimization (single sample, no EM averaging).
    """

    def __init__(self, scaling_network, brownian_bridge, loss_fn,
                 lr=1e-4, device='cuda', img_size=128,
                 gamma=1.0, guidance_scale=1.0,
                 use_amp=False):
        # scaling_network may be None when only compute_brownian_bridge_velocities
        # is used (e.g. test-time inference from a saved s^K artifact). In that
        # case _predict_scaling_factors and train_step will fail; that's expected.
        self.scaling_network = scaling_network.to(device) if scaling_network is not None else None
        self.brownian_bridge = brownian_bridge
        self.loss_fn = loss_fn
        self.device = device
        self.img_size = img_size
        self.gamma = gamma
        self.guidance_scale = guidance_scale
        # Mixed precision: fp16 activations for the network forward + warp + loss;
        # the SDE coefficient math (sigma_sq, delta_t) is forced to fp32 inside
        # brownian_bridge.run_posterior_reverse_process for numerical safety.
        self.use_amp = bool(use_amp) and (str(device) == 'cuda')

        if scaling_network is not None:
            self.optimizer = torch.optim.Adam(
                self.scaling_network.parameters(),
                lr=lr,
                weight_decay=1e-5
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
            )
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        else:
            self.optimizer = None
            self.scheduler = None
            self.scaler = None

    def scheduler_step(self, val_loss):
        """Step the LR scheduler with the validation/epoch loss."""
        self.scheduler.step(val_loss)

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def warp_image(self, source, velocity):
        """Warp source image with velocity field."""
        inshape = (self.img_size, self.img_size)
        return warp_image_with_velocity(source, velocity, inshape=inshape)

    def compute_brownian_bridge_velocities(self, v_reg_list, scaling_factors,
                                            source_img=None, target_imgs=None):
        """
        Compute velocities at each cardiac frame using the Brownian Bridge formula
        with posterior reverse SDE (energy gradient correction).
        """
        bb = self.brownian_bridge
        num_cardiac_frames = len(v_reg_list)

        v_0 = v_reg_list[0].clone()
        v_T = v_reg_list[-1].clone()

        # Run reverse process (posterior if images provided, prior otherwise)
        if source_img is not None and target_imgs is not None:
            trajectory = bb.run_posterior_reverse_process(
                v_reg_list, scaling_factors, v_0, v_T,
                source_img, target_imgs,
                gamma=self.gamma, guidance_scale=self.guidance_scale,
                device=self.device
            )
        else:
            trajectory = bb.run_reverse_process(
                v_reg_list, scaling_factors, v_0, v_T,
                device=self.device
            )

        # Extract v_t at cardiac frame boundaries from trajectory
        cardiac_frame_to_v_t = {}
        cardiac_frame_to_v_t[num_cardiac_frames - 1] = trajectory[0]  # v_T
        cardiac_frame_to_v_t[0] = trajectory[-1]  # v_0

        for traj_idx, t in enumerate(range(bb.T - 1, -1, -1)):
            if t > 0:
                cardiac_idx = bb.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)
                frame_idx = int(round(cardiac_idx))
                if 0 <= frame_idx < num_cardiac_frames and frame_idx not in cardiac_frame_to_v_t:
                    cardiac_frame_to_v_t[frame_idx] = trajectory[traj_idx + 1]

        # Build output lists
        v_t_list = []
        std_tilde_list = []
        for frame_idx in range(num_cardiac_frames):
            m_frame = frame_idx / (num_cardiac_frames - 1) if num_cardiac_frames > 1 else 0
            s_frame = scaling_factors[:, frame_idx]

            if frame_idx == 0 or frame_idx == num_cardiac_frames - 1:
                marginal_std = torch.zeros_like(s_frame)
            else:
                marginal_variance = 2 * s_frame * (m_frame - m_frame ** 2)
                marginal_std = torch.sqrt(torch.clamp(marginal_variance, min=0.0))

            std_tilde_list.append(marginal_std)

            if frame_idx in cardiac_frame_to_v_t:
                v_t_list.append(cardiac_frame_to_v_t[frame_idx])
            else:
                v_t_list.append(v_reg_list[frame_idx])

        return v_t_list, std_tilde_list

    def compute_mean_velocity(self, v_reg_list, scaling_factors_bb,
                              source_img, target_imgs, num_samples):
        """
        Vectorized N-sample mean velocity from the posterior reverse SDE.

        Stacks every input N times along the batch dim (effective batch =
        N*B), then makes ONE SDE call. Each reverse step's
        ``torch.randn_like(v_t)`` then draws N*B independent Gaussians,
        which is statistically identical to N i.i.d. trajectories per
        original sample. The output is reshaped back to ``[N, B, ...]`` and
        averaged along the N axis.

        Eliminates the Python for-loop over samples and lets the GPU
        saturate larger kernels — typically 2-5x faster than a per-sample
        loop. Total memory is roughly the same (same total tensors retained
        for backward).

        Returns:
            v_bar_list:  list of [B, 2, H, W] (gradient-attached)
            u_list:      list of [B, 2, H, W] (detached, sample std per channel)
        """
        N = int(num_samples)
        if N <= 1:
            v_t_list, _ = self.compute_brownian_bridge_velocities(
                v_reg_list, scaling_factors_bb,
                source_img=source_img, target_imgs=target_imgs,
            )
            u_list = [torch.zeros_like(v) for v in v_t_list]
            return v_t_list, u_list

        B = v_reg_list[0].shape[0]

        def _rep(t):
            return t.repeat(N, *([1] * (t.dim() - 1)))

        v_reg_NB           = [_rep(v) for v in v_reg_list]
        scaling_factors_NB = _rep(scaling_factors_bb)
        source_img_NB      = _rep(source_img)
        target_imgs_NB     = [_rep(t) for t in target_imgs]

        v_t_list_NB, _ = self.compute_brownian_bridge_velocities(
            v_reg_NB, scaling_factors_NB,
            source_img=source_img_NB, target_imgs=target_imgs_NB,
        )

        v_bar_list = []
        u_list = []
        for v_NB in v_t_list_NB:
            v_view = v_NB.view(N, B, *v_NB.shape[1:])
            v_bar = v_view.mean(dim=0)
            with torch.no_grad():
                centered = v_view.detach() - v_bar.detach().unsqueeze(0)
                u = (centered ** 2).mean(dim=0).clamp_min(0.0).sqrt()
            v_bar_list.append(v_bar)
            u_list.append(u)

        return v_bar_list, u_list

    def _marginal_std_from_scaling(self, scaling_factors_bb, num_frames):
        """
        Analytical bridge marginal std at each cardiac frame from the current
        scaling factors. Mirrors the std logic inside
        ``compute_brownian_bridge_velocities`` but avoids re-running the SDE.
        """
        out = []
        last_idx = scaling_factors_bb.shape[1] - 1
        for t in range(num_frames):
            m = t / (num_frames - 1) if num_frames > 1 else 0
            s = scaling_factors_bb[:, min(t, last_idx)]
            if t == 0 or t == num_frames - 1:
                out.append(torch.zeros_like(s))
            else:
                var = 2 * s * (m - m ** 2)
                out.append(torch.sqrt(torch.clamp(var, min=0.0)))
        return out

    def _predict_scaling_factors(self, source_img, target_imgs, v_reg_list, v_current_list=None):
        """
        Predict scaling factors from BOTH images AND velocity fields,
        plus the current iterate v^k for the third projection branch.
        """
        # Exclude t=0 (zero velocity) -- network sees only the T registration velocities
        v_reg_no_t0 = v_reg_list[1:]  # List of T tensors [B, 2, H, W]
        v_cur_no_t0 = v_reg_no_t0 if v_current_list is None else v_current_list[1:]

        scaling_factors_raw = self.scaling_network(
            source_img, target_imgs,
            v_reg_list=v_reg_no_t0,
            v_current_list=v_cur_no_t0,
        )
        # Clamp to prevent extreme values that cause NaN in reverse SDE
        scaling_factors_raw = torch.clamp(scaling_factors_raw, min=1e-4, max=1e4)

        # Expand to [B, T, 2, H, W] by repeating across time
        num_time_steps = len(v_reg_list) - 1  # exclude t=0
        scaling_factors = scaling_factors_raw.unsqueeze(1).expand(-1, num_time_steps, -1, -1, -1)
        zero_scaling = torch.zeros_like(scaling_factors[:, 0:1])
        scaling_factors_bb = torch.cat([zero_scaling, scaling_factors], dim=1)

        return scaling_factors_raw, scaling_factors, scaling_factors_bb

    def train_step(self, source_img, target_imgs, v_reg_list,
                   inner_iterations=2000,
                   num_train_samples=1,
                   vis_save_path=None, vis_subject_idx=None, vis_interval=50):
        """
        Iterative MLE training: for k in range(K), predict scaling using
        (images, v_reg, v_current^k), draw N posterior reverse-SDE trajectories,
        average them into v_bar, backprop loss(v_bar), optimizer.step, then
        v_current^{k+1} = v_bar.detach().

        With num_train_samples=1 the loop collapses to a single trajectory
        (matches v2 behavior exactly).

        Returns (last_loss_dict, scaling_factors_bb.detach()) so callers can
        run UQ on the same minibatch under no-graph conditions.
        """
        self.scaling_network.train()
        bb = self.brownian_bridge
        num_cardiac_frames = len(v_reg_list)

        # v^0 = v^r (detached, no grad)
        v_current_list = [v.clone().detach() for v in v_reg_list]
        last_loss_dict = None
        scaling_factors_bb = None

        amp_ctx = (torch.cuda.amp.autocast(enabled=self.use_amp)
                   if self.use_amp else _NullContext())

        inner_pbar = tqdm(range(inner_iterations), desc="  inner iters",
                          leave=False, position=1)
        for k in inner_pbar:
            self.optimizer.zero_grad()

            with amp_ctx:
                # Predict scaling using images, v_reg, and current v^k
                _, scaling_factors, scaling_factors_bb = self._predict_scaling_factors(
                    source_img, target_imgs, v_reg_list,
                    v_current_list=v_current_list,
                )

                # delta_tilde at each cardiac frame
                delta_tilde_per_frame = {}
                for fi in range(1, num_cardiac_frames - 1):
                    t_diff = int(round(fi * bb.T / (num_cardiac_frames - 1)))
                    s_t    = bb.interpolate_scaling_factors(scaling_factors_bb, fi)
                    s_prev = bb.interpolate_scaling_factors(scaling_factors_bb, fi - 1)
                    _, _, _, delta_tilde = bb.compute_coefficients(t_diff, s_t, s_prev)
                    delta_tilde_per_frame[fi] = delta_tilde

                if k >= 1:
                    warped_imgs_NB = [self.warp_image(source_img_NB, v_t) for v_t in v_t_list_NB]
                    total_loss, loss_dict = self.loss_fn.compute_loss(
                        source_img_NB, target_imgs_NB,
                        warped_imgs_NB[1:], v_t_list_NB[1:], sf_NB,
                        delta_tilde_per_frame=delta_tilde_NB,
                        v_t_list=v_t_list_NB, v_reg_list=v_reg_NB,
                        num_cardiac_frames=num_cardiac_frames,
                    )
                #if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                # else:
                #     total_loss.backward()
                #     torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
                #     self.optimizer.step()

                # Sample N posterior reverse-SDE trajectories and compute the
                # MEAN-OF-LOSS estimator of E_{v ~ p_σ}[L_one(v, σ)] (paper
                # Eq. 10, Algorithm 1 line 5). Implementation: tile every
                # input N times along the batch dim so the SDE produces N*B
                # independent samples in one call, then pass the tiled σ /
                # tiled v to compute_loss. The .mean() reductions inside
                # compute_loss naturally average v-dependent terms over the
                # N samples, while σ-only terms (inv_gamma_prior) are
                # invariant under tiling — together this is the unbiased
                # MC estimator of the expected loss.
                #
                # N=1 keeps the simple non-tiled path (no extra repeat).
                if num_train_samples <= 1:
                    v_t_list, std_tilde_list = self.compute_brownian_bridge_velocities(
                        v_reg_list, scaling_factors_bb,
                        source_img=source_img, target_imgs=target_imgs,
                    )
                    u_train_list = None

                    warped_imgs = [self.warp_image(source_img, v_t) for v_t in v_t_list]
                    total_loss, loss_dict = self.loss_fn.compute_loss(
                        source_img, target_imgs, warped_imgs[1:], v_t_list[1:], scaling_factors,
                        delta_tilde_per_frame=delta_tilde_per_frame,
                        v_t_list=v_t_list, v_reg_list=v_reg_list,
                        num_cardiac_frames=num_cardiac_frames,
                    )
                    v_bar_list = v_t_list
                else:
                    N = int(num_train_samples)
                    B = v_reg_list[0].shape[0]

                    def _rep(t):
                        return t.repeat(N, *([1] * (t.dim() - 1)))

                    v_reg_NB        = [_rep(v) for v in v_reg_list]
                    source_img_NB   = _rep(source_img)
                    target_imgs_NB  = [_rep(t) for t in target_imgs]
                    sf_bb_NB        = _rep(scaling_factors_bb)
                    sf_NB           = _rep(scaling_factors)
                    delta_tilde_NB  = {fi: _rep(d) for fi, d in delta_tilde_per_frame.items()}

                    # SDE at effective batch N*B — N i.i.d. trajectories per
                    # original batch element. σ stays in the graph via the
                    # tiled sf_bb_NB, so reparameterization gives path-wise
                    # gradients on σ for every sample.
                    v_t_list_NB, _ = self.compute_brownian_bridge_velocities(
                        v_reg_NB, sf_bb_NB,
                        source_img=source_img_NB, target_imgs=target_imgs_NB,
                    )
                    #v_t_list_NB_prev = v_t_list_NB

                    # warped_imgs_NB = [self.warp_image(source_img_NB, v_t) for v_t in v_t_list_NB]
                    # total_loss, loss_dict = self.loss_fn.compute_loss(
                    #     source_img_NB, target_imgs_NB,
                    #     warped_imgs_NB[1:], v_t_list_NB[1:], sf_NB,
                    #     delta_tilde_per_frame=delta_tilde_NB,
                    #     v_t_list=v_t_list_NB, v_reg_list=v_reg_NB,
                    #     num_cardiac_frames=num_cardiac_frames,
                    # )

                    # Reshape [N*B, ...] -> [N, B, ...] for v_bar / u (used
                    # for v_current^{r+1} and the uncertainty map). Detached
                    # so they don't appear in backward.
                    v_bar_list = []
                    u_train_list = []
                    with torch.no_grad():
                        for v_NB in v_t_list_NB:
                            v_view = v_NB.view(N, B, *v_NB.shape[1:]).detach()
                            v_bar = v_view.mean(dim=0)
                            centered = v_view - v_bar.unsqueeze(0)
                            u = (centered ** 2).mean(dim=0).clamp_min(0.0).sqrt()
                            v_bar_list.append(v_bar)
                            u_train_list.append(u)

                    std_tilde_list = self._marginal_std_from_scaling(
                        scaling_factors_bb, num_cardiac_frames
                    )

                    # Visualize the per-batch v_bar / warps, not the tiled
                    # versions — keeps figures at original batch size.
                    with torch.no_grad():
                        warped_imgs = [self.warp_image(source_img, v_t) for v_t in v_bar_list]
                    v_t_list = v_bar_list

            # if self.use_amp:
            #     self.scaler.scale(total_loss).backward()
            #     self.scaler.unscale_(self.optimizer)
            #     torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            #     total_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
            #     self.optimizer.step()

            # v_current^{k+1} = v_perturbed.detach() for next iteration
            # (When num_train_samples > 1, v_t_list holds the per-frame mean
            # v_bar; matches Algorithm 1 line 7: v^{r+1} <- {v_bar_t}.)
            v_current_list = [v.detach() for v in v_t_list]

            if k >=1:
                loss_dict['total'] = total_loss.item()
                if u_train_list is not None:
                    loss_dict['mean_u'] = float(
                        torch.stack([u.mean() for u in u_train_list]).mean().item()
                    )
                last_loss_dict = loss_dict

                inner_pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'sim': f"{loss_dict['similarity']:.4f}",
                    'bn': f"{loss_dict.get('bridge_norm', 0):.4f}",
                    'inv_gamma': f"{loss_dict['inv_gamma_prior']:.4f}",
                    'lr': f"{self.get_lr():.2e}",
                })

            if vis_save_path is not None and (k % vis_interval == 0
                                              or k == inner_iterations - 1):
                self.visualize_training_step(
                    scaling_factors_bb, std_tilde_list, v_reg_list, v_t_list,
                    source_img, target_imgs, warped_imgs,
                    vis_save_path, epoch=k, subject_idx=vis_subject_idx,
                )

        inner_pbar.close()

        # The σ inside the loop is f_{θ_{k}}(v^{k}) — predicted at the START
        # of iteration k, before that iteration's optimizer.step(). So at
        # exit, scaling_factors_bb is σ^{K-1} (one step stale). Re-predict
        # once with the FINAL θ_K and the final v_current = v^K so the s^K
        # artifact reflects the fully-trained network.
        self.scaling_network.eval()
        with torch.no_grad():
            _, _, scaling_factors_bb = self._predict_scaling_factors(
                source_img, target_imgs, v_reg_list,
                v_current_list=v_current_list,
            )
        self.scaling_network.train()

        return last_loss_dict, scaling_factors_bb.detach()

    def visualize_training_step(self, scaling_factors, std_tilde_list, v_reg_list, v_t_list,
                                 source_img, target_imgs, warped_imgs,
                                 save_path, epoch, subject_idx=None):
        """Visualize scaling factors, std_tilde, and velocities during training."""
        os.makedirs(save_path, exist_ok=True)

        num_frames = len(v_reg_list)
        fig, axes = plt.subplots(7, num_frames, figsize=(2.5 * num_frames, 16))

        with torch.no_grad():
            batch_size = source_img.shape[0]
            if subject_idx is None:
                subject_idx = np.random.randint(0, batch_size)

            src = source_img[subject_idx, 0].cpu().numpy()

            std_mag_max = 0.0
            for t in range(num_frames):
                std = std_tilde_list[t][subject_idx].cpu().numpy()
                std_mag = np.sqrt(std[0] + std[1])
                std_mag_max = max(std_mag_max, std_mag.max())
            std_mag_max = max(std_mag_max, 1e-6)

            for t in range(num_frames):
                if t == 0:
                    target = src
                else:
                    target = target_imgs[t-1][subject_idx, 0].cpu().numpy()

                axes[0, t].imshow(target, cmap='gray', vmin=0, vmax=1)
                axes[0, t].set_title(f't={t}', fontsize=10)
                axes[0, t].axis('off')

                warped = warped_imgs[t][subject_idx, 0].cpu().numpy()
                axes[1, t].imshow(warped, cmap='gray', vmin=0, vmax=1)
                axes[1, t].axis('off')

                v_reg = v_reg_list[t][subject_idx].cpu().numpy()
                v_reg_mag = np.sqrt(v_reg[0]**2 + v_reg[1]**2)
                im2 = axes[2, t].imshow(v_reg_mag, cmap='jet')
                axes[2, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im2, ax=axes[2, t], fraction=0.046)

                s = scaling_factors[subject_idx, t].cpu().numpy()
                s_mag = np.sqrt(s[0]**2 + s[1]**2)
                axes[3, t].imshow(src, cmap='gray', vmin=0, vmax=1)
                im3 = axes[3, t].imshow(s_mag, cmap='viridis', alpha=0.7)
                axes[3, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im3, ax=axes[3, t], fraction=0.046)

                std = std_tilde_list[t][subject_idx].cpu().numpy()
                std_mag = np.sqrt(std[0] + std[1])
                im4 = axes[4, t].imshow(std_mag, cmap='hot', vmin=0.0, vmax=std_mag_max)
                axes[4, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im4, ax=axes[4, t], fraction=0.046)

                v_t = v_t_list[t][subject_idx].cpu().numpy()
                v_t_mag = np.sqrt(v_t[0]**2 + v_t[1]**2)
                im5 = axes[5, t].imshow(v_t_mag, cmap='jet')
                axes[5, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im5, ax=axes[5, t], fraction=0.046)

                diff = np.abs(warped - target)
                im6 = axes[6, t].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
                axes[6, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im6, ax=axes[6, t], fraction=0.046)

        axes[0, 0].set_ylabel('Target', fontsize=12)
        axes[1, 0].set_ylabel('Warped', fontsize=12)
        axes[2, 0].set_ylabel('|v_reg|', fontsize=12)
        axes[3, 0].set_ylabel('|s| (scale)', fontsize=12)
        axes[4, 0].set_ylabel('|std_tilde|', fontsize=12)
        axes[5, 0].set_ylabel('|v_t| (BB)', fontsize=12)
        axes[6, 0].set_ylabel('|warp-target|', fontsize=12)

        plt.suptitle(f'Epoch {epoch + 1} (Subject {subject_idx})', fontsize=14)
        plt.tight_layout()

        save_file = f"{save_path}/training_vis_iter{epoch:04d}.png"
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
        plt.close()

    def visualize_epoch(self, source_img, target_imgs, v_reg_list, save_path, epoch, subject_idx=None):
        """Visualize at the end of an epoch."""
        self.scaling_network.eval()

        with torch.no_grad():
            _, _, scaling_factors_bb = self._predict_scaling_factors(
                source_img, target_imgs, v_reg_list
            )

            # Use prior reverse process for visualization (no energy gradient)
            v_t_list, std_tilde_list = self.compute_brownian_bridge_velocities(
                v_reg_list, scaling_factors_bb
            )

            warped_imgs = []
            for v_t in v_t_list:
                warped = self.warp_image(source_img, v_t)
                warped_imgs.append(warped)

            self.visualize_training_step(
                scaling_factors_bb, std_tilde_list, v_reg_list, v_t_list,
                source_img, target_imgs, warped_imgs,
                save_path, epoch, subject_idx
            )

        self.scaling_network.train()

