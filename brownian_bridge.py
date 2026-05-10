"""
Brownian Bridge Diffusion with learned spatially-varying scaling factors
and Posterior Reverse SDE for uncertainty quantification.

Identical to uncertainty_inv_gamma_prior/brownian_bridge.py except:
- run_posterior_reverse_process: adds energy gradient correction to the
  discrete bridge formula to steer samples toward lower registration error.

The posterior correction modifies the bridge conditional mean:
    mu_posterior = mu_tilde - (sigma^2 / gamma) * grad_E(v_t)

where sigma^2 = 2s/T, and grad_E is detached (only steers samples,
does not affect training gradients which flow through mu_tilde and std_tilde).
"""

import os
import sys

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn.functional as F
import numpy as np

from uncertainty_brain_sde_v3.warping import warp_image_with_velocity
from uncertainty_brain_sde_v3.losses import Grad


class BrownianBridgeLearnedScaling:
    """
    Brownian Bridge Diffusion with learned spatially-varying scaling factors
    and posterior reverse SDE support.
    """

    def __init__(self, num_diffusion_steps=1000, img_size=128, reg_weight=0.1):
        self.T = num_diffusion_steps
        self.m = torch.linspace(0, 1, self.T + 1)
        self.img_size = img_size
        self.grad_loss = Grad(penalty='l2')
        self.reg_weight = reg_weight

    def compute_delta_t(self, t, scaling_factors):
        """Compute delta_t = 2s(m_t - m_t^2) with spatially-varying s."""
        m_t = self.m[t]
        return 2 * scaling_factors * (m_t - m_t ** 2)

    def compute_delta_conditional(self, t, scaling_factors_t, scaling_factors_t_minus_1):
        """Compute delta_{t|t-1} with spatially-varying scaling factors."""
        if t <= 1:
            return torch.zeros_like(scaling_factors_t)

        delta_t = self.compute_delta_t(t, scaling_factors_t)
        delta_t_minus_1 = self.compute_delta_t(t - 1, scaling_factors_t_minus_1)

        m_t = self.m[t]
        m_t_minus_1 = self.m[t - 1]

        ratio = ((1 - m_t) / (1 - m_t_minus_1 + 1e-10)) ** 2
        delta_conditional = delta_t - delta_t_minus_1 * ratio

        return torch.clamp(delta_conditional, min=1e-10)

    def compute_coefficients(self, t, scaling_factors_t, scaling_factors_t_minus_1):
        """Compute coefficients c_xt, c_x0, c_y for the reverse process mean."""
        if t <= 1:
            ones = torch.ones_like(scaling_factors_t)
            zeros = torch.zeros_like(scaling_factors_t)
            return zeros, ones, zeros, zeros

        m_t = self.m[t]
        m_t_minus_1 = self.m[t - 1]

        delta_t = self.compute_delta_t(t, scaling_factors_t)
        delta_t_minus_1 = self.compute_delta_t(t - 1, scaling_factors_t_minus_1)
        delta_conditional = self.compute_delta_conditional(t, scaling_factors_t, scaling_factors_t_minus_1)

        delta_t_safe = torch.clamp(delta_t, min=1e-10)

        alpha = (1 - m_t) / (1 - m_t_minus_1 + 1e-10)
        beta = m_t - alpha * m_t_minus_1

        c_xt = (delta_t_minus_1 / delta_t_safe) * alpha
        c_x0 = (delta_conditional * (1 - m_t_minus_1)) / delta_t_safe
        c_y = (delta_conditional * m_t_minus_1 - delta_t_minus_1 * alpha * beta) / delta_t_safe

        delta_tilde = (delta_conditional * delta_t_minus_1) / delta_t_safe
        delta_tilde = torch.clamp(delta_tilde, min=0.0)

        return c_xt, c_x0, c_y, delta_tilde

    def interpolate_velocity(self, v_reg_list, cardiac_frame_idx):
        """Interpolate velocity field for a given diffusion timestep."""
        K = len(v_reg_list)
        cardiac_frame_idx = max(0, min(K - 1, cardiac_frame_idx))

        idx_low = int(np.floor(cardiac_frame_idx))
        idx_high = int(np.ceil(cardiac_frame_idx))

        if idx_low == idx_high or idx_high >= K:
            return v_reg_list[min(idx_low, K - 1)]

        alpha = cardiac_frame_idx - idx_low
        return v_reg_list[idx_low] * (1 - alpha) + v_reg_list[idx_high] * alpha

    def interpolate_scaling_factors(self, scaling_factors, cardiac_frame_idx):
        """Interpolate scaling factors for a given cardiac frame index."""
        T = scaling_factors.shape[1]
        cardiac_frame_idx = max(0, min(T - 1, cardiac_frame_idx))

        idx_low = int(np.floor(cardiac_frame_idx))
        idx_high = int(np.ceil(cardiac_frame_idx))

        if idx_low == idx_high or idx_high >= T:
            return scaling_factors[:, min(idx_low, T - 1)]

        alpha = cardiac_frame_idx - idx_low
        return scaling_factors[:, idx_low] * (1 - alpha) + scaling_factors[:, idx_high] * alpha

    def diffusion_timestep_to_cardiac_frame(self, t, num_cardiac_frames):
        """Map diffusion timestep t to cardiac frame index."""
        K = num_cardiac_frames - 1
        return K * t / self.T

    def compute_energy_gradient(self, v_t, source_img, target_imgs, cardiac_idx, num_cardiac_frames):
        """
        Compute gradient of registration energy w.r.t. velocity field (detached).
        Used for posterior SDE steering only, not for training gradients.
        """
        target_idx = int(round(cardiac_idx)) - 1
        if target_idx < 0 or target_idx >= len(target_imgs):
            return torch.zeros_like(v_t)

        target = target_imgs[target_idx]

        v_t_grad = v_t.detach().clone().requires_grad_(True)
        inshape = (self.img_size, self.img_size)
        warped = warp_image_with_velocity(source_img, v_t_grad, inshape=inshape)
        mse_loss = F.mse_loss(warped, target)
        reg_loss = self.grad_loss.loss(v_t_grad)
        energy = mse_loss + self.reg_weight * reg_loss
        grad_E = torch.autograd.grad(energy, v_t_grad, create_graph=False)[0]

        return grad_E.detach()

    def run_reverse_process(self, v_reg_list, scaling_factors, v_0, v_T, device='cuda'):
        """
        Run standard (prior) reverse diffusion process. Identical to inv_gamma_prior.

        Uses discrete bridge formula: v_t = mu_tilde + std_tilde * z
        """
        num_cardiac_frames = len(v_reg_list)

        v_t = v_T.clone()
        trajectory = [v_t.clone()]

        for t in range(self.T - 1, -1, -1):
            if t == 0:
                v_t = v_0.clone()
            else:
                cardiac_idx_t = self.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)
                cardiac_idx_t_plus_1 = self.diffusion_timestep_to_cardiac_frame(t + 1, num_cardiac_frames)

                s_t = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_t)
                s_t_plus_1 = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_t_plus_1)

                c_xt, c_x0, c_y, delta_tilde = self.compute_coefficients(t + 1, s_t_plus_1, s_t)

                v_t_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t_plus_1)
                v_t_minus_1_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t)

                if t > 0:
                    z = torch.randn_like(v_t, device=device)
                else:
                    z = torch.zeros_like(v_t, device=device)

                mu_tilde = (c_xt * v_t + c_x0 * v_0 + c_y * v_T +
                           v_t_minus_1_reg - c_xt * v_t_reg)
                std_tilde = torch.sqrt(torch.clamp(delta_tilde, min=1e-10))
                v_t = mu_tilde + std_tilde * z

                del z, mu_tilde

            trajectory.append(v_t.clone())

        return trajectory

    def run_posterior_reverse_process(self, v_reg_list, scaling_factors, v_0, v_T,
                                      source_img, target_imgs,
                                      gamma=1.0, guidance_scale=1.0, device='cuda'):
        """
        Posterior conditional reverse SDE (Eq. 15 from the paper).

        Discretization with Delta_t = -1:
            v_{t-1} = v_t + (v^r_{t-1} - v^r_t)
                      + eta_t/(T-t)                        [h-transform, f_BB]
                      - (sigma^2/gamma)*grad_E              [energy gradient]
                      - sigma^2 * eta_t / delta_t           [marginal score]
                      + sigma * z                           [noise]

        where eta_t = v_t - v^r_t, sigma^2 = 2s/T,
        delta_t = sigma^2 * t * (1 - t/T) = 2s * m_t * (1-m_t).

        This is a direct computation (no accumulator), producing variance
        that peaks at the midpoint t=T/2 as expected from the bridge prior.
        """
        num_cardiac_frames = len(v_reg_list)

        v_t = v_T.clone()
        trajectory = [v_t.clone()]

        for t in range(self.T - 1, -1, -1):
            if t == 0:
                v_t = v_0.clone()
            else:
                # paper_t = t+1: the diffusion step we are currently at
                paper_t = t + 1

                cardiac_idx_current = self.diffusion_timestep_to_cardiac_frame(paper_t, num_cardiac_frames)
                cardiac_idx_prev = self.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)

                # Registered velocities
                v_t_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_current)
                v_t_minus_1_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_prev)

                # Perturbation at current step
                eta_t = v_t - v_t_reg

                # sigma^2 = 2s/T
                # Force fp32 here even if caller is under autocast: s_current is
                # clamped to [1e-4, 1e4] so sigma_sq can hit ~1.4e-5 (fp16
                # subnormal) and delta_t ~1e-6 — fp16 division in `score` would
                # underflow. Coefficients stay fp32; the additive update at the
                # bottom inherits v_t's dtype via explicit casts below.
                s_current = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_current).float()
                sigma_sq = 2.0 * s_current / self.T
                sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-10))

                # delta_t = sigma^2 * paper_t * (1 - paper_t/T) (bridge marginal variance)
                m_t = paper_t / self.T
                delta_t = sigma_sq * paper_t * (1.0 - m_t)
                delta_t = torch.clamp(delta_t, min=1e-10)

                # Energy gradient (detached — steers samples, no training gradients)
                grad_E = self.compute_energy_gradient(
                    v_t, source_img, target_imgs,
                    cardiac_idx_current, num_cardiac_frames
                )

                z = torch.randn_like(v_t, device=device)

                # Eq. 15 three drift terms. Coefficient math is fp32 (above);
                # cast each term to v_t.dtype before summing so v_t stays in its
                # original dtype (fp16 under autocast) — preserves the memory
                # win without losing the fp32 protection on the small-scale
                # divides.
                v_dtype = v_t.dtype
                # 1. h-transform (bridge drift f_BB, sign flipped by Delta_t=-1)
                h_transform = eta_t / (self.T - paper_t + 1e-10)
                # 2. energy gradient correction (fp32 score, then cast)
                energy_correction = (-(guidance_scale / gamma) * sigma_sq * grad_E.float()).to(v_dtype)
                # 3. marginal score: -sigma^2 * eta_t / delta_t (fp32 score, then cast)
                score = (-sigma_sq * eta_t.float() / delta_t).to(v_dtype)
                sigma_z = (sigma * z.float()).to(v_dtype)

                v_t = (v_t + (v_t_minus_1_reg - v_t_reg)
                       + h_transform + energy_correction + score + sigma_z)

                if torch.isnan(v_t).any():
                    print(f"\n  [FATAL] NaN detected in v_t at reverse step t={t}")
                    raise RuntimeError(f"NaN detected at reverse step t={t}")

            trajectory.append(v_t.clone())

        return trajectory

    def run_posterior_forward_process(self, v_reg_list, scaling_factors, v_0, v_T,
                                      source_img, target_imgs,
                                      gamma=1.0, guidance_scale=1.0, device='cuda'):
        """
        Solve the posterior reverse SDE using forward Euler-Maruyama (dt=+1).

        Same SDE as run_posterior_reverse_process but discretized in the
        forward direction: starting from v_0 (t=0) and stepping to v_T (t=T).

        Reverse EM (current code) uses mu_posterior = -b (negated drift), then:
            v_tilde += mu_posterior + sigma*z      (dt = -1 absorbed)

        Forward EM flips the sign:
            v_tilde += -mu_posterior + sigma*z     (dt = +1)
            v_tilde += (+score + h_transform + correction) + sigma*z

        Three drift terms (with forward signs):
        1. Score:       +sigma^2 * (v_tilde - mu) / delta_t
        2. h-transform: +(y - v_tilde) / (T - t)
        3. Energy grad: +(sigma^2 / gamma) * grad_E
        """
        num_cardiac_frames = len(v_reg_list)

        v_t = v_0.clone()
        v_tilde = torch.zeros_like(v_0)
        trajectory = [v_t.clone()]

        for t in range(1, self.T + 1):
            if t == self.T:
                v_t = v_T.clone()
            else:
                # Map diffusion timesteps to cardiac frames
                cardiac_idx_t_minus_1 = self.diffusion_timestep_to_cardiac_frame(t - 1, num_cardiac_frames)
                cardiac_idx_t = self.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)

                # Scaling factors at current position (t-1)
                s_t_minus_1 = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_t_minus_1)

                # Registered velocities
                v_t_minus_1_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t_minus_1)
                v_t_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t)

                # sigma^2 = 2s/T at current position (t-1)
                sigma_sq = 2.0 * s_t_minus_1 / self.T

                z = torch.randn_like(v_t, device=device)

                # Marginal variance at t-1: delta = 2*s*m*(1-m)
                delta_t_minus_1 = self.compute_delta_t(t - 1, s_t_minus_1)
                delta_t_minus_1 = torch.clamp(delta_t_minus_1, min=1e-10)

                m_t_minus_1 = (t - 1) / self.T

                # 1. Score: +sigma^2 * (v_tilde - mu) / delta
                #    mu = (1-m)*v_0 + m*v_T (bridge mean in code convention)
                score = sigma_sq * (((v_t - v_t_minus_1_reg) - (1 - m_t_minus_1) * v_0 - m_t_minus_1 * v_T) / delta_t_minus_1)

                # 2. h-transform: +(y - v_tilde) / (T - (t-1))
                h_transform = (v_T - (v_t - v_t_minus_1_reg)) / (self.T - (t - 1))

                # 3. Energy gradient correction (detached)
                grad_E = self.compute_energy_gradient(
                    v_t, source_img, target_imgs,
                    cardiac_idx_t_minus_1, num_cardiac_frames
                )
                correction = (guidance_scale / gamma) * sigma_sq * grad_E

                # Forward EM step: v_tilde += b*dt + sigma*dW, dt = +1
                mu_forward = score + h_transform + correction
                std = torch.sqrt(torch.clamp(sigma_sq, min=1e-10))
                v_tilde = v_tilde + mu_forward + std * z
                v_t = v_t_reg + v_tilde

            trajectory.append(v_t.clone())

        return trajectory

    def sample_bridge_transition(self, v_reg_list, scaling_factors, device='cuda'):
        """
        Sample velocities using the forward bridge transition probability:

            v_t | v_{t-1} ~ N(mu_t, sigma_t^2),   t = 1, ..., T

        where:
            mu_t      = v^r_t + (T - t) / (T - (t-1)) * (v_{t-1} - v^r_{t-1})
            sigma_t^2 = sigma^2(x) * (T - t) / (T - (t-1))

        v_0 is fixed at the boundary. At t = T the ratio is 0, so
        v_T = v^r_T deterministically (bridge endpoint).

        Args:
            v_reg_list: list of K registered velocities [v^r_0, ..., v^r_{K-1}]
            scaling_factors: [B, K, C, H, W] learned scaling factors
            device: torch device

        Returns:
            dict mapping frame_idx -> sampled velocity tensor [B, 2, H, W]
        """
        num_cardiac_frames = len(v_reg_list)
        T = num_cardiac_frames - 1  # last frame index

        sampled = {}
        sampled[0] = v_reg_list[0].clone()  # v_0 fixed (boundary)

        v_prev = sampled[0]

        for t in range(1, num_cardiac_frames):
            ratio = (T - t) / (T - (t - 1))  # (T-1)/T at t=1, ... , 0 at t=T

            s_t = self.interpolate_scaling_factors(scaling_factors, t)  # [B, 2, H, W]

            # Conditional mean
            mean = v_reg_list[t] + ratio * (v_prev - v_reg_list[t - 1])

            if t < T:
                # Conditional variance: sigma^2(x) * ratio
                variance = s_t * ratio
                std = torch.sqrt(torch.clamp(variance, min=1e-10))
                z = torch.randn_like(mean, device=device)
                v_t = mean + std * z
            else:
                # At t = T: ratio = 0 => v_T = v^r_T deterministically
                v_t = v_reg_list[T].clone()

            sampled[t] = v_t.clone()
            v_prev = v_t

        return sampled

    def warp_image(self, image, velocity):
        """Warp an image using a velocity field."""
        inshape = (self.img_size, self.img_size)
        return warp_image_with_velocity(image, velocity, inshape=inshape)
