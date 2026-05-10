"""
Network components for learned scaling factor prediction.

The ScalingFactorNetwork takes BOTH the image sequence AND the registration
velocity sequence as input.  Each modality is independently projected to a
shared channel dimension and then *added* element-wise before being fed into
a U-Net encoder–decoder.

Input:  source_img + target_imgs + v_reg_list
        images:     [B, 1 + T, H, W]
        velocities: [B, T * 2, H, W]
Output: scaling_factors  ->  [B, 2, H, W]  (shared across time)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with LeakyReLU activation."""

    def __init__(self, ndims, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.main(x))


class ScalingFactorNetwork(nn.Module):
    """
    Network to predict spatially varying scaling factors from both images
    and velocity fields.

    Instead of concatenating images and velocities along the channel axis,
    each modality is projected to a shared dimension ``proj_dim`` by a small
    conv stack and the two projections are **added** element-wise.  This
    forces the network to learn modality-specific features before fusion and
    makes the scaling factor more sensitive to the velocity field produced by
    each registration backbone.

    Architecture: projection-add fusion  ->  U-Net encoder-decoder
    """

    def __init__(self, num_time_steps, img_size=128, num_heads=8, proj_dim=32):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.img_size = img_size
        ndims = 2

        # Channel counts for each modality
        img_channels = 1 + num_time_steps        # source (1) + targets (T)
        vel_channels = num_time_steps * 2         # velocity x,y per frame

        # --- Projection branches ---
        # Two-layer conv projections: raw channels -> proj_dim
        self.img_proj = nn.Sequential(
            ConvBlock(ndims, img_channels, proj_dim),
            ConvBlock(ndims, proj_dim, proj_dim),
        )
        self.vel_proj = nn.Sequential(
            ConvBlock(ndims, vel_channels, proj_dim),
            ConvBlock(ndims, proj_dim, proj_dim),
        )

        # Current-velocity branch (v^k from inner loop). Same channel count as v_reg.
        self.vel_cur_proj = nn.Sequential(
            ConvBlock(ndims, vel_channels, proj_dim),
            ConvBlock(ndims, proj_dim, proj_dim),
        )

        self.img_norm = nn.InstanceNorm2d(proj_dim)
        self.vel_norm = nn.InstanceNorm2d(proj_dim)
        self.vel_cur_norm = nn.InstanceNorm2d(proj_dim)

        # --- U-Net encoder-decoder (starts from proj_dim channels) ---
        self.enc_channels = [proj_dim, 32, 64, 128, 256]

        # Encoder
        self.encoder = nn.ModuleList()
        self.pooling = nn.ModuleList()

        for i in range(len(self.enc_channels) - 1):
            self.encoder.append(ConvBlock(ndims, self.enc_channels[i], self.enc_channels[i+1]))
            self.pooling.append(nn.MaxPool2d(2))

        # Bottleneck
        self.bottleneck = ConvBlock(ndims, self.enc_channels[-1], self.enc_channels[-1])

        # Decoder with skip connections
        self.decoder = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        dec_in_channels = [
            self.enc_channels[-1] + self.enc_channels[-1],  # 256 + 256
            128 + self.enc_channels[-2],  # 128 + 128
            64 + self.enc_channels[-3],   # 64 + 64
            32 + self.enc_channels[-4],   # 32 + 32
        ]
        dec_out_channels = [128, 64, 32, 16]

        for i in range(len(dec_out_channels)):
            self.decoder.append(ConvBlock(ndims, dec_in_channels[i], dec_out_channels[i]))
            self.upsampling.append(nn.Upsample(scale_factor=2, mode='nearest'))

        # Final convolutions
        self.final_conv1 = ConvBlock(ndims, 16, 16)
        self.final_conv2 = ConvBlock(ndims, 16, 16)

        # Output layer: predict scaling factor for velocity components (shared across time)
        self.output_conv = nn.Conv2d(16, 2, kernel_size=3, padding=1)

        # Softplus to ensure positive scaling factors
        self.softplus = nn.Softplus(beta=0.5)

        # Initialize output to produce values around 1
        nn.init.zeros_(self.output_conv.weight)
        nn.init.constant_(self.output_conv.bias, 0.0)

    def forward(self, source_img, target_imgs, v_reg_list=None, v_current_list=None):
        """
        Args:
            source_img: [B, 1, H, W] source image
            target_imgs: List of [B, 1, H, W] target images, length T
                        OR stacked tensor [B, T, 1, H, W]
            v_reg_list: List of T velocity fields [B, 2, H, W] (the registered velocity)
                        OR pre-concatenated tensor [B, T*2, H, W]
            v_current_list: List/tensor of the current velocity v^k.
                        If None, falls back to v_reg_list (i.e. v^0 = v^r).

        Returns:
            scaling_factors: [B, 2, H, W] spatially varying scaling factors
        """
        if v_reg_list is None:
            raise ValueError("v_reg_list must be provided for combined scaling network")

        if v_current_list is None:
            v_current_list = v_reg_list

        # Handle target_imgs as list or tensor
        if isinstance(target_imgs, list):
            targets_cat = torch.cat(target_imgs, dim=1)  # [B, T, H, W]
        else:
            # Assume [B, T, 1, H, W]
            B, T, C, H, W = target_imgs.shape
            targets_cat = target_imgs.squeeze(2)  # [B, T, H, W]

        # Handle v_reg_list as list or pre-concatenated tensor
        if isinstance(v_reg_list, list):
            vel_cat = torch.cat(v_reg_list, dim=1)  # [B, T*2, H, W]
        else:
            vel_cat = v_reg_list  # Already [B, T*2, H, W]

        if isinstance(v_current_list, list):
            vel_cur_cat = torch.cat(v_current_list, dim=1)
        else:
            vel_cur_cat = v_current_list

        # --- Projection + instance-norm + additive fusion (3-way) ---
        img_input = torch.cat([source_img, targets_cat], dim=1)        # [B, 1+T, H, W]
        img_feat     = self.img_norm(self.img_proj(img_input))         # [B, proj_dim, H, W]
        vel_reg_feat = self.vel_norm(self.vel_proj(vel_cat))           # [B, proj_dim, H, W]
        vel_cur_feat = self.vel_cur_norm(self.vel_cur_proj(vel_cur_cat))
        x = img_feat + vel_reg_feat + vel_cur_feat                     # additive 3-way fusion

        # Encoder with skip connection storage
        enc_features = []
        for conv, pool in zip(self.encoder, self.pooling):
            x = conv(x)
            enc_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections (reverse order)
        for i, (conv, upsample) in enumerate(zip(self.decoder, self.upsampling)):
            x = upsample(x)
            skip_idx = len(enc_features) - 1 - i
            if skip_idx >= 0:
                skip = enc_features[skip_idx]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear')
                x = torch.cat([x, skip], dim=1)
            x = conv(x)

        # Final convolutions
        x = self.final_conv1(x)
        x = self.final_conv2(x)

        # Output scaling factors
        x = self.output_conv(x)  # [B, 2, H, W]

        # Ensure positive scaling factors
        x = self.softplus(x)

        return x
