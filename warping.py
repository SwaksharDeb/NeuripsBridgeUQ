"""
Image warping utilities using VecInt and SpatialTransformer.

This module provides warping functionality for images using velocity fields.
"""

import os
import sys

# Add project root to path for utils imports
project_root = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightningTemplate_2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from utils.Int import VecInt, SpatialTransformer


# Global warping instances (shared across modules)
_vec_int = None
_spatial_transformer = None


def warp_image_with_velocity(image, velocity, inshape=(64, 64)):
    """
    Warp an image using a velocity field via VecInt and SpatialTransformer.

    Args:
        image: Input image tensor [B, C, H, W] or [C, H, W] or [H, W]
        velocity: Velocity field tensor [B, 2, H, W] or [2, H, W]
        inshape: Tuple of (H, W) for the spatial dimensions

    Returns:
        Warped image with same shape as input
    """
    global _vec_int, _spatial_transformer

    if _vec_int is None:
        _vec_int = VecInt(inshape, TSteps=7)
        _spatial_transformer = SpatialTransformer(inshape)

    # Move modules to same device as input
    device = velocity.device
    _vec_int.to(device)
    _spatial_transformer.to(device)

    squeeze_batch = False
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
        squeeze_batch = True
    elif image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_batch = True

    if velocity.dim() == 3:
        velocity = velocity.unsqueeze(0)

    disp_list = _vec_int(velocity)
    displacement = disp_list[-1]
    warped, _ = _spatial_transformer(image, displacement)

    if squeeze_batch:
        warped = warped.squeeze(0)

    return warped


def reset_warping_modules():
    """Reset the global warping modules (useful for testing or changing image size)."""
    global _vec_int, _spatial_transformer
    _vec_int = None
    _spatial_transformer = None
