"""
Data utility functions for loading models, data, and computing metrics.

Contains:
- load_model_and_data: Load TLRN model and train/test data loaders
- extract_velocity_fields: Extract velocity fields from the TLRN model
- ncc: Compute Normalized Cross-Correlation
- compute_ncc_vx_calibration: Compute NCC_VX calibration metric
"""

import os
import sys

# Add project root to path for utils imports
project_root = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from box import Box
from tqdm import tqdm

from utils.utils import instantiate_from_config, merge_yaml_files
from utils.utils_train import load_checkpoints


def load_model_and_data(combine_acdc=True):
    """
    Load the TLRN model and train/test data loaders.

    Args:
        combine_acdc: If True (default), combine the original 1200-sample
            training set with the 862-sample ACDC training set (2062 total).
            The combined file lists must exist at:
              TrainingList/train_1200_plus_ACDC.txt
              TrainingList/index_endEstolic_train_1200_plus_ACDC.txt

    Returns:
        model: Loaded TLRN model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration object
    """
    import glob as globmod

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level to LightningTemplate
    target_path = os.path.join(project_root, "2026Experiments", "CINE")

    if combine_acdc:
        setobj = Box({
            "name": "CINE",
            "model": "TLRN",
            "editfile": "basic_MSE_Penp_img1200Reg0.03_plus_ACDC"
        })
        print("Using COMBINED dataset: original 1200 + ACDC 862 = 2062 samples")
    else:
        setobj = Box({
            "name": "CINE",
            "model": "TLRN",
            "editfile": "basic_MSE_Penp_img1200Reg0.03"
        })
        print("Using ORIGINAL dataset: 1200 samples only")

    config_file_basic = f"{target_path}/settings/{setobj.model}/basic.yaml"
    config_file_edit = f"{target_path}/settings/{setobj.model}/{setobj.editfile}.yaml"
    config_file = f"{target_path}/settings/{setobj.model}/config_merged.yaml"
    merge_yaml_files(config_file_basic, config_file_edit, config_file)

    seed_everything(20250101)
    config = OmegaConf.load(config_file)

    # Auto-detect best checkpoint for combined model
    if combine_acdc:
        ckpt_dir = f"{target_path}/outputs/{setobj.model}/{setobj.editfile}/checkpoints"
        best_ckpts = sorted(globmod.glob(f"{ckpt_dir}/epoch=*-val_mse=*.ckpt"))
        if not best_ckpts:
            best_ckpts = sorted(globmod.glob(f"{ckpt_dir}/epoch=*-val_loss=*.ckpt"))
        if best_ckpts:
            best_ckpt = best_ckpts[-1]
            config.model.pretrained_checkpoint = best_ckpt
            print(f"Auto-detected best checkpoint: {best_ckpt}")
        else:
            all_ckpts = sorted(globmod.glob(f"{ckpt_dir}/epoch=*.ckpt"))
            if all_ckpts:
                config.model.pretrained_checkpoint = all_ckpts[-1]
                print(f"Using latest checkpoint: {all_ckpts[-1]}")
            else:
                print(f"WARNING: No combined checkpoints found in {ckpt_dir}")
                print("  Falling back to checkpoint in config yaml")

    config.model.params.workdir = f"{target_path}/outputs/{setobj.model}/{setobj.editfile}"
    config.model.params.setobj_info = f"{setobj.name}_{setobj.model}_{setobj.editfile}"
    config.model.params.test_split = config.data.params.test.params.split
    config.model.params.Start_Subject_ID = 0

    print("Loading model...")
    model = instantiate_from_config(config.model)
    load_checkpoints(model, config.model)
    model.eval()
    model.cuda()

    print("Loading data...")
    data = instantiate_from_config(config.data)
    data.setup()
    train_loader = data.train_dataloader()
    test_loader = data.test_dataloader()

    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")

    return model, train_loader, test_loader, config


def extract_velocity_fields(model, batch):
    """
    Extract velocity fields from the TLRN model.

    Args:
        model: TLRN model
        batch: Data batch containing 'series' and 'lv_segs'

    Returns:
        v_series: List of velocity fields
        Sdef_series: List of deformed images
        series: Original image series
    """
    series = batch['series'].cuda()
    lv_segs = batch['lv_segs'].cuda()

    print(f"Input series shape: {series.shape}")

    with torch.no_grad():
        Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = \
            model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)

    print(f"Number of velocity fields: {len(v_series)}")
    print(f"Velocity field shape: {v_series[0].shape}")

    return v_series, Sdef_series, series


def ncc(a, v, zero_norm=True):
    """
    Compute Normalized Cross-Correlation between two arrays.

    Args:
        a: First array
        v: Second array
        zero_norm: Whether to zero-normalize the arrays

    Returns:
        NCC value
    """
    a = a.flatten()
    v = v.flatten()
    eps = 1e-15

    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a) + eps)
        v = (v - np.mean(v)) / (np.std(v) + eps)
    else:
        a = (a) / (np.std(a) * len(a) + eps)
        v = (v) / (np.std(v) + eps)

    return np.correlate(a, v)[0]


def compute_ncc_vx_calibration(sampled_velocities, v_reg_list, subject_idx=0):
    """
    Compute NCC_VX calibration metric.

    This metric measures the correlation between the variance of sampled
    velocities and the MSE between mean sampled velocity and registered velocity.

    Args:
        sampled_velocities: Dict mapping frame_idx -> list of velocity tensors
        v_reg_list: List of registered velocity fields
        subject_idx: Which subject to analyze

    Returns:
        ncc_vx_per_frame: Dict with per-frame statistics
        overall_ncc_vx: Overall NCC_VX value
        overall_std: Standard deviation across frames
    """
    num_cardiac_frames = len(v_reg_list)
    ncc_vx_per_frame = {}
    all_ncc_values = []

    print(f"\nComputing NCC_VX calibration metric...")

    for frame_idx in tqdm(range(num_cardiac_frames), desc="Computing NCC_VX"):
        if frame_idx not in sampled_velocities:
            continue

        velocities_list = sampled_velocities[frame_idx]
        velocities_at_t = np.stack([v[subject_idx].cpu().numpy() for v in velocities_list], axis=0)

        variance = np.var(velocities_at_t, axis=0)
        variance_magnitude = np.sqrt(variance[0]**2 + variance[1]**2)

        mean_velocity = np.mean(velocities_at_t, axis=0)
        v_reg = v_reg_list[frame_idx][subject_idx].cpu().numpy()

        mse = (mean_velocity - v_reg) ** 2
        mse_magnitude = np.sqrt(mse[0]**2 + mse[1]**2)

        ncc_vx = ncc(variance_magnitude, mse_magnitude, zero_norm=True)

        ncc_vx_per_frame[frame_idx] = {
            'ncc_vx': ncc_vx,
            'mean_variance': variance_magnitude.mean(),
            'mean_mse': mse_magnitude.mean()
        }
        all_ncc_values.append(ncc_vx)

    overall_ncc_vx = np.mean(all_ncc_values) if all_ncc_values else 0.0
    overall_std = np.std(all_ncc_values) if all_ncc_values else 0.0

    print(f"Overall NCC_VX: {overall_ncc_vx:.6f} +/- {overall_std:.6f}")

    return ncc_vx_per_frame, overall_ncc_vx, overall_std
