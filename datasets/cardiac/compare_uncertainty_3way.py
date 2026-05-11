#!/usr/bin/env python3
"""
Three-way comparison: TLRN (inv-gamma prior) vs VoxelMorph Prob vs PULPo.

Loads VxM Prob and PULPo models, runs evaluation on CINE test data, then
overlays results on pre-computed TLRN curves (risk-coverage, sparsification).

Usage:
    python uncertainty_inv_gamma_prior/compare_uncertainty_3way.py \
        --device cuda
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import SimpleITK as sitk

# Add VoxelMorph Prob to path
VXM_PATH = "/scratch/swd9tc/Uncertanity_quantification/voxelmorph_prob"
sys.path.insert(0, VXM_PATH)
from models import VxmProbabilistic, SpatialTransformer

# Add PULPo to path
PULPO_PATH = "/scratch/swd9tc/Uncertanity_quantification/pulbo"
sys.path.insert(0, PULPO_PATH)
from evaluate import Evaluate
from src.models import PULPo

# ============================================================
# Path configuration
# ============================================================
OLD_DATA_ROOT = "/scratch/bsw3ac/nellie/code/cvpr/Project/Foundation/Data"
NEW_DATA_ROOT = "/scratch/swd9tc/Uncertanity_quantification/DataPerNor"
TXT_FOLDER = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightningTemplate/dataload"


def remap_path(path):
    """Remap paths from old location to new location."""
    if OLD_DATA_ROOT in path:
        return path.replace(OLD_DATA_ROOT, NEW_DATA_ROOT)
    return path


def load_sequence(file_path, target_T=15, resize=(64, 64)):
    """Load a cardiac MRI sequence."""
    path = remap_path(file_path)
    if not os.path.isfile(path):
        path = path.replace("DataPd5", "Data")

    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img)
    vol = np.nan_to_num(vol, copy=False).astype(np.float32)

    vmin, vmax = np.min(vol), np.max(vol)
    if abs(vmax - vmin) > 1e-6:
        vol = (vol - vmin) / (vmax - vmin)

    T, H, W = vol.shape

    if T != target_T:
        vol_tensor = torch.from_numpy(vol).float()
        vol_tensor = vol_tensor.permute(1, 2, 0).reshape(1, H * W, T)
        vol_tensor = F.interpolate(vol_tensor, size=target_T, mode='linear', align_corners=True)
        vol = vol_tensor.reshape(H, W, target_T).permute(2, 0, 1).numpy()

    if resize is not None:
        out_h, out_w = resize
        vol_tensor = torch.from_numpy(vol).float().unsqueeze(1)
        vol_tensor = F.interpolate(vol_tensor, size=(out_h, out_w), mode='bilinear', align_corners=False)
        vol = vol_tensor.squeeze(1).numpy()

    return vol


def load_test_files():
    """Load test file list from standard location."""
    test_txt = Path(TXT_FOLDER) / "TestingList/testid_files_frames24_25.txt"
    test_files = []
    if test_txt.exists():
        with open(test_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    test_files.append(line)
    return test_files


# ============================================================
# VoxelMorph Prob evaluation
# ============================================================

def load_vxm_model(checkpoint_path, device='cuda'):
    """Load VxmProbabilistic model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get('args', None)
    if ckpt_args is not None:
        nb_features = ckpt_args.nb_features
        int_steps = ckpt_args.int_steps
    else:
        nb_features = [16, 32, 32, 32]
        int_steps = 7

    model = VxmProbabilistic(
        inshape=(64, 64),
        in_channels=1,
        nb_features=nb_features,
        int_steps=int_steps,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded VxmProbabilistic (features={nb_features}, int_steps={int_steps})")
    return model


@torch.no_grad()
def evaluate_vxm_single_sequence(model, sequence, device='cuda'):
    """Evaluate VoxelMorph Prob on a single cardiac sequence."""
    T, H, W = sequence.shape
    source = torch.from_numpy(sequence[0]).float().unsqueeze(0).unsqueeze(0).to(device)

    fractions = np.linspace(0, 1, 21)
    coverages = np.linspace(0.1, 1, 91)

    spars_results = {}
    rc_results = {}

    for t in range(1, T - 1):
        target = torch.from_numpy(sequence[t]).float().unsqueeze(0).unsqueeze(0).to(device)

        output = model(source, target, sample=False)
        warped = output['y_source']
        sigma = output['sigma']

        sigma_map = sigma[0].mean(dim=0).cpu().numpy()
        eps = 1e-6
        unc_flat = 1.0 / (sigma_map.flatten() + eps)

        target_np = sequence[t]
        warped_np = warped[0, 0].cpu().numpy()
        err_flat = ((warped_np - target_np) ** 2).flatten()
        n_pixels = len(unc_flat)

        # Sparsification
        unc_order = np.argsort(-unc_flat)
        oracle_order = np.argsort(-err_flat)
        overall_mean = np.mean(err_flat)

        unc_curve = np.zeros(len(fractions))
        oracle_curve = np.zeros(len(fractions))
        random_curve = np.zeros(len(fractions))

        for j, frac in enumerate(fractions):
            n_remove = int(frac * n_pixels)
            n_remain = n_pixels - n_remove
            if n_remain == 0:
                unc_curve[j] = 0.0
                oracle_curve[j] = 0.0
                random_curve[j] = 0.0
            else:
                unc_curve[j] = np.mean(err_flat[unc_order[n_remove:]])
                oracle_curve[j] = np.mean(err_flat[oracle_order[n_remove:]])
                random_curve[j] = overall_mean
        random_curve[-1] = 0.0

        ause = np.trapz(unc_curve - oracle_curve, fractions)
        random_area = np.trapz(random_curve - oracle_curve, fractions)
        ause_norm = ause / random_area if random_area > 1e-12 else 0.0

        spars_results[t] = {
            'fractions': fractions,
            'uncertainty_curve': unc_curve,
            'oracle_curve': oracle_curve,
            'random_curve': random_curve,
            'ause': ause,
            'ause_norm': ause_norm,
        }

        # Risk-Coverage
        unc_asc_order = np.argsort(unc_flat)
        oracle_asc_order = np.argsort(err_flat)

        unc_rc_curve = np.zeros(len(coverages))
        oracle_rc_curve = np.zeros(len(coverages))
        random_rc_curve = np.zeros(len(coverages))

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

        rc_results[t] = {
            'coverages': coverages,
            'unc_rc_curve': unc_rc_curve,
            'oracle_rc_curve': oracle_rc_curve,
            'random_rc_curve': random_rc_curve,
            'aurc': aurc,
            'aurc_norm': aurc_norm,
        }

    return spars_results, rc_results


# ============================================================
# PULPo evaluation
# ============================================================

def load_pulpo_model(checkpoint_dir, device='cuda'):
    """Load PULPo model from checkpoint directory."""
    eval_obj = Evaluate()

    checkpoint_parts = checkpoint_dir.rstrip('/').split('/')
    if 'version' in checkpoint_parts[-1]:
        version = checkpoint_parts[-1]
        git_hash = checkpoint_parts[-2]
        model_dir = '/'.join(checkpoint_parts[:-2])
    else:
        git_hash = checkpoint_parts[-1]
        model_dir = '/'.join(checkpoint_parts[:-1])
        version = "version_0"

    print(f"  Loading PULPo from: {model_dir}/{git_hash}/{version}")
    model = eval_obj.load_model(model_dir=model_dir, git_hash=git_hash, version=version)
    model = model.to(device)
    model.eval()
    print(f"  Loaded PULPo model")
    return model


@torch.no_grad()
def evaluate_pulpo_single_sequence(model, sequence, num_samples=20, device='cuda'):
    """
    Evaluate PULPo on a single cardiac sequence.

    Uses output image variance (std of warped images across samples) as uncertainty.
    """
    T, H, W = sequence.shape
    sequence_tensor = torch.from_numpy(sequence).float()
    source = sequence_tensor[0].unsqueeze(0).unsqueeze(0).to(device)

    fractions = np.linspace(0, 1, 21)
    coverages = np.linspace(0.1, 1, 91)

    spars_results = {}
    rc_results = {}

    for t in range(1, T - 1):
        target = sequence_tensor[t].unsqueeze(0).unsqueeze(0).to(device)

        # Get multiple samples from posterior
        outputs, individual_dfs = model.predict_output_samples(source, target, N=num_samples)

        # outputs[0]: [1, N, 1, H, W] — warped images at finest level
        all_outputs = outputs[0][0]  # [N, 1, H, W]

        # Uncertainty: std of warped images across samples
        output_std = torch.std(all_outputs, dim=0).squeeze(0).cpu().numpy()  # [H, W]
        uncertainty = output_std

        # Mean warped image for error computation
        mean_warped = torch.mean(all_outputs, dim=0).squeeze(0).cpu().numpy()  # [H, W]

        # Per-pixel error
        target_np = sequence[t]
        err_flat = ((mean_warped - target_np) ** 2).flatten()
        unc_flat = uncertainty.flatten()
        n_pixels = len(unc_flat)

        if np.max(unc_flat) < 1e-8:
            continue

        # Sparsification
        unc_order = np.argsort(-unc_flat)
        oracle_order = np.argsort(-err_flat)
        overall_mean = np.mean(err_flat)

        unc_curve = np.zeros(len(fractions))
        oracle_curve = np.zeros(len(fractions))
        random_curve = np.zeros(len(fractions))

        for j, frac in enumerate(fractions):
            n_remove = int(frac * n_pixels)
            n_remain = n_pixels - n_remove
            if n_remain == 0:
                unc_curve[j] = 0.0
                oracle_curve[j] = 0.0
                random_curve[j] = 0.0
            else:
                unc_curve[j] = np.mean(err_flat[unc_order[n_remove:]])
                oracle_curve[j] = np.mean(err_flat[oracle_order[n_remove:]])
                random_curve[j] = overall_mean
        random_curve[-1] = 0.0

        ause = np.trapz(unc_curve - oracle_curve, fractions)
        random_area = np.trapz(random_curve - oracle_curve, fractions)
        ause_norm = ause / random_area if random_area > 1e-12 else 0.0

        spars_results[t] = {
            'fractions': fractions,
            'uncertainty_curve': unc_curve,
            'oracle_curve': oracle_curve,
            'random_curve': random_curve,
            'ause': ause,
            'ause_norm': ause_norm,
        }

        # Risk-Coverage
        unc_asc_order = np.argsort(unc_flat)
        oracle_asc_order = np.argsort(err_flat)

        unc_rc_curve = np.zeros(len(coverages))
        oracle_rc_curve = np.zeros(len(coverages))
        random_rc_curve = np.zeros(len(coverages))

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

        rc_results[t] = {
            'coverages': coverages,
            'unc_rc_curve': unc_rc_curve,
            'oracle_rc_curve': oracle_rc_curve,
            'random_rc_curve': random_rc_curve,
            'aurc': aurc,
            'aurc_norm': aurc_norm,
        }

    return spars_results, rc_results


# ============================================================
# Common utilities
# ============================================================

def aggregate_curves(all_results, curve_type='sparsification'):
    """Average per-subject curve results across all subjects, per frame."""
    all_frames = set()
    for res in all_results:
        all_frames.update(res.keys())
    all_frames = sorted(all_frames)

    avg_per_frame = {}

    if curve_type == 'sparsification':
        for fi in all_frames:
            unc_curves = [r[fi]['uncertainty_curve'] for r in all_results if fi in r]
            oracle_curves = [r[fi]['oracle_curve'] for r in all_results if fi in r]
            random_curves = [r[fi]['random_curve'] for r in all_results if fi in r]
            ause_norms = [r[fi]['ause_norm'] for r in all_results if fi in r]
            if unc_curves:
                avg_per_frame[fi] = {
                    'fractions': all_results[0][fi]['fractions'],
                    'uncertainty_curve': np.mean(unc_curves, axis=0),
                    'oracle_curve': np.mean(oracle_curves, axis=0),
                    'random_curve': np.mean(random_curves, axis=0),
                    'ause_norm': np.mean(ause_norms),
                    'num_subjects': len(unc_curves),
                }
    else:  # risk_coverage
        for fi in all_frames:
            unc_curves = [r[fi]['unc_rc_curve'] for r in all_results if fi in r]
            oracle_curves = [r[fi]['oracle_rc_curve'] for r in all_results if fi in r]
            random_curves = [r[fi]['random_rc_curve'] for r in all_results if fi in r]
            aurc_norms = [r[fi]['aurc_norm'] for r in all_results if fi in r]
            if unc_curves:
                avg_per_frame[fi] = {
                    'coverages': all_results[0][fi]['coverages'],
                    'unc_rc_curve': np.mean(unc_curves, axis=0),
                    'oracle_rc_curve': np.mean(oracle_curves, axis=0),
                    'random_rc_curve': np.mean(random_curves, axis=0),
                    'aurc_norm': np.mean(aurc_norms),
                    'num_subjects': len(unc_curves),
                }

    return avg_per_frame


def load_tlrn_curves(curves_dir):
    """Load pre-saved TLRN averaged curve data from .npz files."""
    spars_path = os.path.join(curves_dir, 'avg_sparsification_curves.npz')
    rc_path = os.path.join(curves_dir, 'avg_risk_coverage_curves.npz')

    tlrn_spars = {}
    if os.path.exists(spars_path):
        data = np.load(spars_path, allow_pickle=True)
        for fi in data['frame_indices']:
            fi = int(fi)
            tlrn_spars[fi] = {
                'fractions': data[f'frame_{fi}_fractions'],
                'uncertainty_curve': data[f'frame_{fi}_uncertainty_curve'],
                'oracle_curve': data[f'frame_{fi}_oracle_curve'],
                'random_curve': data[f'frame_{fi}_random_curve'],
                'ause_norm': float(data[f'frame_{fi}_ause_norm']),
            }

    tlrn_rc = {}
    if os.path.exists(rc_path):
        data = np.load(rc_path, allow_pickle=True)
        for fi in data['frame_indices']:
            fi = int(fi)
            tlrn_rc[fi] = {
                'coverages': data[f'frame_{fi}_coverages'],
                'unc_rc_curve': data[f'frame_{fi}_unc_rc_curve'],
                'oracle_rc_curve': data[f'frame_{fi}_oracle_rc_curve'],
                'random_rc_curve': data[f'frame_{fi}_random_rc_curve'],
                'aurc_norm': float(data[f'frame_{fi}_aurc_norm']),
            }

    return tlrn_spars, tlrn_rc


def load_cached_curves(npz_path):
    """Load cached model curves from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    spars = {}
    rc = {}

    # Check if it contains sparsification or risk-coverage data
    for fi in data.get('frame_indices', []):
        fi = int(fi)
        if f'frame_{fi}_fractions' in data:
            spars[fi] = {
                'fractions': data[f'frame_{fi}_fractions'],
                'uncertainty_curve': data[f'frame_{fi}_uncertainty_curve'],
                'oracle_curve': data[f'frame_{fi}_oracle_curve'],
                'random_curve': data[f'frame_{fi}_random_curve'],
                'ause_norm': float(data[f'frame_{fi}_ause_norm']),
            }
        if f'frame_{fi}_coverages' in data:
            rc[fi] = {
                'coverages': data[f'frame_{fi}_coverages'],
                'unc_rc_curve': data[f'frame_{fi}_unc_rc_curve'],
                'oracle_rc_curve': data[f'frame_{fi}_oracle_rc_curve'],
                'random_rc_curve': data[f'frame_{fi}_random_rc_curve'],
                'aurc_norm': float(data[f'frame_{fi}_aurc_norm']),
            }

    num_subjects = int(data.get('num_subjects', 0))
    return spars, rc, num_subjects


def save_curves_npz(spars, rc, output_dir, prefix, num_subjects):
    """Save averaged curves as .npz for caching."""
    # Sparsification
    npz_data = {}
    for fi, res in sorted(spars.items()):
        npz_data[f'frame_{fi}_fractions'] = res['fractions']
        npz_data[f'frame_{fi}_uncertainty_curve'] = res['uncertainty_curve']
        npz_data[f'frame_{fi}_oracle_curve'] = res['oracle_curve']
        npz_data[f'frame_{fi}_random_curve'] = res['random_curve']
        npz_data[f'frame_{fi}_ause_norm'] = np.array(res['ause_norm'])
    npz_data['frame_indices'] = np.array(sorted(spars.keys()))
    npz_data['num_subjects'] = np.array(num_subjects)
    np.savez(os.path.join(output_dir, f'{prefix}_avg_sparsification_curves.npz'), **npz_data)

    # Risk-coverage
    npz_data = {}
    for fi, res in sorted(rc.items()):
        npz_data[f'frame_{fi}_coverages'] = res['coverages']
        npz_data[f'frame_{fi}_unc_rc_curve'] = res['unc_rc_curve']
        npz_data[f'frame_{fi}_oracle_rc_curve'] = res['oracle_rc_curve']
        npz_data[f'frame_{fi}_random_rc_curve'] = res['random_rc_curve']
        npz_data[f'frame_{fi}_aurc_norm'] = np.array(res['aurc_norm'])
    npz_data['frame_indices'] = np.array(sorted(rc.keys()))
    npz_data['num_subjects'] = np.array(num_subjects)
    np.savez(os.path.join(output_dir, f'{prefix}_avg_risk_coverage_curves.npz'), **npz_data)


# ============================================================
# 3-way plotting functions
# ============================================================

# Colors: TLRN=blue, VxM=orange, PULPo=purple
# Oracle always green, Random always red dashed

def _plot_single_rc_frame(fi, tlrn_rc, vxm_rc, pulpo_rc, save_path):
    """Plot a single risk-coverage frame and save as individual image."""
    t = tlrn_rc[fi]
    v = vxm_rc[fi]
    p = pulpo_rc[fi]

    # Determine scale factor from max y value
    all_y = np.concatenate([t['oracle_rc_curve'], t['random_rc_curve'],
                            t['unc_rc_curve'], v['unc_rc_curve'],
                            p['unc_rc_curve'], p['oracle_rc_curve']])
    max_val = np.max(all_y)
    if max_val > 0:
        exponent = int(np.floor(np.log10(max_val)))
    else:
        exponent = 0
    scale = 10 ** exponent

    # Scaled curves
    t_oracle_s = t['oracle_rc_curve'] / scale
    p_oracle_s = p['oracle_rc_curve'] / scale
    t_random_s = t['random_rc_curve'] / scale
    t_unc_s = t['unc_rc_curve'] / scale
    v_unc_s = v['unc_rc_curve'] / scale
    p_unc_s = p['unc_rc_curve'] / scale

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Shaded area between each method and its oracle (visualises nAURC)
    ax.fill_between(t['coverages'], t_oracle_s, t_unc_s, color='blue', alpha=0.12)
    ax.fill_between(v['coverages'], t_oracle_s, v_unc_s, color='orange', alpha=0.10)
    ax.fill_between(p['coverages'], p_oracle_s, p_unc_s, color='purple', alpha=0.10)

    ax.plot(t['coverages'], t_oracle_s, 'g-', linewidth=1.5, label='TLRN Oracle')
    ax.plot(p['coverages'], p_oracle_s, color='green', linestyle=':',
            linewidth=1.5, label='PULPo Oracle')
    ax.plot(t['coverages'], t_random_s, 'r--', linewidth=1, label='Random')
    ax.plot(t['coverages'], t_unc_s, 'b-', linewidth=1.5,
            label=f'TLRN (nAURC={t["aurc_norm"]:.3f})')
    ax.plot(v['coverages'], v_unc_s, color='orange', linestyle='-',
            linewidth=1.5, label=f'VxM (nAURC={v["aurc_norm"]:.3f})')
    ax.plot(p['coverages'], p_unc_s, color='purple', linestyle='-',
            linewidth=1.5, label=f'PULPo (nAURC={p["aurc_norm"]:.3f})')

    if exponent != 0:
        ax.text(0.03, 0.97, f'$\\times 10^{{{exponent}}}$',
                transform=ax.transAxes, fontsize=7, va='top', ha='left')

    ax.legend(fontsize=5, loc='upper right', handlelength=1.2, borderpad=0.3, labelspacing=0.2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def _plot_single_spars_frame(fi, tlrn_spars, vxm_spars, pulpo_spars, save_path):
    """Plot a single sparsification frame and save as individual image."""
    t = tlrn_spars[fi]
    v = vxm_spars[fi]
    p = pulpo_spars[fi]

    # Determine scale factor from max y value
    all_y = np.concatenate([t['oracle_curve'], t['random_curve'],
                            t['uncertainty_curve'], v['uncertainty_curve'],
                            p['uncertainty_curve']])
    max_val = np.max(all_y)
    if max_val > 0:
        exponent = int(np.floor(np.log10(max_val)))
    else:
        exponent = 0
    scale = 10 ** exponent

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(t['fractions'], t['oracle_curve'] / scale, 'g-', linewidth=1.5, label='Oracle')
    ax.plot(t['fractions'], t['random_curve'] / scale, 'r--', linewidth=1, label='Random')
    ax.plot(t['fractions'], t['uncertainty_curve'] / scale, 'b-', linewidth=1.5,
            label=f'TLRN (nAUSE={t["ause_norm"]:.3f})')
    ax.plot(v['fractions'], v['uncertainty_curve'] / scale, color='orange', linestyle='-',
            linewidth=1.5, label=f'VxM (nAUSE={v["ause_norm"]:.3f})')
    ax.plot(p['fractions'], p['uncertainty_curve'] / scale, color='purple', linestyle='-',
            linewidth=1.5, label=f'PULPo (nAUSE={p["ause_norm"]:.3f})')

    if exponent != 0:
        ax.text(0.03, 0.97, f'$\\times 10^{{{exponent}}}$',
                transform=ax.transAxes, fontsize=7, va='top', ha='left')

    ax.legend(fontsize=5, loc='upper left', handlelength=1.2, borderpad=0.3, labelspacing=0.2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_3way_rc_per_frame(tlrn_rc, vxm_rc, pulpo_rc, save_dir, tlrn_n, vxm_n, pulpo_n):
    """Per-frame grid of risk-coverage plots with three methods overlaid."""
    common_frames = sorted(set(tlrn_rc.keys()) & set(vxm_rc.keys()) & set(pulpo_rc.keys()))
    num_plots = len(common_frames)
    if num_plots == 0:
        return

    # Save individual per-frame images
    rc_individual_dir = os.path.join(save_dir, 'rc_per_frame')
    os.makedirs(rc_individual_dir, exist_ok=True)
    for fi in common_frames:
        _plot_single_rc_frame(fi, tlrn_rc, vxm_rc, pulpo_rc,
                              os.path.join(rc_individual_dir, f'risk_coverage_frame_{fi}.png'))

    # Grid plot
    cols = min(5, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.0 * rows), squeeze=False)

    for idx, fi in enumerate(common_frames):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        t = tlrn_rc[fi]
        v = vxm_rc[fi]
        p = pulpo_rc[fi]

        # Determine scale factor from max y value
        all_y = np.concatenate([t['oracle_rc_curve'], t['random_rc_curve'],
                                t['unc_rc_curve'], v['unc_rc_curve'],
                                p['unc_rc_curve'], p['oracle_rc_curve']])
        max_val = np.max(all_y)
        if max_val > 0:
            exponent = int(np.floor(np.log10(max_val)))
        else:
            exponent = 0
        scale = 10 ** exponent

        # Scaled curves
        t_oracle_s = t['oracle_rc_curve'] / scale
        p_oracle_s = p['oracle_rc_curve'] / scale
        t_random_s = t['random_rc_curve'] / scale
        t_unc_s = t['unc_rc_curve'] / scale
        v_unc_s = v['unc_rc_curve'] / scale
        p_unc_s = p['unc_rc_curve'] / scale

        # Shaded area between each method and its oracle (visualises nAURC)
        ax.fill_between(t['coverages'], t_oracle_s, t_unc_s, color='blue', alpha=0.12)
        ax.fill_between(v['coverages'], t_oracle_s, v_unc_s, color='orange', alpha=0.10)
        ax.fill_between(p['coverages'], p_oracle_s, p_unc_s, color='purple', alpha=0.10)

        # Oracle & Random
        ax.plot(t['coverages'], t_oracle_s, 'g-', linewidth=1.5, label='TLRN Oracle')
        ax.plot(p['coverages'], p_oracle_s, color='green', linestyle=':',
                linewidth=1.5, label='PULPo Oracle')
        ax.plot(t['coverages'], t_random_s, 'r--', linewidth=1, label='Random')

        # TLRN
        ax.plot(t['coverages'], t_unc_s, 'b-', linewidth=1.5, label='TLRN')

        # VxM
        ax.plot(v['coverages'], v_unc_s, color='orange', linestyle='-',
                linewidth=1.5, label='VxM')

        # PULPo
        ax.plot(p['coverages'], p_unc_s, color='purple', linestyle='-',
                linewidth=1.5, label='PULPo')

        # Scale annotation in top left
        if exponent != 0:
            ax.text(0.03, 0.97, f'$\\times 10^{{{exponent}}}$',
                    transform=ax.transAxes, fontsize=6, va='top', ha='left')

        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=4, loc='upper right', handlelength=1.2, borderpad=0.2, labelspacing=0.15)

    for idx in range(num_plots, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.subplots_adjust(hspace=0.15, wspace=0.2)
    plt.savefig(os.path.join(save_dir, 'comparison_3way_risk_coverage_per_frame.png'),
                dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_3way_rc_average(tlrn_rc, vxm_rc, pulpo_rc, save_dir, tlrn_n, vxm_n, pulpo_n):
    """Single averaged risk-coverage plot with three methods."""
    common_frames = sorted(set(tlrn_rc.keys()) & set(vxm_rc.keys()) & set(pulpo_rc.keys()))
    if not common_frames:
        return

    coverages = tlrn_rc[common_frames[0]]['coverages']

    t_avg_unc = np.mean([tlrn_rc[f]['unc_rc_curve'] for f in common_frames], axis=0)
    t_avg_oracle = np.mean([tlrn_rc[f]['oracle_rc_curve'] for f in common_frames], axis=0)
    t_avg_random = np.mean([tlrn_rc[f]['random_rc_curve'] for f in common_frames], axis=0)
    t_avg_naurc = np.mean([tlrn_rc[f]['aurc_norm'] for f in common_frames])

    v_avg_unc = np.mean([vxm_rc[f]['unc_rc_curve'] for f in common_frames], axis=0)
    v_avg_naurc = np.mean([vxm_rc[f]['aurc_norm'] for f in common_frames])

    p_avg_unc = np.mean([pulpo_rc[f]['unc_rc_curve'] for f in common_frames], axis=0)
    p_avg_oracle = np.mean([pulpo_rc[f]['oracle_rc_curve'] for f in common_frames], axis=0)
    p_avg_naurc = np.mean([pulpo_rc[f]['aurc_norm'] for f in common_frames])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, t_avg_oracle, 'g-', linewidth=2, label='TLRN Oracle')
    ax.plot(coverages, p_avg_oracle, color='green', linestyle=':', linewidth=2, label='PULPo Oracle')
    ax.plot(coverages, t_avg_random, 'r--', linewidth=1.5, label='Random')
    ax.plot(coverages, t_avg_unc, 'b-', linewidth=2,
            label=f'TLRN (nAURC={t_avg_naurc:.3f})')
    ax.plot(coverages, v_avg_unc, color='orange', linestyle='-', linewidth=2,
            label=f'VxM (nAURC={v_avg_naurc:.3f})')
    ax.plot(coverages, p_avg_unc, color='purple', linestyle='-', linewidth=2,
            label=f'PULPo (nAURC={p_avg_naurc:.3f})')

    ax.set_xlabel('Coverage (fraction of retained pixels)')
    ax.set_ylabel('Risk (Cumulative Mean Error)')
    ax.set_title(f'Risk-Coverage (Avg over {len(common_frames)} frames)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_3way_risk_coverage_average.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_3way_rc_single_row(tlrn_rc, vxm_rc, pulpo_rc, save_dir, tlrn_n, vxm_n, pulpo_n):
    """Compact single-row risk-coverage for all frames."""
    common_frames = sorted(set(tlrn_rc.keys()) & set(vxm_rc.keys()) & set(pulpo_rc.keys()))
    num_plots = len(common_frames)
    if num_plots == 0:
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(2.5 * num_plots, 2.8), squeeze=False)

    for idx, fi in enumerate(common_frames):
        ax = axes[0, idx]
        t = tlrn_rc[fi]
        v = vxm_rc[fi]
        p = pulpo_rc[fi]

        all_vals = np.concatenate([
            t['unc_rc_curve'], t['oracle_rc_curve'], t['random_rc_curve'],
            v['unc_rc_curve'], p['unc_rc_curve'], p['oracle_rc_curve']
        ])
        max_val = np.max(all_vals)
        exponent = int(np.floor(np.log10(max_val))) if max_val > 0 else 0
        scale = 10 ** exponent

        ax.plot(t['coverages'], t['oracle_rc_curve'] / scale, 'g-', linewidth=1.5)
        ax.plot(p['coverages'], p['oracle_rc_curve'] / scale,
                color='green', linestyle=':', linewidth=1.5)
        ax.plot(t['coverages'], t['random_rc_curve'] / scale, 'r--', linewidth=1)
        ax.plot(t['coverages'], t['unc_rc_curve'] / scale, 'b-', linewidth=1.5)
        ax.plot(v['coverages'], v['unc_rc_curve'] / scale,
                color='orange', linestyle='-', linewidth=1.5)
        ax.plot(p['coverages'], p['unc_rc_curve'] / scale,
                color='purple', linestyle='-', linewidth=1.5)

        ax.set_xlabel('Coverage', fontsize=8)
        ax.set_title(f'Frame {fi}\nT:{t["aurc_norm"]:.3f} V:{v["aurc_norm"]:.3f} P:{p["aurc_norm"]:.3f}',
                      fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if exponent != 0:
            ax.text(0.02, 0.95, f'$\\times 10^{{{exponent}}}$',
                    transform=ax.transAxes, fontsize=7, va='top', ha='left')
        if idx == 0:
            ax.set_ylabel('Risk', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_3way_risk_coverage_single_row.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_3way_spars_per_frame(tlrn_spars, vxm_spars, pulpo_spars, save_dir, tlrn_n, vxm_n, pulpo_n):
    """Per-frame grid of sparsification plots with three methods overlaid."""
    common_frames = sorted(set(tlrn_spars.keys()) & set(vxm_spars.keys()) & set(pulpo_spars.keys()))
    num_plots = len(common_frames)
    if num_plots == 0:
        return

    # Save individual per-frame images
    spars_individual_dir = os.path.join(save_dir, 'spars_per_frame')
    os.makedirs(spars_individual_dir, exist_ok=True)
    for fi in common_frames:
        _plot_single_spars_frame(fi, tlrn_spars, vxm_spars, pulpo_spars,
                                 os.path.join(spars_individual_dir, f'sparsification_frame_{fi}.png'))

    # Grid plot
    cols = min(5, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.0 * rows), squeeze=False)

    for idx, fi in enumerate(common_frames):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        t = tlrn_spars[fi]
        v = vxm_spars[fi]
        p = pulpo_spars[fi]

        # Determine scale factor from max y value
        all_y = np.concatenate([t['oracle_curve'], t['random_curve'],
                                t['uncertainty_curve'], v['uncertainty_curve'],
                                p['uncertainty_curve']])
        max_val = np.max(all_y)
        if max_val > 0:
            exponent = int(np.floor(np.log10(max_val)))
        else:
            exponent = 0
        scale = 10 ** exponent

        # Oracle & Random (from TLRN)
        ax.plot(t['fractions'], t['oracle_curve'] / scale, 'g-', linewidth=1.5, label='Oracle')
        ax.plot(t['fractions'], t['random_curve'] / scale, 'r--', linewidth=1, label='Random')

        # Three methods
        ax.plot(t['fractions'], t['uncertainty_curve'] / scale, 'b-', linewidth=1.5, label='TLRN')
        ax.plot(v['fractions'], v['uncertainty_curve'] / scale, color='orange', linestyle='-',
                linewidth=1.5, label='VxM')
        ax.plot(p['fractions'], p['uncertainty_curve'] / scale, color='purple', linestyle='-',
                linewidth=1.5, label='PULPo')

        # Scale annotation in top left
        if exponent != 0:
            ax.text(0.03, 0.97, f'$\\times 10^{{{exponent}}}$',
                    transform=ax.transAxes, fontsize=6, va='top', ha='left')

        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=4, loc='upper left', handlelength=1.2, borderpad=0.2, labelspacing=0.15)

    for idx in range(num_plots, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.subplots_adjust(hspace=0.15, wspace=0.2)
    plt.savefig(os.path.join(save_dir, 'comparison_3way_sparsification_per_frame.png'),
                dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_3way_spars_average(tlrn_spars, vxm_spars, pulpo_spars, save_dir, tlrn_n, vxm_n, pulpo_n):
    """Single averaged sparsification plot with three methods."""
    common_frames = sorted(set(tlrn_spars.keys()) & set(vxm_spars.keys()) & set(pulpo_spars.keys()))
    if not common_frames:
        return

    fractions = tlrn_spars[common_frames[0]]['fractions']

    t_avg_unc = np.mean([tlrn_spars[f]['uncertainty_curve'] for f in common_frames], axis=0)
    t_avg_oracle = np.mean([tlrn_spars[f]['oracle_curve'] for f in common_frames], axis=0)
    t_avg_random = np.mean([tlrn_spars[f]['random_curve'] for f in common_frames], axis=0)
    t_avg_nause = np.mean([tlrn_spars[f]['ause_norm'] for f in common_frames])

    v_avg_unc = np.mean([vxm_spars[f]['uncertainty_curve'] for f in common_frames], axis=0)
    v_avg_nause = np.mean([vxm_spars[f]['ause_norm'] for f in common_frames])

    p_avg_unc = np.mean([pulpo_spars[f]['uncertainty_curve'] for f in common_frames], axis=0)
    p_avg_nause = np.mean([pulpo_spars[f]['ause_norm'] for f in common_frames])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions, t_avg_oracle, 'g-', linewidth=2, label='Oracle')
    ax.plot(fractions, t_avg_random, 'r--', linewidth=1.5, label='Random')
    ax.plot(fractions, t_avg_unc, 'b-', linewidth=2,
            label=f'TLRN (nAUSE={t_avg_nause:.3f})')
    ax.plot(fractions, v_avg_unc, color='orange', linestyle='-', linewidth=2,
            label=f'VxM (nAUSE={v_avg_nause:.3f})')
    ax.plot(fractions, p_avg_unc, color='purple', linestyle='-', linewidth=2,
            label=f'PULPo (nAUSE={p_avg_nause:.3f})')

    ax.set_xlabel('Fraction of Removed Pixels')
    ax.set_ylabel('Mean Remaining Error (MSE)')
    ax.set_title(f'Sparsification (Avg over {len(common_frames)} frames)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_3way_sparsification_average.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="3-way comparison: TLRN (inv-gamma) vs VxM Prob vs PULPo"
    )
    parser.add_argument("--vxm_checkpoint", type=str,
        default=os.path.join(VXM_PATH,
                             "runs_vxm_prob/vxm_prob_cine_20260130_031022/checkpoints/best_model.pt"))
    parser.add_argument("--pulpo_checkpoint", type=str,
        default=os.path.join(PULPO_PATH, "runs/pulbo_cine/version_0"))
    parser.add_argument("--tlrn_curves_dir", type=str,
        default="/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/"
                "2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03/"
                "visualization/uncertainty_learned_scaling_inv_gamma/test_results")
    parser.add_argument("--output_dir", type=str,
        default="/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/"
                "uncertainty_inv_gamma_prior/comparison_results_3way")
    parser.add_argument("--max_sequences", type=int, default=-1,
        help="Max sequences to evaluate (-1 for all)")
    parser.add_argument("--pulpo_num_samples", type=int, default=20,
        help="Number of posterior samples for PULPo uncertainty")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_T", type=int, default=15)
    parser.add_argument("--skip_vxm_eval", action='store_true',
        help="Skip VxM evaluation and load from cached .npz")
    parser.add_argument("--skip_pulpo_eval", action='store_true',
        help="Skip PULPo evaluation and load from cached .npz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    # ============================================================
    # Step 1: Load TLRN pre-computed curves
    # ============================================================
    print("=" * 60)
    print("Loading TLRN (inv-gamma prior) pre-computed curves...")
    print("=" * 60)
    tlrn_spars, tlrn_rc = load_tlrn_curves(args.tlrn_curves_dir)

    spars_npz_path = os.path.join(args.tlrn_curves_dir, 'avg_sparsification_curves.npz')
    tlrn_num_subjects = 0
    if os.path.exists(spars_npz_path):
        npz = np.load(spars_npz_path, allow_pickle=True)
        tlrn_num_subjects = int(npz.get('num_subjects', 0))

    print(f"  TLRN frames: {sorted(tlrn_spars.keys())}")
    print(f"  TLRN subjects: {tlrn_num_subjects}")

    if not tlrn_spars or not tlrn_rc:
        print("ERROR: TLRN .npz curves not found. Run uncertainty_inv_gamma_prior/test.py "
              "with SUBJECT_IDX=None first to generate them.")
        sys.exit(1)

    # Load test files (shared by VxM and PULPo)
    test_files = load_test_files()
    if args.max_sequences > 0:
        test_files = test_files[:args.max_sequences]

    # ============================================================
    # Step 2: Evaluate VoxelMorph Prob (or load cached)
    # ============================================================
    vxm_spars_cache = os.path.join(args.output_dir, 'vxm_avg_sparsification_curves.npz')
    vxm_rc_cache = os.path.join(args.output_dir, 'vxm_avg_risk_coverage_curves.npz')

    if args.skip_vxm_eval and os.path.exists(vxm_spars_cache) and os.path.exists(vxm_rc_cache):
        print("\n" + "=" * 60)
        print("Loading cached VxM curves...")
        print("=" * 60)
        vxm_spars, _, vxm_num_subjects = load_cached_curves(vxm_spars_cache)
        _, vxm_rc, _ = load_cached_curves(vxm_rc_cache)
        print(f"  VxM frames: {sorted(vxm_spars.keys())}")
        print(f"  VxM subjects: {vxm_num_subjects}")
    else:
        print("\n" + "=" * 60)
        print("Evaluating VoxelMorph Probabilistic...")
        print("=" * 60)
        vxm_model = load_vxm_model(args.vxm_checkpoint, device)

        print(f"  Test sequences: {len(test_files)}")
        all_vxm_spars = []
        all_vxm_rc = []

        for seq_idx, filepath in enumerate(tqdm(test_files, desc="VxM Eval")):
            try:
                sequence = load_sequence(filepath, target_T=args.target_T, resize=(64, 64))
                spars_res, rc_res = evaluate_vxm_single_sequence(
                    vxm_model, sequence, device=device
                )
                all_vxm_spars.append(spars_res)
                all_vxm_rc.append(rc_res)
            except Exception as e:
                print(f"  Warning: Error on sequence {seq_idx}: {e}")
                continue

        vxm_num_subjects = len(all_vxm_spars)
        print(f"\nSuccessfully evaluated {vxm_num_subjects} VxM sequences")

        vxm_spars = aggregate_curves(all_vxm_spars, 'sparsification')
        vxm_rc = aggregate_curves(all_vxm_rc, 'risk_coverage')

        save_curves_npz(vxm_spars, vxm_rc, args.output_dir, 'vxm', vxm_num_subjects)
        print(f"  Cached VxM curves to: {args.output_dir}")

        del vxm_model
        torch.cuda.empty_cache()

    # ============================================================
    # Step 3: Evaluate PULPo (or load cached)
    # ============================================================
    pulpo_spars_cache = os.path.join(args.output_dir, 'pulpo_avg_sparsification_curves.npz')
    pulpo_rc_cache = os.path.join(args.output_dir, 'pulpo_avg_risk_coverage_curves.npz')

    if args.skip_pulpo_eval and os.path.exists(pulpo_spars_cache) and os.path.exists(pulpo_rc_cache):
        print("\n" + "=" * 60)
        print("Loading cached PULPo curves...")
        print("=" * 60)
        pulpo_spars, _, pulpo_num_subjects = load_cached_curves(pulpo_spars_cache)
        _, pulpo_rc, _ = load_cached_curves(pulpo_rc_cache)
        print(f"  PULPo frames: {sorted(pulpo_spars.keys())}")
        print(f"  PULPo subjects: {pulpo_num_subjects}")
    else:
        print("\n" + "=" * 60)
        print("Evaluating PULPo...")
        print("=" * 60)
        pulpo_model = load_pulpo_model(args.pulpo_checkpoint, device)

        print(f"  Test sequences: {len(test_files)}")
        print(f"  Posterior samples per frame: {args.pulpo_num_samples}")
        all_pulpo_spars = []
        all_pulpo_rc = []

        for seq_idx, filepath in enumerate(tqdm(test_files, desc="PULPo Eval")):
            try:
                sequence = load_sequence(filepath, target_T=args.target_T, resize=(64, 64))
                spars_res, rc_res = evaluate_pulpo_single_sequence(
                    pulpo_model, sequence, num_samples=args.pulpo_num_samples, device=device
                )
                all_pulpo_spars.append(spars_res)
                all_pulpo_rc.append(rc_res)
            except Exception as e:
                print(f"  Warning: Error on sequence {seq_idx}: {e}")
                continue

        pulpo_num_subjects = len(all_pulpo_spars)
        print(f"\nSuccessfully evaluated {pulpo_num_subjects} PULPo sequences")

        pulpo_spars = aggregate_curves(all_pulpo_spars, 'sparsification')
        pulpo_rc = aggregate_curves(all_pulpo_rc, 'risk_coverage')

        save_curves_npz(pulpo_spars, pulpo_rc, args.output_dir, 'pulpo', pulpo_num_subjects)
        print(f"  Cached PULPo curves to: {args.output_dir}")

        del pulpo_model
        torch.cuda.empty_cache()

    # ============================================================
    # Step 4: Generate 3-way comparison plots
    # ============================================================
    print("\n" + "=" * 60)
    print("Generating 3-way comparison plots...")
    print("=" * 60)

    plot_3way_rc_per_frame(tlrn_rc, vxm_rc, pulpo_rc, args.output_dir,
                            tlrn_num_subjects, vxm_num_subjects, pulpo_num_subjects)
    print("  Saved: comparison_3way_risk_coverage_per_frame.png")
    print("  Saved: rc_per_frame/ (individual frame images)")

    plot_3way_rc_average(tlrn_rc, vxm_rc, pulpo_rc, args.output_dir,
                          tlrn_num_subjects, vxm_num_subjects, pulpo_num_subjects)
    print("  Saved: comparison_3way_risk_coverage_average.png")

    plot_3way_rc_single_row(tlrn_rc, vxm_rc, pulpo_rc, args.output_dir,
                             tlrn_num_subjects, vxm_num_subjects, pulpo_num_subjects)
    print("  Saved: comparison_3way_risk_coverage_single_row.png")

    plot_3way_spars_per_frame(tlrn_spars, vxm_spars, pulpo_spars, args.output_dir,
                               tlrn_num_subjects, vxm_num_subjects, pulpo_num_subjects)
    print("  Saved: comparison_3way_sparsification_per_frame.png")
    print("  Saved: spars_per_frame/ (individual frame images)")

    plot_3way_spars_average(tlrn_spars, vxm_spars, pulpo_spars, args.output_dir,
                             tlrn_num_subjects, vxm_num_subjects, pulpo_num_subjects)
    print("  Saved: comparison_3way_sparsification_average.png")

    # ============================================================
    # Summary table
    # ============================================================
    common_frames = sorted(set(tlrn_spars.keys()) & set(vxm_spars.keys()) & set(pulpo_spars.keys()))
    print(f"\n{'='*85}")
    print(f"{'3-WAY COMPARISON SUMMARY':^85}")
    print(f"{'='*85}")
    print(f"{'Frame':>6} | {'TLRN nAUSE':>11} {'VxM nAUSE':>11} {'PULPo nAUSE':>12} | "
          f"{'TLRN nAURC':>11} {'VxM nAURC':>11} {'PULPo nAURC':>12}")
    print(f"{'-'*85}")
    for fi in common_frames:
        t_nause = tlrn_spars[fi]['ause_norm']
        v_nause = vxm_spars[fi]['ause_norm']
        p_nause = pulpo_spars[fi]['ause_norm']
        t_naurc = tlrn_rc[fi]['aurc_norm'] if fi in tlrn_rc else float('nan')
        v_naurc = vxm_rc[fi]['aurc_norm'] if fi in vxm_rc else float('nan')
        p_naurc = pulpo_rc[fi]['aurc_norm'] if fi in pulpo_rc else float('nan')
        print(f"{fi:>6} | {t_nause:>11.4f} {v_nause:>11.4f} {p_nause:>12.4f} | "
              f"{t_naurc:>11.4f} {v_naurc:>11.4f} {p_naurc:>12.4f}")

    t_avg_nause = np.mean([tlrn_spars[f]['ause_norm'] for f in common_frames])
    v_avg_nause = np.mean([vxm_spars[f]['ause_norm'] for f in common_frames])
    p_avg_nause = np.mean([pulpo_spars[f]['ause_norm'] for f in common_frames])
    t_avg_naurc = np.mean([tlrn_rc[f]['aurc_norm'] for f in common_frames if f in tlrn_rc])
    v_avg_naurc = np.mean([vxm_rc[f]['aurc_norm'] for f in common_frames if f in vxm_rc])
    p_avg_naurc = np.mean([pulpo_rc[f]['aurc_norm'] for f in common_frames if f in pulpo_rc])
    print(f"{'-'*85}")
    print(f"{'Avg':>6} | {t_avg_nause:>11.4f} {v_avg_nause:>11.4f} {p_avg_nause:>12.4f} | "
          f"{t_avg_naurc:>11.4f} {v_avg_naurc:>11.4f} {p_avg_naurc:>12.4f}")
    print(f"{'='*85}")

    # Save summary text
    summary_file = os.path.join(args.output_dir, 'comparison_3way_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("3-Way Uncertainty Comparison: TLRN (inv-gamma) vs VxM Prob vs PULPo\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"TLRN subjects: {tlrn_num_subjects}\n")
        f.write(f"VxM subjects: {vxm_num_subjects}\n")
        f.write(f"PULPo subjects: {pulpo_num_subjects}\n\n")
        f.write(f"{'Frame':>6} | {'TLRN nAUSE':>11} {'VxM nAUSE':>11} {'PULPo nAUSE':>12} | "
                f"{'TLRN nAURC':>11} {'VxM nAURC':>11} {'PULPo nAURC':>12}\n")
        f.write("-" * 85 + "\n")
        for fi in common_frames:
            t_nause = tlrn_spars[fi]['ause_norm']
            v_nause = vxm_spars[fi]['ause_norm']
            p_nause = pulpo_spars[fi]['ause_norm']
            t_naurc = tlrn_rc[fi]['aurc_norm'] if fi in tlrn_rc else float('nan')
            v_naurc = vxm_rc[fi]['aurc_norm'] if fi in vxm_rc else float('nan')
            p_naurc = pulpo_rc[fi]['aurc_norm'] if fi in pulpo_rc else float('nan')
            f.write(f"{fi:>6} | {t_nause:>11.4f} {v_nause:>11.4f} {p_nause:>12.4f} | "
                    f"{t_naurc:>11.4f} {v_naurc:>11.4f} {p_naurc:>12.4f}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Avg':>6} | {t_avg_nause:>11.4f} {v_avg_nause:>11.4f} {p_avg_nause:>12.4f} | "
                f"{t_avg_naurc:>11.4f} {v_avg_naurc:>11.4f} {p_avg_naurc:>12.4f}\n")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
