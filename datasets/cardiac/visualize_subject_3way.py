#!/usr/bin/env python3
"""
Per-subject visualisation comparing TLRN, VoxelMorph Prob, and PULPo.

For a chosen test subject, generates per-frame images of:
  - Source / target images
  - Registered images (all 3 methods)
  - Velocity-field magnitude (all 3 methods)
  - Uncertainty maps (all 3 methods)
  - Registration error |target - registered| (all 3 methods)

Saves a combined grid and individual images.

Usage:
    python uncertainty_inv_gamma_prior/visualize_subject_3way.py \
        --subject_idx 0 --device cuda
"""

import os
import sys
import argparse

# Project root
sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import SimpleITK as sitk

# TLRN components — imported BEFORE adding VxM/PULPo to sys.path
# so that 'models.TLRN' resolves to the project package, not voxelmorph's models.py
from uncertainty_plot.networks import ScalingFactorNetwork
from uncertainty_plot.losses import LTMALoss
from uncertainty_plot.brownian_bridge import BrownianBridgeLearnedScaling
from uncertainty_plot.trainer import ScalingFactorTrainer
from uncertainty_plot.data_utils import load_model_and_data
from utils.Int import VecInt, SpatialTransformer as TLRNSpatialTransformer

# Path constants (imports are deferred to load functions to avoid sys.path conflicts)
VXM_PATH = "/scratch/swd9tc/Uncertanity_quantification/voxelmorph_prob"
PULPO_PATH = "/scratch/swd9tc/Uncertanity_quantification/pulbo"

# ============================================================
# Path configuration
# ============================================================
OLD_DATA_ROOT = "/scratch/bsw3ac/nellie/code/cvpr/Project/Foundation/Data"
NEW_DATA_ROOT = "/scratch/swd9tc/Uncertanity_quantification/DataPerNor"
TXT_FOLDER = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightningTemplate/dataload"


def remap_path(path):
    if OLD_DATA_ROOT in path:
        return path.replace(OLD_DATA_ROOT, NEW_DATA_ROOT)
    return path


def load_sequence(file_path, target_T=15, resize=(64, 64)):
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
        vt = torch.from_numpy(vol).float().permute(1, 2, 0).reshape(1, H * W, T)
        vt = F.interpolate(vt, size=target_T, mode='linear', align_corners=True)
        vol = vt.reshape(H, W, target_T).permute(2, 0, 1).numpy()
    if resize is not None:
        oh, ow = resize
        vt = torch.from_numpy(vol).float().unsqueeze(1)
        vt = F.interpolate(vt, size=(oh, ow), mode='bilinear', align_corners=False)
        vol = vt.squeeze(1).numpy()
    return vol


def load_test_files():
    from pathlib import Path
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
# Model loaders
# ============================================================

def load_vxm_model(checkpoint_path, device='cuda'):
    # Use importlib to load VxM models.py directly, avoiding name collision
    # with TLRN's 'models' package already in sys.modules
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vxm_models", os.path.join(VXM_PATH, "models.py"))
    vxm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vxm_mod)
    VxmProbabilistic = vxm_mod.VxmProbabilistic

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get('args', None)
    if ckpt_args is not None:
        nb_features = ckpt_args.nb_features
        int_steps = ckpt_args.int_steps
    else:
        nb_features = [16, 32, 32, 32]
        int_steps = 7
    model = VxmProbabilistic(
        inshape=(64, 64), in_channels=1,
        nb_features=nb_features, int_steps=int_steps,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded VxmProbabilistic")
    return model


def load_pulpo_model(checkpoint_dir, device='cuda'):
    # Lazy import to avoid sys.path conflict with TLRN 'models' package
    if PULPO_PATH not in sys.path:
        sys.path.insert(0, PULPO_PATH)
    from evaluate import Evaluate

    eval_obj = Evaluate()
    parts = checkpoint_dir.rstrip('/').split('/')
    if 'version' in parts[-1]:
        version = parts[-1]
        git_hash = parts[-2]
        model_dir = '/'.join(parts[:-2])
    else:
        git_hash = parts[-1]
        model_dir = '/'.join(parts[:-1])
        version = "version_0"
    model = eval_obj.load_model(model_dir=model_dir, git_hash=git_hash, version=version)
    model = model.to(device)
    model.eval()
    print(f"  Loaded PULPo")
    return model


def load_tlrn_model(checkpoint_path, device='cuda'):
    """Load TLRN registration model + scaling network + brownian bridge."""
    model, _, test_loader, config = load_model_and_data()

    # Determine dimensions
    first_batch = next(iter(test_loader))
    first_series = first_batch['series'].to(device)
    first_lv = first_batch.get('lv_segs')
    if first_lv is not None:
        first_lv = first_lv.to(device)
    else:
        first_lv = torch.zeros_like(first_series[:, 0:1]).to(device)
    with torch.no_grad():
        _, fv, _, _, _ = model.sequence_register_no_avg_lowf_addlatentf(first_series, first_lv)
    num_time_steps = len(fv)
    img_size = first_series.shape[-1]
    del first_series, first_lv, fv
    torch.cuda.empty_cache()

    # Scaling network
    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps, img_size=img_size, num_heads=8
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    scaling_network.load_state_dict(ckpt['model_state_dict'])
    scaling_network.to(device)
    scaling_network.eval()
    ckpt_config = ckpt.get('config', {})

    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=14, img_size=img_size
    )
    loss_fn = LTMALoss(
        lambda_sim=ckpt_config.get('lambda_sim', 1.0),
        lambda_reg=ckpt_config.get('lambda_reg', 0.0),
        lambda_scale=ckpt_config.get('lambda_scale', 0.001),
        lambda_low_structure=ckpt_config.get('lambda_low_structure', 0.0),
        use_ncc=False,
    )
    trainer = ScalingFactorTrainer(
        scaling_network=scaling_network, brownian_bridge=brownian_bridge,
        loss_fn=loss_fn, lr=1e-4, device=device, img_size=img_size,
    )
    print(f"  Loaded TLRN (epoch {ckpt.get('epoch', '?')})")
    return model, test_loader, config, trainer, brownian_bridge, num_time_steps, img_size


# ============================================================
# Per-model inference
# ============================================================

@torch.no_grad()
def run_vxm(model, sequence, device):
    """Returns per-frame dicts with registered, vel_mag, uncertainty, error."""
    T = sequence.shape[0]
    source = torch.from_numpy(sequence[0]).float().unsqueeze(0).unsqueeze(0).to(device)
    results = {}
    for t in range(1, T - 1):
        target = torch.from_numpy(sequence[t]).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(source, target, sample=False)
        reg = out['y_source'][0, 0].cpu().numpy()
        flow = out['preint_flow'][0]  # [2, H, W]
        vel_mag = torch.norm(flow, dim=0).cpu().numpy()
        sigma = out['sigma'][0].mean(dim=0).cpu().numpy()  # [H, W]
        err = np.abs(sequence[t] - reg)
        results[t] = {
            'registered': reg, 'vel_mag': vel_mag,
            'uncertainty': sigma, 'error': err,
        }
    return results


@torch.no_grad()
def run_pulpo(model, sequence, num_samples, device):
    T = sequence.shape[0]
    source = torch.from_numpy(sequence[0]).float().unsqueeze(0).unsqueeze(0).to(device)
    results = {}
    for t in range(1, T - 1):
        target = torch.from_numpy(sequence[t]).float().unsqueeze(0).unsqueeze(0).to(device)
        outputs, individual_dfs = model.predict_output_samples(source, target, N=num_samples)
        all_out = outputs[0][0]  # [N, 1, H, W]
        mean_warped = torch.mean(all_out, dim=0).squeeze(0).cpu().numpy()
        unc = torch.std(all_out, dim=0).squeeze(0).cpu().numpy()

        # Deformation field magnitude (average across samples)
        avg_dfs = {k: individual_dfs[k].mean(dim=1) for k in individual_dfs}
        _, final_dfs = model.combine_dfs(avg_dfs)
        df = final_dfs[0][0]  # [2, H, W]
        vel_mag = torch.norm(df, dim=0).cpu().numpy()

        err = np.abs(sequence[t] - mean_warped)
        results[t] = {
            'registered': mean_warped, 'vel_mag': vel_mag,
            'uncertainty': unc, 'error': err,
        }
    return results


@torch.no_grad()
def run_tlrn(reg_model, test_loader, trainer, brownian_bridge,
             subject_idx, num_runs, num_time_steps, img_size, device):
    """Run TLRN registration + uncertainty for a single subject."""
    # Find the batch containing subject_idx
    batch_size = None
    target_batch_idx = None
    subj_in_batch = None

    for bi, batch in enumerate(test_loader):
        bs = batch['series'].shape[0]
        if batch_size is None:
            batch_size = bs
        if subject_idx < (bi + 1) * bs:
            target_batch_idx = bi
            subj_in_batch = subject_idx - bi * bs
            break

    if target_batch_idx is None:
        raise ValueError(f"subject_idx {subject_idx} out of range")

    # Re-iterate to target batch
    for bi, batch in enumerate(test_loader):
        if bi == target_batch_idx:
            break

    series = batch['series'].to(device)
    lv_segs = batch.get('lv_segs')
    if lv_segs is not None:
        lv_segs = lv_segs.to(device)
    else:
        lv_segs = torch.zeros_like(series[:, 0:1]).to(device)

    # Registration
    Sdef_series, v_series, _, _, _ = reg_model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)

    source_img = series[:, 0:1]
    target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]

    zero_velocity = torch.zeros_like(v_series[0])
    v_series_bb = [zero_velocity] + v_series

    _, scaling_factors_bb = trainer.validate(source_img, target_imgs, v_series_bb)

    # Brownian bridge sampling
    v_0 = zero_velocity.clone()
    v_T = v_series[-1].clone()
    num_ts_bb = len(v_series_bb)
    sampled_velocities = {i: [] for i in range(num_ts_bb)}

    for run in tqdm(range(num_runs), desc="TLRN UQ runs"):
        trajectory = brownian_bridge.run_reverse_process(
            v_series_bb, scaling_factors_bb, v_0, v_T, device=device
        )
        for fi in range(num_ts_bb):
            traj_idx = 14 - int(fi * 14 / (num_ts_bb - 1)) if num_ts_bb > 1 else 0
            traj_idx = min(max(traj_idx, 0), len(trajectory) - 1)
            sampled_velocities[fi].append(trajectory[traj_idx].clone())
        del trajectory
        if (run + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Extract per-frame results
    si = subj_in_batch
    inshape = (img_size, img_size)
    vec_int = VecInt(inshape, TSteps=7).to(device)
    spatial_tf = TLRNSpatialTransformer(inshape).to(device)
    src = series[si:si+1, 0:1]  # [1, 1, H, W]

    results = {}
    for frame_idx in range(1, series.shape[1] - 1):
        # Registered image from network
        reg = Sdef_series[frame_idx - 1][si, 0].cpu().numpy()

        # Velocity magnitude
        vel = v_series[frame_idx - 1][si]  # [2, H, W]
        vel_mag = torch.norm(vel, dim=0).cpu().numpy()

        # Uncertainty from BB samples
        bb_idx = frame_idx  # in v_series_bb, index 0 is zero-vel, 1..14 are frames
        vels = sampled_velocities[bb_idx]
        n_samples = min(50, len(vels))
        indices = np.random.choice(len(vels), n_samples, replace=False)
        warped_imgs = []
        for idx in indices:
            v = vels[idx][si:si+1]
            disp = vec_int(v)[-1]
            w, _ = spatial_tf(src, disp)
            warped_imgs.append(w[0, 0].cpu().numpy())
        unc = np.std(np.stack(warped_imgs, axis=0), axis=0)

        target_np = series[si, frame_idx].cpu().numpy()
        err = np.abs(target_np - reg)

        results[frame_idx] = {
            'registered': reg, 'vel_mag': vel_mag,
            'uncertainty': unc, 'error': err,
        }

    # Also return the sequence for plotting source/target
    seq_np = series[si].cpu().numpy()  # [T, H, W]
    return results, seq_np


# ============================================================
# Plotting utilities
# ============================================================

def _save_individual_image(arr, path, cmap='gray', vmin=None, vmax=None):
    """Save a single 2D array as a compact image."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def _save_overlay_image(bg, overlay, path, cmap='hot', vmin=0, vmax=None, alpha=0.8):
    """Save overlay heatmap on grayscale background."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(bg, cmap='gray')
    ax.imshow(overlay, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    ax.axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def _plot_overlay(ax, bg, overlay, cmap='hot', vmin=0, vmax=None, alpha=0.8):
    """Draw overlay heatmap on grayscale background into an axes."""
    ax.imshow(bg, cmap='gray')
    ax.imshow(overlay, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)


def _compute_vmax(results_list, key, frames):
    """Compute a shared vmax across all methods and frames for consistent colouring."""
    vals = []
    for res in results_list:
        for t in frames:
            if t in res:
                vals.append(np.max(res[t][key]))
    return max(vals) if vals else 1.0


def save_individual_images(sequence, tlrn_res, vxm_res, pulpo_res, frames, save_dir):
    """Save every image as a separate file."""
    os.makedirs(save_dir, exist_ok=True)

    # Shared vmax for consistent colour scale across methods
    unc_vmax = _compute_vmax([tlrn_res, vxm_res, pulpo_res], 'uncertainty', frames)
    vel_vmax = _compute_vmax([tlrn_res, vxm_res, pulpo_res], 'vel_mag', frames)

    for t in frames:
        _save_individual_image(sequence[0], os.path.join(save_dir, f'frame_{t}_source.png'))
        _save_individual_image(sequence[t], os.path.join(save_dir, f'frame_{t}_target.png'))

        for method, res in [('tlrn', tlrn_res), ('vxm', vxm_res), ('pulpo', pulpo_res)]:
            d = res[t]
            _save_individual_image(d['registered'], os.path.join(save_dir, f'frame_{t}_{method}_registered.png'))
            _save_individual_image(d['vel_mag'], os.path.join(save_dir, f'frame_{t}_{method}_velocity_mag.png'), cmap='jet', vmin=0, vmax=vel_vmax)
            _save_overlay_image(sequence[0], d['uncertainty'], os.path.join(save_dir, f'frame_{t}_{method}_uncertainty.png'), cmap='hot', vmin=0, vmax=unc_vmax)
            _save_individual_image(d['error'], os.path.join(save_dir, f'frame_{t}_{method}_error.png'), cmap='hot')


def plot_per_frame_grid(sequence, tlrn_res, vxm_res, pulpo_res, frame_idx,
                        save_path, unc_vmax=None, vel_vmax=None):
    """
    5-row x 3-col grid for a single frame:
        Row 0: Source | Target | (blank)
        Row 1: TLRN Reg | VxM Reg | PULPo Reg
        Row 2: TLRN VelMag | VxM VelMag | PULPo VelMag
        Row 3: TLRN Unc | VxM Unc | PULPo Unc  (overlaid on source)
        Row 4: TLRN Err | VxM Err | PULPo Err
    """
    methods = [('TLRN', tlrn_res), ('VxM', vxm_res), ('PULPo', pulpo_res)]
    fig, axes = plt.subplots(5, 3, figsize=(5.4, 9))

    # Row 0: source, target, blank
    axes[0, 0].imshow(sequence[0], cmap='gray')
    axes[0, 0].set_title('Source', fontsize=7)
    axes[0, 1].imshow(sequence[frame_idx], cmap='gray')
    axes[0, 1].set_title('Target', fontsize=7)
    axes[0, 2].set_visible(False)

    source_img = sequence[0]

    for col, (name, res) in enumerate(methods):
        d = res[frame_idx]
        # Row 1: Registered
        axes[1, col].imshow(d['registered'], cmap='gray')
        if col == 0:
            axes[1, col].set_title(name, fontsize=7)
        else:
            axes[1, col].set_title(name, fontsize=7)
        # Row 2: Velocity magnitude
        axes[2, col].imshow(d['vel_mag'], cmap='jet', vmin=0, vmax=vel_vmax)
        # Row 3: Uncertainty overlaid on source
        _plot_overlay(axes[3, col], source_img, d['uncertainty'],
                      cmap='hot', vmin=0, vmax=unc_vmax, alpha=0.8)
        # Row 4: Error
        axes[4, col].imshow(d['error'], cmap='hot')

    for ax in axes.flat:
        ax.axis('off')

    fig.subplots_adjust(hspace=0.08, wspace=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_summary_grid(sequence, tlrn_res, vxm_res, pulpo_res, frames, save_path,
                      unc_vmax=None, vel_vmax=None):
    """
    Summary grid: columns = frames, rows grouped by method.
    Layout per method-block (3 rows): Registered, Velocity Mag, Uncertainty
    Plus top 2 rows: Source (repeated), Target per frame.
    Total rows = 2 + 3*3 = 11
    """
    n_frames = len(frames)
    methods = [('TLRN', tlrn_res), ('VxM', vxm_res), ('PULPo', pulpo_res)]
    n_rows = 2 + 3 * 3  # source + target + 3 methods * 3 outputs each
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(1.6 * n_frames, 1.4 * n_rows))

    if n_frames == 1:
        axes = axes[:, np.newaxis]

    source_img = sequence[0]

    for col, t in enumerate(frames):
        # Row 0: Source
        axes[0, col].imshow(source_img, cmap='gray')
        # Row 1: Target
        axes[1, col].imshow(sequence[t], cmap='gray')

        for mi, (mname, res) in enumerate(methods):
            d = res[t]
            base_row = 2 + mi * 3
            axes[base_row, col].imshow(d['registered'], cmap='gray')
            axes[base_row + 1, col].imshow(d['vel_mag'], cmap='jet', vmin=0, vmax=vel_vmax)
            _plot_overlay(axes[base_row + 2, col], source_img, d['uncertainty'],
                          cmap='hot', vmin=0, vmax=unc_vmax, alpha=0.8)

    # Row labels on first column
    row_labels = ['Source', 'Target']
    for mname, _ in methods:
        row_labels += [f'{mname} Reg', f'{mname} Vel', f'{mname} Unc']
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=5, rotation=90, labelpad=2)

    # Frame labels on top
    for col, t in enumerate(frames):
        axes[0, col].set_title(f'Frame {t}', fontsize=5)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-subject 3-way visualisation: TLRN vs VxM vs PULPo"
    )
    parser.add_argument("--subject_idx", type=int, default=10,
                        help="Subject index in test file list (0-indexed)")
    parser.add_argument("--vxm_checkpoint", type=str,
        default=os.path.join(VXM_PATH,
                             "runs_vxm_prob/vxm_prob_cine_20260130_031022/checkpoints/best_model.pt"))
    parser.add_argument("--pulpo_checkpoint", type=str,
        default=os.path.join(PULPO_PATH, "runs/pulbo_cine/version_0"))
    parser.add_argument("--tlrn_checkpoint", type=str,
        default="2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03/"
                "visualization/uncertainty_sde_combined/checkpoints/"
                "checkpoint_epoch_1800.pth")
    parser.add_argument("--output_dir", type=str,
        default="/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/"
                "uncertainty_inv_gamma_prior/subject_visualizations")
                
    parser.add_argument("--pulpo_num_samples", type=int, default=20)
    parser.add_argument("--tlrn_num_runs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_T", type=int, default=15)
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    subj_dir = os.path.join(args.output_dir, f'subject_{args.subject_idx}')
    indiv_dir = os.path.join(subj_dir, 'individual')
    frame_grid_dir = os.path.join(subj_dir, 'frame_grids')
    os.makedirs(indiv_dir, exist_ok=True)
    os.makedirs(frame_grid_dir, exist_ok=True)

    # Load test files (shared across all models)
    test_files = load_test_files()
    if args.subject_idx >= len(test_files):
        print(f"ERROR: subject_idx {args.subject_idx} >= number of test files ({len(test_files)})")
        sys.exit(1)
    print(f"Subject {args.subject_idx}: {test_files[args.subject_idx]}")

    # ============================================================
    # TLRN — loaded FIRST so that 'models.TLRN' import resolves
    # before VxM/PULPo add their paths to sys.path
    # ============================================================
    print("\n" + "=" * 60)
    print("Running TLRN...")
    print("=" * 60)
    (reg_model, test_loader, config, trainer,
     bb, num_ts, img_sz) = load_tlrn_model(args.tlrn_checkpoint, device)
    tlrn_results, tlrn_seq = run_tlrn(
        reg_model, test_loader, trainer, bb,
        args.subject_idx, args.tlrn_num_runs, num_ts, img_sz, device,
    )
    del reg_model, trainer, bb
    torch.cuda.empty_cache()
    print(f"  TLRN frames: {sorted(tlrn_results.keys())}")

    # Load sequence for VxM and PULPo (they use raw numpy via load_sequence)
    sequence = load_sequence(test_files[args.subject_idx],
                             target_T=args.target_T, resize=(64, 64))
    print(f"Sequence shape: {sequence.shape}")

    # ============================================================
    # VoxelMorph Probabilistic
    # ============================================================
    print("\n" + "=" * 60)
    print("Running VoxelMorph Probabilistic...")
    print("=" * 60)
    vxm_model = load_vxm_model(args.vxm_checkpoint, device)
    vxm_results = run_vxm(vxm_model, sequence, device)
    del vxm_model
    torch.cuda.empty_cache()
    print(f"  VxM frames: {sorted(vxm_results.keys())}")

    # ============================================================
    # PULPo
    # ============================================================
    print("\n" + "=" * 60)
    print("Running PULPo...")
    print("=" * 60)
    pulpo_model = load_pulpo_model(args.pulpo_checkpoint, device)
    pulpo_results = run_pulpo(pulpo_model, sequence, args.pulpo_num_samples, device)
    del pulpo_model
    torch.cuda.empty_cache()
    print(f"  PULPo frames: {sorted(pulpo_results.keys())}")

    # Use TLRN sequence (from data loader) as ground truth for consistency
    sequence_for_plot = tlrn_seq

    # Common frames across all three methods
    common_frames = sorted(
        set(tlrn_results.keys()) & set(vxm_results.keys()) & set(pulpo_results.keys())
    )
    print(f"\nCommon frames: {common_frames}")

    # Compute shared vmax for consistent colour scales
    all_res = [tlrn_results, vxm_results, pulpo_results]
    unc_vmax = _compute_vmax(all_res, 'uncertainty', common_frames)
    vel_vmax = _compute_vmax(all_res, 'vel_mag', common_frames)
    print(f"  Shared scales: unc_vmax={unc_vmax:.4f}, vel_vmax={vel_vmax:.4f}")

    # ============================================================
    # Save individual images
    # ============================================================
    print("\nSaving individual images...")
    save_individual_images(sequence_for_plot, tlrn_results, vxm_results,
                           pulpo_results, common_frames, indiv_dir)

    # ============================================================
    # Per-frame grid plots
    # ============================================================
    print("Saving per-frame grid plots...")
    for t in common_frames:
        plot_per_frame_grid(
            sequence_for_plot, tlrn_results, vxm_results, pulpo_results, t,
            os.path.join(frame_grid_dir, f'frame_{t}_grid.png'),
            unc_vmax=unc_vmax, vel_vmax=vel_vmax,
        )

    # ============================================================
    # Summary grid
    # ============================================================
    print("Saving summary grid...")
    plot_summary_grid(
        sequence_for_plot, tlrn_results, vxm_results, pulpo_results, common_frames,
        os.path.join(subj_dir, f'subject_{args.subject_idx}_summary_grid.png'),
        unc_vmax=unc_vmax, vel_vmax=vel_vmax,
    )

    print(f"\nDone! Results saved to: {subj_dir}")


if __name__ == '__main__':
    main()
