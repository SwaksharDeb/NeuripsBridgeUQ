"""
Correlation bar-plot script -- computes Pearson (r) and Spearman (rho)
correlations between uncertainty maps and registration error maps for
multiple methods (TLRN, VM, LTMA, TM) across selected frames.

Inspired by the correlation analysis in:
  "Uncertainty Estimation for Pretrained Medical Image Registration Models
   via Transformation Equivariance" (arXiv 2509.23355)

Run with:
  python -m uncertainty_sde_combined.test_correlation_barplot [--num_runs N]
"""

import os
import sys
import argparse
import importlib
import importlib.util
import gc

import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


# ==========================================================================
# Method configurations
# ==========================================================================

METHOD_CONFIGS = {
    'TLRN': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2',
        'checkpoint': '2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03/visualization/uncertainty_sde_combined/checkpoints/checkpoint_epoch_500.pth',
        'file_paths_key': 'path',
        'use_svf_module': False,
    },
    'VM': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/voxel_and_R2R/voxelmorph',
        'checkpoint': '/scratch/swd9tc/Uncertanity_quantification/voxel_and_R2R/voxelmorph/visualization_voxelmorph/uncertainty_sde_combined/checkpoints/checkpoint_epoch_500.pth',
        'file_paths_key': 'paths',
        'use_svf_module': True,
    },
    'LTMA': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA',
        'checkpoint': '/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/2026Experiments/CINE/outputs/TLMA_TGrad/config/visualization/uncertainty_sde_combined/checkpoints/checkpoint_epoch_500.pth',
        'file_paths_key': 'path',
        'use_svf_module': False,
    },
    'TM': {
        'sys_path': '/scratch/swd9tc/Uncertanity_quantification/TM',
        'checkpoint': '/scratch/swd9tc/Uncertanity_quantification/TM/2026Experiments/CINE/outputs/TransMorph/basic/visualization/uncertainty_sde_combined/checkpoints/checkpoint_epoch_900.pth',
        'file_paths_key': 'path',
        'use_svf_module': False,
    },
}


# ==========================================================================
# Per-subject per-frame correlation computation
# ==========================================================================

def _compute_correlations(sampled_velocities, Sdef_series, series, subject_idx):
    """Compute Pearson (r), Spearman (rho), nAURC, and mean error per frame.

    Returns dict: frame_idx -> {
        'pearson': r, 'spearman': rho, 'naurc': float, 'mean_error': float
    }
    """
    num_frames_sdef = len(Sdef_series)
    frame_correlations = {}
    coverages = np.linspace(0.15, 1, 86)

    for sdef_idx in range(num_frames_sdef):
        frame_idx = sdef_idx + 1

        if sdef_idx == num_frames_sdef - 1:
            continue
        if frame_idx not in sampled_velocities or len(sampled_velocities[frame_idx]) == 0:
            continue

        # Per-pixel uncertainty (velocity-space std)
        velocities_at_t = np.stack(
            [v[subject_idx].cpu().numpy() for v in sampled_velocities[frame_idx]], axis=0
        )
        variance = np.var(velocities_at_t, axis=0)
        uncertainty = np.sqrt(variance[0] + variance[1])

        if np.max(uncertainty) < 1e-8:
            continue

        # Per-pixel registration error
        target = series[subject_idx, frame_idx].cpu().numpy()
        deformed = Sdef_series[sdef_idx][subject_idx, 0].cpu().numpy()
        pixel_error = (deformed - target) ** 2

        unc_flat = uncertainty.flatten()
        err_flat = pixel_error.flatten()
        n_pixels = len(unc_flat)

        # Pearson correlation (linear)
        r_pearson, _ = pearsonr(unc_flat, err_flat)
        # Spearman rank correlation (monotonic)
        r_spearman, _ = spearmanr(unc_flat, err_flat)

        # Mean registration error
        mean_error = np.mean(err_flat)

        # nAURC (normalised Area Under Risk-Coverage curve)
        eps = 1e-6
        unc_confidence = 1.0 / (unc_flat + eps)
        unc_asc_order = np.argsort(unc_confidence)
        oracle_asc_order = np.argsort(err_flat)

        unc_rc = np.zeros(len(coverages))
        oracle_rc = np.zeros(len(coverages))
        random_rc = np.zeros(len(coverages))

        for j, cov in enumerate(coverages):
            n_retain = max(int(cov * n_pixels), 1)
            unc_rc[j] = np.mean(err_flat[unc_asc_order[:n_retain]])
            oracle_rc[j] = np.mean(err_flat[oracle_asc_order[:n_retain]])
            random_rc[j] = mean_error

        aurc = np.trapz(unc_rc, coverages)
        oracle_aurc = np.trapz(oracle_rc, coverages)
        random_aurc = np.trapz(random_rc, coverages)
        naurc = (aurc - oracle_aurc) / (random_aurc - oracle_aurc) \
            if (random_aurc - oracle_aurc) > 1e-12 else 0.0

        frame_correlations[frame_idx] = {
            'pearson': r_pearson,
            'spearman': r_spearman,
            'naurc': naurc,
            'mean_error': mean_error,
        }

    return frame_correlations


# ==========================================================================
# Load and run one method
# ==========================================================================

def _run_method(method_name, method_cfg, num_runs, num_diffusion_steps, use_train=False):
    """
    Load a method, run inference + UQ, return per-subject per-frame correlations.
    Returns: (list_of_dicts, available_frame_indices)
        Each dict maps frame_idx -> pearson_corr for one subject.
    """
    method_sys_path = method_cfg['sys_path']

    # Save original sys.path and reset to avoid cross-contamination
    # from previous method's paths leaking into imports.
    original_sys_path = sys.path.copy()
    # Remove all other method paths, keep only this method's path
    for other_cfg in METHOD_CONFIGS.values():
        other_path = other_cfg['sys_path']
        if other_path != method_sys_path:
            while other_path in sys.path:
                sys.path.remove(other_path)
    if method_sys_path not in sys.path:
        sys.path.insert(0, method_sys_path)

    # Force-reload method-specific modules so we get the right version.
    # Must also clear shared top-level names (utils, networks, dataload, etc.)
    # that differ across projects.
    _purge_prefixes = [
        'uncertainty_sde_combined', 'uncertainty_inv_gamma_prior',
        'uncertainty_sde_fix_u',
        'networks', 'utils', 'dataload', 'models',
        'visualization_voxelmorph',
    ]
    for mod_name in list(sys.modules.keys()):
        if any(mod_name == pfx or mod_name.startswith(pfx + '.')
               for pfx in _purge_prefixes):
            del sys.modules[mod_name]

    from uncertainty_sde_combined_acdc.networks import ScalingFactorNetwork
    from uncertainty_sde_combined_acdc.losses import LTMALoss
    from uncertainty_sde_combined_acdc.brownian_bridge import BrownianBridgeLearnedScaling
    from uncertainty_sde_combined_acdc.trainer import ScalingFactorTrainer
    from uncertainty_sde_combined_acdc.data_utils import load_model_and_data

    # Load model
    model, train_loader, test_loader, config = load_model_and_data()
    data_loader = train_loader if use_train else test_loader

    first_batch = next(iter(data_loader))
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
    print(f"  [{method_name}] Image size: {img_size}, time steps: {num_time_steps}")

    del first_series, first_lv_segs, first_v_series
    torch.cuda.empty_cache()

    # Scaling network + checkpoint
    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps, img_size=img_size, num_heads=8
    )

    checkpoint_path = method_cfg['checkpoint']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[{method_name}] Checkpoint not found: {checkpoint_path}")

    print(f"  [{method_name}] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    scaling_network.load_state_dict(checkpoint['model_state_dict'])
    scaling_network.cuda().eval()

    ckpt_config = checkpoint.get('config', {})

    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=num_diffusion_steps, img_size=img_size
    )

    loss_fn = LTMALoss(
        lambda_sim=ckpt_config.get('lambda_sim', 1.0),
        lambda_reg=ckpt_config.get('lambda_reg', 0.0),
        lambda_scale=ckpt_config.get('lambda_scale', 0.001),
        lambda_low_structure=ckpt_config.get('lambda_low_structure', 0.0),
        use_ncc=False,
    )

    trainer_kwargs = dict(
        scaling_network=scaling_network,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        lr=1e-4,
        device='cuda',
        img_size=img_size,
    )
    if method_cfg['use_svf_module']:
        trainer_kwargs['svf_module'] = model.pair_model.MSvf

    trainer = ScalingFactorTrainer(**trainer_kwargs)

    # Run over batches
    all_correlations = []  # list of dicts: {frame_idx: corr}

    total_batches = len(data_loader)
    for batch_idx, test_batch in enumerate(data_loader):
        print(f"  [{method_name}] Batch {batch_idx + 1}/{total_batches}")

        series = test_batch['series'].cuda()
        lv_segs = test_batch.get('lv_segs')
        if lv_segs is not None:
            lv_segs = lv_segs.cuda()
        else:
            lv_segs = torch.zeros_like(series[:, 0:1]).cuda()

        batch_size = series.shape[0]

        with torch.no_grad():
            Sdef_series, v_series, _, _, _ = \
                model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)

        source_img = series[:, 0:1]
        target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]

        zero_velocity = torch.zeros_like(v_series[0])
        v_series_bb = [zero_velocity] + v_series

        _, scaling_factors_bb = trainer.validate(source_img, target_imgs, v_series_bb)

        num_time_steps_bb = len(v_series_bb)
        sampled_velocities = {i: [] for i in range(num_time_steps_bb)}

        for run in tqdm(range(num_runs), desc=f"    UQ runs", leave=False):
            sampled = brownian_bridge.sample_bridge_transition(
                v_series_bb, scaling_factors_bb, device='cuda'
            )
            for frame_idx in range(num_time_steps_bb):
                sampled_velocities[frame_idx].append(sampled[frame_idx].clone())
            del sampled
            if (run + 1) % 10 == 0:
                torch.cuda.empty_cache()

        for subj_idx in range(batch_size):
            corrs = _compute_correlations(
                sampled_velocities, Sdef_series, series, subj_idx
            )
            all_correlations.append(corrs)

        del series, lv_segs, Sdef_series, v_series, v_series_bb
        del source_img, target_imgs, scaling_factors_bb, sampled_velocities
        torch.cuda.empty_cache()
        gc.collect()

    # Collect all frame indices seen
    all_frame_indices = set()
    for c in all_correlations:
        all_frame_indices.update(c.keys())
    all_frame_indices = sorted(all_frame_indices)

    # Clean up
    del model, scaling_network, brownian_bridge, trainer, loss_fn
    torch.cuda.empty_cache()
    gc.collect()

    # Restore original sys.path
    sys.path[:] = original_sys_path

    return all_correlations, all_frame_indices


# ==========================================================================
# Plotting
# ==========================================================================

def _plot_correlation_barplots(method_correlations, selected_frames, save_path):
    """
    Save .npz data, then produce the final plot with:
      - Left y-axis:  Pearson r / Spearman rho bars (absolute values, negative tick labels)
      - Right y-axis 1: Mean AURC line
      - Right y-axis 2: Mean mError line

    method_correlations: dict  method_name -> list of dicts
        {frame_idx: {'pearson': r, 'spearman': rho, 'naurc': float, 'mean_error': float}}
    selected_frames: list of frame indices to plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(method_correlations.keys())
    all_metric_keys = ['pearson', 'spearman', 'naurc', 'mean_error']

    # Compute mean and std across subjects for every metric
    stats = {}  # stats[method][metric] = {'mean': [...], 'std': [...]}
    for m in methods:
        stats[m] = {}
        for mk in all_metric_keys:
            means, stds = [], []
            for fidx in selected_frames:
                vals = [c[fidx][mk] for c in method_correlations[m]
                        if fidx in c and mk in c[fidx]]
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals) if vals else 0.0)
            stats[m][mk] = {'mean': np.array(means), 'std': np.array(stds)}

    # ── Save .npz (all frames, not just selected) ──
    npz_path = save_path.replace('.png', '.npz')
    # Gather all available frame indices across all methods/subjects
    all_frames_set = set()
    for m in methods:
        for subj_dict in method_correlations[m]:
            all_frames_set.update(subj_dict.keys())
    all_frames_sorted = sorted(all_frames_set)

    npz_data = {
        'methods': np.array(methods),
        'all_frames': np.array(all_frames_sorted),
        'selected_frames': np.array(selected_frames),
    }
    for m in methods:
        for mk in all_metric_keys:
            # Save per-subject data for ALL frames
            for fidx in all_frames_sorted:
                vals = [c[fidx][mk] for c in method_correlations[m]
                        if fidx in c and mk in c[fidx]]
                npz_data[f'{m}_{mk}_frame_{fidx}_per_subject'] = np.array(vals)
            # Save mean/std for selected frames (for backward compat)
            npz_data[f'{m}_{mk}_mean'] = stats[m][mk]['mean']
            npz_data[f'{m}_{mk}_std'] = stats[m][mk]['std']
    np.savez(npz_path, **npz_data)
    print(f"Saved correlation data to: {npz_path}")
    print(f"  All frames saved: {all_frames_sorted}")
    print(f"  Selected frames for plot: {selected_frames}")

    # ── Style ──
    matplotlib.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 26,
        'axes.labelsize': 23,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 16,
    })

    # ── Plot ──
    frame_labels = [str(f) for f in selected_frames]
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(14 * n_methods, 12))
    if n_methods == 1:
        axes = [axes]

    bar_width = 0.35
    x = np.arange(len(frame_labels)) * 1.4  # add spacing between frame groups

    for idx, method in enumerate(methods):
        ax = axes[idx]

        pearson_abs = np.abs(stats[method]['pearson']['mean'])
        spearman_abs = np.abs(stats[method]['spearman']['mean'])
        naurc_means = stats[method]['naurc']['mean']
        merror_means = stats[method]['mean_error']['mean']

        # ── Left y-axis: correlation bars (upward bars, negative tick labels) ──
        bars1 = ax.bar(x - bar_width / 2, pearson_abs, bar_width,
                       label='Pearson r', color='#4169E1', edgecolor='white', linewidth=0.3)
        bars2 = ax.bar(x + bar_width / 2, spearman_abs, bar_width,
                       label='Spearman ρ', color='#FFA500', edgecolor='white', linewidth=0.3)

        ax.set_ylim(0, 0.45)
        yticks = np.linspace(0, 0.45, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
        ax.tick_params(axis='y', labelcolor='#4169E1')

        # Vertical separators between frame groups
        for i in range(len(x) - 1):
            mid = (x[i] + x[i + 1]) / 2
            ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)

        ax.set_xlabel('Frame Index', fontsize=23)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels)
        ax.set_title(f'({chr(97 + idx)}) {method}', fontweight='bold', fontsize=32)

        # ── Right y-axis 1: Mean AURC (scaled x10 so axis reads x10^-1) ──
        ax2 = ax.twinx()
        ax2.plot(x, naurc_means * 10, 's-', color='#006400', linewidth=2,
                 markersize=6, label='Mean AURC', markerfacecolor='#006400')
        ax2.tick_params(axis='y', labelcolor='#006400')

        # ── Right y-axis 2: Mean mError ──
        ax3 = ax.twinx()
        ax3.spines['right'].set_visible(False)
        ax3.plot(x, merror_means, 's--', color='#CC0000', linewidth=2,
                 markersize=6, label='Mean mError', markerfacecolor='#CC0000')
        ax3.set_yticks([])
        ax3.set_yticklabels([])

        # Only show y-axis labels, legend, and annotations on the first subplot
        if idx == 0:
            ax.set_ylabel('Correlation\n(Pearson/Spearman)', color='#4169E1', fontsize=28)
            ax2.set_ylabel('Mean AURC', color='#006400', fontsize=28)
            ax2.annotate(r'$\times 10^{-1}$', xy=(1.02, 1.02), xycoords='axes fraction',
                         fontsize=18, color='#006400', ha='left', va='bottom')
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles3, labels3 = ax3.get_legend_handles_labels()
            ax.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3,
                      loc='upper left', fontsize=23, framealpha=0.9, ncol=2)
        else:
            ax.set_ylabel('')
            ax2.set_ylabel('')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved correlation bar plot to: {save_path}")
    print(f"Saved correlation bar plot to: {pdf_path}")


# ==========================================================================
# Plot-only from saved .npz
# ==========================================================================

def _plot_from_npz(npz_path, save_path):
    """Re-plot from a previously saved .npz file (no model inference needed)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    selected_frames = list(data['selected_frames'])
    frame_labels = [str(f) for f in selected_frames]

    matplotlib.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 26,
        'axes.labelsize': 23,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 16,
    })

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(14 * n_methods, 12))
    if n_methods == 1:
        axes = [axes]

    bar_width = 0.60
    x = np.arange(len(frame_labels)) * 1.7  # add spacing between frame groups

    for idx, method in enumerate(methods):
        ax = axes[idx]

        pearson_abs = np.abs(data[f'{method}_pearson_mean'])
        spearman_abs = np.abs(data[f'{method}_spearman_mean'])
        naurc_means = data[f'{method}_naurc_mean']
        merror_means = data[f'{method}_mean_error_mean']

        bars1 = ax.bar(x - bar_width / 2, pearson_abs, bar_width,
                       label='Pearson r', color='#4169E1', edgecolor='white', linewidth=0.3)
        bars2 = ax.bar(x + bar_width / 2, spearman_abs, bar_width,
                       label='Spearman ρ', color='#FFA500', edgecolor='white', linewidth=0.3)

        ax.set_ylabel('Correlation\n(Pearson/Spearman)', color='#4169E1', fontsize=28)
        ax.set_ylim(0, 0.45)
        yticks = np.linspace(0, 0.45, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
        ax.tick_params(axis='y', labelcolor='#4169E1')

        # Vertical separators between frame groups
        for i in range(len(x) - 1):
            mid = (x[i] + x[i + 1]) / 2
            ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)

        ax.set_xlabel('Frame Index', fontsize=23)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels)
        ax.set_title(f'({chr(97 + idx)}) {method}', fontweight='bold', fontsize=32)

        ax2 = ax.twinx()
        ax2.plot(x, naurc_means * 10, 's-', color='#006400', linewidth=2,
                 markersize=6, label='Mean AURC', markerfacecolor='#006400')
        ax2.set_ylabel('Mean AURC\n/ Mean mError (Pixel)', color='#006400', fontsize=28)
        ax2.tick_params(axis='y', labelcolor='#006400')
        ax2.annotate(r'$\times 10^{-1}$', xy=(1.02, 1.02), xycoords='axes fraction',
                     fontsize=18, color='#006400', ha='left', va='bottom')

        ax3 = ax.twinx()
        ax3.spines['right'].set_visible(False)
        ax3.plot(x, merror_means, 's--', color='#CC0000', linewidth=2,
                 markersize=6, label='Mean mError', markerfacecolor='#CC0000')
        ax3.set_yticks([])
        ax3.set_yticklabels([])

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3,
                  loc='upper left', fontsize=27, framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved correlation bar plot to: {save_path}")
    print(f"Saved correlation bar plot to: {pdf_path}")


def _get_metric_for_frames(data, method, metric, selected_frames):
    """Compute mean and std of a metric for given frames from per-subject data."""
    means, stds = [], []
    for f in selected_frames:
        key = f'{method}_{metric}_frame_{f}_per_subject'
        vals = data[key]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(means), np.array(stds)


def _plot_combined_from_npz(npz_path, save_path, selected_frames=None):
    """Single combined bar plot with all methods in one figure.

    X-axis: frame groups. Within each frame, grouped bars per method.
    Pearson = solid bar, Spearman = lighter shade bar (same color per method).
    Mean AURC and Mean mError overlaid as lines.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    if selected_frames is None:
        selected_frames = list(data['selected_frames'])
    n_methods = len(methods)
    n_frames = len(selected_frames)

    matplotlib.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 26,
        'axes.labelsize': 23,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 18,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })

    fig, ax = plt.subplots(figsize=(22, 10))

    method_colors = {
        'TLRN': '#2274A5', 'VM': '#E36414', 'LTMA': '#606C38', 'TM': '#9B2226'
    }
    method_markers = {'TLRN': 's', 'VM': 'o', 'LTMA': '^', 'TM': 'D'}

    bar_width = 1.0
    group_width = n_methods * 2 * bar_width
    frame_positions = np.arange(n_frames) * (group_width + 3.0)

    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        # Blend with white to create a solid light color (PDF-safe, no alpha)
        r, g, b, _ = to_rgba(color)
        blend = 0.75  # how much white to mix in
        light_color = (r + blend * (1 - r), g + blend * (1 - g), b + blend * (1 - b), 1.0)
        pearson_means, _ = _get_metric_for_frames(data, method, 'pearson', selected_frames)
        spearman_means, _ = _get_metric_for_frames(data, method, 'spearman', selected_frames)
        pearson_abs = np.abs(pearson_means)
        spearman_abs = np.abs(spearman_means)

        offset_pearson = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
        offset_spearman = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2

        ax.bar(frame_positions + offset_pearson, pearson_abs, bar_width,
               color=color, edgecolor='black', linewidth=4,
               label=f'{method} (Pearson)')
        ax.bar(frame_positions + offset_spearman, spearman_abs, bar_width,
               color=light_color, edgecolor=color, linewidth=4,
               hatch='///', label=f'{method} (Spearman)')

    # Y-axis (correlation)
    ax.set_ylim(0, 0.38)
    yticks = np.linspace(0, 0.35, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
    ax.set_ylabel('Correlation (Pearson / Spearman)', color='black', fontsize=28)
    ax.tick_params(axis='y', labelcolor='black')
    # Vertical separators between frame groups
    for i in range(n_frames - 1):
        mid = (frame_positions[i] + frame_positions[i + 1]) / 2
        ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)

    # X-axis
    ax.set_xticks(frame_positions)
    ax.set_xticklabels([str(f) for f in selected_frames])
    ax.set_xlabel('Frame Index', fontsize=23)

    # Right y-axis: Mean AURC per method
    ax2 = ax.twinx()
    all_naurc = []
    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        naurc_means, _ = _get_metric_for_frames(data, method, 'naurc', selected_frames)
        all_naurc.extend(naurc_means * 10)
        ax2.plot(frame_positions, naurc_means * 10, marker=marker, linestyle='--',
                 color=color, linewidth=4, markersize=20, alpha=0.85,
                 markeredgecolor='black', markeredgewidth=1.2)
    naurc_min = min(all_naurc)
    naurc_max = max(all_naurc)
    naurc_margin = (naurc_max - naurc_min) * 0.15
    ax2.set_ylim(naurc_min - naurc_margin, naurc_max + naurc_margin)
    ax2.set_ylabel('Mean AURC', color='black', fontsize=28)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.annotate(r'$\times 10^{-1}$', xy=(1.02, 1.02), xycoords='axes fraction',
                 fontsize=18, color='black', ha='left', va='bottom')

    # Mean mError per method (hidden axis)
    ax3 = ax.twinx()
    ax3.spines['right'].set_visible(False)
    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        merror_means, _ = _get_metric_for_frames(data, method, 'mean_error', selected_frames)
        ax3.plot(frame_positions, merror_means, marker=marker, linestyle='--',
                 color=color, linewidth=2.5, markersize=9, alpha=0.5,
                 markeredgecolor='black', markeredgewidth=0.6)
    ax3.set_yticks([])
    ax3.set_yticklabels([])

    ax.legend(loc='upper center', fontsize=22, framealpha=0.9, ncol=4,
              edgecolor='gray')
    ax.grid(False)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#F0F0F0')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined bar plot to: {save_path}")
    print(f"Saved combined bar plot to: {pdf_path}")


def _plot_combined_no_merror_from_npz(npz_path, save_path, selected_frames=None):
    """Same as _plot_combined_from_npz but without Mean mError lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    if selected_frames is None:
        selected_frames = list(data['selected_frames'])
    n_methods = len(methods)
    n_frames = len(selected_frames)

    matplotlib.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 26,
        'axes.labelsize': 23,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 18,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })

    fig, ax = plt.subplots(figsize=(22, 10))

    method_colors = {
        'TLRN': '#2274A5', 'VM': '#E36414', 'LTMA': '#606C38', 'TM': '#9B2226'
    }
    method_markers = {'TLRN': 's', 'VM': 'o', 'LTMA': '^', 'TM': 'D'}

    bar_width = 1.0
    group_width = n_methods * 2 * bar_width
    frame_positions = np.arange(n_frames) * (group_width + 3.0)

    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        # Blend with white to create a solid light color (PDF-safe, no alpha)
        r, g, b, _ = to_rgba(color)
        blend = 0.75  # how much white to mix in
        light_color = (r + blend * (1 - r), g + blend * (1 - g), b + blend * (1 - b), 1.0)
        pearson_means, _ = _get_metric_for_frames(data, method, 'pearson', selected_frames)
        spearman_means, _ = _get_metric_for_frames(data, method, 'spearman', selected_frames)
        pearson_abs = np.abs(pearson_means)
        spearman_abs = np.abs(spearman_means)

        offset_pearson = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
        offset_spearman = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2

        ax.bar(frame_positions + offset_pearson, pearson_abs, bar_width,
               color=color, edgecolor='black', linewidth=4,
               label=f'{method} (Pearson)')
        ax.bar(frame_positions + offset_spearman, spearman_abs, bar_width,
               color=light_color, edgecolor=color, linewidth=4,
               hatch='///', label=f'{method} (Spearman)')

    # Y-axis (correlation)
    ax.set_ylim(0, 0.38)
    yticks = np.linspace(0, 0.35, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
    ax.set_ylabel('Correlation (Pearson / Spearman)', color='black', fontsize=28)
    ax.tick_params(axis='y', labelcolor='black')
    # Vertical separators between frame groups
    for i in range(n_frames - 1):
        mid = (frame_positions[i] + frame_positions[i + 1]) / 2
        ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)

    # X-axis
    ax.set_xticks(frame_positions)
    ax.set_xticklabels([str(f) for f in selected_frames])
    ax.set_xlabel('Frame Index', fontsize=23)

    # Right y-axis: Mean AURC per method
    ax2 = ax.twinx()
    all_naurc = []
    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        naurc_means, _ = _get_metric_for_frames(data, method, 'naurc', selected_frames)
        all_naurc.extend(naurc_means * 10)
        ax2.plot(frame_positions, naurc_means * 10, marker=marker, linestyle='--',
                 color=color, linewidth=4, markersize=20, alpha=0.85,
                 markeredgecolor='black', markeredgewidth=1.2)
    naurc_min = min(all_naurc)
    naurc_max = max(all_naurc)
    naurc_margin = (naurc_max - naurc_min) * 0.15
    ax2.set_ylim(naurc_min - naurc_margin, naurc_max + naurc_margin)
    ax2.set_ylabel('Mean AURC', color='black', fontsize=28)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.annotate(r'$\times 10^{-1}$', xy=(1.02, 1.02), xycoords='axes fraction',
                 fontsize=18, color='black', ha='left', va='bottom')

    ax.legend(loc='upper center', fontsize=22, framealpha=0.9, ncol=4,
              edgecolor='gray')
    ax.grid(False)

    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#F0F0F0')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined (no mError) bar plot to: {save_path}")
    print(f"Saved combined (no mError) bar plot to: {pdf_path}")
    print(f"Saved combined (no mError) bar plot to: {svg_path}")


def _plot_combined_no_merror_two_datasets(npz_path, save_path, selected_frames=None):
    """Two-row bar plot: top = Cardiac Cine MRI, bottom = Longitudinal Brain Disease Progression.
    Currently uses the same data for both (brain is placeholder)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    if selected_frames is None:
        selected_frames = list(data['selected_frames'])
    n_methods = len(methods)
    n_frames = len(selected_frames)

    matplotlib.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 26,
        'axes.labelsize': 23,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 18,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })

    method_colors = {
        'TLRN': '#2274A5', 'VM': '#E36414', 'LTMA': '#606C38', 'TM': '#9B2226'
    }
    method_markers = {'TLRN': 's', 'VM': 'o', 'LTMA': '^', 'TM': 'D'}

    dataset_titles = [
        'Cardiac Cine MRI',
        'Longitudinal Brain Disease Progression',
    ]
    # Use same data for both (brain is placeholder)
    datasets = [data, data]

    fig, axes = plt.subplots(2, 1, figsize=(22, 18), sharex=False,
                             gridspec_kw={'hspace': 0.08, 'height_ratios': [1, 1]})

    for row, (ax, ds_data, title) in enumerate(zip(axes, datasets, dataset_titles)):
        bar_width = 1.0
        group_width = n_methods * 2 * bar_width
        frame_positions = np.arange(n_frames) * (group_width + 3.0)

        for m_idx, method in enumerate(methods):
            color = method_colors.get(method, '#999999')
            r, g, b, _ = to_rgba(color)
            blend = 0.75
            light_color = (r + blend * (1 - r), g + blend * (1 - g), b + blend * (1 - b), 1.0)
            pearson_means, _ = _get_metric_for_frames(ds_data, method, 'pearson', selected_frames)
            spearman_means, _ = _get_metric_for_frames(ds_data, method, 'spearman', selected_frames)
            pearson_abs = np.abs(pearson_means)
            spearman_abs = np.abs(spearman_means)

            offset_pearson = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
            offset_spearman = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2

            # Only add legend labels on the second row (where legend is placed)
            label_p = f'{method} (Pearson)' if row == 1 else None
            label_s = f'{method} (Spearman)' if row == 1 else None

            ax.bar(frame_positions + offset_pearson, pearson_abs, bar_width,
                   color=color, edgecolor='black', linewidth=4,
                   label=label_p)
            ax.bar(frame_positions + offset_spearman, spearman_abs, bar_width,
                   color=light_color, edgecolor=color, linewidth=4,
                   hatch='///', label=label_s)

        # Y-axis (correlation)
        ax.set_ylim(0, 0.38)
        yticks = np.linspace(0, 0.35, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
        ax.set_ylabel('Correlation (Pearson / Spearman)', color='black', fontsize=22)
        ax.tick_params(axis='y', labelcolor='black')

        # Vertical separators between frame groups
        for i in range(n_frames - 1):
            mid = (frame_positions[i] + frame_positions[i + 1]) / 2
            ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)

        # X-axis: only show ticks and label on bottom plot
        ax.set_xticks(frame_positions)
        if row == 0:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xticklabels([str(f) for f in selected_frames])
            ax.set_xlabel('Frame Index', fontsize=23)

        # Right y-axis: Mean AURC per method
        ax2 = ax.twinx()
        all_naurc = []
        for m_idx, method in enumerate(methods):
            color = method_colors.get(method, '#999999')
            marker = method_markers.get(method, 's')
            naurc_means, _ = _get_metric_for_frames(ds_data, method, 'naurc', selected_frames)
            all_naurc.extend(naurc_means * 10)
            ax2.plot(frame_positions, naurc_means * 10, marker=marker, linestyle='--',
                     color=color, linewidth=4, markersize=20, alpha=0.85,
                     markeredgecolor='black', markeredgewidth=1.2)
        naurc_min = min(all_naurc)
        naurc_max = max(all_naurc)
        naurc_margin = (naurc_max - naurc_min) * 0.15
        ax2.set_ylim(naurc_min - naurc_margin, naurc_max + naurc_margin)
        ax2.set_ylabel('Mean AURC', color='black', fontsize=22)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.annotate(r'$\times 10^{-1}$', xy=(1.03, 1.02), xycoords='axes fraction',
                     fontsize=18, color='black', ha='right', va='bottom')

        # Title for each subplot (same style as y-axis labels, no bold)
        ax.set_title(title, fontsize=28, fontweight='normal', pad=8)

        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.set_facecolor('#F0F0F0')

    # Legend in the center of the second (bottom) plot
    axes[1].legend(loc='upper center', fontsize=22, framealpha=0.9, ncol=4,
                   edgecolor='gray')

    fig.patch.set_facecolor('white')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved two-dataset bar plot to: {save_path}")
    print(f"Saved two-dataset bar plot to: {pdf_path}")
    print(f"Saved two-dataset bar plot to: {svg_path}")


def _plot_combined_v2_colored_labels(npz_path, save_path):
    """Version 2: Color-coded y-axis labels (blue left, dark green right)
    to indicate which data maps to which axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    selected_frames = list(data['selected_frames'])
    n_methods = len(methods)
    n_frames = len(selected_frames)

    matplotlib.rcParams.update({
        'font.size': 23, 'axes.titlesize': 26, 'axes.labelsize': 23,
        'xtick.labelsize': 23, 'ytick.labelsize': 23, 'legend.fontsize': 18,
        'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    })

    fig, ax = plt.subplots(figsize=(22, 10))
    method_colors = {'TLRN': '#fc8d59', 'VM': '#b0b0b0', 'LTMA': '#6baed6', 'TM': '#78c679'}
    method_markers = {'TLRN': 's', 'VM': 'o', 'LTMA': '^', 'TM': 'D'}
    bar_width = 1.0
    group_width = n_methods * 2 * bar_width
    frame_positions = np.arange(n_frames) * (group_width + 3.0)

    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        # Blend with white to create a solid light color (PDF-safe, no alpha)
        r, g, b, _ = to_rgba(color)
        blend = 0.75  # how much white to mix in
        light_color = (r + blend * (1 - r), g + blend * (1 - g), b + blend * (1 - b), 1.0)
        pearson_abs = np.abs(data[f'{method}_pearson_mean'])
        spearman_abs = np.abs(data[f'{method}_spearman_mean'])
        off_p = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
        off_s = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2
        ax.bar(frame_positions + off_p, pearson_abs, bar_width,
               color=color, edgecolor='black', linewidth=0.5,
               label=f'{method} (Pearson)')
        ax.bar(frame_positions + off_s, spearman_abs, bar_width,
               color=light_color, edgecolor='black', linewidth=0.5,
               label=f'{method} (Spearman)')

    ax.set_ylim(0, 0.38)
    yticks = np.linspace(0, 0.35, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
    # Colored label to indicate "bars use this axis"
    ax.set_ylabel('Correlation (Bars)', color='#1a3a6b', fontsize=28, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#1a3a6b')
    # Vertical separators between frame groups
    for i in range(n_frames - 1):
        mid = (frame_positions[i] + frame_positions[i + 1]) / 2
        ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)
    ax.set_xticks(frame_positions)
    ax.set_xticklabels([str(f) for f in selected_frames])
    ax.set_xlabel('Frame Index', fontsize=23)

    ax2 = ax.twinx()
    all_naurc = []
    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        naurc_means = data[f'{method}_naurc_mean']
        all_naurc.extend(naurc_means * 10)
        ax2.plot(frame_positions, naurc_means * 10, marker=marker, linestyle='--',
                 color=color, linewidth=4, markersize=20, alpha=0.85,
                 markeredgecolor='black', markeredgewidth=1.2)
    naurc_min, naurc_max = min(all_naurc), max(all_naurc)
    naurc_margin = (naurc_max - naurc_min) * 0.15
    ax2.set_ylim(naurc_min - naurc_margin, naurc_max + naurc_margin)
    # Colored label to indicate "lines use this axis"
    ax2.set_ylabel('Mean AURC (Lines)', color='#1a5c1a', fontsize=28, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1a5c1a')
    ax2.annotate(r'$\times 10^{-1}$', xy=(1.02, 1.02), xycoords='axes fraction',
                 fontsize=18, color='#1a5c1a', ha='left', va='bottom')

    ax.legend(loc='upper center', fontsize=22, framealpha=0.9, ncol=4, edgecolor='gray')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#F0F0F0')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved V2 (colored labels) to: {save_path}")


def _plot_combined_v3_arrow_annotations(npz_path, save_path):
    """Version 3: Black axis labels with small arrow annotations pointing
    from 'Bars →' and '← Lines' to the respective axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    selected_frames = list(data['selected_frames'])
    n_methods = len(methods)
    n_frames = len(selected_frames)

    matplotlib.rcParams.update({
        'font.size': 23, 'axes.titlesize': 26, 'axes.labelsize': 23,
        'xtick.labelsize': 23, 'ytick.labelsize': 23, 'legend.fontsize': 18,
        'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    })

    fig, ax = plt.subplots(figsize=(22, 10))
    method_colors = {'TLRN': '#fc8d59', 'VM': '#b0b0b0', 'LTMA': '#6baed6', 'TM': '#78c679'}
    method_markers = {'TLRN': 's', 'VM': 'o', 'LTMA': '^', 'TM': 'D'}
    bar_width = 1.0
    group_width = n_methods * 2 * bar_width
    frame_positions = np.arange(n_frames) * (group_width + 3.0)

    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        # Blend with white to create a solid light color (PDF-safe, no alpha)
        r, g, b, _ = to_rgba(color)
        blend = 0.75  # how much white to mix in
        light_color = (r + blend * (1 - r), g + blend * (1 - g), b + blend * (1 - b), 1.0)
        pearson_abs = np.abs(data[f'{method}_pearson_mean'])
        spearman_abs = np.abs(data[f'{method}_spearman_mean'])
        off_p = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
        off_s = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2
        ax.bar(frame_positions + off_p, pearson_abs, bar_width,
               color=color, edgecolor='black', linewidth=0.5,
               label=f'{method} (Pearson)')
        ax.bar(frame_positions + off_s, spearman_abs, bar_width,
               color=light_color, edgecolor='black', linewidth=0.5,
               label=f'{method} (Spearman)')

    ax.set_ylim(0, 0.38)
    yticks = np.linspace(0, 0.35, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
    ax.set_ylabel('Correlation (Pearson / Spearman)', color='black', fontsize=28)
    ax.tick_params(axis='y', labelcolor='black')
    # Vertical separators between frame groups
    for i in range(n_frames - 1):
        mid = (frame_positions[i] + frame_positions[i + 1]) / 2
        ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)
    ax.set_xticks(frame_positions)
    ax.set_xticklabels([str(f) for f in selected_frames])
    ax.set_xlabel('Frame Index', fontsize=23)

    # Arrow annotation on left axis
    ax.annotate('← Bars', xy=(0.0, 0.5), xycoords='axes fraction',
                xytext=(-0.12, 0.5), textcoords='axes fraction',
                fontsize=16, color='#555555', ha='center', va='center',
                rotation=90, fontstyle='italic')

    ax2 = ax.twinx()
    all_naurc = []
    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        naurc_means = data[f'{method}_naurc_mean']
        all_naurc.extend(naurc_means * 10)
        ax2.plot(frame_positions, naurc_means * 10, marker=marker, linestyle='--',
                 color=color, linewidth=4, markersize=20, alpha=0.85,
                 markeredgecolor='black', markeredgewidth=1.2)
    naurc_min, naurc_max = min(all_naurc), max(all_naurc)
    naurc_margin = (naurc_max - naurc_min) * 0.15
    ax2.set_ylim(naurc_min - naurc_margin, naurc_max + naurc_margin)
    ax2.set_ylabel('Mean AURC', color='black', fontsize=28)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.annotate(r'$\times 10^{-1}$', xy=(1.02, 1.02), xycoords='axes fraction',
                 fontsize=18, color='black', ha='left', va='bottom')

    # Arrow annotation on right axis
    ax2.annotate('Lines →', xy=(1.0, 0.5), xycoords='axes fraction',
                 xytext=(1.12, 0.5), textcoords='axes fraction',
                 fontsize=16, color='#555555', ha='center', va='center',
                 rotation=90, fontstyle='italic')

    ax.legend(loc='upper center', fontsize=22, framealpha=0.9, ncol=4, edgecolor='gray')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#F0F0F0')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved V3 (arrow annotations) to: {save_path}")


def _plot_combined_v4_legend_hint(npz_path, save_path):
    """Version 4: Add 'Bars = Correlation (left axis)' and
    'Lines = Mean AURC (right axis)' as extra entries in the legend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    selected_frames = list(data['selected_frames'])
    n_methods = len(methods)
    n_frames = len(selected_frames)

    matplotlib.rcParams.update({
        'font.size': 23, 'axes.titlesize': 26, 'axes.labelsize': 23,
        'xtick.labelsize': 23, 'ytick.labelsize': 23, 'legend.fontsize': 18,
        'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    })

    fig, ax = plt.subplots(figsize=(22, 10))
    method_colors = {'TLRN': '#fc8d59', 'VM': '#b0b0b0', 'LTMA': '#6baed6', 'TM': '#78c679'}
    method_markers = {'TLRN': 's', 'VM': 'o', 'LTMA': '^', 'TM': 'D'}
    bar_width = 1.0
    group_width = n_methods * 2 * bar_width
    frame_positions = np.arange(n_frames) * (group_width + 3.0)

    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        # Blend with white to create a solid light color (PDF-safe, no alpha)
        r, g, b, _ = to_rgba(color)
        blend = 0.75  # how much white to mix in
        light_color = (r + blend * (1 - r), g + blend * (1 - g), b + blend * (1 - b), 1.0)
        pearson_abs = np.abs(data[f'{method}_pearson_mean'])
        spearman_abs = np.abs(data[f'{method}_spearman_mean'])
        off_p = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
        off_s = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2
        ax.bar(frame_positions + off_p, pearson_abs, bar_width,
               color=color, edgecolor='black', linewidth=0.5,
               label=f'{method} (Pearson)')
        ax.bar(frame_positions + off_s, spearman_abs, bar_width,
               color=light_color, edgecolor='black', linewidth=0.5,
               label=f'{method} (Spearman)')

    ax.set_ylim(0, 0.38)
    yticks = np.linspace(0, 0.35, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'−.{int(v*100):02d}' if v > 0 else '.00' for v in yticks])
    ax.set_ylabel('Correlation (Pearson / Spearman)', color='black', fontsize=28)
    ax.tick_params(axis='y', labelcolor='black')
    # Vertical separators between frame groups
    for i in range(n_frames - 1):
        mid = (frame_positions[i] + frame_positions[i + 1]) / 2
        ax.axvline(mid, color='#888888', linewidth=0.5, zorder=1)
    ax.set_xticks(frame_positions)
    ax.set_xticklabels([str(f) for f in selected_frames])
    ax.set_xlabel('Frame Index', fontsize=23)

    ax2 = ax.twinx()
    all_naurc = []
    for m_idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        naurc_means = data[f'{method}_naurc_mean']
        all_naurc.extend(naurc_means * 10)
        ax2.plot(frame_positions, naurc_means * 10, marker=marker, linestyle='--',
                 color=color, linewidth=4, markersize=20, alpha=0.85,
                 markeredgecolor='black', markeredgewidth=1.2)
    naurc_min, naurc_max = min(all_naurc), max(all_naurc)
    naurc_margin = (naurc_max - naurc_min) * 0.15
    ax2.set_ylim(naurc_min - naurc_margin, naurc_max + naurc_margin)
    ax2.set_ylabel('Mean AURC', color='black', fontsize=28)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.annotate(r'$\times 10^{-1}$', xy=(1.02, 1.02), xycoords='axes fraction',
                 fontsize=18, color='black', ha='left', va='bottom')

    # Build legend with extra "Bars = left axis" and "Lines = right axis" hints
    handles, labels = ax.get_legend_handles_labels()
    # Add separator-style legend entries
    bar_hint = Patch(facecolor='gray', edgecolor='black', linewidth=0.5,
                     label='Bars → left axis')
    line_hint = Line2D([0], [0], color='gray', linewidth=3, marker='s',
                       markersize=8, markeredgecolor='black',
                       label='Lines → right axis')
    handles.extend([bar_hint, line_hint])
    labels.extend(['Bars → left axis', 'Lines → right axis'])
    ax.legend(handles, labels, loc='upper center', fontsize=20,
              framealpha=0.9, ncol=5, edgecolor='gray')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor('#F0F0F0')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved V4 (legend hint) to: {save_path}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correlation bar-plot: uncertainty vs registration error across methods"
    )
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of UQ sampling runs per method (default: 100)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_train", action="store_true",
                        help="Use training set instead of test set")
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run even if .npz already exists")
    parser.add_argument("--frames", type=int, nargs='+', default=[2, 3, 5, 7, 9, 10],
                        help="Frame indices to plot (default: 2 3 5 7 9 10)")
    args = parser.parse_args()

    output_dir = '2026Experiments/CINE/outputs/correlation_barplots'
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, 'uncertainty_error_correlation.npz')
    png_path = os.path.join(output_dir, 'uncertainty_error_correlation.png')

    # ------------------------------------------------------------------
    # If .npz exists, just re-plot and exit
    # ------------------------------------------------------------------
    selected_frames = sorted(args.frames)
    print(f"Selected frames for plotting: {selected_frames}")

    if os.path.exists(npz_path) and not args.rerun:
        print(f"Found existing data: {npz_path}")
        print("Plotting from saved data (use --rerun to force re-computation)")
        # Verify requested frames exist in npz
        data = np.load(npz_path, allow_pickle=True)
        all_frames_in_npz = list(data['all_frames']) if 'all_frames' in data else list(data['selected_frames'])
        missing = [f for f in selected_frames if f not in all_frames_in_npz]
        if missing:
            print(f"WARNING: Frames {missing} not in .npz (available: {all_frames_in_npz}). Use --rerun to recompute.")
            return
        _plot_from_npz(npz_path, png_path)
        combined_path = os.path.join(output_dir, 'uncertainty_error_correlation_combined.png')
        _plot_combined_from_npz(npz_path, combined_path, selected_frames=selected_frames)
        no_merror_path = os.path.join(output_dir, 'uncertainty_error_correlation_combined_no_merror.png')
        _plot_combined_no_merror_from_npz(npz_path, no_merror_path, selected_frames=selected_frames)
        two_ds_path = os.path.join(output_dir, 'uncertainty_error_correlation_two_datasets.png')
        _plot_combined_no_merror_two_datasets(npz_path, two_ds_path, selected_frames=selected_frames)
        # Variant plots for comparison
        v2_path = os.path.join(output_dir, 'combined_v2_colored_labels.png')
        _plot_combined_v2_colored_labels(npz_path, v2_path)
        v3_path = os.path.join(output_dir, 'combined_v3_arrow_annotations.png')
        _plot_combined_v3_arrow_annotations(npz_path, v3_path)
        v4_path = os.path.join(output_dir, 'combined_v4_legend_hint.png')
        _plot_combined_v4_legend_hint(npz_path, v4_path)
        print(f"\nDone! All outputs saved to: {output_dir}")
        return

    # ------------------------------------------------------------------
    # Run each method and collect correlations
    # ------------------------------------------------------------------
    NUM_DIFFUSION_STEPS = 14
    NUM_RUNS = args.num_runs

    method_correlations = {}
    all_available_frames = None

    for method_name, method_cfg in METHOD_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running method: {method_name}")
        print(f"{'='*60}")

        correlations, frame_indices = _run_method(
            method_name, method_cfg, NUM_RUNS, NUM_DIFFUSION_STEPS,
            use_train=args.use_train,
        )
        method_correlations[method_name] = correlations

        if all_available_frames is None:
            all_available_frames = frame_indices
        print(f"  [{method_name}] Done. {len(correlations)} subjects, frames: {frame_indices}")

    # ------------------------------------------------------------------
    # Use all available frames for .npz, selected_frames from --frames for plotting
    # ------------------------------------------------------------------
    print(f"\nAll available frames: {all_available_frames}")
    print(f"Selected frames for bar plot: {selected_frames}")

    # ------------------------------------------------------------------
    # Plot (also saves .npz)
    # ------------------------------------------------------------------
    _plot_correlation_barplots(
        method_correlations, selected_frames, png_path
    )

    combined_path = os.path.join(output_dir, 'uncertainty_error_correlation_combined.png')
    _plot_combined_from_npz(npz_path, combined_path, selected_frames=selected_frames)

    no_merror_path = os.path.join(output_dir, 'uncertainty_error_correlation_combined_no_merror.png')
    _plot_combined_no_merror_from_npz(npz_path, no_merror_path, selected_frames=selected_frames)

    two_ds_path = os.path.join(output_dir, 'uncertainty_error_correlation_two_datasets.png')
    _plot_combined_no_merror_two_datasets(npz_path, two_ds_path, selected_frames=selected_frames)

    print(f"\nDone! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
