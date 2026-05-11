"""
Dice / HD95 box-plot script -- computes per-frame Dice score and HD95 for
multiple methods (TLRN, VM, LTMA, TM), each with Baseline and +BridgeUQ
variants, and produces a combined boxplot with scatter.

Run with:
  python uncertainty_sde_combined_acdc/test_dice_hd95_boxplot.py [--num_runs N]
"""

import os
import sys
import argparse
import gc

import torch
import numpy as np
from tqdm import tqdm


# ==========================================================================
# Method configurations (same as test_correlation_barplot.py)
# ==========================================================================

METHOD_CONFIGS = {
    'TLRN': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2',
        'checkpoint': '2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03_plus_ACDC/visualization/uncertainty_sde_combined_acdc/checkpoints/checkpoint_epoch_300.pth',
        'file_paths_key': 'path',
        'use_svf_module': False,
    },
    'VM': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/voxel_and_R2R/voxelmorph',
        'checkpoint': '/scratch/swd9tc/Uncertanity_quantification/voxel_and_R2R/voxelmorph/visualization_voxelmorph/uncertainty_sde_combined_acdc/checkpoints/checkpoint_epoch_300.pth',
        'file_paths_key': 'paths',
        'use_svf_module': True,
    },
    'LTMA': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA',
        'checkpoint': '/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/2026Experiments/CINE/outputs/TLMA_TGrad/config_plus_ACDC/visualization/uncertainty_sde_combined_acdc/checkpoints/checkpoint_epoch_300.pth',
        'file_paths_key': 'path',
        'use_svf_module': False,
    },
    'TM': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/TM',
        'checkpoint': '2026Experiments/CINE/outputs/TransMorph/basic_plus_ACDC/visualization/uncertainty_sde_combined_acdc/checkpoints/checkpoint_epoch_300.pth',
        'file_paths_key': 'path',
        'use_svf_module': False,
    },
}


# ==========================================================================
# Per-subject per-frame Dice & HD95 computation
# ==========================================================================

def _compute_dice_hd95(v_series, Sdef_series, lv_segs, series,
                       sampled_velocities, subject_idx, inshape, device,
                       num_runs, svf_module=None):
    """Compute per-frame Dice and HD95 for baseline (reg network) and
    +BridgeUQ (mean velocity) for one subject.

    Args:
        svf_module: If provided (e.g. model.pair_model.MSvf for VM), use it
            to integrate velocities and warp segmentations. Otherwise fall
            back to standalone VecInt + SpatialTransformer.

    Returns dict: frame_idx -> {
        'dice_baseline': float, 'hd95_baseline': float,
        'dice_bridge': float,   'hd95_bridge': float,
    }
    """
    from utils.Int import (
        VecInt, SpatialTransformer,
        dice_coefficient_tensor, hausdorff_distance_95,
    )

    num_frames = len(v_series)

    if svf_module is None:
        vec_int = VecInt(inshape, TSteps=7).to(device)
        spatial_transformer = SpatialTransformer(inshape).to(device)

    def _integrate_and_warp(velocity, seg):
        """Integrate velocity -> displacement, then warp seg."""
        if svf_module is not None:
            displacement, _ = svf_module(velocity)
            warped, _ = svf_module.transformer(seg, displacement)
        else:
            disp_list = vec_int(velocity)
            displacement = disp_list[-1]
            warped, _ = spatial_transformer(seg, displacement)
        return warped

    frame_metrics = {}

    for frame_idx in range(num_frames):
        # Need segmentations
        if lv_segs is None or lv_segs.shape[1] <= frame_idx + 1:
            continue

        source_seg = lv_segs[subject_idx:subject_idx+1, 0:1]       # [1,1,H,W]
        target_seg = lv_segs[subject_idx:subject_idx+1, frame_idx+1:frame_idx+2]

        target_seg_bin = (target_seg > 0.5).float()

        # --- Baseline: registration network velocity ---
        v_reg = v_series[frame_idx][subject_idx:subject_idx+1]
        warped_seg = _integrate_and_warp(v_reg, source_seg)
        warped_seg_bin = (warped_seg > 0.5).float()

        dice_base = dice_coefficient_tensor(warped_seg_bin, target_seg_bin, return_mean=True)
        hd95_base = hausdorff_distance_95(warped_seg_bin, target_seg_bin, return_mean=True)

        # --- +BridgeUQ: mean sampled velocity ---
        bb_frame_idx = frame_idx + 1  # v_series_bb = [zero] + v_series
        if bb_frame_idx in sampled_velocities and len(sampled_velocities[bb_frame_idx]) > 0:
            velocities_at_frame = [
                v[subject_idx:subject_idx+1] for v in sampled_velocities[bb_frame_idx]
            ]
            mean_velocity = torch.stack(velocities_at_frame, dim=0).mean(dim=0)

            warped_seg_mean = _integrate_and_warp(mean_velocity, source_seg)
            warped_seg_mean_bin = (warped_seg_mean > 0.5).float()

            dice_bridge = dice_coefficient_tensor(warped_seg_mean_bin, target_seg_bin, return_mean=True)
            hd95_bridge = hausdorff_distance_95(warped_seg_mean_bin, target_seg_bin, return_mean=True)
        else:
            dice_bridge = dice_base
            hd95_bridge = hd95_base

        frame_metrics[frame_idx + 1] = {
            'dice_baseline': dice_base,
            'hd95_baseline': hd95_base,
            'dice_bridge': dice_bridge,
            'hd95_bridge': hd95_bridge,
        }

    return frame_metrics


# ==========================================================================
# Load and run one method
# ==========================================================================

def _run_method(method_name, method_cfg, num_runs, num_diffusion_steps, use_train=False):
    """
    Load a method, run inference + UQ, return per-subject per-frame
    Dice & HD95 metrics.

    Returns: (list_of_dicts, available_frame_indices)
    """
    method_sys_path = method_cfg['sys_path']

    original_sys_path = sys.path.copy()
    for other_cfg in METHOD_CONFIGS.values():
        other_path = other_cfg['sys_path']
        if other_path != method_sys_path:
            while other_path in sys.path:
                sys.path.remove(other_path)
    if method_sys_path not in sys.path:
        sys.path.insert(0, method_sys_path)

    _purge_prefixes = [
        'uncertainty_sde_combined', 'uncertainty_sde_combined_acdc',
        'uncertainty_inv_gamma_prior',
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
    inshape = (img_size, img_size)
    print(f"  [{method_name}] Image size: {img_size}, time steps: {num_time_steps}")

    del first_series, first_lv_segs, first_v_series
    torch.cuda.empty_cache()

    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps, img_size=img_size, num_heads=8
    )

    checkpoint_path = method_cfg['checkpoint']
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(method_sys_path, checkpoint_path)
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

    svf_module = model.pair_model.MSvf if method_cfg['use_svf_module'] else None

    trainer = ScalingFactorTrainer(
        scaling_network=scaling_network,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        lr=1e-4,
        device='cuda',
        img_size=img_size,
    )

    all_metrics = []  # list of dicts per subject

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
            metrics = _compute_dice_hd95(
                v_series, Sdef_series, lv_segs, series,
                sampled_velocities, subj_idx, inshape, 'cuda',
                num_runs, svf_module=svf_module,
            )
            all_metrics.append(metrics)

        del series, lv_segs, Sdef_series, v_series, v_series_bb
        del source_img, target_imgs, scaling_factors_bb, sampled_velocities
        torch.cuda.empty_cache()
        gc.collect()

    all_frame_indices = set()
    for m in all_metrics:
        all_frame_indices.update(m.keys())
    all_frame_indices = sorted(all_frame_indices)

    del model, scaling_network, brownian_bridge, trainer, loss_fn
    torch.cuda.empty_cache()
    gc.collect()

    sys.path[:] = original_sys_path

    return all_metrics, all_frame_indices


# ==========================================================================
# Plotting
# ==========================================================================

def _save_avg_dice_txt(method_to_frame_vals, methods, selected_frames, save_path):
    """Write average Dice (across all subjects, frames, structures) per method & variant.

    method_to_frame_vals: dict
        method_name -> {'dice_baseline': {fidx: np.array}, 'dice_bridge': {fidx: np.array}}
    """
    lines = []
    lines.append("Cardiac Dice Score Summary")
    lines.append("=" * 60)
    lines.append("Average across all subjects, all time frames, all structures")
    lines.append(f"Frames: {list(selected_frames)}")
    lines.append("Structures: ['LV']")
    lines.append("")
    lines.append(f"{'Method':<10} {'Variant':<12} {'Mean':>8} {'Std':>8} {'N':>8}")
    lines.append("-" * 60)
    for method in methods:
        for variant, mk in [('Baseline', 'dice_baseline'), ('+BridgeUQ', 'dice_bridge')]:
            vals = []
            per_frame = method_to_frame_vals.get(method, {}).get(mk, {})
            for fidx in selected_frames:
                arr = per_frame.get(fidx)
                if arr is not None and len(arr) > 0:
                    vals.extend(np.asarray(arr).tolist())
            arr = np.array(vals) if vals else np.array([0.0])
            lines.append(f"{method:<10} {variant:<12} {np.mean(arr):>8.4f} "
                         f"{np.std(arr):>8.4f} {len(arr):>8d}")
    text = "\n".join(lines) + "\n"
    with open(save_path, 'w') as f:
        f.write(text)
    print(f"Saved average Dice summary to: {save_path}")


def _plot_dice_hd95_boxplot(method_metrics, selected_frames, save_path):
    """
    Save .npz data and produce boxplot with scatter for Dice and HD95
    across methods and frames.

    method_metrics: dict  method_name -> list of dicts
        {frame_idx: {'dice_baseline', 'hd95_baseline', 'dice_bridge', 'hd95_bridge'}}
    selected_frames: list of frame indices to plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.patches import Patch

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

    methods = list(method_metrics.keys())
    n_frames = len(selected_frames)
    frames = np.arange(1, n_frames + 1)

    # ── Save .npz ──
    npz_path = save_path.replace('.png', '.npz')
    npz_data = {
        'methods': np.array(methods),
        'selected_frames': np.array(selected_frames),
    }

    # Build per-method per-variant data arrays: lists of arrays per frame
    # dice_data[label] = list of arrays (one per frame), each array = per-subject values
    # Same for hd95_data
    dice_data = {}
    hd95_data = {}

    color_map = {
        'VM':   ('#888888', '#505050'),   # gray
        'LTMA': ('#c02080', '#901060'),   # magenta
        'TLRN': ('#7040b0', '#501880'),   # purple
        'TM':   ('#309050', '#186838'),   # green
    }

    for method in methods:
        for variant, metric_key in [('Baseline', 'dice_baseline'),
                                    ('+BridgeUQ', 'dice_bridge')]:
            label = f'{method} ({variant})'
            frame_data = []
            for fidx in selected_frames:
                vals = [m[fidx][metric_key] for m in method_metrics[method]
                        if fidx in m and metric_key in m[fidx]]
                frame_data.append(np.array(vals) if vals else np.array([0.0]))
            dice_data[label] = frame_data

        for variant, metric_key in [('Baseline', 'hd95_baseline'),
                                    ('+BridgeUQ', 'hd95_bridge')]:
            label = f'{method} ({variant})'
            frame_data = []
            for fidx in selected_frames:
                vals = [m[fidx][metric_key] for m in method_metrics[method]
                        if fidx in m and metric_key in m[fidx]]
                # Filter out 0.0 values (artifact from empty-mask HD95)
                arr = np.array(vals) if vals else np.array([])
                arr = arr[arr > 0.0] if len(arr) > 0 else arr
                frame_data.append(arr if len(arr) > 0 else np.array([np.nan]))
            hd95_data[label] = frame_data

    # Save per-subject data to npz
    for method in methods:
        for variant_suffix, metric_keys in [
            ('baseline', ('dice_baseline', 'hd95_baseline')),
            ('bridge', ('dice_bridge', 'hd95_bridge')),
        ]:
            for fidx in selected_frames:
                for mk in metric_keys:
                    vals = [m[fidx][mk] for m in method_metrics[method]
                            if fidx in m and mk in m[fidx]]
                    npz_data[f'{method}_{mk}_frame_{fidx}'] = np.array(vals)
    np.savez(npz_path, **npz_data)
    print(f"Saved data to: {npz_path}")

    # ── Save average Dice summary ──
    txt_path = save_path.replace('.png', '_avg.txt')
    method_to_frame_vals = {}
    for method in methods:
        method_to_frame_vals[method] = {'dice_baseline': {}, 'dice_bridge': {}}
        for mk in ('dice_baseline', 'dice_bridge'):
            for fidx in selected_frames:
                vals = [m[fidx][mk] for m in method_metrics[method]
                        if fidx in m and mk in m[fidx]]
                method_to_frame_vals[method][mk][fidx] = np.array(vals) if vals else np.array([])
    _save_avg_dice_txt(method_to_frame_vals, methods, selected_frames, txt_path)

    # ── Plot ──
    method_names = list(dice_data.keys())
    n_methods = len(method_names)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))

    group_width = 0.75
    box_width = group_width / n_methods
    offsets = np.linspace(-group_width/2 + box_width/2,
                          group_width/2 - box_width/2, n_methods)

    def plot_boxplot(ax, all_data, ylabel, ylim, yticks, log_scale=False):
        for mi, mname in enumerate(method_names):
            # Extract model key (VM, LTMA, TLRN, TM)
            model_key = mname.split(' (')[0]
            base_color, dark_color = color_map.get(model_key, ('#999999', '#666666'))
            is_bridge = '+BridgeUQ' in mname
            alpha_box = 0.85 if not is_bridge else 0.35

            positions = frames + offsets[mi]

            bp = ax.boxplot(
                all_data[mname],
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=False,
                zorder=3,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(color=dark_color, linewidth=0.9),
                capprops=dict(color=dark_color, linewidth=0.9),
                boxprops=dict(facecolor=base_color, edgecolor=dark_color,
                              alpha=alpha_box, linewidth=0.9),
            )

            # Colored jittered data points above/below whiskers (outliers)
            for f in range(n_frames):
                frame_samples = all_data[mname][f]
                if len(frame_samples) < 2:
                    continue
                q1 = np.percentile(frame_samples, 25)
                q3 = np.percentile(frame_samples, 75)
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                outlier_mask = (frame_samples < lower_fence) | (frame_samples > upper_fence)
                outliers = frame_samples[outlier_mask]
                if len(outliers) > 0:
                    jitter = np.random.uniform(-box_width * 0.2,
                                               box_width * 0.2,
                                               len(outliers))
                    ax.scatter(
                        positions[f] + jitter, outliers,
                        s=8, alpha=0.4, color='black', zorder=4,
                        edgecolors='none', marker='o',
                    )

        # Vertical separators
        for f in range(1, n_frames):
            ax.axvline(f + 0.5, color='#888888', linewidth=0.15, zorder=1)

        ax.set_xlim(0.3, n_frames + 0.7)
        ax.set_xticks(frames)
        ax.set_ylabel(ylabel, fontsize=20)
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(ylim)
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{v:.2f}' for v in yticks])
        else:
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_facecolor('#f5f5f5')
        ax.set_axisbelow(True)
        ax.grid(False)

    # Compute y-axis limits from dice data
    all_dice = [v for label in dice_data for arr in dice_data[label] for v in arr]

    dice_lo = max(0, np.percentile(all_dice, 1) - 0.05) if all_dice else 0.5
    dice_hi = min(1.0, np.percentile(all_dice, 99) + 0.05) if all_dice else 1.0

    dice_yticks = np.linspace(
        np.round(dice_lo, 2), np.round(dice_hi, 2), 5
    )

    # Top: Dice for Cardiac Cine MRI
    np.random.seed(42)
    plot_boxplot(ax1, dice_data, 'Dice Score',
                 (dice_lo, dice_hi), dice_yticks)
    ax1.set_title('Cardiac Cine MRI', fontsize=22)

    # # Bottom: Dice for Longitudinal Brain (same data as placeholder)
    # np.random.seed(123)
    # plot_boxplot(ax2, dice_data, 'Dice Score',
    #              (dice_lo, dice_hi), dice_yticks)
    # ax2.set_title('Longitudinal Brain Diseases Progression', fontsize=22)

    ax1.set_xlabel('Frame Index', fontsize=20)
    ax1.set_xticklabels([str(f) for f in selected_frames], fontsize=20)

    # Shared legend at bottom
    legend_elements = []
    for model_key in color_map:
        if model_key not in methods:
            continue
        base_c, dark_c = color_map[model_key]
        legend_elements.append(
            Patch(facecolor=base_c, edgecolor=dark_c, alpha=0.85,
                  label=f'{model_key} (Baseline)'))
        legend_elements.append(
            Patch(facecolor=base_c, edgecolor=dark_c, alpha=0.35,
                  label=f'{model_key} (+BridgeUQ)'))

    ax1.legend(
        handles=legend_elements,
        loc='lower right',
        ncol=4,
        fontsize=16,
        frameon=True,
        fancybox=False,
        edgecolor='#cccccc',
        framealpha=0.9,
        handlelength=1.5,
        handleheight=1.0,
        columnspacing=1.0,
    )

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.12)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot to: {save_path}")
    print(f"Saved boxplot to: {pdf_path}")
    print(f"Saved boxplot to: {svg_path}")


def _plot_from_npz(npz_path, save_path):
    """Re-plot from a previously saved .npz file (no model inference needed)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.patches import Patch

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

    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    # selected_frames = list(data['selected_frames'])[:-2]
    selected_frames = list(data['selected_frames'])
    n_frames = len(selected_frames)
    frames = np.arange(1, n_frames + 1)

    # ── Save average Dice summary ──
    txt_path = save_path.replace('.png', '_avg.txt')
    method_to_frame_vals = {}
    for method in methods:
        method_to_frame_vals[method] = {'dice_baseline': {}, 'dice_bridge': {}}
        for mk in ('dice_baseline', 'dice_bridge'):
            for fidx in selected_frames:
                key = f'{method}_{mk}_frame_{fidx}'
                arr = data[key] if key in data else np.array([])
                method_to_frame_vals[method][mk][fidx] = arr
    _save_avg_dice_txt(method_to_frame_vals, methods, selected_frames, txt_path)

    color_map = {
        'VM':   ('#888888', '#505050'),   # gray
        'LTMA': ('#c02080', '#901060'),   # magenta
        'TLRN': ('#7040b0', '#501880'),   # purple
        'TM':   ('#309050', '#186838'),   # green
    }

    # Reconstruct dice_data and hd95_data from npz
    dice_data = {}
    hd95_data = {}
    for method in methods:
        for variant, mk in [('Baseline', 'dice_baseline'), ('+BridgeUQ', 'dice_bridge')]:
            label = f'{method} ({variant})'
            frame_data = []
            for fidx in selected_frames:
                key = f'{method}_{mk}_frame_{fidx}'
                vals = data[key] if key in data else np.array([0.0])
                frame_data.append(vals)
            dice_data[label] = frame_data

        for variant, mk in [('Baseline', 'hd95_baseline'), ('+BridgeUQ', 'hd95_bridge')]:
            label = f'{method} ({variant})'
            frame_data = []
            for fidx in selected_frames:
                key = f'{method}_{mk}_frame_{fidx}'
                arr = data[key] if key in data else np.array([])
                # Filter out 0.0 values (artifact from empty-mask HD95)
                arr = arr[arr > 0.0] if len(arr) > 0 else arr
                frame_data.append(arr if len(arr) > 0 else np.array([np.nan]))
            hd95_data[label] = frame_data

    method_names = list(dice_data.keys())
    n_methods = len(method_names)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))

    group_width = 0.75
    box_width = group_width / n_methods
    offsets = np.linspace(-group_width/2 + box_width/2,
                          group_width/2 - box_width/2, n_methods)

    def plot_boxplot(ax, all_data, ylabel, ylim, yticks, log_scale=False):
        for mi, mname in enumerate(method_names):
            model_key = mname.split(' (')[0]
            base_color, dark_color = color_map.get(model_key, ('#999999', '#666666'))
            is_bridge = '+BridgeUQ' in mname
            alpha_box = 0.85 if not is_bridge else 0.35

            positions = frames + offsets[mi]

            bp = ax.boxplot(
                all_data[mname],
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=False,
                zorder=3,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(color=dark_color, linewidth=0.9),
                capprops=dict(color=dark_color, linewidth=0.9),
                boxprops=dict(facecolor=base_color, edgecolor=dark_color,
                              alpha=alpha_box, linewidth=0.9),
            )

            # Colored jittered data points above/below whiskers (outliers)
            for f in range(n_frames):
                frame_samples = all_data[mname][f]
                if len(frame_samples) < 2:
                    continue
                q1 = np.percentile(frame_samples, 25)
                q3 = np.percentile(frame_samples, 75)
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                outlier_mask = (frame_samples < lower_fence) | (frame_samples > upper_fence)
                outliers = frame_samples[outlier_mask]
                if len(outliers) > 0:
                    jitter = np.random.uniform(-box_width * 0.2,
                                               box_width * 0.2,
                                               len(outliers))
                    ax.scatter(
                        positions[f] + jitter, outliers,
                        s=8, alpha=0.4, color='black', zorder=4,
                        edgecolors='none', marker='o',
                    )

        for f in range(1, n_frames):
            ax.axvline(f + 0.5, color='#888888', linewidth=0.15, zorder=1)

        ax.set_xlim(0.3, n_frames + 0.7)
        ax.set_xticks(frames)
        ax.set_ylabel(ylabel, fontsize=20)
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(ylim)
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{v:.2f}' for v in yticks])
        else:
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_facecolor('#f5f5f5')
        ax.set_axisbelow(True)
        ax.grid(False)

    all_dice = [v for label in dice_data for arr in dice_data[label] for v in arr]

    dice_lo = max(0, np.percentile(all_dice, 1) - 0.05) if all_dice else 0.5
    dice_hi = min(1.0, np.percentile(all_dice, 99) + 0.05) if all_dice else 1.0

    dice_yticks = np.linspace(np.round(dice_lo, 2), np.round(dice_hi, 2), 5)

    # Top: Dice for Cardiac Cine MRI
    np.random.seed(42)
    plot_boxplot(ax1, dice_data, 'Dice Score',
                 (dice_lo, dice_hi), dice_yticks)
    ax1.set_title('Cardiac Cine MRI', fontsize=22)

    # # Bottom: Dice for Longitudinal Brain (same data as placeholder)
    # np.random.seed(123)
    # plot_boxplot(ax2, dice_data, 'Dice Score',
    #              (dice_lo, dice_hi), dice_yticks)
    # ax2.set_title('Longitudinal Brain Diseases Progression', fontsize=22)

    ax1.set_xlabel('Frame Index', fontsize=20)
    ax1.set_xticklabels([str(f) for f in selected_frames], fontsize=20)

    legend_elements = []
    for model_key in color_map:
        if model_key not in methods:
            continue
        base_c, dark_c = color_map[model_key]
        legend_elements.append(
            Patch(facecolor=base_c, edgecolor=dark_c, alpha=0.85,
                  label=f'{model_key} (Baseline)'))
        legend_elements.append(
            Patch(facecolor=base_c, edgecolor=dark_c, alpha=0.35,
                  label=f'{model_key} (+BridgeUQ)'))

    ax1.legend(
        handles=legend_elements,
        loc='lower right',
        ncol=4,
        fontsize=16,
        frameon=True,
        fancybox=False,
        edgecolor='#cccccc',
        framealpha=0.9,
        handlelength=1.5,
        handleheight=1.0,
        columnspacing=1.0,
    )

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.12)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot to: {save_path}")
    print(f"Saved boxplot to: {pdf_path}")
    print(f"Saved boxplot to: {svg_path}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dice & HD95 boxplot: per-frame registration quality across methods"
    )
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of UQ sampling runs per method (default: 100)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_train", action="store_true",
                        help="Use training set instead of test set")
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run even if .npz already exists")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), 'dice_hd95_boxplots')
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, 'dice_hd95_by_frame.npz')
    png_path = os.path.join(output_dir, 'dice_hd95_by_frame.png')

    # ------------------------------------------------------------------
    # If .npz exists, just re-plot and exit
    # ------------------------------------------------------------------
    if os.path.exists(npz_path) and not args.rerun:
        print(f"Found existing data: {npz_path}")
        print("Plotting from saved data (use --rerun to force re-computation)")
        _plot_from_npz(npz_path, png_path)
        print(f"\nDone! All outputs saved to: {output_dir}")
        return

    # ------------------------------------------------------------------
    # Run each method and collect metrics
    # ------------------------------------------------------------------
    NUM_DIFFUSION_STEPS = 14
    NUM_RUNS = args.num_runs

    method_metrics = {}
    all_available_frames = None

    for method_name, method_cfg in METHOD_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running method: {method_name}")
        print(f"{'='*60}")

        metrics, frame_indices = _run_method(
            method_name, method_cfg, NUM_RUNS, NUM_DIFFUSION_STEPS,
            use_train=args.use_train,
        )
        method_metrics[method_name] = metrics

        if all_available_frames is None:
            all_available_frames = frame_indices
        print(f"  [{method_name}] Done. {len(metrics)} subjects, frames: {frame_indices}")

    # ------------------------------------------------------------------
    # Use all available frames
    # ------------------------------------------------------------------
    # selected_frames = sorted(all_available_frames)[:-2]
    selected_frames = sorted(all_available_frames)
    print(f"\nSelected frames for boxplot: {selected_frames}")

    # ------------------------------------------------------------------
    # Plot (also saves .npz)
    # ------------------------------------------------------------------
    _plot_dice_hd95_boxplot(method_metrics, selected_frames, png_path)

    print(f"\nDone! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
