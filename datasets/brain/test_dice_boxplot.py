"""
Dice box-plot script for longitudinal brain registration -- computes per-frame
Dice score (lateral ventricles) for multiple methods (TLRN, VM, LTMA, TM),
each with Baseline and +BridgeUQ variants, using FreeSurfer segmentation masks.

Run with:
  python -m uncertainty_brain_sde.test_dice_boxplot [--num_runs N]
  python -m uncertainty_brain_sde.test_dice_boxplot --replot   # re-plot from saved .npz
"""

import os
import sys
import argparse
import gc
import re

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm


# ==========================================================================
# Method configurations
# ==========================================================================

METHOD_CONFIGS = {
    'TLRN': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2',
        'data_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/dataset/precomputed_brain_z64_tlrn',
        'ckpt_dir': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/2026Experiments/BRAIN/outputs/TLRN/basic_brain/visualization/uncertainty_brain_sde/checkpoints',
        'vec_int_api': 'list',  # vec_int returns list, use [-1]
        'st_api': 'tuple',      # spatial_transformer returns (warped, _)
        'vec_int_kwarg': 'TSteps',
    },
    'VM': {
        'sys_path': '/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty',
        'data_path': '/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty/dataset/precomputed_brain_z64',
        'ckpt_dir': '/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty/outputs/uncertainty_brain_sde/checkpoints',
        'vec_int_api': 'direct',  # vec_int returns displacement directly
        'st_api': 'direct',       # spatial_transformer returns tensor directly
        'vec_int_kwarg': 'nsteps',
    },
    'LTMA': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA',
        'data_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/dataset/precomputed_brain_z64_ltma',
        'ckpt_dir': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/2026Experiments/BRAIN/outputs/TLMA_TGrad/basic_brain/visualization/uncertainty_brain_sde/checkpoints',
        'vec_int_api': 'list',
        'st_api': 'tuple',
        'vec_int_kwarg': 'TSteps',
    },
    'TM': {
        'sys_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/TM',
        'data_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/TM/dataset/precomputed_brain_z64_tm',
        'ckpt_dir': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/TM/2026Experiments/BRAIN/outputs/TransMorph/basic_brain/visualization/uncertainty_brain_sde/checkpoints',
        'vec_int_api': 'list',
        'st_api': 'tuple',
        'vec_int_kwarg': 'TSteps',
    },
}


# ==========================================================================
# Lateral ventricle segmentation helpers
# ==========================================================================

FREESURFER_SEG_DIR = '/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty/freesurfer_seg'
LEFT_VENTRICLE_LABEL = 4    # Left-Lateral-Ventricle
RIGHT_VENTRICLE_LABEL = 43  # Right-Lateral-Ventricle
AXIAL_SLICE_IDX = 64

# Cyclic sequence: [t0, t1, t2, t3, t2', t1', t0']
# target frame index (0-based in target_imgs) -> timepoint name
CYCLIC_TO_TIMEPOINT = {0: 't1', 1: 't2', 2: 't3', 3: 't2', 4: 't1', 5: 't0'}

# Structure names for plotting
STRUCTURES = ['LV', 'RV']  # Left Ventricle, Right Ventricle
STRUCTURE_LABELS = {'LV': LEFT_VENTRICLE_LABEL, 'RV': RIGHT_VENTRICLE_LABEL}


def _get_seg_base_name(filename):
    """Extract base name (with class prefix) from precomputed filename."""
    basename = os.path.basename(filename)
    return basename.replace('_precomputed.pt', '').replace('.pt', '')


def load_seg_slice(seg_base_name, timepoint, axial_idx=AXIAL_SLICE_IDX):
    """Load full segmentation slice for a given subject/timepoint.
    Returns the raw label slice (not binarized), or None if file missing."""
    seg_file = os.path.join(FREESURFER_SEG_DIR, f'{seg_base_name}_{timepoint}_aseg.nii.gz')
    if not os.path.exists(seg_file):
        return None
    seg_vol = nib.load(seg_file).get_fdata()
    return seg_vol[:, :, axial_idx]


def make_binary_mask(seg_slice, label):
    """Create binary mask for a single label from a segmentation slice."""
    return (seg_slice == label).astype(np.float32)


def compute_dice(pred_mask, target_mask):
    """Compute Dice score between two binary masks."""
    pred = pred_mask.flatten()
    target = target_mask.flatten()
    intersection = np.sum(pred * target)
    total = np.sum(pred) + np.sum(target)
    if total < 1e-8:
        return 1.0
    return 2.0 * intersection / total


# ==========================================================================
# Per-subject per-frame Dice computation
# ==========================================================================

def _compute_dice_per_frame(v_series_bb, sampled_velocities, subject_idx,
                            filename, inshape, device, method_cfg):
    """Compute per-frame per-structure Dice for baseline and +BridgeUQ.

    Returns dict: frame_idx -> {
        'LV_dice_baseline': float, 'LV_dice_bridge': float,
        'RV_dice_baseline': float, 'RV_dice_bridge': float,
    }
    or empty dict if no seg files found.
    """
    seg_base_name = _get_seg_base_name(filename)
    source_seg_slice = load_seg_slice(seg_base_name, 't0')
    if source_seg_slice is None:
        return {}

    # Import VecInt/SpatialTransformer from the appropriate method
    vec_int_kwarg = method_cfg['vec_int_kwarg']
    if method_cfg['sys_path'] == '/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty':
        from uncertainty_brain_sde.pretrained_registration.layers import VecInt, SpatialTransformer
    else:
        from utils.Int import VecInt, SpatialTransformer

    vec_int = VecInt(inshape, **{vec_int_kwarg: 7}).to(device)
    spatial_transformer = SpatialTransformer(inshape).to(device)

    is_list_api = method_cfg['vec_int_api'] == 'list'
    is_tuple_st = method_cfg['st_api'] == 'tuple'

    def _integrate(velocity):
        result = vec_int(velocity)
        return result[-1] if is_list_api else result

    def _warp(seg_tensor, displacement):
        result = spatial_transformer(seg_tensor, displacement)
        return result[0] if is_tuple_st else result

    # Precompute source masks per structure
    src_masks = {}
    for struct in STRUCTURES:
        label = STRUCTURE_LABELS[struct]
        mask = make_binary_mask(source_seg_slice, label)
        src_masks[struct] = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)

    num_target_frames = len(v_series_bb) - 2  # exclude zero endpoints
    frame_metrics = {}

    for frame_idx in range(num_target_frames):
        tp_name = CYCLIC_TO_TIMEPOINT.get(frame_idx)
        if tp_name is None:
            continue
        target_seg_slice = load_seg_slice(seg_base_name, tp_name)
        if target_seg_slice is None:
            continue

        # Baseline: registration network velocity
        v_reg = v_series_bb[frame_idx + 1][subject_idx:subject_idx + 1]
        displacement = _integrate(v_reg)

        # +BridgeUQ: mean sampled velocity
        bb_idx = frame_idx + 1
        disp_mean = None
        if bb_idx in sampled_velocities and len(sampled_velocities[bb_idx]) > 0:
            vels = [v[subject_idx:subject_idx + 1] for v in sampled_velocities[bb_idx]]
            mean_vel = torch.stack(vels, dim=0).mean(dim=0)
            disp_mean = _integrate(mean_vel)

        metrics = {}
        for struct in STRUCTURES:
            label = STRUCTURE_LABELS[struct]
            target_mask = make_binary_mask(target_seg_slice, label)

            # Baseline dice
            warped = _warp(src_masks[struct], displacement)
            warped_bin = (warped[0, 0].detach().cpu().numpy() > 0.5).astype(np.float32)
            dice_base = compute_dice(warped_bin, target_mask)
            metrics[f'{struct}_dice_baseline'] = dice_base

            # Bridge dice
            if disp_mean is not None:
                warped_mean = _warp(src_masks[struct], disp_mean)
                warped_mean_bin = (warped_mean[0, 0].detach().cpu().numpy() > 0.5).astype(np.float32)
                metrics[f'{struct}_dice_bridge'] = compute_dice(warped_mean_bin, target_mask)
            else:
                metrics[f'{struct}_dice_bridge'] = dice_base

        frame_metrics[frame_idx + 1] = metrics

    return frame_metrics


# ==========================================================================
# Load and run one method
# ==========================================================================

def _run_method(method_name, method_cfg, num_runs, num_diffusion_steps):
    """
    Load precomputed data and BridgeUQ checkpoint for one method, run UQ,
    return per-subject per-frame Dice metrics.
    """
    method_sys_path = method_cfg['sys_path']

    original_sys_path = sys.path.copy()
    # Remove other method paths
    for other_cfg in METHOD_CONFIGS.values():
        other_path = other_cfg['sys_path']
        if other_path != method_sys_path:
            while other_path in sys.path:
                sys.path.remove(other_path)
    if method_sys_path not in sys.path:
        sys.path.insert(0, method_sys_path)

    # Purge cached modules
    _purge_prefixes = [
        'uncertainty_brain_sde', 'networks', 'utils', 'dataload', 'models',
    ]
    for mod_name in list(sys.modules.keys()):
        if any(mod_name == pfx or mod_name.startswith(pfx + '.')
               for pfx in _purge_prefixes):
            del sys.modules[mod_name]

    from uncertainty_brain_sde.networks import ScalingFactorNetwork
    from uncertainty_brain_sde.losses import LTMALoss
    from uncertainty_brain_sde.brownian_bridge import BrownianBridgeLearnedScaling
    from uncertainty_brain_sde.trainer import ScalingFactorTrainer
    from uncertainty_brain_sde.data_utils import BrainDataset
    from torch.utils.data import DataLoader, ConcatDataset

    BATCH_SIZE = 4
    LAMBDA_SCALE = 3
    ALPHA_SCALE = 0.0001
    GAMMA = 0.0001 / 2
    GUIDANCE_SCALE = 1.0

    def collate_brain_batch(batch):
        series = torch.stack([b['series'] for b in batch])
        labels = torch.tensor([b['label'] for b in batch])
        filenames = [b['filename'] for b in batch]
        ptids = [b['ptid'] for b in batch]
        age_lists = [b['age_list'] for b in batch]
        num_frames = len(batch[0]['v_series'])
        v_series = [torch.stack([b['v_series'][t] for b in batch]) for t in range(num_frames)]
        return {'series': series, 'v_series': v_series, 'label': labels,
                'filename': filenames, 'ptid': ptids, 'age_list': age_lists}

    data_path = method_cfg['data_path']
    print(f"  [{method_name}] Loading data from: {data_path}")

    train_ds = BrainDataset(data_path, split='all', mode='train', dataset='full')
    val_ds = BrainDataset(data_path, split='all', mode='val', dataset='full')
    test_ds = BrainDataset(data_path, split='all', mode='test', dataset='full')
    eval_dataset = ConcatDataset([train_ds, val_ds, test_ds])

    data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_brain_batch, num_workers=2, pin_memory=True)
    print(f"  [{method_name}] {len(eval_dataset)} samples")

    first_batch = next(iter(data_loader))
    num_time_steps = len(first_batch['v_series']) - 1
    img_size = first_batch['series'].shape[-1]
    inshape = (img_size, img_size)
    print(f"  [{method_name}] Image size: {img_size}, time steps: {num_time_steps}")
    del first_batch

    scaling_network = ScalingFactorNetwork(
        num_time_steps=num_time_steps, img_size=img_size, num_heads=8
    )

    # Auto-detect latest checkpoint
    ckpt_dir = method_cfg['ckpt_dir']
    import glob as globmod
    ckpt_files = globmod.glob(f"{ckpt_dir}/checkpoint_epoch_*.pth")
    if not ckpt_files:
        raise FileNotFoundError(f"[{method_name}] No checkpoints in {ckpt_dir}")

    def _epoch_num(p):
        m = re.search(r'checkpoint_epoch_(\d+)\.pth', p)
        return int(m.group(1)) if m else 0

    checkpoint_path = max(ckpt_files, key=_epoch_num)
    print(f"  [{method_name}] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    scaling_network.load_state_dict(checkpoint['model_state_dict'])
    scaling_network.cuda().eval()

    ckpt_config = checkpoint.get('config', {})

    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=num_diffusion_steps, img_size=img_size
    )

    loss_fn = LTMALoss(
        lambda_sim=1.0,
        lambda_reg=0.0,
        lambda_scale=ckpt_config.get('lambda_scale', LAMBDA_SCALE),
        alpha_scale=ckpt_config.get('alpha_scale', ALPHA_SCALE),
        use_ncc=False,
    )

    trainer = ScalingFactorTrainer(
        scaling_network=scaling_network,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        lr=1e-4,
        device='cuda',
        img_size=img_size,
        gamma=ckpt_config.get('gamma', GAMMA),
        guidance_scale=ckpt_config.get('guidance_scale', GUIDANCE_SCALE),
    )

    all_metrics = []
    total_batches = len(data_loader)

    for batch_idx, batch in enumerate(data_loader):
        print(f"  [{method_name}] Batch {batch_idx + 1}/{total_batches}")

        series = batch['series'].cuda()
        v_series = [v.cuda() for v in batch['v_series']]
        filenames_batch = batch['filename']
        batch_size = series.shape[0]

        source_img = series[:, 0:1]
        target_imgs = [series[:, t:t + 1] for t in range(1, series.shape[1])]
        v_series_bb = v_series

        _, scaling_factors_bb = trainer.validate(source_img, target_imgs, v_series_bb)

        num_frames_bb = len(v_series_bb)
        sampled_velocities = {i: [] for i in range(num_frames_bb)}

        for run in tqdm(range(num_runs), desc=f"    UQ runs", leave=False):
            sampled = brownian_bridge.sample_bridge_transition(
                v_series_bb, scaling_factors_bb, device='cuda'
            )
            for fi in range(num_frames_bb):
                sampled_velocities[fi].append(sampled[fi].clone())
            del sampled
            if (run + 1) % 10 == 0:
                torch.cuda.empty_cache()

        for subj_idx in range(batch_size):
            metrics = _compute_dice_per_frame(
                v_series_bb, sampled_velocities, subj_idx,
                filenames_batch[subj_idx], inshape, 'cuda', method_cfg,
            )
            if metrics:  # only include subjects with seg
                all_metrics.append(metrics)

        del series, v_series, v_series_bb, source_img, target_imgs
        del scaling_factors_bb, sampled_velocities
        torch.cuda.empty_cache()
        gc.collect()

    all_frame_indices = set()
    for m in all_metrics:
        all_frame_indices.update(m.keys())
    all_frame_indices = sorted(all_frame_indices)

    print(f"  [{method_name}] Done. {len(all_metrics)} subjects with seg, frames: {all_frame_indices}")

    del scaling_network, brownian_bridge, trainer, loss_fn
    torch.cuda.empty_cache()
    gc.collect()

    sys.path[:] = original_sys_path

    return all_metrics, all_frame_indices


# ==========================================================================
# Plotting -- NeurIPS two-panel style
# ==========================================================================

# Method display order: (label, baseline_color, bridge_color)
PLOT_METHODS = [
    ('VM',   '#4d4d4d', '#bfbfbf'),
    ('LTMA', '#c2185b', '#f48fb1'),
    ('TLRN', '#5e35b1', '#b39ddb'),
    ('TM',   '#2e7d32', '#a5d6a7'),
]

BOX_WIDTH = 0.09
GROUP_GAP = 1.0

BOXPLOT_KW = dict(
    widths=BOX_WIDTH * 0.9,
    patch_artist=True,
    whis=(5, 100),
    flierprops=dict(
        marker='o', markersize=1.2,
        markerfacecolor='gray', markeredgecolor='gray', alpha=0.5,
    ),
    medianprops=dict(color='white', linewidth=0.8),
    boxprops=dict(linewidth=0.5),
    whiskerprops=dict(linewidth=0.5),
    capprops=dict(linewidth=0.5),
)

NEURIPS_RC = {
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def _build_panel_data(dice_per_struct, methods_raw, selected_frames, struct):
    """Build panel_data[method_variant_idx][frame_idx] -> np.array of Dice."""
    display_name = lambda m: 'VM' if m in ('VxM', 'VM') else m
    n_frames = len(selected_frames)

    # Build mapping: display_method_name -> raw_method_name
    raw_for = {}
    for m in methods_raw:
        raw_for[display_name(m)] = m

    panel_data = []  # one list per method-variant (8 total)
    for model_key, _, _ in PLOT_METHODS:
        raw_m = raw_for.get(model_key)
        for suffix in ['dice_baseline', 'dice_bridge']:
            mk = f'{struct}_{suffix}'
            frame_arrays = []
            for fidx in selected_frames:
                if raw_m is not None:
                    vals = dice_per_struct.get((raw_m, mk, fidx), np.array([0.0]))
                else:
                    vals = np.array([0.0])
                frame_arrays.append(vals)
            panel_data.append(frame_arrays)
    return panel_data


def _draw_panel(ax, panel_data, subtitle, n_frames, ylim):
    """Draw one panel (Left or Right Ventricle)."""
    n_variants = len(panel_data)  # 8 = 4 methods * 2 variants
    positions, arrays, colors = [], [], []

    # Build color list matching panel_data order
    color_list = []
    for _, base_c, bridge_c in PLOT_METHODS:
        color_list.append(base_c)
        color_list.append(bridge_c)

    for f in range(n_frames):
        base = f * GROUP_GAP + 1
        for m in range(n_variants):
            offset = (m - (n_variants - 1) / 2) * BOX_WIDTH
            positions.append(base + offset)
            arrays.append(panel_data[m][f])
            colors.append(color_list[m])

    bp = ax.boxplot(arrays, positions=positions, **BOXPLOT_KW)

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor('black')

    ax.set_xticks([f * GROUP_GAP + 1 for f in range(n_frames)])
    ax.set_xticklabels([str(f + 1) for f in range(n_frames)])
    ax.set_xlabel(f'Frame Index ({subtitle})')
    ax.set_ylim(ylim)
    ax.set_xlim(0.4, n_frames * GROUP_GAP + 0.6)


def _do_plot(panel_data_left, panel_data_right, n_frames, ylim, save_path):
    """Create the two-panel figure and save."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(NEURIPS_RC)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), sharey=True)

    _draw_panel(axes[0], panel_data_left,  'Left Ventricle',  n_frames, ylim)
    _draw_panel(axes[1], panel_data_right, 'Right Ventricle', n_frames, ylim)
    axes[0].set_ylabel('Dice Score')

    # Legend above figure
    handles = []
    for model_key, base_c, bridge_c in PLOT_METHODS:
        handles.append(Patch(facecolor=base_c, edgecolor='black',
                             label=f'{model_key} (Baseline)'))
        handles.append(Patch(facecolor=bridge_c, edgecolor='black',
                             label=f'{model_key} (+BridgeUQ)'))

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.4,
    )

    plt.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.17, wspace=0.06)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot to: {save_path}")
    print(f"Saved boxplot to: {pdf_path}")
    print(f"Saved boxplot to: {svg_path}")


def _save_avg_dice_txt(dice_lookup, methods, selected_frames, structures, save_path):
    """Write average Dice (across all subjects, frames, structures) per method & variant."""
    lines = []
    lines.append("Brain Dice Score Summary")
    lines.append("=" * 60)
    lines.append("Average across all subjects, all time frames, all structures")
    lines.append(f"Frames: {list(selected_frames)}")
    lines.append(f"Structures: {list(structures)}")
    lines.append("")
    lines.append(f"{'Method':<10} {'Variant':<12} {'Mean':>8} {'Std':>8} {'N':>8}")
    lines.append("-" * 60)
    for method in methods:
        for variant, suffix in [('Baseline', 'dice_baseline'), ('+BridgeUQ', 'dice_bridge')]:
            vals = []
            for struct in structures:
                mk = f'{struct}_{suffix}'
                for fidx in selected_frames:
                    arr = dice_lookup.get((method, mk, fidx))
                    if arr is not None and len(arr) > 0:
                        vals.extend(arr.tolist())
            arr = np.array(vals) if vals else np.array([0.0])
            lines.append(f"{method:<10} {variant:<12} {np.mean(arr):>8.4f} "
                         f"{np.std(arr):>8.4f} {len(arr):>8d}")
    text = "\n".join(lines) + "\n"
    with open(save_path, 'w') as f:
        f.write(text)
    print(f"Saved average Dice summary to: {save_path}")


def _plot_dice_boxplot(method_metrics, selected_frames, save_path):
    """Produce two-panel Dice boxplot from live method_metrics."""
    methods = list(method_metrics.keys())
    n_frames = len(selected_frames)

    # Save .npz
    npz_path = save_path.replace('.png', '.npz')
    npz_data = {
        'methods': np.array(methods),
        'selected_frames': np.array(selected_frames),
        'structures': np.array(STRUCTURES),
    }
    for method in methods:
        for struct in STRUCTURES:
            for suffix in ['dice_baseline', 'dice_bridge']:
                mk = f'{struct}_{suffix}'
                for fidx in selected_frames:
                    vals = [m[fidx][mk] for m in method_metrics[method]
                            if fidx in m and mk in m[fidx]]
                    npz_data[f'{method}_{mk}_frame_{fidx}'] = np.array(vals)
    np.savez(npz_path, **npz_data)
    print(f"Saved data to: {npz_path}")

    # Build lookup: (method, metric_key, frame_idx) -> np.array
    dice_lookup = {}
    for method in methods:
        for struct in STRUCTURES:
            for suffix in ['dice_baseline', 'dice_bridge']:
                mk = f'{struct}_{suffix}'
                for fidx in selected_frames:
                    vals = [m[fidx][mk] for m in method_metrics[method]
                            if fidx in m and mk in m[fidx]]
                    dice_lookup[(method, mk, fidx)] = np.array(vals) if vals else np.array([0.0])

    # Compute shared y-limits
    all_vals = [v for arr in dice_lookup.values() for v in arr]
    ylo = max(0, np.percentile(all_vals, 1) - 0.05) if all_vals else 0.5
    yhi = min(1.0, np.percentile(all_vals, 99) + 0.05) if all_vals else 1.0

    panel_left  = _build_panel_data(dice_lookup, methods, selected_frames, 'LV')
    panel_right = _build_panel_data(dice_lookup, methods, selected_frames, 'RV')

    _do_plot(panel_left, panel_right, n_frames, (ylo, yhi), save_path)

    txt_path = save_path.replace('.png', '_avg.txt')
    _save_avg_dice_txt(dice_lookup, methods, selected_frames, STRUCTURES, txt_path)


def _plot_from_npz(npz_path, save_path):
    """Re-plot from a previously saved .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    methods = list(data['methods'])
    selected_frames = list(data['selected_frames'])
    structures = list(data['structures']) if 'structures' in data else STRUCTURES
    n_frames = len(selected_frames)

    display_name = lambda m: 'VM' if m == 'VxM' else m

    # Build lookup
    dice_lookup = {}
    for method in methods:
        for struct in structures:
            for suffix in ['dice_baseline', 'dice_bridge']:
                mk = f'{struct}_{suffix}'
                for fidx in selected_frames:
                    key = f'{method}_{mk}_frame_{fidx}'
                    vals = data[key] if key in data else np.array([0.0])
                    dice_lookup[(method, mk, fidx)] = vals

    all_vals = [v for arr in dice_lookup.values() for v in arr]
    ylo = max(0, np.percentile(all_vals, 1) - 0.05) if all_vals else 0.5
    yhi = min(1.0, np.percentile(all_vals, 99) + 0.05) if all_vals else 1.0

    panel_left  = _build_panel_data(dice_lookup, methods, selected_frames, 'LV')
    panel_right = _build_panel_data(dice_lookup, methods, selected_frames, 'RV')

    _do_plot(panel_left, panel_right, n_frames, (ylo, yhi), save_path)

    txt_path = save_path.replace('.png', '_avg.txt')
    _save_avg_dice_txt(dice_lookup, methods, selected_frames, structures, txt_path)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Brain Dice boxplot: per-frame lateral ventricle Dice across methods"
    )
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of UQ sampling runs per method (default: 100)")
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run even if .npz already exists")
    parser.add_argument("--replot", action="store_true",
                        help="Just re-plot from existing .npz")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dice_boxplots')
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, 'brain_dice_by_frame.npz')
    png_path = os.path.join(output_dir, 'brain_dice_by_frame.png')

    # If .npz exists and --replot or no --rerun, just re-plot
    if os.path.exists(npz_path) and (args.replot or not args.rerun):
        print(f"Found existing data: {npz_path}")
        print("Plotting from saved data (use --rerun to force re-computation)")
        _plot_from_npz(npz_path, png_path)
        print(f"\nDone! Outputs saved to: {output_dir}")
        return

    NUM_DIFFUSION_STEPS = 7
    NUM_RUNS = args.num_runs

    method_metrics = {}
    all_available_frames = None

    for method_name, method_cfg in METHOD_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Running method: {method_name}")
        print(f"{'=' * 60}")

        metrics, frame_indices = _run_method(
            method_name, method_cfg, NUM_RUNS, NUM_DIFFUSION_STEPS,
        )
        method_metrics[method_name] = metrics

        if all_available_frames is None:
            all_available_frames = frame_indices
        print(f"  [{method_name}] {len(metrics)} subjects with seg, frames: {frame_indices}")

    selected_frames = sorted(all_available_frames)
    print(f"\nSelected frames for boxplot: {selected_frames}")

    _plot_dice_boxplot(method_metrics, selected_frames, png_path)

    # Print summary table
    print(f"\n{'=' * 100}")
    print("DICE SUMMARY (mean +/- std)")
    print(f"{'=' * 100}")
    for struct in STRUCTURES:
        struct_label = 'Left Ventricle' if struct == 'LV' else 'Right Ventricle'
        print(f"\n--- {struct_label} ---")
        print(f"{'Method':<10} {'Variant':<12} {'Overall':>12}  " +
              "  ".join([f"Frame {f}" for f in selected_frames]))
        print("-" * 80)
        for method_name in method_metrics:
            for variant, suffix in [('Baseline', 'dice_baseline'), ('+BridgeUQ', 'dice_bridge')]:
                mk = f'{struct}_{suffix}'
                all_vals = []
                frame_strs = []
                for fidx in selected_frames:
                    vals = [m[fidx][mk] for m in method_metrics[method_name]
                            if fidx in m and mk in m[fidx]]
                    arr = np.array(vals) if vals else np.array([0.0])
                    all_vals.extend(vals)
                    frame_strs.append(f"{np.mean(arr):.3f}")
                overall = np.array(all_vals)
                print(f"{method_name:<10} {variant:<12} {np.mean(overall):.3f}+/-{np.std(overall):.3f}  " +
                      "  ".join(frame_strs))

    print(f"\nDone! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
