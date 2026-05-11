"""
Combined Dice boxplot -- NeurIPS style.

Top row:    Cardiac Cine MRI (single panel, 12 frames)
Bottom row: Longitudinal Brain (two panels: Left Ventricle, Right Ventricle)

Reads pre-computed .npz files; no model inference needed.

Usage:
    python -m uncertainty_brain_sde.plot_combined_dice
"""

import os
import numpy as np


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CARDIAC_NPZ = os.path.join(
    PROJECT_ROOT,
    'uncertainty_sde_combined_acdc', 'dice_hd95_boxplots', 'dice_hd95_by_frame.npz',
)
BRAIN_NPZ = os.path.join(
    PROJECT_ROOT,
    'uncertainty_brain_sde', 'dice_boxplots', 'brain_dice_by_frame.npz',
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'dice_boxplots')


# ──────────────────────────────────────────────────────────────
# NeurIPS rc params
# ──────────────────────────────────────────────────────────────

NEURIPS_RC = {
    'font.family': 'serif',
    'font.size': 7,
    'axes.labelsize': 7.5,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
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


# ──────────────────────────────────────────────────────────────
# Method definitions (label, baseline colour, bridge colour)
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────

def _load_cardiac(npz_path):
    """Load cardiac dice data.

    Returns panel_data[method_variant_idx][frame_idx] -> np.array
    and n_frames.
    """
    data = np.load(npz_path, allow_pickle=True)
    methods_raw = list(data['methods'])
    selected_frames = list(data['selected_frames'])
    n_frames = len(selected_frames)

    # Build raw method -> display name mapping
    raw_for = {}
    for m in methods_raw:
        dn = 'VM' if m in ('VxM', 'VM') else m
        raw_for[dn] = m

    panel_data = []
    for model_key, _, _ in PLOT_METHODS:
        raw_m = raw_for.get(model_key)
        for suffix in ['dice_baseline', 'dice_bridge']:
            frame_arrays = []
            for fidx in selected_frames:
                key = f'{raw_m}_{suffix}_frame_{fidx}' if raw_m else None
                if key and key in data:
                    frame_arrays.append(data[key])
                else:
                    frame_arrays.append(np.array([0.0]))
            panel_data.append(frame_arrays)

    return panel_data, n_frames


def _load_brain(npz_path):
    """Load brain dice data for LV and RV.

    Returns (panel_lv, panel_rv, n_frames) where each panel is
    panel_data[method_variant_idx][frame_idx] -> np.array.
    """
    data = np.load(npz_path, allow_pickle=True)
    methods_raw = list(data['methods'])
    selected_frames = list(data['selected_frames'])
    structures = list(data['structures']) if 'structures' in data else ['LV', 'RV']
    n_frames = len(selected_frames)

    raw_for = {}
    for m in methods_raw:
        dn = 'VM' if m in ('VxM', 'VM') else m
        raw_for[dn] = m

    panels = {}
    for struct in structures:
        panel_data = []
        for model_key, _, _ in PLOT_METHODS:
            raw_m = raw_for.get(model_key)
            for suffix in ['dice_baseline', 'dice_bridge']:
                mk = f'{struct}_{suffix}'
                frame_arrays = []
                for fidx in selected_frames:
                    key = f'{raw_m}_{mk}_frame_{fidx}' if raw_m else None
                    if key and key in data:
                        frame_arrays.append(data[key])
                    else:
                        frame_arrays.append(np.array([0.0]))
                panel_data.append(frame_arrays)
        panels[struct] = panel_data

    return panels.get('LV', []), panels.get('RV', []), n_frames


# ──────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────

def _draw_panel(ax, panel_data, n_frames, xlabel=None, ylabel=None,
                ylim=None, title=None):
    """Draw one boxplot panel."""
    n_variants = len(panel_data)  # 8 = 4 methods * 2 variants

    color_list = []
    for _, base_c, bridge_c in PLOT_METHODS:
        color_list.append(base_c)
        color_list.append(bridge_c)

    positions, arrays, colors = [], [], []
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

    # Thin vertical separators between frame groups
    for f in range(1, n_frames):
        ax.axvline(f * GROUP_GAP + 0.5, color='#cccccc', linewidth=0.3, zorder=1)

    ax.set_xticks([f * GROUP_GAP + 1 for f in range(n_frames)])
    ax.set_xticklabels([str(f + 1) for f in range(n_frames)])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.set_xlim(0.4, n_frames * GROUP_GAP + 0.6)


def _compute_ylim(all_panels):
    """Compute shared y-limits from all panel data."""
    all_vals = []
    for panel in all_panels:
        for method_frames in panel:
            for arr in method_frames:
                all_vals.extend(arr.tolist())
    if not all_vals:
        return (0.4, 1.0)
    lo = max(0, np.percentile(all_vals, 1) - 0.05)
    hi = min(1.0, np.percentile(all_vals, 99) + 0.05)
    return (lo, hi)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(NEURIPS_RC)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print(f"Loading cardiac data from: {CARDIAC_NPZ}")
    cardiac_panel, cardiac_n_frames = _load_cardiac(CARDIAC_NPZ)

    print(f"Loading brain data from: {BRAIN_NPZ}")
    brain_lv, brain_rv, brain_n_frames = _load_brain(BRAIN_NPZ)

    # Compute y-limits per row
    cardiac_ylim = _compute_ylim([cardiac_panel])
    # Fixed brain ylim so boxes are more visible (outlier dots may go below)
    brain_ylim = (0.55, 1.005)

    # ── Figure layout ──
    # Top row: 1 wide panel (cardiac, 12 frames)
    # Bottom row: 2 panels side-by-side (brain LV + RV, 5 frames each)
    #
    # Use gridspec: top row spans full width, bottom row split in half
    fig = plt.figure(figsize=(5.5, 3.2))

    # ── Legend at the very top ──
    handles = []
    for model_key, base_c, bridge_c in PLOT_METHODS:
        handles.append(Patch(facecolor=base_c, edgecolor='black',
                             label=f'{model_key} (Baseline)'))
        handles.append(Patch(facecolor=bridge_c, edgecolor='black',
                             label=f'{model_key} (+BridgeUQ)'))

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.53, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.3,
    )

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 0.85],
        width_ratios=[1, 1],
        hspace=0.55,
        wspace=0.06,
        left=0.08, right=0.99, top=0.85, bottom=0.1,
    )

    # Top: cardiac spans both columns
    ax_cardiac = fig.add_subplot(gs[0, :])
    _draw_panel(
        ax_cardiac, cardiac_panel, cardiac_n_frames,
        ylabel='Dice Score',
        ylim=cardiac_ylim,
        title='(a) Cardiac Cine MRI',
    )
    ax_cardiac.set_xlabel('Frame Index', labelpad=1)

    # Bottom left: brain LV
    ax_lv = fig.add_subplot(gs[1, 0])
    _draw_panel(
        ax_lv, brain_lv, brain_n_frames,
        xlabel='Frame Index (Left Ventricle)',
        ylabel='Dice Score',
        ylim=brain_ylim,
    )

    # Bottom right: brain RV (shared y-axis)
    ax_rv = fig.add_subplot(gs[1, 1], sharey=ax_lv)
    _draw_panel(
        ax_rv, brain_rv, brain_n_frames,
        xlabel='Frame Index (Right Ventricle)',
        ylim=brain_ylim,
    )
    plt.setp(ax_rv.get_yticklabels(), visible=False)

    # Centered title spanning both bottom panels
    # Get the x-center between ax_lv and ax_rv, and place above them
    lv_bbox = ax_lv.get_position()
    rv_bbox = ax_rv.get_position()
    center_x = (lv_bbox.x0 + rv_bbox.x1) / 2
    fig.text(center_x, lv_bbox.y1 + 0.01, '(b) Longitudinal Brain Diseases Progression',
             ha='center', va='bottom', fontsize=8)

    # ── Save ──
    for ext in ['png', 'pdf', 'svg']:
        out_path = os.path.join(OUTPUT_DIR, f'combined_dice_boxplot.{ext}')
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")

    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()
