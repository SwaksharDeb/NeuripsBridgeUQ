#!/usr/bin/env python3
"""
Generate all 9 style combinations for the comparison figure.
Saves each as a separate PNG for visual comparison.

Usage:
    python uncertainty_sde_combined/compare_styles.py
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Reuse data loading from compare_uncertainty
sys.path.insert(0, os.path.dirname(__file__))
from compare_uncertainty import (
    load_curves, build_es_frame_summary, DEFAULT_BACKBONE_DIRS,
    C_ORACLE, C_UNC, C_RAND,
)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': '600',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.4,
})


# ============================================================
# Style definitions
# ============================================================

# σ-band styles (how the ±σ shading around the mean looks)
BAND_STYLES = {
    'A_gradient': {
        'label': 'A: Gradient fade',
        'draw': '_draw_band_gradient',
    },
    'B_edgelines': {
        'label': 'B: Thin edge lines',
        'draw': '_draw_band_edgelines',
    },
    'C_warm': {
        'label': 'C: Soft warm tint',
        'draw': '_draw_band_warm',
    },
}

# nAURC/nAUSE area styles (the region between oracle and uncertainty)
AREA_STYLES = {
    'D_solid': {
        'label': 'D: Soft solid fill',
        'draw': '_draw_area_solid',
    },
    'E_sparse_hatch': {
        'label': 'E: Sparse thin hatch',
        'draw': '_draw_area_sparse_hatch',
    },
    'F_dotted': {
        'label': 'F: Dotted pattern',
        'draw': '_draw_area_dotted',
    },
}


# ============================================================
# Drawing functions for σ-band
# ============================================================

def _draw_band_gradient(ax, x, mean, lo, hi):
    """Option A: Two nested bands for gradient effect."""
    # Outer band: ±1σ, very light
    ax.fill_between(x, lo, hi,
                    color='#A8CFF0', alpha=0.18, linewidth=0,
                    label='Uncertainty ($\\pm 1\\sigma$)')
    # Inner band: ±0.5σ, slightly stronger
    inner_lo = (mean + lo) / 2
    inner_hi = (mean + hi) / 2
    ax.fill_between(x, inner_lo, inner_hi,
                    color='#7BB3E8', alpha=0.25, linewidth=0)


def _draw_band_edgelines(ax, x, mean, lo, hi):
    """Option B: Solid fill with thin crisp edge lines."""
    ax.fill_between(x, lo, hi,
                    color='#85B7EB', alpha=0.22, linewidth=0,
                    label='Uncertainty ($\\pm 1\\sigma$)')
    ax.plot(x, lo, color=C_UNC, linewidth=0.5, alpha=0.35, zorder=2)
    ax.plot(x, hi, color=C_UNC, linewidth=0.5, alpha=0.35, zorder=2)


def _draw_band_warm(ax, x, mean, lo, hi):
    """Option C: Softer, warmer sky-blue tint."""
    ax.fill_between(x, lo, hi,
                    color='#B8D4F0', alpha=0.40, linewidth=0,
                    label='Uncertainty ($\\pm 1\\sigma$)')


# ============================================================
# Drawing functions for nAURC/nAUSE area
# ============================================================

def _draw_area_solid(ax, x, oracle, unc_mean):
    """Option D: Soft semi-transparent solid fill."""
    ax.fill_between(x, oracle, unc_mean,
                    color='#C8C0F0', alpha=0.22, linewidth=0,
                    label='nAURC area', zorder=1.5)


def _draw_area_sparse_hatch(ax, x, oracle, unc_mean):
    """Option E: Sparser, thinner diagonal hatch lines."""
    ax.fill_between(x, oracle, unc_mean,
                    facecolor='none', edgecolor='#9990CC', linewidth=0.0,
                    hatch='//', alpha=0.30, label='nAURC area', zorder=1.5)


def _draw_area_dotted(ax, x, oracle, unc_mean):
    """Option F: Dotted fill pattern."""
    ax.fill_between(x, oracle, unc_mean,
                    facecolor='none', edgecolor='#9990CC', linewidth=0.0,
                    hatch='...', alpha=0.35, label='nAURC area', zorder=1.5)


# Map names to functions
BAND_FUNCS = {
    'A_gradient': _draw_band_gradient,
    'B_edgelines': _draw_band_edgelines,
    'C_warm': _draw_band_warm,
}

AREA_FUNCS = {
    'D_solid': _draw_area_solid,
    'E_sparse_hatch': _draw_area_sparse_hatch,
    'F_dotted': _draw_area_dotted,
}


# ============================================================
# Plot one combination
# ============================================================

def plot_one_style(backbones, band_key, area_key, output_dir):
    """Generate a single comparison figure with the given style combo."""
    band_func = BAND_FUNCS[band_key]
    area_func = AREA_FUNCS[area_key]
    combo_name = f'{band_key}__{area_key}'

    n_cols = len(backbones)
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(5.0 * n_cols, 8.0),
                             constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    band_mult = 1.0

    for col, bb in enumerate(backbones):
        name = bb['name']
        rc = bb['rc_avg']
        sp = bb['sp_avg']

        # --- Top row: Risk-Coverage ---
        ax = axes[0, col]
        if rc is not None:
            x = rc['x']
            rc_lo = np.clip(rc['unc_mean'] - band_mult * rc['unc_std'],
                            rc['oracle'], rc['random'])
            rc_hi = np.clip(rc['unc_mean'] + band_mult * rc['unc_std'],
                            rc['oracle'], rc['random'])

            band_func(ax, x, rc['unc_mean'], rc_lo, rc_hi)
            area_func(ax, x, rc['oracle'], rc['unc_mean'])

            ax.plot(x, rc['oracle'], color=C_ORACLE, linewidth=2.0,
                    label='Oracle', zorder=3)
            ax.plot(x, rc['unc_mean'], color=C_UNC, linewidth=2.2,
                    label='Uncertainty (mean)', zorder=3)
            ax.plot(x, rc['random'], color=C_RAND, linewidth=1.6,
                    linestyle='--', label='Random', zorder=3)

            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Coverage (fraction of retained voxels)')
            if col == 0:
                ax.set_ylabel('Risk (cumulative mean error)')
            ax.set_title(f'{name}', fontsize=13, fontweight='600', pad=12)

            ax.text(0.96, 0.07, f'nAURC = {rc["metric"]:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=10, fontweight='500', color='#333',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              edgecolor='#bbb', alpha=0.90, linewidth=0.6))
        else:
            ax.set_title(f'{name}', fontsize=13, fontweight='600', pad=12)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='#999')

        # --- Bottom row: Sparsification ---
        ax2 = axes[1, col]
        if sp is not None:
            x = sp['x']
            sp_lo = np.clip(sp['unc_mean'] - band_mult * sp['unc_std'],
                            sp['oracle'], sp['random'])
            sp_hi = np.clip(sp['unc_mean'] + band_mult * sp['unc_std'],
                            sp['oracle'], sp['random'])

            band_func(ax2, x, sp['unc_mean'], sp_lo, sp_hi)
            area_func(ax2, x, sp['oracle'], sp['unc_mean'])

            ax2.plot(x, sp['oracle'], color=C_ORACLE, linewidth=2.0, zorder=3)
            ax2.plot(x, sp['unc_mean'], color=C_UNC, linewidth=2.2, zorder=3)
            ax2.plot(x, sp['random'], color=C_RAND, linewidth=1.6,
                     linestyle='--', zorder=3)

            ax2.set_xlim(x.min(), x.max())
            ax2.set_ylim(bottom=0)
            ax2.set_xlabel('Fraction of removed voxels')
            if col == 0:
                ax2.set_ylabel('Mean remaining error (MSE)')

            ax2.text(0.96, 0.93, f'nAUSE = {sp["metric"]:.3f}',
                     transform=ax2.transAxes, ha='right', va='top',
                     fontsize=10, fontweight='500', color='#333',
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                               edgecolor='#bbb', alpha=0.90, linewidth=0.6))
        else:
            ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes,
                     ha='center', va='center', fontsize=11, color='#999')

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=5,
                   frameon=True, fancybox=True, edgecolor='#bbb',
                   fontsize=10, bbox_to_anchor=(0.5, -0.02),
                   handlelength=2.2, columnspacing=1.8)

    # Row labels
    fig.text(-0.008, 0.73, 'Risk\u2013Coverage', fontsize=13, fontweight='600',
             rotation=90, va='center', ha='center', color='#333')
    fig.text(-0.008, 0.28, 'Sparsification', fontsize=13, fontweight='600',
             rotation=90, va='center', ha='center', color='#333')

    # Combo label at very top
    band_label = BAND_STYLES[band_key]['label']
    area_label = AREA_STYLES[area_key]['label']
    fig.suptitle(f'{band_label}  +  {area_label}',
                 fontsize=11, fontweight='400', color='#666', y=1.01)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'style_{combo_name}.png')
    pdf_path = os.path.join(output_dir, f'style_{combo_name}.pdf')
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.18)
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.18)
    plt.close()
    return png_path


# ============================================================
# Main
# ============================================================

def main():
    output_dir = os.path.join(
        "/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/"
        "uncertainty_sde_combined/comparison_results/style_variants")

    # Load data
    print("Loading curves...")
    backbones = []
    for name, curves_dir in [
        ('VoxelMorph (VM)', DEFAULT_BACKBONE_DIRS['VoxelMorph (VM)']),
        ('LTMA',            DEFAULT_BACKBONE_DIRS['LTMA']),
        ('TLRN',            DEFAULT_BACKBONE_DIRS['TLRN']),
    ]:
        if not os.path.isdir(curves_dir):
            backbones.append({'name': name, 'rc_avg': None, 'sp_avg': None})
            continue
        spars, rc, es_spars, es_rc, num_subjects = load_curves(curves_dir)
        rc_avg = build_es_frame_summary(rc, es_rc, 'risk_coverage') if rc else None
        sp_avg = build_es_frame_summary(spars, es_spars, 'sparsification') if spars else None
        backbones.append({'name': name, 'rc_avg': rc_avg, 'sp_avg': sp_avg})

    # Generate all 9 combinations
    print(f"\nGenerating 9 style variants to: {output_dir}\n")
    for bk in BAND_FUNCS:
        for ak in AREA_FUNCS:
            path = plot_one_style(backbones, bk, ak, output_dir)
            bl = BAND_STYLES[bk]['label']
            al = AREA_STYLES[ak]['label']
            print(f"  {bl:25s} + {al:25s} -> {os.path.basename(path)}")

    print(f"\nDone! All 9 variants saved to:\n  {output_dir}")


if __name__ == "__main__":
    main()
