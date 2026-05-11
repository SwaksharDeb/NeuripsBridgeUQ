#!/usr/bin/env python3
"""
Publication-quality comparison figure: Risk-Coverage & Sparsification
across multiple registration backbones (VoxelMorph, LTMA, TLRN).

Loads pre-computed averaged curve .npz files from each backbone's
uncertainty_sde_combined/test_results/ directory.

Usage:
    python uncertainty_sde_combined/compare_uncertainty.py [--output_dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- style ----------
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
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.4,
})

# ---------- colors ----------
C_ORACLE = '#0F6E56'
C_UNC    = '#185FA5'
C_RAND   = '#B03030'
C_BAND   = '#85B7EB'
C_HATCH  = '#7F77DD'

# ---------- default paths ----------
BASE = "/scratch/swd9tc/Uncertanity_quantification"

DEFAULT_BACKBONE_DIRS = {
    'VoxelMorph (VM)': os.path.join(
        BASE, "voxel_and_R2R/voxelmorph/visualization_voxelmorph/"
              "visualization/uncertainty_sde_combined/test_results"),
    'LTMA': os.path.join(
        BASE, "LightingTemplate_LTMA/2026Experiments/CINE/outputs/"
              "TLMA_TGrad/config/visualization/uncertainty_sde_combined/test_results"),
    'TransMorph': os.path.join(
        BASE, "LightingTemplate_2/2026Experiments/CINE/outputs/TLRN/"
              "basic_MSE_Penp_img1200Reg0.03/visualization/"
              "uncertainty_sde_combined/test_results"),
    'TLRN': os.path.join(
        BASE, "LightingTemplate_2/2026Experiments/CINE/outputs/TLRN/"
              "basic_MSE_Penp_img1200Reg0.03/visualization/"
              "uncertainty_sde_combined/test_results"),
}


# ============================================================
# Data loading
# ============================================================

def load_curves(curves_dir):
    """Load pre-saved averaged curve data from .npz files.

    Returns:
        spars: dict {frame_idx -> {fractions, uncertainty_curve, oracle_curve,
                                    random_curve, ause_norm}}
        rc:    dict {frame_idx -> {coverages, unc_rc_curve, oracle_rc_curve,
                                    random_rc_curve, aurc_norm}}
        num_subjects: int
    """
    spars_path = os.path.join(curves_dir, 'avg_sparsification_curves.npz')
    rc_path = os.path.join(curves_dir, 'avg_risk_coverage_curves.npz')

    spars = {}
    es_spars = {}   # per-subject ES frame curves
    num_subjects = 0
    if os.path.exists(spars_path):
        data = np.load(spars_path, allow_pickle=True)
        num_subjects = int(data.get('num_subjects', 0))
        for fi in data['frame_indices']:
            fi = int(fi)
            spars[fi] = {
                'fractions': data[f'frame_{fi}_fractions'],
                'uncertainty_curve': data[f'frame_{fi}_uncertainty_curve'],
                'oracle_curve': data[f'frame_{fi}_oracle_curve'],
                'random_curve': data[f'frame_{fi}_random_curve'],
                'ause_norm': float(data[f'frame_{fi}_ause_norm']),
            }
        # Load ES frame per-subject curves (if available)
        if 'es_unc_per_subject' in data:
            es_spars = {
                'es_frame': int(data['es_frame']),
                'unc_per_subject': data['es_unc_per_subject'],
                'oracle_per_subject': data['es_oracle_per_subject'],
                'random_per_subject': data['es_random_per_subject'],
            }

    rc = {}
    es_rc = {}   # per-subject ES frame curves
    if os.path.exists(rc_path):
        data = np.load(rc_path, allow_pickle=True)
        if num_subjects == 0:
            num_subjects = int(data.get('num_subjects', 0))
        for fi in data['frame_indices']:
            fi = int(fi)
            rc[fi] = {
                'coverages': data[f'frame_{fi}_coverages'],
                'unc_rc_curve': data[f'frame_{fi}_unc_rc_curve'],
                'oracle_rc_curve': data[f'frame_{fi}_oracle_rc_curve'],
                'random_rc_curve': data[f'frame_{fi}_random_rc_curve'],
                'aurc_norm': float(data[f'frame_{fi}_aurc_norm']),
            }
        if 'es_unc_per_subject' in data:
            es_rc = {
                'es_frame': int(data['es_frame']),
                'unc_per_subject': data['es_unc_per_subject'],
                'oracle_per_subject': data['es_oracle_per_subject'],
                'random_per_subject': data['es_random_per_subject'],
            }

    return spars, rc, es_spars, es_rc, num_subjects


def build_es_frame_summary(per_frame, es_per_subject, curve_type='risk_coverage'):
    """Build plot data from the ES frame per-subject curves.

    Mean and std are both computed from the same ES frame across subjects,
    so the shading is consistent with the plotted mean curve.
    Falls back to all-frames average (no std) if ES data is unavailable.

    Args:
        per_frame: dict {frame_idx -> curve data} (averaged over subjects).
        es_per_subject: dict with per-subject arrays from the ES frame,
                        or empty dict if unavailable.
        curve_type: 'risk_coverage' or 'sparsification'.
    """
    frames = sorted(per_frame.keys())
    if not frames:
        return None

    if curve_type == 'risk_coverage':
        if es_per_subject and 'unc_per_subject' in es_per_subject:
            x = per_frame[frames[0]]['coverages']
            unc_subj = np.array(es_per_subject['unc_per_subject'])       # [N_subj, N_pts]
            oracle_subj = np.array(es_per_subject['oracle_per_subject'])
            random_subj = np.array(es_per_subject['random_per_subject'])

            avg_unc = unc_subj.mean(axis=0)
            avg_oracle = oracle_subj.mean(axis=0)
            avg_random = random_subj.mean(axis=0)

            # Normalized std (shape variability, not magnitude)
            subj_means = unc_subj.mean(axis=1, keepdims=True)  # [N_subj, 1]
            subj_means = np.clip(subj_means, 1e-12, None)
            normed = unc_subj / subj_means
            std_normed = normed.std(axis=0)
            std_unc = std_normed * avg_unc
        else:
            x = per_frame[frames[0]]['coverages']
            avg_unc = np.mean([per_frame[f]['unc_rc_curve'] for f in frames], axis=0)
            avg_oracle = np.mean([per_frame[f]['oracle_rc_curve'] for f in frames], axis=0)
            avg_random = np.mean([per_frame[f]['random_rc_curve'] for f in frames], axis=0)
            std_unc = np.zeros_like(avg_unc)

        aurc = np.trapz(avg_unc, x)
        oracle_aurc = np.trapz(avg_oracle, x)
        random_aurc = np.trapz(avg_random, x)
        naurc = (aurc - oracle_aurc) / (random_aurc - oracle_aurc) \
            if (random_aurc - oracle_aurc) > 1e-12 else 0.0

        return {
            'x': x,
            'unc_mean': avg_unc,
            'unc_std': std_unc,
            'oracle': avg_oracle,
            'random': avg_random,
            'metric': naurc,
            'metric_name': 'nAURC',
        }
    else:  # sparsification
        if es_per_subject and 'unc_per_subject' in es_per_subject:
            x = per_frame[frames[0]]['fractions']
            unc_subj = np.array(es_per_subject['unc_per_subject'])
            oracle_subj = np.array(es_per_subject['oracle_per_subject'])
            random_subj = np.array(es_per_subject['random_per_subject'])

            avg_unc = unc_subj.mean(axis=0)
            avg_oracle = oracle_subj.mean(axis=0)
            avg_random = random_subj.mean(axis=0)

            # Normalized std
            subj_means = unc_subj.mean(axis=1, keepdims=True)
            subj_means = np.clip(subj_means, 1e-12, None)
            normed = unc_subj / subj_means
            std_normed = normed.std(axis=0)
            std_unc = std_normed * avg_unc
        else:
            x = per_frame[frames[0]]['fractions']
            avg_unc = np.mean([per_frame[f]['uncertainty_curve'] for f in frames], axis=0)
            avg_oracle = np.mean([per_frame[f]['oracle_curve'] for f in frames], axis=0)
            avg_random = np.mean([per_frame[f]['random_curve'] for f in frames], axis=0)
            std_unc = np.zeros_like(avg_unc)

        ause = np.trapz(avg_unc - avg_oracle, x)
        random_area = np.trapz(avg_random - avg_oracle, x)
        nause = ause / random_area if random_area > 1e-12 else 0.0

        return {
            'x': x,
            'unc_mean': avg_unc,
            'unc_std': std_unc,
            'oracle': avg_oracle,
            'random': avg_random,
            'metric': nause,
            'metric_name': 'nAUSE',
        }


# ============================================================
# Plotting
# ============================================================

def plot_comparison(backbones, output_dir):
    """Create 2-row x N-col publication figure.

    Top row: Risk-Coverage, Bottom row: Sparsification.
    One column per backbone.
    """
    n_cols = len(backbones)
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(5.0 * n_cols, 8.0),
                             constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    band_mult = 1.0  # ±1σ

    for col, bb in enumerate(backbones):
        name = bb['name']
        rc = bb['rc_avg']
        sp = bb['sp_avg']

        # --- Top row: Risk-Coverage ---
        ax = axes[0, col]
        if rc is not None:
            x = rc['x']
            # Clip shading to stay between oracle and random
            rc_lo = np.clip(rc['unc_mean'] - band_mult * rc['unc_std'],
                            rc['oracle'], rc['random'])
            rc_hi = np.clip(rc['unc_mean'] + band_mult * rc['unc_std'],
                            rc['oracle'], rc['random'])

            # Layer 1: gradient σ-band (outer light + inner stronger)
            ax.fill_between(x, rc_lo, rc_hi,
                            color='#A8CFF0', alpha=0.18, linewidth=0,
                            label='Uncertainty ($\\pm 1\\sigma$)')
            rc_inner_lo = (rc['unc_mean'] + rc_lo) / 2
            rc_inner_hi = (rc['unc_mean'] + rc_hi) / 2
            ax.fill_between(x, rc_inner_lo, rc_inner_hi,
                            color='#7BB3E8', alpha=0.25, linewidth=0)
            # Layer 2: soft solid nAURC area
            ax.fill_between(x, rc['oracle'], rc['unc_mean'],
                            color='#C8C0F0', alpha=0.22, linewidth=0,
                            label='nAURC area', zorder=1.5)
            # Layer 3: curves on top
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

            # Gradient σ-band
            ax2.fill_between(x, sp_lo, sp_hi,
                             color='#A8CFF0', alpha=0.18, linewidth=0)
            sp_inner_lo = (sp['unc_mean'] + sp_lo) / 2
            sp_inner_hi = (sp['unc_mean'] + sp_hi) / 2
            ax2.fill_between(x, sp_inner_lo, sp_inner_hi,
                             color='#7BB3E8', alpha=0.25, linewidth=0)
            # Soft solid nAUSE area
            ax2.fill_between(x, sp['oracle'], sp['unc_mean'],
                             color='#C8C0F0', alpha=0.22, linewidth=0,
                             zorder=1.5)
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

    # --- Shared legend ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=5,
                   frameon=True, fancybox=True, edgecolor='#bbb',
                   fontsize=10, bbox_to_anchor=(0.5, -0.02),
                   handlelength=2.2, columnspacing=1.8)

    # Row labels (left margin)
    fig.text(-0.008, 0.73, 'Risk\u2013Coverage', fontsize=13, fontweight='600',
             rotation=90, va='center', ha='center', color='#333')
    fig.text(-0.008, 0.28, 'Sparsification', fontsize=13, fontweight='600',
             rotation=90, va='center', ha='center', color='#333')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, 'comparison_calibration.pdf')
    png_path = os.path.join(output_dir, 'comparison_calibration.png')
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.18)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.18)
    plt.close()
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare uncertainty calibration across registration backbones"
    )
    parser.add_argument("--vm_dir", type=str, default=DEFAULT_BACKBONE_DIRS['VoxelMorph (VM)'],
                        help="Path to VoxelMorph test_results dir with .npz files")
    parser.add_argument("--transmorph_dir", type=str, default=DEFAULT_BACKBONE_DIRS['TransMorph'],
                        help="Path to TransMorph test_results dir with .npz files")
    parser.add_argument("--ltma_dir", type=str, default=DEFAULT_BACKBONE_DIRS['LTMA'],
                        help="Path to LTMA test_results dir with .npz files")
    parser.add_argument("--tlrn_dir", type=str, default=DEFAULT_BACKBONE_DIRS['TLRN'],
                        help="Path to TLRN test_results dir with .npz files")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(BASE, "LightingTemplate_2/"
                                             "uncertainty_sde_combined/comparison_results"),
                        help="Directory to save comparison figure")
    args = parser.parse_args()

    # Define backbone order (left to right)
    backbone_configs = [
        ('VoxelMorph (VM)', args.vm_dir),
        ('TransMorph',      args.transmorph_dir),
        ('LTMA',            args.ltma_dir),
        ('TLRN',            args.tlrn_dir),
    ]

    print("=" * 60)
    print("Loading pre-computed curves for each backbone...")
    print("=" * 60)

    backbones = []
    for name, curves_dir in backbone_configs:
        print(f"\n  {name}: {curves_dir}")
        if not os.path.isdir(curves_dir):
            print(f"    WARNING: directory not found, skipping")
            backbones.append({
                'name': name, 'rc_avg': None, 'sp_avg': None,
                'num_subjects': 0,
            })
            continue

        spars, rc, es_spars, es_rc, num_subjects = load_curves(curves_dir)
        rc_avg = build_es_frame_summary(rc, es_rc, 'risk_coverage') if rc else None
        sp_avg = build_es_frame_summary(spars, es_spars, 'sparsification') if spars else None

        print(f"    Frames (RC): {sorted(rc.keys()) if rc else 'none'}")
        print(f"    Frames (SP): {sorted(spars.keys()) if spars else 'none'}")
        print(f"    Subjects: {num_subjects}")
        if rc_avg:
            print(f"    nAURC = {rc_avg['metric']:.4f}")
        if sp_avg:
            print(f"    nAUSE = {sp_avg['metric']:.4f}")

        backbones.append({
            'name': name,
            'rc_avg': rc_avg,
            'sp_avg': sp_avg,
            'num_subjects': num_subjects,
        })

    # Filter out backbones with no data at all
    has_data = [bb for bb in backbones if bb['rc_avg'] is not None or bb['sp_avg'] is not None]
    if not has_data:
        print("\nERROR: No curve data found for any backbone. "
              "Run each model's test_fast.py first.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Generating comparison figure...")
    print(f"{'=' * 60}")
    plot_comparison(backbones, args.output_dir)

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"{'COMPARISON SUMMARY':^60}")
    print(f"{'=' * 60}")
    print(f"{'Backbone':<20} {'nAURC':>10} {'nAUSE':>10} {'Subjects':>10}")
    print(f"{'-' * 60}")
    for bb in backbones:
        naurc = f"{bb['rc_avg']['metric']:.4f}" if bb['rc_avg'] else 'N/A'
        nause = f"{bb['sp_avg']['metric']:.4f}" if bb['sp_avg'] else 'N/A'
        print(f"{bb['name']:<20} {naurc:>10} {nause:>10} {bb['num_subjects']:>10}")
    print(f"{'=' * 60}")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
