"""
Orchestrator script that runs both SDE Bridge UQ and IR-SGMCMC tests
for a given batch/subject, then saves individual per-timepoint:
  - Risk-coverage curves
  - Sparsification plots
  - Uncertainty maps

Usage:
    cd /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2
    python -m uncertainty_sde_combined_acdc.run_both_tests --batch_idx 12 --subject_idx 5
"""

import os
import re
import sys
import argparse
import subprocess
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Matplotlib style (matching uncertainty_sde_velInput/compare_uncertainty.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
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

# Color palette (matching compare_uncertainty.py)
CLR_ORACLE = '#0F6E56'
CLR_SDE = '#185FA5'
CLR_IR = '#B03030'
CLR_VXM = '#2E8B57'          # sea green for DIF-VM
CLR_RANDOM = '#888888'
CLR_BAND_SDE = '#85B7EB'
CLR_BAND_IR = '#E8A0A0'
CLR_BAND_VXM = '#90D4A8'

# Sparsification-specific colors (distinct from risk-coverage)
CLR_SDE_SPARS = '#7B2D8E'   # purple
CLR_IR_SPARS = '#D4760A'    # orange
CLR_VXM_SPARS = '#2E8B57'   # sea green
CLR_BAND_SDE_SPARS = '#C9A0D8'
CLR_BAND_IR_SPARS = '#F0C888'
CLR_BAND_VXM_SPARS = '#90D4A8'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"
SDE_MODULE = "uncertainty_sde_combined_acdc.test_fast"
IR_SGMCMC_DIR = "/scratch/swd9tc/Uncertanity_quantification/ir_sgmcmc"
IR_SGMCMC_MODULE = "cardiac.test_cardiac"
VXM_PROB_DIR = "/scratch/swd9tc/Uncertanity_quantification/voxelmorph_prob"
VXM_PROB_SCRIPT = "test_cine.py"
PULPO_DIR = "/scratch/swd9tc/Uncertanity_quantification/pulbo"
PULPO_SCRIPT = "test.py"

BASE_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "uncertainty_sde_combined_acdc",
    "comparison_results",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SDE Bridge UQ and IR-SGMCMC tests, then save "
                    "individual per-timepoint visualizations."
    )
    parser.add_argument("--batch_idx", type=int, default=12,
                        help="Batch index in data loader (default: 12)")
    parser.add_argument("--subject_idx", type=int, default=13,
                        help="Subject index within batch (default: 5)")
    parser.add_argument("--num_runs_sde", type=int, default=100,
                        help="Number of UQ runs for SDE test (default: 3000)")
    parser.add_argument("--num_samples_ir", type=int, default=1000,
                        help="Number of posterior samples for IR-SGMCMC (default: 2000)")
    parser.add_argument("--use_train", action="store_true",
                        help="Use training set instead of test set")
    parser.add_argument("--sampling_method", type=str,
                        default="posterior_reverse_sde",
                        choices=["bridge_transition", "posterior_reverse_sde"],
                        help="SDE sampling method (default: posterior_reverse_sde)")
    parser.add_argument("--skip_sde", action="store_true",
                        help="Skip running SDE test (reuse existing results)")
    parser.add_argument("--skip_ir", action="store_true",
                        help="Skip running IR-SGMCMC test (reuse existing results)")
    parser.add_argument("--skip_vxm", action="store_true",
                        help="Skip running DIF-VM test (reuse existing results)")
    parser.add_argument("--num_samples_vxm", type=int, default=100,
                        help="Number of samples for DIF-VM (default: 100)")
    parser.add_argument("--skip_pulpo", action="store_true",
                        help="Skip running PULBO test (reuse existing results)")
    parser.add_argument("--num_samples_pulpo", type=int, default=100,
                        help="Number of posterior samples for PULBO (default: 100)")
    parser.add_argument("--frames", type=int, nargs='+', default=None,
                        help="Select specific frames to plot (e.g. --frames 1 3 5 7). "
                             "Default: all available frames.")
    # IR-SGMCMC hyperparameters
    parser.add_argument("--ir_lr_vi", type=float, default=0.01,
                        help="IR-SGMCMC: VI learning rate (default: 0.01)")
    parser.add_argument("--ir_w_reg", type=float, default=0.6,
                        help="IR-SGMCMC: regularization weight (default: 0.6)")
    parser.add_argument("--ir_sobolev_lambda", type=float, default=0.5,
                        help="IR-SGMCMC: Sobolev smoothing lambda (default: 1.0)")
    parser.add_argument("--ir_sobolev_s", type=int, default=3,
                        help="IR-SGMCMC: Sobolev order (default: 5)")
    parser.add_argument("--ir_no_iters_VI", type=int, default=1024,
                        help="IR-SGMCMC: number of VI iterations (default: 1024)")
    parser.add_argument("--ir_sigma_v_init", type=float, default=0.1,
                        help="IR-SGMCMC: initial sigma for var params (default: 0.1)")
    parser.add_argument("--ir_gmm_components", type=int, default=4,
                        help="IR-SGMCMC: GMM components (default: 4)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run sub-processes
# ---------------------------------------------------------------------------

def run_sde_test(args, sde_output_dir):
    """Run uncertainty_sde_combined_acdc/test_fast.py as a subprocess.

    test_fast.py processes ALL subjects and writes aggregate plots to a fixed
    location, so per-subject flags are not supported and are commented out below.
    """
    cmd = [
        sys.executable, "-m", SDE_MODULE,
        # "--batch_idx", str(args.batch_idx),         # not supported by test_fast.py (processes all subjects)
        # "--subject_idx", str(args.subject_idx),     # not supported by test_fast.py (processes all subjects)
        "--num_runs", str(args.num_runs_sde),
        "--output_dir", sde_output_dir,
        # "--save_individual",                         # not supported by test_fast.py (aggregate outputs only)
        "--sampling_method", args.sampling_method,
    ]
    if args.use_train:
        cmd.append("--use_train")

    print(f"\n{'='*70}")
    print("STEP 1: Running SDE Bridge UQ test")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output dir: {sde_output_dir}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"WARNING: SDE test exited with code {result.returncode}")
    return result.returncode


def run_ir_sgmcmc_test(args, ir_output_dir):
    """Run ir_sgmcmc/cardiac/test_cardiac.py as a subprocess."""
    cmd = [
        sys.executable, "-m", IR_SGMCMC_MODULE,
        "--batch_idx", str(args.batch_idx),
        "--subject_idx", str(args.subject_idx),
        "--num_samples", str(args.num_samples_ir),
        "--save_dir", ir_output_dir,
        "--save_individual",
        "--combine_acdc",
        "--lr_vi", str(args.ir_lr_vi),
        "--w_reg", str(args.ir_w_reg),
        "--sobolev_lambda", str(args.ir_sobolev_lambda),
        "--sobolev_s", str(args.ir_sobolev_s),
        "--no_iters_VI", str(args.ir_no_iters_VI),
        "--sigma_v_init", str(args.ir_sigma_v_init),
        "--gmm_components", str(args.ir_gmm_components),
    ]
    if args.use_train:
        cmd.append("--use_train")

    print(f"\n{'='*70}")
    print("STEP 2: Running IR-SGMCMC test")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output dir: {ir_output_dir}\n")

    result = subprocess.run(cmd, cwd=IR_SGMCMC_DIR)
    if result.returncode != 0:
        print(f"WARNING: IR-SGMCMC test exited with code {result.returncode}")
    return result.returncode


def run_vxm_prob_test(args, vxm_output_dir):
    """Run voxelmorph_prob/test_cine.py as a subprocess."""
    cmd = [
        sys.executable, VXM_PROB_SCRIPT,
        "--batch_idx", str(args.batch_idx),
        "--subject_idx", str(args.subject_idx),
        "--num_samples", str(args.num_samples_vxm),
        "--output_dir", vxm_output_dir,
        "--save_individual",
        "--direct",
    ]
    if args.use_train:
        cmd.append("--use_train")

    print(f"\n{'='*70}")
    print("STEP 3: Running DIF-VMabilistic test")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output dir: {vxm_output_dir}\n")

    result = subprocess.run(cmd, cwd=VXM_PROB_DIR)
    if result.returncode != 0:
        print(f"WARNING: DIF-VM test exited with code {result.returncode}")
    return result.returncode


def run_pulpo_test(args, pulpo_output_dir):
    """Run pulbo/test.py as a subprocess in eval_all (all subjects) mode.

    PULBO's aggregate npz files land in --cache_dir, so we point cache_dir at
    pulpo_output_dir directly so downstream comparison code can locate them.
    """
    cmd = [
        sys.executable, PULPO_SCRIPT,
        "--num_samples", str(args.num_samples_pulpo),
        "--output_dir", pulpo_output_dir,
        "--cache_dir", pulpo_output_dir,
        "--combine_acdc",
    ]

    print(f"\n{'='*70}")
    print("STEP 4: Running PULBO test")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output dir: {pulpo_output_dir}\n")

    result = subprocess.run(cmd, cwd=PULPO_DIR)
    if result.returncode != 0:
        print(f"WARNING: PULBO test exited with code {result.returncode}")
    return result.returncode


# ---------------------------------------------------------------------------
# Per-timepoint visualization helpers
# ---------------------------------------------------------------------------

def _find_pulpo_results(pulpo_output_dir):
    """Parse PULBO aggregate outputs from pulpo_output_dir (the --cache_dir it
    was launched with). PULBO writes avg_sparsification_curves.npz and
    avg_risk_coverage_curves.npz there (see pulbo/test.py:_render_final_aggregate_plots)."""
    results = {}
    spars_npz = os.path.join(pulpo_output_dir, "avg_sparsification_curves.npz")
    if os.path.exists(spars_npz):
        results["sparsification_npz"] = spars_npz
    rc_npz = os.path.join(pulpo_output_dir, "avg_risk_coverage_curves.npz")
    if os.path.exists(rc_npz):
        results["risk_coverage_npz"] = rc_npz
    return results


def _find_sde_results(sde_output_dir):
    """Parse SDE sparsification / risk-coverage from saved text + images."""
    results = {}

    # Read sparsification metrics
    spars_file = os.path.join(sde_output_dir, "sparsification_metrics.txt")
    if os.path.exists(spars_file):
        results["sparsification_metrics_file"] = spars_file

    rc_file = os.path.join(sde_output_dir, "risk_coverage_metrics.txt")
    if os.path.exists(rc_file):
        results["risk_coverage_metrics_file"] = rc_file

    # Per-frame uncertainty/error maps
    for f in sorted(glob.glob(os.path.join(sde_output_dir, "unc_err_maps_frame_*.png"))):
        frame_idx = int(os.path.basename(f).split("_")[-1].replace(".png", ""))
        results.setdefault("unc_err_maps", {})[frame_idx] = f

    # Individual images directory
    indiv_dir = os.path.join(sde_output_dir, "individual_images")
    if os.path.isdir(indiv_dir):
        results["individual_images_dir"] = indiv_dir
        for f in sorted(glob.glob(os.path.join(indiv_dir, "uncertainty_frame_*.png"))):
            parts = os.path.basename(f).replace(".png", "").split("_")
            frame_idx = int(parts[-1])
            results.setdefault("uncertainty_frames", {})[frame_idx] = f

    return results


def _find_ir_results(ir_output_dir):
    """Parse IR-SGMCMC results from save directory."""
    results = {}

    # Subject subdirectory (batch*_subject*)
    subdirs = sorted(glob.glob(os.path.join(ir_output_dir, "batch*_subject*")))
    if not subdirs:
        subdirs = [ir_output_dir]
    subj_dir = subdirs[0]
    results["subject_dir"] = subj_dir

    # Per-frame directories
    for fd in sorted(glob.glob(os.path.join(subj_dir, "frame_*"))):
        frame_idx = int(os.path.basename(fd).split("_")[1])
        results.setdefault("frame_dirs", {})[frame_idx] = fd

    # Training vis directory with test_vis_pair*.png
    tv_dir = os.path.join(subj_dir, "training_vis")
    if os.path.isdir(tv_dir):
        results["training_vis_dir"] = tv_dir

    # Aggregate curves
    for name in ["sparsification_curves.npz", "risk_coverage_curves.npz"]:
        p = os.path.join(subj_dir, name)
        if os.path.exists(p):
            results[name.replace(".npz", "")] = p

    return results


def _find_vxm_results(vxm_output_dir):
    """Parse DIF-VM results from output directory."""
    results = {}

    # Look for the subject subdirectory
    subdirs = sorted(glob.glob(os.path.join(vxm_output_dir, "batch*")))
    if subdirs:
        subj_dir = subdirs[0]
    else:
        subj_dir = vxm_output_dir
    results["subject_dir"] = subj_dir

    # Per-frame curves
    curves_dir = os.path.join(subj_dir, "per_frame_curves")
    if os.path.isdir(curves_dir):
        results["curves_dir"] = curves_dir

    # Individual images
    indiv_dir = os.path.join(subj_dir, "individual_images")
    if os.path.isdir(indiv_dir):
        results["individual_images_dir"] = indiv_dir
        for f in sorted(glob.glob(os.path.join(indiv_dir, "uncertainty_frame_*.png"))):
            idx = int(os.path.basename(f).replace(".png", "").split("_")[-1])
            results.setdefault("uncertainty_frames", {})[idx] = f

    return results


def _load_per_frame_npz(curves_dir, prefix, frame_idx):
    """Load a per-frame npz file. Returns NpzFile or None."""
    path = os.path.join(curves_dir, f"{prefix}_frame_{frame_idx:02d}.npz")
    if os.path.exists(path):
        return np.load(path)
    return None


def _extract_frame_from_aggregate(aggregate_npz, prefix, frame_idx):
    """Fallback loader: pull a single frame's curves out of test_fast.py's
    aggregate npz (avg_sparsification_curves.npz / avg_risk_coverage_curves.npz),
    which stores per-frame subject-averaged curves under keys
    'frame_{fi}_<suffix>'. Returns a dict with the same keys as the per-frame
    files, or None if the frame is not present."""
    if aggregate_npz is None:
        return None

    if prefix == "sparsification":
        suffixes = {
            'fractions': 'fractions',
            'uncertainty_curve': 'uncertainty_curve',
            'oracle_curve': 'oracle_curve',
            'random_curve': 'random_curve',
            'ause_norm': 'ause_norm',
        }
    elif prefix == "risk_coverage":
        suffixes = {
            'coverages': 'coverages',
            'unc_rc_curve': 'unc_rc_curve',
            'oracle_rc_curve': 'oracle_rc_curve',
            'random_rc_curve': 'random_rc_curve',
            'aurc_norm': 'aurc_norm',
        }
    else:
        return None

    out = {}
    for out_key, npz_suffix in suffixes.items():
        key = f'frame_{frame_idx}_{npz_suffix}'
        if key not in aggregate_npz.files:
            return None
        out[out_key] = np.asarray(aggregate_npz[key])
    return out


_LATEX_SUB_RE = re.compile(r'\$_\{([^}]+)\}\$')


def _plain_label(label):
    return _LATEX_SUB_RE.sub(r'_\1', label)


def _save_nausc_per_frame_txt(sp_methods, sp_frames, save_dir):
    """Write per-frame nAUSC values for each method to a .txt file.

    sp_methods: list of (label, sp_dict, color, fill) where sp_dict maps
        frame_idx -> npz with 'ause_norm' field.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "nausc_per_frame.txt")

    method_width = max(
        (len(_plain_label(label)) for label, *_ in sp_methods),
        default=len("Method"),
    )
    method_width = max(method_width, len("Method"))
    cell_width = 9

    header = f"{'Method':<{method_width}}  " + \
        "  ".join(f"t{f:<{cell_width - 1}}" for f in sp_frames) + \
        f"  {'mean':>{cell_width}}"

    lines = [
        "Per-frame nAUSC values",
        "=" * len(header),
        f"Frames: {list(sp_frames)}",
        "",
        header,
        "-" * len(header),
    ]

    for label, sp_dict, *_ in sp_methods:
        plain = _plain_label(label)
        cells, vals = [], []
        for f in sp_frames:
            d = sp_dict.get(f)
            keys = getattr(d, 'files', None) if d is not None else None
            if keys is None and isinstance(d, dict):
                keys = d.keys()
            if d is not None and keys is not None and 'ause_norm' in keys:
                v = float(d['ause_norm'])
                cells.append(f"{v:>{cell_width}.4f}")
                vals.append(v)
            else:
                cells.append(f"{'NA':>{cell_width}}")
        mean_str = f"{np.mean(vals):>{cell_width}.4f}" if vals else f"{'NA':>{cell_width}}"
        lines.append(f"{plain:<{method_width}}  " + "  ".join(cells) + f"  {mean_str}")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved per-frame nAUSC to: {out_path}")


def _save_legend(handles, labels, save_path):
    """Save a standalone 3-column legend as a separate image."""
    fig_leg = plt.figure(figsize=(6, 0.5))
    fig_leg.legend(handles, labels, loc='center', ncol=3,
                   frameon=True, fancybox=True, edgecolor='#bbb',
                   fontsize=10, handlelength=2.2, columnspacing=1.8)
    fig_leg.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig_leg)


def _plot_overlay_sparsification(sde_data, ir_data, frame_idx, save_path):
    """Plot overlaid sparsification curves for both methods at one frame.
    Each method draws its own oracle/random/uncertainty in the method's color."""
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    handles, labels = [], []

    if sde_data is not None:
        nause = float(sde_data["ause_norm"])
        h_sde_orc, = ax.plot(sde_data["fractions"], sde_data["oracle_curve"],
                             color=CLR_SDE_SPARS, linestyle=':', linewidth=1.6)
        h_sde_rnd, = ax.plot(sde_data["fractions"], sde_data["random_curve"],
                             color=CLR_SDE_SPARS, linestyle='--', linewidth=1.6)
        h_sde_unc, = ax.plot(sde_data["fractions"], sde_data["uncertainty_curve"],
                             color=CLR_SDE_SPARS, linestyle='-', linewidth=5.5)
        ax.fill_between(sde_data["fractions"], sde_data["oracle_curve"],
                        sde_data["uncertainty_curve"], alpha=0.15, color=CLR_BAND_SDE_SPARS)
        handles += [h_sde_unc, h_sde_orc, h_sde_rnd]
        labels += ["SDE Bridge", "SDE Oracle", "SDE Random"]

    if ir_data is not None:
        nause = float(ir_data["ause_norm"])
        h_ir_orc, = ax.plot(ir_data["fractions"], ir_data["oracle_curve"],
                            color=CLR_IR_SPARS, linestyle=':', linewidth=1.6)
        h_ir_rnd, = ax.plot(ir_data["fractions"], ir_data["random_curve"],
                            color=CLR_IR_SPARS, linestyle='--', linewidth=1.6)
        h_ir_unc, = ax.plot(ir_data["fractions"], ir_data["uncertainty_curve"],
                            color=CLR_IR_SPARS, linestyle='-', linewidth=5.5)
        ax.fill_between(ir_data["fractions"], ir_data["oracle_curve"],
                        ir_data["uncertainty_curve"], alpha=0.15, color=CLR_BAND_IR_SPARS)
        handles += [h_ir_unc, h_ir_orc, h_ir_rnd]
        labels += ["IR-SGMCMC", "IR Oracle", "IR Random"]

    ax.set_xlabel("Fraction of Removed Pixels")
    ax.set_ylabel("Mean Squared Error")

    parts = []
    if sde_data is not None:
        parts.append(f"SDE nAUSE={float(sde_data['ause_norm']):.3f}")
    if ir_data is not None:
        parts.append(f"IR nAUSE={float(ir_data['ause_norm']):.3f}")
    ax.set_title(" | ".join(parts), fontsize=12, fontweight='600')

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.18)
    svg_path = save_path.replace('.png', '.svg')
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)

    legend_path = save_path.replace('.png', '_legend.png')
    _save_legend(handles, labels, legend_path)


def _plot_overlay_risk_coverage(sde_data, ir_data, frame_idx, save_path):
    """Plot overlaid risk-coverage curves for both methods at one frame.
    Each method draws its own oracle/random/uncertainty in the method's color."""
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    handles, labels = [], []

    if sde_data is not None:
        naurc = float(sde_data["aurc_norm"])
        h_sde_orc, = ax.plot(sde_data["coverages"], sde_data["oracle_rc_curve"],
                             color=CLR_SDE, linestyle=':', linewidth=1.6)
        h_sde_rnd, = ax.plot(sde_data["coverages"], sde_data["random_rc_curve"],
                             color=CLR_SDE, linestyle='--', linewidth=1.6)
        h_sde_unc, = ax.plot(sde_data["coverages"], sde_data["unc_rc_curve"],
                             color=CLR_SDE, linestyle='-', linewidth=5.5)
        ax.fill_between(sde_data["coverages"], sde_data["oracle_rc_curve"],
                        sde_data["unc_rc_curve"], alpha=0.15, color=CLR_BAND_SDE)
        handles += [h_sde_unc, h_sde_orc, h_sde_rnd]
        labels += [f"SDE Bridge (nAURC={naurc:.3f})", "SDE Oracle", "SDE Random"]

    if ir_data is not None:
        naurc = float(ir_data["aurc_norm"])
        h_ir_orc, = ax.plot(ir_data["coverages"], ir_data["oracle_rc_curve"],
                            color=CLR_IR, linestyle=':', linewidth=1.6)
        h_ir_rnd, = ax.plot(ir_data["coverages"], ir_data["random_rc_curve"],
                            color=CLR_IR, linestyle='--', linewidth=1.6)
        h_ir_unc, = ax.plot(ir_data["coverages"], ir_data["unc_rc_curve"],
                            color=CLR_IR, linestyle='-', linewidth=5.5)
        ax.fill_between(ir_data["coverages"], ir_data["oracle_rc_curve"],
                        ir_data["unc_rc_curve"], alpha=0.15, color=CLR_BAND_IR)
        handles += [h_ir_unc, h_ir_orc, h_ir_rnd]
        labels += [f"IR-SGMCMC (nAURC={naurc:.3f})", "IR Oracle", "IR Random"]

    ax.set_xlabel("Coverage (fraction of retained pixels)")
    ax.set_ylabel("Risk (Cumulative Mean Error)")

    parts = []
    if sde_data is not None:
        parts.append(f"SDE nAURC={float(sde_data['aurc_norm']):.3f}")
    if ir_data is not None:
        parts.append(f"IR nAURC={float(ir_data['aurc_norm']):.3f}")
    ax.set_title(" | ".join(parts), fontsize=12, fontweight='600')

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.18)
    svg_path = save_path.replace('.png', '.svg')
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)

    legend_path = save_path.replace('.png', '_legend.png')
    _save_legend(handles, labels, legend_path)


def save_individual_timepoint_plots(sde_output_dir, ir_output_dir, save_dir,
                                    selected_frames=None, vxm_output_dir=None,
                                    pulpo_output_dir=None):
    """
    Load results from all methods and produce per-timepoint comparison plots:
      1. Uncertainty map (side-by-side: SDE vs IR-SGMCMC vs DIF-VM)
      2. Sparsification curve (all methods overlaid)
      3. Risk-coverage curve (all methods overlaid)
    """
    os.makedirs(save_dir, exist_ok=True)

    sde_res = _find_sde_results(sde_output_dir)
    ir_res = _find_ir_results(ir_output_dir)
    vxm_res = _find_vxm_results(vxm_output_dir) if vxm_output_dir else {}

    ir_subj_dir = ir_res.get("subject_dir", ir_output_dir)

    # ------------------------------------------------------------------
    # Per-frame curve directories
    # ------------------------------------------------------------------
    sde_curves_dir = os.path.join(sde_output_dir, "per_frame_curves")
    ir_curves_dir = os.path.join(ir_subj_dir, "per_frame_curves")
    vxm_subj_dir = vxm_res.get("subject_dir", vxm_output_dir or "")
    vxm_curves_dir = vxm_res.get("curves_dir", os.path.join(vxm_subj_dir, "per_frame_curves"))

    # ------------------------------------------------------------------
    # Load averaged sparsification/risk-coverage npz for summary plots
    # ------------------------------------------------------------------
    ir_spars_data = None
    ir_rc_data = None
    ir_spars_npz = os.path.join(ir_subj_dir, "sparsification_curves.npz")
    ir_rc_npz = os.path.join(ir_subj_dir, "risk_coverage_curves.npz")
    if os.path.exists(ir_spars_npz):
        ir_spars_data = np.load(ir_spars_npz)
    if os.path.exists(ir_rc_npz):
        ir_rc_data = np.load(ir_rc_npz)

    sde_spars_data = None
    sde_rc_data = None
    for name in ["avg_sparsification_curves.npz", "sparsification_curves.npz"]:
        p = os.path.join(sde_output_dir, name)
        if os.path.exists(p):
            sde_spars_data = np.load(p)
            break
    for name in ["avg_risk_coverage_curves.npz", "risk_coverage_curves.npz"]:
        p = os.path.join(sde_output_dir, name)
        if os.path.exists(p):
            sde_rc_data = np.load(p)
            break

    # PULBO aggregate npz (same layout as test_fast.py's avg_*_curves.npz)
    pulpo_spars_data = None
    pulpo_rc_data = None
    if pulpo_output_dir:
        pulpo_res = _find_pulpo_results(pulpo_output_dir)
        if pulpo_res.get("sparsification_npz"):
            pulpo_spars_data = np.load(pulpo_res["sparsification_npz"])
        if pulpo_res.get("risk_coverage_npz"):
            pulpo_rc_data = np.load(pulpo_res["risk_coverage_npz"])

    # ------------------------------------------------------------------
    # Determine all frame indices present in either method
    # ------------------------------------------------------------------
    all_frames = set()

    # SDE frames from unc_err_maps and individual uncertainty images
    sde_unc_frames = sde_res.get("unc_err_maps", {})
    all_frames.update(sde_unc_frames.keys())
    sde_unc_indiv = sde_res.get("uncertainty_frames", {})
    all_frames.update(sde_unc_indiv.keys())

    # IR frames from frame directories and individual images
    ir_frame_dirs = ir_res.get("frame_dirs", {})
    all_frames.update(ir_frame_dirs.keys())

    # Also check per_frame_curves dirs for frame indices
    for d in [sde_curves_dir, ir_curves_dir, vxm_curves_dir]:
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*_frame_*.npz")):
                idx = int(os.path.basename(f).split("_frame_")[1].replace(".npz", ""))
                all_frames.add(idx)

    # Frames from test_fast.py's aggregate npz (subject-averaged per-frame curves)
    for agg in (sde_spars_data, sde_rc_data, pulpo_spars_data, pulpo_rc_data):
        if agg is not None and 'frame_indices' in agg.files:
            all_frames.update(int(x) for x in np.asarray(agg['frame_indices']).tolist())

    # IR individual images
    ir_indiv_dir = os.path.join(ir_subj_dir, "individual_images")
    ir_unc_indiv = {}
    if os.path.isdir(ir_indiv_dir):
        for f in sorted(glob.glob(os.path.join(ir_indiv_dir, "uncertainty_frame_*.png"))):
            idx = int(os.path.basename(f).replace(".png", "").split("_")[-1])
            ir_unc_indiv[idx] = f
            all_frames.add(idx)

    if not all_frames:
        print("No per-frame results found from either method. Skipping per-timepoint plots.")
        return

    all_frames = sorted(all_frames)
    if selected_frames is not None:
        all_frames = [f for f in all_frames if f in selected_frames]
        print(f"\nFiltered to selected frames: {all_frames}")
    else:
        print(f"\nGenerating per-timepoint plots for frames: {all_frames}")

    for frame_idx in all_frames:
        frame_dir = os.path.join(save_dir, f"frame_{frame_idx:02d}")
        os.makedirs(frame_dir, exist_ok=True)

        # ==============================================================
        # 1. Uncertainty Map side-by-side comparison
        # ==============================================================
        sde_unc_path = sde_unc_indiv.get(frame_idx)
        ir_unc_path = ir_unc_indiv.get(frame_idx)
        sde_unc_err_path = sde_unc_frames.get(frame_idx)

        # Fallback: IR registration result image
        ir_reg_path = None
        ir_fdir = ir_frame_dirs.get(frame_idx)
        if ir_fdir and os.path.isdir(ir_fdir):
            cands = glob.glob(os.path.join(ir_fdir, "registration_result_*.png"))
            if cands:
                ir_reg_path = cands[0]

        has_sde_unc = sde_unc_path is not None or sde_unc_err_path is not None
        has_ir_unc = ir_unc_path is not None or ir_reg_path is not None

        if has_sde_unc or has_ir_unc:
            n_cols = int(has_sde_unc) + int(has_ir_unc)
            fig, axes = plt.subplots(1, max(n_cols, 1), figsize=(7 * max(n_cols, 1), 6))
            if n_cols == 1:
                axes = [axes]

            col = 0
            if has_sde_unc:
                img_path = sde_unc_path or sde_unc_err_path
                img = plt.imread(img_path)
                axes[col].imshow(img)
                axes[col].set_title("SDE Bridge UQ", fontsize=12, fontweight='600')
                axes[col].axis("off")
                col += 1
            if has_ir_unc:
                img_path = ir_unc_path or ir_reg_path
                img = plt.imread(img_path)
                axes[col].imshow(img)
                axes[col].set_title("IR-SGMCMC", fontsize=12, fontweight='600')
                axes[col].axis("off")

            plt.tight_layout()
            fig.savefig(os.path.join(frame_dir, f"uncertainty_map_frame_{frame_idx:02d}.png"),
                        bbox_inches="tight", pad_inches=0.18)
            fig.savefig(os.path.join(frame_dir, f"uncertainty_map_frame_{frame_idx:02d}.svg"),
                        bbox_inches="tight", pad_inches=0.18)
            plt.close()

        # ==============================================================
        # 2. Overlay sparsification curves for this frame
        # ==============================================================
        sde_spars_frame = _load_per_frame_npz(sde_curves_dir, "sparsification", frame_idx)
        ir_spars_frame = _load_per_frame_npz(ir_curves_dir, "sparsification", frame_idx)

        if sde_spars_frame is not None or ir_spars_frame is not None:
            _plot_overlay_sparsification(
                sde_spars_frame, ir_spars_frame, frame_idx,
                os.path.join(frame_dir, f"sparsification_overlay_frame_{frame_idx:02d}.png")
            )

        # ==============================================================
        # 3. Overlay risk-coverage curves for this frame
        # ==============================================================
        sde_rc_frame = _load_per_frame_npz(sde_curves_dir, "risk_coverage", frame_idx)
        ir_rc_frame = _load_per_frame_npz(ir_curves_dir, "risk_coverage", frame_idx)

        if sde_rc_frame is not None or ir_rc_frame is not None:
            _plot_overlay_risk_coverage(
                sde_rc_frame, ir_rc_frame, frame_idx,
                os.path.join(frame_dir, f"risk_coverage_overlay_frame_{frame_idx:02d}.png")
            )

        # ==============================================================
        # 4. Copy standalone per-frame images
        # ==============================================================
        if sde_unc_path:
            _copy_file(sde_unc_path,
                       os.path.join(frame_dir, f"sde_uncertainty_frame_{frame_idx:02d}.png"))
        if sde_unc_err_path:
            _copy_file(sde_unc_err_path,
                       os.path.join(frame_dir, f"sde_unc_err_maps_frame_{frame_idx:02d}.png"))
        if ir_unc_path:
            _copy_file(ir_unc_path,
                       os.path.join(frame_dir, f"ir_uncertainty_frame_{frame_idx:02d}.png"))
        if ir_reg_path:
            _copy_file(ir_reg_path,
                       os.path.join(frame_dir, f"ir_registration_result_frame_{frame_idx:02d}.png"))

        ir_tv_dir = ir_res.get("training_vis_dir")
        if ir_tv_dir:
            tv_path = os.path.join(ir_tv_dir, f"test_vis_pair{frame_idx - 1:04d}.png")
            if os.path.exists(tv_path):
                _copy_file(tv_path,
                           os.path.join(frame_dir, f"ir_test_vis_frame_{frame_idx:02d}.png"))

    # ------------------------------------------------------------------
    # 4b. Publication risk-coverage grid: 2 rows (method) x N cols (frame)
    #     Standard format: 3 curves per panel (Oracle green, Unc blue, Random red dashed)
    #     Single fill_between for nAURC area, nAURC value in bottom-right box
    # ------------------------------------------------------------------
    sde_rc_all = {}
    ir_rc_all = {}
    vxm_rc_all = {}
    pulpo_rc_all = {}
    for frame_idx in all_frames:
        # Prefer test_fast.py's subject-averaged per-frame curves (from aggregate npz)
        # over any stale per_frame_curves/*.npz files from a prior test.py run.
        sde_rc_f = _extract_frame_from_aggregate(sde_rc_data, "risk_coverage", frame_idx)
        if sde_rc_f is None:
            sde_rc_f = _load_per_frame_npz(sde_curves_dir, "risk_coverage", frame_idx)
        if sde_rc_f is not None:
            sde_rc_all[frame_idx] = sde_rc_f
        ir_rc_f = _load_per_frame_npz(ir_curves_dir, "risk_coverage", frame_idx)
        if ir_rc_f is not None:
            ir_rc_all[frame_idx] = ir_rc_f
        vxm_rc_f = _load_per_frame_npz(vxm_curves_dir, "risk_coverage", frame_idx)
        if vxm_rc_f is not None:
            vxm_rc_all[frame_idx] = vxm_rc_f
        pulpo_rc_f = _extract_frame_from_aggregate(pulpo_rc_data, "risk_coverage", frame_idx)
        if pulpo_rc_f is not None:
            pulpo_rc_all[frame_idx] = pulpo_rc_f

    if sde_rc_all or ir_rc_all or vxm_rc_all or pulpo_rc_all:
        rc_frames = sorted(set(sde_rc_all.keys()) | set(ir_rc_all.keys())
                           | set(vxm_rc_all.keys()) | set(pulpo_rc_all.keys()))
        n_frames = len(rc_frames)

        method_rows = [
            (r'BridgeUQ$_{VM}$', sde_rc_all, '#185FA5', '#85B7EB'),   # blue unc, light blue fill
            ('IR-SGMCMC', ir_rc_all, '#B03030', '#E8A0A0'),          # red unc, light red fill
            ('DIF-VM', vxm_rc_all, '#2E8B57', '#90D4A8'),          # green unc, light green fill
            ('PULBO', pulpo_rc_all, '#8A4FBF', '#C9A0D8'),           # purple unc, light purple fill
        ]
        method_rows = [r for r in method_rows if r[1]]
        n_rows = len(method_rows)

        fig, axes = plt.subplots(n_rows, n_frames,
                                 figsize=(2.8 * n_frames, 2.8 * n_rows),
                                 squeeze=False)

        # Determine global y scale factor from all data
        all_y_vals = []
        for _, rc_dict, _, _ in method_rows:
            for fidx in rc_frames:
                d = rc_dict.get(fidx)
                if d is not None:
                    all_y_vals.extend(d['random_rc_curve'].tolist())
        y_max = max(all_y_vals) if all_y_vals else 1.0
        if y_max > 0:
            y_exp = int(np.floor(np.log10(y_max)))
        else:
            y_exp = 0
        y_scale = 10 ** y_exp

        for row, (mname, rc_dict, c_unc, c_fill) in enumerate(method_rows):
            for col, fidx in enumerate(rc_frames):
                ax = axes[row, col]
                d = rc_dict.get(fidx)

                if d is not None:
                    x = d['coverages']
                    unc = d['unc_rc_curve'] / y_scale
                    orc = d['oracle_rc_curve'] / y_scale
                    rnd = d['random_rc_curve'] / y_scale
                    naurc = float(d['aurc_norm'])

                    # nAURC shaded area with hatching
                    ax.fill_between(x, orc, unc, alpha=0.12, color=c_fill,
                                    hatch='///', edgecolor=c_unc, linewidth=0)

                    # 3 standard curves
                    ax.plot(x, orc, color='black', linestyle=':', linewidth=2, label='Oracle')
                    ax.plot(x, unc, color=c_unc, linestyle='-', linewidth=4,
                            label=f'{mname}')
                    ax.plot(x, rnd, color='r', linestyle='--', linewidth=1.5,
                            marker='x', markersize=5, markevery=8,
                            label='Random', zorder=2)

                    ax.set_xlim(0, 1)
                    ax.set_ylim(bottom=0)

                    # "Random" label box on the random line (use data coords for y)
                    rnd_mid = rnd[len(rnd) // 2]
                    ax.annotate('Random', xy=(0.5, rnd_mid), xycoords=('axes fraction', 'data'),
                                ha='center', va='center',
                                fontsize=11, fontweight='600', color='red',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='#F0F0F0', edgecolor='red',
                                          alpha=1.0, linewidth=0.8))

                    # nAURC value in the middle of the shaded area
                    mid_idx = len(x) // 2
                    naurc_y = (orc[mid_idx] + unc[mid_idx]) / 2
                    ax.annotate(f'nAURC = {naurc:.3f}',
                                xy=(0.5, naurc_y), xycoords=('axes fraction', 'data'),
                                ha='center', va='center',
                                fontsize=15, fontweight='500', color='#333',
                                bbox=dict(boxstyle='round,pad=0.35',
                                          facecolor='white', edgecolor='#bbb',
                                          alpha=0.90, linewidth=0.6))
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                            ha='center', va='center', fontsize=10, color='#999')

                # Format y-ticks: strip leading zero (0.5 → .5)
                from matplotlib.ticker import FuncFormatter
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda v, _: f'.{str(round(v, 2)).split(".")[-1]}' if 0 < v < 1
                    else ('0' if v == 0 else f'{v:g}')
                ))

                # Scale exponent annotation on leftmost column
                if col == 0 and y_exp != 0:
                    if row == 0:
                        y_pos = 0.85
                    elif row == n_rows - 1:
                        y_pos = 0.90  # DIF-VM (last row): slightly higher
                    else:
                        y_pos = 0.85  # IR-SGMCMC
                    ax.text(0.03, y_pos, r'$\times 10^{' + str(y_exp) + r'}$',
                            transform=ax.transAxes, fontsize=12,
                            va='top', ha='left', color='#333')

                # Frame title on top row
                if row == 0:
                    ax.set_title(f'$t = {fidx}$', fontsize=18, fontweight='600',
                                 pad=8)

                # Ticks only on first column and bottom row
                if col == 0 and row == n_rows - 1:
                    ax.set_xlabel('Coverage', fontsize=16)
                    ax.tick_params(axis='both', labelsize=13)
                elif col == 0:
                    ax.tick_params(axis='y', labelsize=13)
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(labelleft=False, labelbottom=False)

                # Grid
                ax.grid(True, alpha=0.3)

        # Row labels on the right margin
        for row, (mname, _, _, _) in enumerate(method_rows):
            axes[row, -1].annotate(
                mname, xy=(1.04, 0.5), xycoords='axes fraction',
                fontsize=18, rotation=-90,
                ha='left', va='center')

        # Shared legend: collect unique entries from all rows
        all_handles, all_labels = [], []
        seen = set()
        for row in range(n_rows):
            h, l = axes[row, 0].get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in seen:
                    all_handles.append(hi)
                    all_labels.append(li)
                    seen.add(li)
        if all_handles:
            fig.legend(all_handles, all_labels, loc='lower center',
                       ncol=len(all_handles), frameon=True, fancybox=True,
                       edgecolor='#bbb', fontsize=14,
                       bbox_to_anchor=(0.58, 0.02),
                       handlelength=2.2, columnspacing=1.8)

        fig.subplots_adjust(left=0.05, right=0.92, bottom=0.08, top=0.90,
                            wspace=0.05, hspace=0.12)

        # Single shared y-label for both rows
        fig.text(0.005, 0.5, 'Risk (cumulative mean error)', fontsize=16,
                 rotation=90, va='center', ha='left')

        p = os.path.join(save_dir, "risk_coverage_all_frames.png")
        fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.15)
        fig.savefig(p.replace('.png', '.svg'), bbox_inches='tight', pad_inches=0.15)
        fig.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        print(f"  All-frames risk-coverage saved: {p}")

    # ------------------------------------------------------------------
    # 4b2. Publication sparsification grid: 1 row x N cols
    #      All methods overlaid in a single panel per frame
    # ------------------------------------------------------------------
    sde_sp_all = {}
    ir_sp_all = {}
    vxm_sp_all = {}
    pulpo_sp_all = {}
    for frame_idx in all_frames:
        # Prefer test_fast.py's subject-averaged per-frame curves (from aggregate npz)
        # over any stale per_frame_curves/*.npz files from a prior test.py run.
        sde_sp_f = _extract_frame_from_aggregate(sde_spars_data, "sparsification", frame_idx)
        if sde_sp_f is None:
            sde_sp_f = _load_per_frame_npz(sde_curves_dir, "sparsification", frame_idx)
        if sde_sp_f is not None:
            sde_sp_all[frame_idx] = sde_sp_f
        ir_sp_f = _load_per_frame_npz(ir_curves_dir, "sparsification", frame_idx)
        if ir_sp_f is not None:
            ir_sp_all[frame_idx] = ir_sp_f
        vxm_sp_f = _load_per_frame_npz(vxm_curves_dir, "sparsification", frame_idx)
        if vxm_sp_f is not None:
            vxm_sp_all[frame_idx] = vxm_sp_f
        pulpo_sp_f = _extract_frame_from_aggregate(pulpo_spars_data, "sparsification", frame_idx)
        if pulpo_sp_f is not None:
            pulpo_sp_all[frame_idx] = pulpo_sp_f

    if sde_sp_all or ir_sp_all or vxm_sp_all or pulpo_sp_all:
        sp_frames = sorted(set(sde_sp_all.keys())
                           | set(ir_sp_all.keys())
                           | set(vxm_sp_all.keys())
                           | set(pulpo_sp_all.keys()))
        n_frames = len(sp_frames)

        sp_methods = [
            (r'BridgeUQ$_{VM}$', sde_sp_all, '#185FA5', '#85B7EB'),
            ('IR-SGMCMC',        ir_sp_all,  '#B03030', '#E8A0A0'),
            ('DIF-VM',           vxm_sp_all, '#2E8B57', '#90D4A8'),
            ('PULBO',            pulpo_sp_all, '#8A4FBF', '#C9A0D8'),
        ]
        sp_methods = [m for m in sp_methods if m[1]]

        _save_nausc_per_frame_txt(sp_methods, sp_frames, save_dir)

        fig, axes = plt.subplots(1, n_frames,
                                 figsize=(2.8 * n_frames, 3.2),
                                 squeeze=False)
        axes = axes[0]

        from matplotlib.ticker import FuncFormatter

        # ProbFlow-style normalization: divide each method's curves by its own
        # mean error at fraction=0 so every curve starts at 1.0.
        for col, fidx in enumerate(sp_frames):
            ax = axes[col]
            for mname, sp_dict, c_unc, c_fill in sp_methods:
                d = sp_dict.get(fidx)
                if d is None:
                    continue
                x = np.asarray(d['fractions'])
                unc_raw = np.asarray(d['uncertainty_curve'])
                orc_raw = np.asarray(d['oracle_curve'])
                rnd_raw = np.asarray(d['random_curve'])

                # Trim trailing "empty set" artifact: the last sample often has
                # any of {random, oracle, uncertainty} collapsing to 0 (e.g.
                # IR-SGMCMC's random drops to 0 at x=0.9; SDE/VXM drop all to 0
                # at x=1.0). That causes a spurious near-vertical line.
                while (x.size > 2 and
                       (rnd_raw[-1] == 0 or unc_raw[-1] == 0) and
                       (rnd_raw[-2] != 0 and unc_raw[-2] != 0)):
                    x = x[:-1]
                    unc_raw = unc_raw[:-1]
                    orc_raw = orc_raw[:-1]
                    rnd_raw = rnd_raw[:-1]

                # Reference = full-pixel mean error (all three curves share this
                # value at fraction=0 by construction; pick uncertainty[0])
                ref = float(unc_raw[0]) if unc_raw[0] != 0 else 1.0
                unc = unc_raw / ref
                orc = orc_raw / ref
                rnd = rnd_raw / ref
                nause = float(d['ause_norm'])

                # nAUSE area between oracle and uncertainty (method color)
                ax.fill_between(x, orc, unc, alpha=0.10, color=c_fill,
                                linewidth=0)
                # Oracle: black dotted (shared style across all methods)
                ax.plot(x, orc, color='black', linestyle=':', linewidth=1,
                        zorder=2)
                # Random: method color, dashed, thin
                ax.plot(x, rnd, color=c_unc, linestyle='--', linewidth=1.3,
                        alpha=0.75)
                # Uncertainty curve (solid, thick) — the focal line
                lw_unc = 3.2 if mname == r'BridgeUQ$_{VM}$' else 2.0
                ax.plot(x, unc, color=c_unc, linestyle='-', linewidth=lw_unc,
                        label=mname, zorder=3)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)

            # y-tick formatter: strip leading zero (0.5 -> .5)
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f'.{str(round(v, 2)).split(".")[-1]}' if 0 < v < 1
                else ('0' if v == 0 else f'{v:g}')
            ))

            # Frame title
            ax.set_title(f'$t = {fidx}$', fontsize=18, fontweight='600', pad=8)

            # x-label and tick labels only on first column
            if col == 0:
                ax.set_xlabel('Fraction of Removed Voxels', fontsize=16)
                ax.tick_params(axis='both', labelsize=13)
            else:
                ax.tick_params(labelleft=False, labelbottom=False)

            ax.grid(True, alpha=0.3)

        # Shared legend at the bottom: methods + shared Oracle entry
        all_handles, all_labels = [], []
        seen = set()
        for a in axes:
            h, l = a.get_legend_handles_labels()
            for hi, li in zip(h, l):
                key = li.split(' (nAUSE')[0]
                if key not in seen:
                    all_handles.append(hi)
                    all_labels.append(li)
                    seen.add(key)
        # Append Oracle (black dotted) — shared across all methods
        from matplotlib.lines import Line2D
        oracle_handle = Line2D([0], [0], color='black', linestyle=':', linewidth=1)
        all_handles.append(oracle_handle)
        all_labels.append('Oracle')
        if all_handles:
            fig.legend(all_handles, all_labels, loc='lower center',
                       ncol=len(all_handles), frameon=True, fancybox=True,
                       edgecolor='#bbb', fontsize=13,
                       bbox_to_anchor=(0.58, 0.06),
                       handlelength=2.2, columnspacing=1.8)

        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.22, top=0.88,
                            wspace=0.06)

        # Single shared y-label
        fig.text(0.005, 0.5, 'MSE (normalized)', fontsize=16,
                 rotation=90, va='center', ha='left')

        p = os.path.join(save_dir, "sparsification_all_frames.png")
        fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.15)
        fig.savefig(p.replace('.png', '.svg'), bbox_inches='tight', pad_inches=0.15)
        fig.savefig(p.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        print(f"  All-frames sparsification saved: {p}")

    # ------------------------------------------------------------------
    # 4c. Per-frame nAURC plot (test_all_models.py style)
    #     One subplot per method, nAURC vs frame index with triangle markers
    # ------------------------------------------------------------------
    if sde_rc_all or ir_rc_all or vxm_rc_all or pulpo_rc_all:
        method_naurc = []  # list of (name, color, marker, {frame: naurc})
        for method_name, rc_dict, color, marker in [
            ("SDE Bridge", sde_rc_all, CLR_SDE, 's'),
            ("IR-SGMCMC", ir_rc_all, CLR_IR, '^'),
            ("DIF-VM", vxm_rc_all, CLR_VXM, 'D'),
            ("PULBO", pulpo_rc_all, '#8A4FBF', 'o'),
        ]:
            if not rc_dict:
                continue
            naurcs = {fidx: float(rc_dict[fidx]['aurc_norm']) for fidx in rc_dict}
            method_naurc.append((method_name, color, marker, naurcs))

        if method_naurc:
            plot_frames = sorted(
                set().union(*(d.keys() for _, _, _, d in method_naurc))
            )
            frame_labels = [str(f) for f in plot_frames]
            x = np.arange(len(plot_frames))

            # Compute shared y-limits
            all_vals = [v for _, _, _, d in method_naurc for v in d.values()]
            data_min = min(all_vals) if all_vals else 0.0
            data_max = max(all_vals + [1.0])
            y_range = data_max - data_min
            y_lo = data_min - y_range * 0.15
            y_hi = data_max + y_range * 0.10

            n_methods = len(method_naurc)

            # --- Combined subplot figure ---
            fig_sub, axes_sub = plt.subplots(n_methods, 1,
                                             figsize=(6, 4 * n_methods),
                                             sharey=True, sharex=True)
            if n_methods == 1:
                axes_sub = [axes_sub]

            for idx, (mname, color, marker, naurcs) in enumerate(method_naurc):
                naurc_vals = np.array([naurcs.get(f, 0.0) for f in plot_frames])

                # Subplot panel
                ax_s = axes_sub[idx]
                ax_s.plot(x, naurc_vals, marker=marker, linestyle='-', color=color,
                          linewidth=4, markersize=16, markeredgecolor='black',
                          markeredgewidth=1.2)
                ax_s.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax_s.set_ylim(y_lo, y_hi)
                ax_s.set_xticks(x)
                ax_s.set_xticklabels(frame_labels)
                if idx == n_methods - 1:
                    ax_s.set_xlabel(r'Frame Index ($t$)', fontsize=20)
                ax_s.set_ylabel('nAURC', fontsize=20)
                ax_s.set_title(mname, fontsize=26, fontweight='bold')
                ax_s.tick_params(axis='both', labelsize=18)
                ax_s.grid(True, alpha=0.3)
                for spine in ax_s.spines.values():
                    spine.set_linewidth(2.5)

                # Individual figure
                fig_m, ax_m = plt.subplots(figsize=(6, 4))
                ax_m.plot(x, naurc_vals, marker=marker, linestyle='-', color=color,
                          linewidth=4, markersize=16, markeredgecolor='black',
                          markeredgewidth=1.2)
                ax_m.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax_m.set_ylim(y_lo, y_hi)
                ax_m.set_xticks(x)
                ax_m.set_xticklabels(frame_labels)
                ax_m.set_xlabel(r'Frame Index ($t$)', fontsize=20)
                ax_m.set_ylabel('nAURC', fontsize=20)
                ax_m.set_title(mname, fontsize=26, fontweight='bold')
                ax_m.tick_params(axis='both', labelsize=18)
                ax_m.grid(True, alpha=0.3)
                for spine in ax_m.spines.values():
                    spine.set_linewidth(2.5)
                plt.tight_layout()
                tag = mname.lower().replace('-', '_').replace(' ', '_')
                mp = os.path.join(save_dir, f"naurc_{tag}.png")
                plt.savefig(mp, dpi=300, bbox_inches='tight', facecolor='white')
                plt.savefig(mp.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
                plt.savefig(mp.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
                plt.close(fig_m)
                print(f"  nAURC ({mname}) saved: {mp}")

            fig_sub.tight_layout()
            combined_path = os.path.join(save_dir, "naurc_comparison.png")
            fig_sub.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
            fig_sub.savefig(combined_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
            fig_sub.savefig(combined_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.close(fig_sub)
            print(f"  nAURC combined saved: {combined_path}")

            # --- Single overlay plot with both methods ---
            fig_ov, ax_ov = plt.subplots(figsize=(8, 5))
            for mname, color, marker, naurcs in method_naurc:
                naurc_vals = np.array([naurcs.get(f, 0.0) for f in plot_frames])
                ax_ov.plot(x, naurc_vals, marker=marker, linestyle='-', color=color,
                           linewidth=4, markersize=16, markeredgecolor='black',
                           markeredgewidth=1.2, label=mname)
            ax_ov.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax_ov.set_ylim(y_lo, y_hi)
            ax_ov.set_xticks(x)
            ax_ov.set_xticklabels(frame_labels)
            ax_ov.set_xlabel(r'Frame Index ($t$)', fontsize=20)
            ax_ov.set_ylabel('nAURC', fontsize=20)
            ax_ov.tick_params(axis='both', labelsize=18)
            ax_ov.legend(fontsize=16, framealpha=0.9, edgecolor='gray', loc='lower right')
            ax_ov.grid(True, alpha=0.3)
            for spine in ax_ov.spines.values():
                spine.set_linewidth(2.5)
            plt.tight_layout()
            ov_path = os.path.join(save_dir, "naurc_both_methods.png")
            plt.savefig(ov_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(ov_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
            plt.savefig(ov_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.close(fig_ov)
            print(f"  nAURC both methods saved: {ov_path}")

    # ------------------------------------------------------------------
    # 5. Averaged sparsification comparison (both methods on one plot)
    # ------------------------------------------------------------------
    if (sde_spars_data is not None or ir_spars_data is not None
            or pulpo_spars_data is not None):
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        handles, labels = [], []

        if sde_spars_data is not None:
            frac = sde_spars_data["fractions"]
            nause = float(sde_spars_data['avg_nause'])
            ax.plot(frac, sde_spars_data["avg_oracle"], color=CLR_SDE, linestyle=':', linewidth=1.6)
            ax.plot(frac, sde_spars_data["avg_random"], color=CLR_SDE, linestyle='--', linewidth=1.6)
            h_sde, = ax.plot(frac, sde_spars_data["avg_unc"], color=CLR_SDE, linewidth=2.2)
            ax.fill_between(frac, sde_spars_data["avg_oracle"], sde_spars_data["avg_unc"],
                            alpha=0.15, color=CLR_BAND_SDE)
            handles.append(h_sde)
            labels.append("SDE Bridge")

        if ir_spars_data is not None:
            frac_ir = ir_spars_data["fractions"]
            nause = float(ir_spars_data['avg_nause'])
            ax.plot(frac_ir, ir_spars_data["avg_oracle"], color=CLR_IR, linestyle=':', linewidth=1.6)
            ax.plot(frac_ir, ir_spars_data["avg_random"], color=CLR_IR, linestyle='--', linewidth=1.6)
            h_ir, = ax.plot(frac_ir, ir_spars_data["avg_unc"], color=CLR_IR, linewidth=2.2)
            ax.fill_between(frac_ir, ir_spars_data["avg_oracle"], ir_spars_data["avg_unc"],
                            alpha=0.15, color=CLR_BAND_IR)
            handles.append(h_ir)
            labels.append("IR-SGMCMC")

        if pulpo_spars_data is not None:
            frac_p = pulpo_spars_data["fractions"]
            nause = float(pulpo_spars_data['avg_nause'])
            ax.plot(frac_p, pulpo_spars_data["avg_oracle"], color='#8A4FBF', linestyle=':', linewidth=1.6)
            ax.plot(frac_p, pulpo_spars_data["avg_random"], color='#8A4FBF', linestyle='--', linewidth=1.6)
            h_p, = ax.plot(frac_p, pulpo_spars_data["avg_unc"], color='#8A4FBF', linewidth=2.2)
            ax.fill_between(frac_p, pulpo_spars_data["avg_oracle"], pulpo_spars_data["avg_unc"],
                            alpha=0.15, color='#C9A0D8')
            handles.append(h_p)
            labels.append("PULBO")

        ax.set_xlabel("Fraction of Removed Pixels")
        ax.set_ylabel("Mean Squared Error")

        parts = []
        if sde_spars_data is not None:
            parts.append(f"SDE nAUSE={float(sde_spars_data['avg_nause']):.3f}")
        if ir_spars_data is not None:
            parts.append(f"IR nAUSE={float(ir_spars_data['avg_nause']):.3f}")
        if pulpo_spars_data is not None:
            parts.append(f"PULBO nAUSE={float(pulpo_spars_data['avg_nause']):.3f}")
        ax.set_title(" | ".join(parts), fontsize=12, fontweight='600')

        fig.savefig(os.path.join(save_dir, "sparsification_comparison_avg.png"),
                    bbox_inches="tight", pad_inches=0.18)
        fig.savefig(os.path.join(save_dir, "sparsification_comparison_avg.svg"),
                    bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
        _save_legend(handles, labels,
                     os.path.join(save_dir, "sparsification_comparison_avg_legend.png"))

        # Per-frame nAUSE bar comparison
        sde_per_nause = sde_spars_data.get("per_frame_nause") if sde_spars_data is not None else None
        ir_per_nause = ir_spars_data.get("per_frame_nause") if ir_spars_data is not None else None
        pulpo_per_nause = pulpo_spars_data.get("per_frame_nause") if pulpo_spars_data is not None else None
        if sde_per_nause is not None or ir_per_nause is not None or pulpo_per_nause is not None:
            fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
            pf_handles, pf_labels = [], []
            if sde_per_nause is not None:
                h, = ax.plot(range(1, len(sde_per_nause) + 1), sde_per_nause,
                             color=CLR_SDE, marker='o', linewidth=2.2)
                pf_handles.append(h); pf_labels.append("SDE Bridge")
            if ir_per_nause is not None:
                h, = ax.plot(range(1, len(ir_per_nause) + 1), ir_per_nause,
                             color=CLR_IR, marker='s', linewidth=2.2)
                pf_handles.append(h); pf_labels.append("IR-SGMCMC")
            if pulpo_per_nause is not None:
                h, = ax.plot(range(1, len(pulpo_per_nause) + 1), pulpo_per_nause,
                             color='#8A4FBF', marker='D', linewidth=2.2)
                pf_handles.append(h); pf_labels.append("PULBO")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("nAUSE")
            ax.set_title("nAUSE", fontsize=12, fontweight='600')
            fig.savefig(os.path.join(save_dir, "sparsification_per_frame_comparison.png"),
                        bbox_inches="tight", pad_inches=0.18)
            fig.savefig(os.path.join(save_dir, "sparsification_per_frame_comparison.svg"),
                        bbox_inches="tight", pad_inches=0.18)
            plt.close(fig)
            _save_legend(pf_handles, pf_labels,
                         os.path.join(save_dir, "sparsification_per_frame_comparison_legend.png"))

    # ------------------------------------------------------------------
    # 6. Averaged risk-coverage comparison
    # ------------------------------------------------------------------
    if sde_rc_data is not None or ir_rc_data is not None or pulpo_rc_data is not None:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        handles, labels = [], []

        if sde_rc_data is not None:
            cov = sde_rc_data["coverages"]
            naurc = float(sde_rc_data['avg_naurc'])
            ax.plot(cov, sde_rc_data["avg_oracle_rc"], color=CLR_SDE, linestyle=':', linewidth=1.6)
            ax.plot(cov, sde_rc_data["avg_random_rc"], color=CLR_SDE, linestyle='--', linewidth=1.6)
            h_sde, = ax.plot(cov, sde_rc_data["avg_unc_rc"], color=CLR_SDE, linewidth=2.2)
            ax.fill_between(cov, sde_rc_data["avg_oracle_rc"], sde_rc_data["avg_unc_rc"],
                            alpha=0.15, color=CLR_BAND_SDE)
            handles.append(h_sde)
            labels.append(f"SDE Bridge (nAURC={naurc:.3f})")

        if ir_rc_data is not None:
            cov_ir = ir_rc_data["coverages"]
            naurc = float(ir_rc_data['avg_naurc'])
            ax.plot(cov_ir, ir_rc_data["avg_oracle_rc"], color=CLR_IR, linestyle=':', linewidth=1.6)
            ax.plot(cov_ir, ir_rc_data["avg_random_rc"], color=CLR_IR, linestyle='--', linewidth=1.6)
            h_ir, = ax.plot(cov_ir, ir_rc_data["avg_unc_rc"], color=CLR_IR, linewidth=2.2)
            ax.fill_between(cov_ir, ir_rc_data["avg_oracle_rc"], ir_rc_data["avg_unc_rc"],
                            alpha=0.15, color=CLR_BAND_IR)
            handles.append(h_ir)
            labels.append(f"IR-SGMCMC (nAURC={naurc:.3f})")

        if pulpo_rc_data is not None:
            cov_p = pulpo_rc_data["coverages"]
            naurc = float(pulpo_rc_data['avg_naurc'])
            ax.plot(cov_p, pulpo_rc_data["avg_oracle_rc"], color='#8A4FBF', linestyle=':', linewidth=1.6)
            ax.plot(cov_p, pulpo_rc_data["avg_random_rc"], color='#8A4FBF', linestyle='--', linewidth=1.6)
            h_p, = ax.plot(cov_p, pulpo_rc_data["avg_unc_rc"], color='#8A4FBF', linewidth=2.2)
            ax.fill_between(cov_p, pulpo_rc_data["avg_oracle_rc"], pulpo_rc_data["avg_unc_rc"],
                            alpha=0.15, color='#C9A0D8')
            handles.append(h_p)
            labels.append(f"PULBO (nAURC={naurc:.3f})")

        ax.set_xlabel("Coverage (fraction of retained pixels)")
        ax.set_ylabel("Risk (Cumulative Mean Error)")

        parts = []
        if sde_rc_data is not None:
            parts.append(f"SDE nAURC={float(sde_rc_data['avg_naurc']):.3f}")
        if ir_rc_data is not None:
            parts.append(f"IR nAURC={float(ir_rc_data['avg_naurc']):.3f}")
        if pulpo_rc_data is not None:
            parts.append(f"PULBO nAURC={float(pulpo_rc_data['avg_naurc']):.3f}")
        ax.set_title(" | ".join(parts), fontsize=12, fontweight='600')

        fig.savefig(os.path.join(save_dir, "risk_coverage_comparison_avg.png"),
                    bbox_inches="tight", pad_inches=0.18)
        fig.savefig(os.path.join(save_dir, "risk_coverage_comparison_avg.svg"),
                    bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
        _save_legend(handles, labels,
                     os.path.join(save_dir, "risk_coverage_comparison_avg_legend.png"))

        # Per-frame nAURC comparison
        sde_per_naurc = sde_rc_data.get("per_frame_naurc") if sde_rc_data is not None else None
        ir_per_naurc = ir_rc_data.get("per_frame_naurc") if ir_rc_data is not None else None
        pulpo_per_naurc = pulpo_rc_data.get("per_frame_naurc") if pulpo_rc_data is not None else None
        if sde_per_naurc is not None or ir_per_naurc is not None or pulpo_per_naurc is not None:
            fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
            pf_handles, pf_labels = [], []
            if sde_per_naurc is not None:
                h, = ax.plot(range(1, len(sde_per_naurc) + 1), sde_per_naurc,
                             color=CLR_SDE, marker='o', linewidth=2.2)
                pf_handles.append(h); pf_labels.append("SDE Bridge")
            if ir_per_naurc is not None:
                h, = ax.plot(range(1, len(ir_per_naurc) + 1), ir_per_naurc,
                             color=CLR_IR, marker='s', linewidth=2.2)
                pf_handles.append(h); pf_labels.append("IR-SGMCMC")
            if pulpo_per_naurc is not None:
                h, = ax.plot(range(1, len(pulpo_per_naurc) + 1), pulpo_per_naurc,
                             color='#8A4FBF', marker='D', linewidth=2.2)
                pf_handles.append(h); pf_labels.append("PULBO")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("nAURC")
            ax.set_title("nAURC", fontsize=12, fontweight='600')
            fig.savefig(os.path.join(save_dir, "risk_coverage_per_frame_comparison.png"),
                        bbox_inches="tight", pad_inches=0.18)
            fig.savefig(os.path.join(save_dir, "risk_coverage_per_frame_comparison.svg"),
                        bbox_inches="tight", pad_inches=0.18)
            plt.close(fig)
            _save_legend(pf_handles, pf_labels,
                         os.path.join(save_dir, "risk_coverage_per_frame_comparison_legend.png"))

    # ------------------------------------------------------------------
    # 7. Summary text
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 8. Velocity/displacement–label correlation comparison (SG-MCMC method)
    #    Scatter plot of (ud, ul) per structure per frame
    # ------------------------------------------------------------------
    sde_corr_npz = os.path.join(sde_output_dir, "velocity_label_correlation.npz")
    ir_corr_npz = os.path.join(ir_subj_dir, "velocity_label_correlation.npz")
    sde_corr = np.load(sde_corr_npz, allow_pickle=True) if os.path.exists(sde_corr_npz) else None
    ir_corr = np.load(ir_corr_npz, allow_pickle=True) if os.path.exists(ir_corr_npz) else None

    if sde_corr is not None or ir_corr is not None:
        # Per-frame Pearson and Spearman comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        handles, labels_leg = [], []

        if sde_corr is not None and 'frames' in sde_corr:
            h, = axes[0].plot(sde_corr["frames"], sde_corr["pearson"],
                              color=CLR_SDE, marker='o', linewidth=2.2)
            axes[1].plot(sde_corr["frames"], sde_corr["spearman"],
                         color=CLR_SDE, marker='o', linewidth=2.2)
            handles.append(h)
            labels_leg.append(f"SDE Bridge (mean r={float(sde_corr['mean_pearson']):.3f})")

        if ir_corr is not None and 'frames' in ir_corr:
            h, = axes[0].plot(ir_corr["frames"], ir_corr["pearson"],
                              color=CLR_IR, marker='s', linewidth=2.2)
            axes[1].plot(ir_corr["frames"], ir_corr["spearman"],
                         color=CLR_IR, marker='s', linewidth=2.2)
            handles.append(h)
            labels_leg.append(f"IR-SGMCMC (mean r={float(ir_corr['mean_pearson']):.3f})")

        axes[0].set_xlabel("Frame Index")
        axes[0].set_ylabel("Pearson r")
        axes[0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        parts_p = []
        if sde_corr is not None and 'mean_pearson' in sde_corr:
            parts_p.append(f"SDE={float(sde_corr['mean_pearson']):.3f}")
        if ir_corr is not None and 'mean_pearson' in ir_corr:
            parts_p.append(f"IR={float(ir_corr['mean_pearson']):.3f}")
        axes[0].set_title("Pearson: " + " | ".join(parts_p), fontsize=12, fontweight='600')

        axes[1].set_xlabel("Frame Index")
        axes[1].set_ylabel("Spearman rho")
        axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        parts_s = []
        if sde_corr is not None and 'mean_spearman' in sde_corr:
            parts_s.append(f"SDE={float(sde_corr['mean_spearman']):.3f}")
        if ir_corr is not None and 'mean_spearman' in ir_corr:
            parts_s.append(f"IR={float(ir_corr['mean_spearman']):.3f}")
        axes[1].set_title("Spearman: " + " | ".join(parts_s), fontsize=12, fontweight='600')

        fig.savefig(os.path.join(save_dir, "correlation_comparison.png"),
                    bbox_inches="tight", pad_inches=0.18)
        fig.savefig(os.path.join(save_dir, "correlation_comparison.svg"),
                    bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)

        _save_legend(handles, labels_leg,
                     os.path.join(save_dir, "correlation_comparison_legend.png"))

        print(f"  Correlation comparison saved")

    # ------------------------------------------------------------------
    # 9. Summary text
    # ------------------------------------------------------------------
    summary_path = os.path.join(save_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Comparison: SDE Bridge UQ vs IR-SGMCMC vs DIF-VM vs PULBO\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"SDE results dir: {sde_output_dir}\n")
        f.write(f"IR-SGMCMC results dir: {ir_output_dir}\n")
        f.write(f"DIF-VM results dir: {vxm_output_dir or 'N/A'}\n")
        f.write(f"PULBO results dir: {pulpo_output_dir or 'N/A'}\n\n")

        if sde_spars_data is not None:
            f.write(f"SDE Bridge   - Avg nAUSE: {float(sde_spars_data['avg_nause']):.4f}\n")
        if ir_spars_data is not None:
            f.write(f"IR-SGMCMC    - Avg nAUSE: {float(ir_spars_data['avg_nause']):.4f}\n")
        if pulpo_spars_data is not None:
            f.write(f"PULBO        - Avg nAUSE: {float(pulpo_spars_data['avg_nause']):.4f}\n")
        if sde_rc_data is not None:
            f.write(f"SDE Bridge   - Avg nAURC: {float(sde_rc_data['avg_naurc']):.4f}\n")
        if ir_rc_data is not None:
            f.write(f"IR-SGMCMC    - Avg nAURC: {float(ir_rc_data['avg_naurc']):.4f}\n")
        if pulpo_rc_data is not None:
            f.write(f"PULBO        - Avg nAURC: {float(pulpo_rc_data['avg_naurc']):.4f}\n")

        if sde_corr is not None and 'mean_pearson' in sde_corr:
            f.write(f"SDE Bridge   - Pearson r (ud vs ul): {float(sde_corr['mean_pearson']):.4f}\n")
            f.write(f"SDE Bridge   - Spearman (ud vs ul):  {float(sde_corr['mean_spearman']):.4f}\n")
        if ir_corr is not None and 'mean_pearson' in ir_corr:
            f.write(f"IR-SGMCMC    - Pearson r (ud vs ul): {float(ir_corr['mean_pearson']):.4f}\n")
            f.write(f"IR-SGMCMC    - Spearman (ud vs ul):  {float(ir_corr['mean_spearman']):.4f}\n")

        f.write(f"\nFrames with results: {all_frames}\n")
        f.write(f"\nPer-frame outputs saved under: {save_dir}/frame_XX/\n")
        f.write("  Each frame directory contains:\n")
        f.write("    - uncertainty_map_frame_XX.png  (side-by-side comparison)\n")
        f.write("    - sparsification_overlay_frame_XX.png  (both methods overlaid)\n")
        f.write("    - risk_coverage_overlay_frame_XX.png  (both methods overlaid)\n")
        f.write("    - sde_uncertainty_frame_XX.png  (standalone SDE)\n")
        f.write("    - ir_uncertainty_frame_XX.png  (standalone IR-SGMCMC)\n")

    print(f"\nComparison summary saved to: {summary_path}")


def _copy_file(src, dst):
    """Copy a file (symlink if possible, else shutil copy)."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    tag = f"batch{args.batch_idx}_subject{args.subject_idx}"
    run_output_dir = os.path.join(BASE_OUTPUT_DIR, tag)
    os.makedirs(run_output_dir, exist_ok=True)

    sde_output_dir = os.path.join(run_output_dir, "sde_bridge_uq")
    ir_output_dir = os.path.join(run_output_dir, "ir_sgmcmc")
    vxm_output_dir = os.path.join(run_output_dir, "vxm_prob")
    pulpo_output_dir = os.path.join(run_output_dir, "pulbo")

    print(f"Output root: {run_output_dir}")
    print(f"  SDE results:      {sde_output_dir}")
    print(f"  IR-SGMCMC results: {ir_output_dir}")
    print(f"  DIF-VM results:  {vxm_output_dir}")
    print(f"  PULBO results:   {pulpo_output_dir}")

    # ------------------------------------------------------------------
    # Run all test scripts
    # ------------------------------------------------------------------
    if not args.skip_sde:
        run_sde_test(args, sde_output_dir)
    else:
        print("\nSkipping SDE test (--skip_sde)")

    if not args.skip_ir:
        run_ir_sgmcmc_test(args, ir_output_dir)
    else:
        print("\nSkipping IR-SGMCMC test (--skip_ir)")

    if not args.skip_vxm:
        run_vxm_prob_test(args, vxm_output_dir)
    else:
        print("\nSkipping DIF-VM test (--skip_vxm)")

    if not args.skip_pulpo:
        run_pulpo_test(args, pulpo_output_dir)
    else:
        print("\nSkipping PULBO test (--skip_pulpo)")

    # ------------------------------------------------------------------
    # Generate per-timepoint comparison visualizations
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 5: Generating per-timepoint comparison visualizations")
    print(f"{'='*70}")

    comparison_dir = os.path.join(run_output_dir, "per_timepoint")
    save_individual_timepoint_plots(sde_output_dir, ir_output_dir, comparison_dir,
                                    selected_frames=args.frames,
                                    vxm_output_dir=vxm_output_dir,
                                    pulpo_output_dir=pulpo_output_dir)

    print(f"\nAll results saved under: {run_output_dir}")
    print(f"  sde_bridge_uq/       - Full SDE Bridge UQ test outputs")
    print(f"  ir_sgmcmc/           - Full IR-SGMCMC test outputs")
    print(f"  vxm_prob/            - Full DIF-VM test outputs")
    print(f"  pulbo/               - Full PULBO test outputs")
    print(f"  per_timepoint/       - Per-frame comparison plots")
    print(f"    frame_XX/          - Individual frame visualizations")
    print(f"    sparsification_comparison_avg.png")
    print(f"    risk_coverage_comparison_avg.png")
    print(f"    comparison_summary.txt")
    print("\nDone!")


if __name__ == "__main__":
    main()
