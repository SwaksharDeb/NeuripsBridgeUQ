"""
Cardiac analogue of uncertainty_brain_sde_v2/run_both_tests.py.

Loads per-frame averaged sparsification curves from four BridgeUQ backbones
(TLRN, LTMA, TM, VxM) plus optionally aggregates the low-dimensional UQ
method's per-pair shards, then plots a single multi-panel figure overlaying
the methods' sparsification curves per frame.

Usage:
    cd /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2
    python -m uncertainty_sde_combined_acdc_v2.run_both_tests_cardiac
    python -m uncertainty_sde_combined_acdc_v2.run_both_tests_cardiac --frames 1 3 5
    python -m uncertainty_sde_combined_acdc_v2.run_both_tests_cardiac --plot_wang
    python -m uncertainty_sde_combined_acdc_v2.run_both_tests_cardiac --rerun_tlrn --num_runs 100
"""

import os
import re
import sys
import glob
import argparse
import subprocess
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


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


PROJECT_ROOT = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"

METHOD_CONFIG = {
    'tlrn': {
        'label': r'BridgeUQ$_{TLRN}$',
        'color': '#185FA5',
        'fill':  '#85B7EB',
        'marker': 'o',
        'linestyle': '-',
        'lw': 2.0,
        'workdir': os.path.join(
            PROJECT_ROOT,
            "2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03_plus_ACDC",
        ),
        'rerun_cwd': PROJECT_ROOT,
        'rerun_module': "uncertainty_sde_combined_acdc_v2.test_fast",
    },
    'ltma': {
        'label': r'BridgeUQ$_{LTMA}$',
        'color': '#B03030',
        'fill':  '#E8A0A0',
        'marker': 's',
        'linestyle': '-',
        'lw': 2.0,
        'workdir': os.path.join(
            PROJECT_ROOT,
            "2026Experiments/CINE/outputs/TLMA_TGrad/config_plus_ACDC",
        ),
        'rerun_cwd': PROJECT_ROOT,
        'rerun_module': "uncertainty_sde_combined_acdc_v2.test_fast",
    },
    'tm': {
        'label': r'BridgeUQ$_{TM}$',
        'color': '#2E8B57',
        'fill':  '#90D4A8',
        'marker': '^',
        'linestyle': '-',
        'lw': 2.0,
        'workdir': os.path.join(
            PROJECT_ROOT,
            "2026Experiments/CINE/outputs/TransMorph/basic_plus_ACDC",
        ),
        'rerun_cwd': PROJECT_ROOT,
        'rerun_module': "uncertainty_sde_combined_acdc_v2.test_fast",
    },
    'vxm': {
        'label': r'BridgeUQ$_{VxM}$',
        'color': '#8A4FBF',
        'fill':  '#C9A0D8',
        'marker': 'D',
        'linestyle': '-',
        'lw': 2.0,
        'workdir': os.path.join(
            PROJECT_ROOT,
            "2026Experiments/CINE/outputs/VoxelMorph/voxelmorph",
        ),
        'rerun_cwd': PROJECT_ROOT,
        'rerun_module': "uncertainty_sde_combined_acdc_v2.test_fast",
    },
    'lowdim': {
        'label': r'Wang et al.',
        'color': '#D4760A',
        'fill':  '#F0C888',
        'marker': 'v',
        'linestyle': (0, (5, 2)),
        'lw': 3.0,
        'shards_root': "/scratch/swd9tc/Uncertanity_quantification/"
                       "low_dimension_uncertainty/outputs",
    },
}

PLOT_ORDER = ['tlrn', 'ltma', 'tm', 'vxm', 'lowdim']

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "uncertainty_sde_combined_acdc_v2", "comparison_results"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cardiac UQ orchestrator: overlay sparsification curves "
                    "from four BridgeUQ backbones + optional Wang et al."
    )
    p.add_argument("--frames", type=int, nargs='+', default=None,
                   help="Subset of frames to plot (e.g. --frames 1 3 5). "
                        "Default: all frames common to TLRN.")
    p.add_argument("--num_runs", type=int, default=100,
                   help="Forwarded to test_fast.py when --rerun_* is used.")
    p.add_argument("--use_train", action="store_true",
                   help="Forwarded to test_fast.py when --rerun_* is used.")
    p.add_argument("--rerun_tlrn", action="store_true")
    p.add_argument("--rerun_ltma", action="store_true")
    p.add_argument("--rerun_tm",   action="store_true")
    p.add_argument("--rerun_vxm",  action="store_true")
    p.add_argument("--plot_wang", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Overlay Wang et al. curve and its highlighted region. "
                        "Use --plot_wang to enable, --no-plot_wang to disable.")
    p.add_argument("--batch_idx", type=int, default=15,
                   help="Plot the per-subject sparsification curves for a "
                        "specific subject. Requires --subject_idx. When set, "
                        "loads {workdir}/visualization/uncertainty_sde_combined_acdc_v2/"
                        "test_results/batch{B}_subject_{N}/per_frame_curves/"
                        "sparsification_frame_*.npz instead of the aggregate. "
                        "(Wang et al. has no per-subject equivalent and is "
                        "automatically disabled in this mode.)")
    p.add_argument("--subject_idx", type=int, default=9,
                   help="See --batch_idx.")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def _aggregate_npz_path(workdir):
    return os.path.join(
        workdir, "visualization", "uncertainty_sde_combined_acdc_v2",
        "aggregate", "avg_sparsification_curves.npz",
    )


def _per_subject_dir(workdir, subject_id):
    return os.path.join(
        workdir, "visualization", "uncertainty_sde_combined_acdc_v2",
        "test_results", subject_id, "per_frame_curves",
    )


def _run_test_fast(method_key, args):
    cfg = METHOD_CONFIG[method_key]
    cmd = [sys.executable, "-m", cfg['rerun_module'],
           "--num_runs", str(args.num_runs)]
    if args.use_train:
        cmd.append("--use_train")
    print(f"\n[rerun:{method_key}] cwd={cfg['rerun_cwd']}")
    print(f"[rerun:{method_key}] cmd={' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=cfg['rerun_cwd']).returncode
    if rc != 0:
        print(f"[rerun:{method_key}] WARNING: exited with code {rc}")
    return rc


def _load_bridgeuq(npz_path):
    """Load a BridgeUQ test_fast.py output NPZ.

    Returns: dict[frame_idx] -> {fractions, unc, oracle, random, nause}.
    """
    if not os.path.isfile(npz_path):
        print(f"  [skip] missing NPZ: {npz_path}")
        return {}
    d = np.load(npz_path, allow_pickle=True)
    files = set(d.files)
    if 'frame_indices' in files:
        frames = [int(f) for f in d['frame_indices']]
    else:
        frames = sorted({int(m.group(1))
                         for k in d.files
                         for m in [re.match(r'frame_(\d+)_fractions$', k)]
                         if m})
    out = {}
    for fi in frames:
        try:
            out[fi] = {
                'fractions': np.asarray(d[f'frame_{fi}_fractions']),
                'unc':       np.asarray(d[f'frame_{fi}_uncertainty_curve']),
                'oracle':    np.asarray(d[f'frame_{fi}_oracle_curve']),
                'random':    np.asarray(d[f'frame_{fi}_random_curve']),
                'nause':     float(d[f'frame_{fi}_ause_norm']),
            }
        except KeyError as e:
            print(f"  [skip] {npz_path} frame {fi}: missing {e}")
    return out


_PER_SUBJ_FRAME_RE = re.compile(r'sparsification_frame_(\d+)\.npz$')


def _load_bridgeuq_per_subject(per_subj_dir):
    """Load a BridgeUQ test.py per-subject sparsification dump.

    Reads {per_subj_dir}/sparsification_frame_NN.npz files. Each .npz holds
    fractions / uncertainty_curve / oracle_curve / random_curve / ause_norm.

    Returns dict[frame_idx] -> {fractions, unc, oracle, random, nause}.
    """
    if not os.path.isdir(per_subj_dir):
        print(f"  [skip] missing per-subject dir: {per_subj_dir}")
        return {}
    files = sorted(glob.glob(os.path.join(per_subj_dir,
                                          "sparsification_frame_*.npz")))
    if not files:
        print(f"  [skip] no sparsification_frame_*.npz in {per_subj_dir}")
        return {}
    out = {}
    for fpath in files:
        m = _PER_SUBJ_FRAME_RE.search(os.path.basename(fpath))
        if not m:
            continue
        fi = int(m.group(1))
        try:
            with np.load(fpath) as d:
                out[fi] = {
                    'fractions': np.asarray(d['fractions']),
                    'unc':       np.asarray(d['uncertainty_curve']),
                    'oracle':    np.asarray(d['oracle_curve']),
                    'random':    np.asarray(d['random_curve']),
                    'nause':     float(d['ause_norm']),
                }
        except KeyError as e:
            print(f"  [skip] {fpath}: missing {e}")
    return out


_LOWDIM_FNAME_RE = re.compile(r'subj(\d+)_frame(\d+)\.npz$')


def _aggregate_lowdim(shards_root):
    """Aggregate low-dim per-pair NPZ shards into per-frame averaged curves."""
    pattern = os.path.join(shards_root, "shard*", "curves", "subj*_frame*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [skip] low-dim shards empty: {pattern}")
        return {}

    by_frame = defaultdict(list)
    for f in files:
        m = _LOWDIM_FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        fi = int(m.group(2))
        by_frame[fi].append(f)

    out = {}
    for fi, group in by_frame.items():
        unc_stack, ora_stack, rnd_stack, ause_list = [], [], [], []
        fractions = None
        for fpath in group:
            d = np.load(fpath)
            if fractions is None:
                fractions = np.asarray(d['fractions'])
            unc_stack.append(np.asarray(d['unc_curve']))
            ora_stack.append(np.asarray(d['oracle_curve']))
            rnd_stack.append(np.asarray(d['random_curve']))
            ause_list.append(float(d['ause_norm']))
        out[fi] = {
            'fractions': fractions,
            'unc':       np.mean(np.stack(unc_stack, axis=0), axis=0),
            'oracle':    np.mean(np.stack(ora_stack, axis=0), axis=0),
            'random':    np.mean(np.stack(rnd_stack, axis=0), axis=0),
            'nause':     float(np.mean(ause_list)),
        }
    return out


def _trim_trailing_zeros(x, unc, orc, rnd):
    while (x.size > 2 and
           (rnd[-1] == 0 or unc[-1] == 0) and
           (rnd[-2] != 0 and unc[-2] != 0)):
        x = x[:-1]; unc = unc[:-1]; orc = orc[:-1]; rnd = rnd[:-1]
    return x, unc, orc, rnd


_LABEL_RE = re.compile(r'\$_\{([^}]+)\}\$')


def _plain_label(label):
    """Strip LaTeX subscripts so the .txt file is readable as plain text."""
    return _LABEL_RE.sub(r'_\1', label)


def _save_nause_txt(method_data, frames, save_dir, plot_order):
    """Write per-frame nAUSC for each method to a .txt file."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "nausc_per_frame.txt")

    method_width = max(
        len(_plain_label(METHOD_CONFIG[mkey]['label'])) for mkey in plot_order
    )
    method_width = max(method_width, len("Method"))
    frame_width = 8

    header = f"{'Method':<{method_width}}  " + \
        "  ".join(f"t{f:<{frame_width - 1}}" for f in frames) + \
        f"  {'mean':>{frame_width}}"

    lines = [
        "Per-frame nAUSC values",
        "=" * len(header),
        f"Frames: {list(frames)}",
        "",
        header,
        "-" * len(header),
    ]

    for mkey in plot_order:
        label = _plain_label(METHOD_CONFIG[mkey]['label'])
        per_frame = method_data.get(mkey, {})
        cells = []
        vals = []
        for f in frames:
            if f in per_frame:
                v = per_frame[f]['nause']
                cells.append(f"{v:>{frame_width}.4f}")
                vals.append(v)
            else:
                cells.append(f"{'NA':>{frame_width}}")
        mean_str = f"{np.mean(vals):>{frame_width}.4f}" if vals else f"{'NA':>{frame_width}}"
        lines.append(f"{label:<{method_width}}  " + "  ".join(cells) + f"  {mean_str}")

    text = "\n".join(lines) + "\n"
    with open(out_path, 'w') as f:
        f.write(text)
    print(f"\nSaved per-frame nAUSC to: {out_path}")


def _plot_sparsification_comparison(method_data, frames, save_dir, plot_order,
                                    filename_suffix=""):
    os.makedirs(save_dir, exist_ok=True)

    n_frames = len(frames)
    fig, axes = plt.subplots(1, n_frames,
                             figsize=(2.8 * n_frames, 3.4),
                             squeeze=False)
    axes = axes[0]

    for col, fi in enumerate(frames):
        ax = axes[col]
        rand_drawn = False
        lowdim_curve = None
        for mkey in plot_order:
            cfg = METHOD_CONFIG[mkey]
            d = method_data.get(mkey, {}).get(fi)
            if d is None:
                continue

            x = np.asarray(d['fractions'])
            unc = np.asarray(d['unc'])
            orc = np.asarray(d['oracle'])
            rnd = np.asarray(d['random'])
            x, unc, orc, rnd = _trim_trailing_zeros(x, unc, orc, rnd)

            ref = float(unc[0]) if unc[0] != 0 else 1.0
            unc_n = unc / ref
            orc_n = orc / ref
            rnd_n = rnd / ref

            if not rand_drawn:
                ax.plot(x, rnd_n, color='#888888', linestyle='--',
                        linewidth=1.2, alpha=0.85, zorder=2, label='Random')
                rand_drawn = True

            ax.plot(x, orc_n,
                    color=cfg['color'],
                    linestyle=':',
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=2 if mkey != 'lowdim' else 3)

            ax.plot(x, unc_n,
                    color=cfg['color'],
                    linestyle=cfg['linestyle'],
                    linewidth=cfg['lw'],
                    marker=cfg['marker'],
                    markersize=5.5,
                    markevery=2,
                    markerfacecolor=cfg['color'],
                    markeredgecolor='white',
                    markeredgewidth=0.6,
                    label=cfg['label'],
                    zorder=4 if mkey == 'lowdim' else 3)

            if mkey == 'lowdim':
                lowdim_curve = (x, unc_n, cfg['color'], cfg['fill'])

        is_edge_panel = (col == 0 or col == n_frames - 1)
        if lowdim_curve is not None and not is_edge_panel:
            x_ld, y_ld, c_ld, fill_ld = lowdim_curve
            min_idx = int(np.argmin(y_ld))
            if min_idx < len(y_ld) - 1 and y_ld[-1] > y_ld[min_idx] + 1e-3:
                ax.axvspan(x_ld[min_idx], x_ld[-1],
                           color=fill_ld, alpha=0.22, zorder=1, linewidth=0)
                ax.fill_between(x_ld[min_idx:], y_ld[min_idx],
                                y_ld[min_idx:],
                                color=c_ld, alpha=0.20,
                                linewidth=0, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f'.{str(round(v, 2)).split(".")[-1]}' if 0 < v < 1
            else ('0' if v == 0 else f'{v:g}')
        ))
        ax.set_title(f'$t = {fi}$', fontsize=18, fontweight='600', pad=8)
        if col == 0:
            ax.set_xlabel('Fraction of Removed Voxels', fontsize=16)
            ax.tick_params(axis='both', labelsize=13)
        else:
            ax.tick_params(labelleft=False, labelbottom=False)
        ax.grid(True, alpha=0.3)

    method_handles, method_labels = [], []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li in seen or li == 'Random':
                continue
            seen.add(li)
            method_handles.append(hi); method_labels.append(li)

    oracle_handle = Line2D([0], [0], color='black', linestyle=':',
                           linewidth=1.4, label='Oracle (per method)')
    handles = method_handles + [oracle_handle]
    labels = method_labels + ['Oracle (per method)']

    fig.legend(handles, labels, loc='lower center',
               ncol=len(handles), frameon=True, fancybox=True,
               edgecolor='#bbb', fontsize=11,
               bbox_to_anchor=(0.60, 0.12),
               handlelength=2.6, columnspacing=1.4)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.28, top=0.88,
                        wspace=0.06)
    fig.text(0.005, 0.55, 'MSE (normalized)', fontsize=16,
             rotation=90, va='center', ha='left')

    base = os.path.join(save_dir, f"sparsification_comparison{filename_suffix}")
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(f"{base}.{ext}", bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"\nSaved: {base}.{{png,svg,pdf}}")


def main():
    args = parse_args()

    rerun_map = {'tlrn': args.rerun_tlrn, 'ltma': args.rerun_ltma,
                 'tm': args.rerun_tm,    'vxm':  args.rerun_vxm}
    for mkey, do_rerun in rerun_map.items():
        if do_rerun:
            _run_test_fast(mkey, args)

    # Per-subject mode: --batch_idx + --subject_idx select one subject;
    # Wang et al. (lowdim) has no per-subject equivalent so it's disabled.
    per_subject_mode = (args.batch_idx is not None and args.subject_idx is not None)
    if (args.batch_idx is None) ^ (args.subject_idx is None):
        print("ERROR: --batch_idx and --subject_idx must be set together.")
        sys.exit(1)
    if per_subject_mode and args.plot_wang:
        print("[info] disabling Wang et al. in per-subject mode (no per-subject curves).")
    plot_wang = args.plot_wang and not per_subject_mode

    plot_order = [m for m in PLOT_ORDER if plot_wang or m != 'lowdim']

    if per_subject_mode:
        subject_id = f"batch{args.batch_idx}_subject_{args.subject_idx}"
        print(f"\nPer-subject mode: {subject_id}")

    method_data = {}
    for mkey in plot_order:
        cfg = METHOD_CONFIG[mkey]
        if mkey == 'lowdim':
            method_data[mkey] = _aggregate_lowdim(cfg['shards_root'])
        elif per_subject_mode:
            method_data[mkey] = _load_bridgeuq_per_subject(
                _per_subject_dir(cfg['workdir'], subject_id)
            )
        else:
            method_data[mkey] = _load_bridgeuq(_aggregate_npz_path(cfg['workdir']))

    available = {mkey: set(method_data[mkey].keys()) for mkey in plot_order}
    union_frames = sorted(set().union(*available.values()))
    if not union_frames:
        print("ERROR: no frames found in any method's output.")
        sys.exit(1)

    if args.frames is not None:
        frames = [f for f in args.frames if f in union_frames]
        missing = [f for f in args.frames if f not in union_frames]
        if missing:
            print(f"WARNING: requested frames not present: {missing}")
    else:
        frames = union_frames

    if not frames:
        print("ERROR: no frames remain after applying --frames.")
        sys.exit(1)

    print("\nPer-method nAUSE summary:")
    for mkey in plot_order:
        per_frame = method_data.get(mkey, {})
        if not per_frame:
            print(f"  {METHOD_CONFIG[mkey]['label']}: (no data)")
            continue
        vals = [per_frame[f]['nause'] for f in frames if f in per_frame]
        if vals:
            mean_nause = float(np.mean(vals))
            per_str = ", ".join(
                f"t{f}={per_frame[f]['nause']:.3f}"
                for f in frames if f in per_frame
            )
            print(f"  {METHOD_CONFIG[mkey]['label']}: "
                  f"mean={mean_nause:.3f}  [{per_str}]")
        else:
            print(f"  {METHOD_CONFIG[mkey]['label']}: (no overlap with selected frames)")

    _save_nause_txt(method_data, frames, args.output_dir, plot_order)

    suffix = f"_{subject_id}" if per_subject_mode else ""
    _plot_sparsification_comparison(method_data, frames, args.output_dir,
                                    plot_order, filename_suffix=suffix)


if __name__ == "__main__":
    main()
