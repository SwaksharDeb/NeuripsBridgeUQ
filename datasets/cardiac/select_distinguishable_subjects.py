"""
Rank test subjects by how visibly different their per-pixel uncertainty maps
are across the four BridgeUQ backbones (TLRN, LTMA, TM, VxM).

Reads the per-subject correlation_frame_NN.npz dumps written by test_fast.py
(when invoked with --save_ud_maps) at:
  {workdir}/visualization/uncertainty_sde_combined_acdc_v2/test_results/
      {subject_id}/per_frame_curves/correlation_frame_NN.npz
where each .npz holds ud_map [H, W] (std of velocity magnitude across UQ
samples) and cardiac_mask [H, W] (LV segmentation).

For each subject_id present in all four backbones:

  distinctness  = mean over frames of the mean cross-method per-pixel std
                  of RMS-normalized ud_map values inside the (dilated)
                  cardiac mask. Higher => the four methods disagree more.
  bg_ratio_max  = max over methods of (mean BG ud_map) / (mean FG ud_map).
                  Lower => uncertainty stays on the heart.
  composite     = distinctness / (bg_ratio_max + 0.05). Used as the rank key.

Writes:
  selected_subjects/ranked_subjects.txt   ranking + top-K
  selected_subjects/ranked_subjects.npz   raw per-subject scores
  selected_subjects/montage_top{K}.png    optional 4xK montage of top-K
                                          frame-mean ud_maps (one row per
                                          method) so you can eyeball them.

Usage:
    cd /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2
    python -m uncertainty_sde_combined_acdc_v2.select_distinguishable_subjects \
        --top_k 8
"""

import os
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECT_ROOT = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"

METHODS = {
    'tlrn': {
        'label': r'BridgeUQ$_{TLRN}$',
        'test_results': f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/TLRN/"
                        "basic_MSE_Penp_img1200Reg0.03_plus_ACDC/visualization/"
                        "uncertainty_sde_combined_acdc_v2/test_results",
    },
    'ltma': {
        'label': r'BridgeUQ$_{LTMA}$',
        'test_results': f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/TLMA_TGrad/"
                        "config_plus_ACDC/visualization/"
                        "uncertainty_sde_combined_acdc_v2/test_results",
    },
    'tm': {
        'label': r'BridgeUQ$_{TM}$',
        'test_results': f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/TransMorph/"
                        "basic_plus_ACDC/visualization/"
                        "uncertainty_sde_combined_acdc_v2/test_results",
    },
    'voxelmorph': {
        'label': r'BridgeUQ$_{VxM}$',
        'test_results': f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/VoxelMorph/"
                        "voxelmorph/visualization/"
                        "uncertainty_sde_combined_acdc_v2/test_results",
    },
}

DEFAULT_REPORT_DIR = os.path.join(
    PROJECT_ROOT, "uncertainty_sde_combined_acdc_v2", "selected_subjects"
)

_FRAME_RE = re.compile(r'correlation_frame_(\d+)\.npz$')


def load_target_images(backbone='voxelmorph'):
    """Build {subject_id -> series[T_total, H, W]} for the test loader.

    Uses load_backbone_and_data so the iteration order / subject_id format
    matches test_fast.py's. Backbone choice does not affect the images, only
    which model gets (uselessly) instantiated; voxelmorph is the cheapest.
    Returns an empty dict on failure (e.g. dataset not available).
    """
    try:
        from uncertainty_sde_combined_acdc_v2.backbones import load_backbone_and_data
        _, _, test_loader, _ = load_backbone_and_data(
            backbone=backbone, combine_acdc=True
        )
    except Exception as e:
        print(f"  [warn] could not load target images: {e}")
        return {}

    out = {}
    for batch_idx, batch in enumerate(test_loader):
        series = batch['series']                # [B, T_total, H, W]
        B = series.shape[0]
        for i in range(B):
            sid = f"batch{batch_idx}_subject_{i}"
            out[sid] = series[i].cpu().numpy()  # [T_total, H, W]
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Rank subjects by cross-method ud_map distinctness."
    )
    p.add_argument("--methods", nargs='+',
                   default=list(METHODS.keys()),
                   choices=list(METHODS.keys()))
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--bg_dilate", type=int, default=6,
                   help="Pixels to dilate cardiac mask before treating the rest "
                        "as background. Larger = stricter background.")
    p.add_argument("--max_bg_ratio", type=float, default=1e9,
                   help="Reject subjects whose worst-method BG/FG ratio exceeds "
                        "this. Default disables the filter; pass e.g. 1.0 to enable.")
    p.add_argument("--min_cardiac_pixels", type=int, default=20)
    p.add_argument("--report_dir", default=DEFAULT_REPORT_DIR)
    p.add_argument("--no_montage", action="store_true",
                   help="Skip the (4 methods x top_k) frame-mean montage.")
    p.add_argument("--no_per_subject_grids", action="store_true",
                   help="Skip the per-subject (methods x frames) grids.")
    p.add_argument("--independent_scale", action="store_true",
                   help="Auto-scale each panel of the per-subject grid "
                        "independently. Default: shared vmin/vmax per subject.")
    p.add_argument("--contour", action="store_true",
                   help="Overlay the cardiac-mask boundary as a white contour. "
                        "Disabled by default since some envs lack a working "
                        "contourpy/libstdc++ pair.")
    p.add_argument("--no_target_row", action="store_true",
                   help="Skip the grayscale target-image row at the top of "
                        "each per-subject grid. (Disable if dataset / "
                        "registration backbone import is failing.)")
    return p.parse_args()


def collect_subjects(test_results_dir):
    """Return dict subject_id -> {frame_idx -> {ud_map, cardiac_mask}}."""
    out = {}
    if not os.path.isdir(test_results_dir):
        print(f"  [warn] missing dir: {test_results_dir}")
        return out
    for entry in sorted(os.listdir(test_results_dir)):
        per_frame_dir = os.path.join(test_results_dir, entry, "per_frame_curves")
        if not os.path.isdir(per_frame_dir):
            continue
        files = sorted(glob.glob(os.path.join(per_frame_dir,
                                              "correlation_frame_*.npz")))
        if not files:
            continue
        frames = {}
        for f in files:
            m = _FRAME_RE.search(os.path.basename(f))
            if not m:
                continue
            fi = int(m.group(1))
            with np.load(f) as d:
                if 'ud_map' not in d.files or 'cardiac_mask' not in d.files:
                    continue
                frames[fi] = {
                    'ud_map':       d['ud_map'].astype(np.float32),
                    'cardiac_mask': d['cardiac_mask'].astype(bool),
                }
        if frames:
            out[entry] = frames
    return out


def _rms_normalize(m):
    s = float(np.sqrt(np.mean(m * m)))
    return m / s if s > 1e-12 else m


def score_subject(per_method_maps, bg_dilate, min_cardiac_pixels):
    methods = list(per_method_maps.keys())
    if len(methods) < 2:
        return None
    common_frames = set.intersection(
        *(set(per_method_maps[m].keys()) for m in methods)
    )
    if not common_frames:
        return None

    distinct_per_frame = []
    bg_ratio_acc = defaultdict(list)
    fg_unc_acc   = defaultdict(list)

    for fi in sorted(common_frames):
        cardiac_mask = per_method_maps[methods[0]][fi]['cardiac_mask']
        if cardiac_mask.sum() < min_cardiac_pixels:
            continue
        fg_mask = binary_dilation(cardiac_mask, iterations=bg_dilate)
        bg_mask = ~fg_mask
        if bg_mask.sum() < min_cardiac_pixels:
            continue

        normalized_fg_stack = []
        for m in methods:
            ud = per_method_maps[m][fi]['ud_map']
            ud_norm = _rms_normalize(ud)
            normalized_fg_stack.append(ud_norm[fg_mask])

            fg_mean = float(ud[fg_mask].mean())
            bg_mean = float(ud[bg_mask].mean())
            fg_unc_acc[m].append(fg_mean)
            bg_ratio_acc[m].append(bg_mean / (fg_mean + 1e-12))

        stack = np.stack(normalized_fg_stack, axis=0)   # [M, N_fg]
        distinct_per_frame.append(float(stack.std(axis=0).mean()))

    if not distinct_per_frame:
        return None

    distinctness = float(np.mean(distinct_per_frame))
    bg_ratio_per = {m: float(np.mean(v)) for m, v in bg_ratio_acc.items()}
    fg_unc_per   = {m: float(np.mean(v)) for m, v in fg_unc_acc.items()}
    bg_ratio_max = max(bg_ratio_per.values())
    fg_unc_max   = max(fg_unc_per.values())
    composite    = distinctness / (bg_ratio_max + 0.05)
    return {
        'distinctness':        distinctness,
        'bg_ratio_max':        bg_ratio_max,
        'bg_ratio_per_method': bg_ratio_per,
        'fg_unc_max':          fg_unc_max,
        'composite':           composite,
        'n_frames':            len(distinct_per_frame),
    }


def write_report(scored, keepers, args, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    txt_path = os.path.join(report_dir, "ranked_subjects.txt")
    npz_path = os.path.join(report_dir, "ranked_subjects.npz")

    with open(txt_path, 'w') as f:
        f.write("Subjects ranked by cross-method ud_map distinctness\n")
        f.write("=" * 100 + "\n")
        f.write(f"Methods: {args.methods}\n")
        f.write(f"BG dilate: {args.bg_dilate} px,  max_bg_ratio: {args.max_bg_ratio},  "
                f"min_cardiac_pixels: {args.min_cardiac_pixels}\n\n")

        top = keepers[:args.top_k]
        f.write(f"TOP {len(top)}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'rank':>4}  {'subject_id':<28} {'composite':>10} "
                f"{'distinct':>10} {'bg_ratio':>10} {'fg_unc':>10} {'frames':>7}\n")
        for r, s in enumerate(top, 1):
            f.write(f"{r:>4}  {s['subject_id']:<28} {s['composite']:>10.4f} "
                    f"{s['distinctness']:>10.4f} {s['bg_ratio_max']:>10.4f} "
                    f"{s['fg_unc_max']:>10.4f} {s['n_frames']:>7d}\n")

        f.write("\nFULL ranking (passed BG filter)\n")
        f.write("-" * 100 + "\n")
        for r, s in enumerate(keepers, 1):
            per_m = ", ".join(f"{m}={v:.3f}"
                              for m, v in s['bg_ratio_per_method'].items())
            f.write(f"{r:>4}. {s['subject_id']}  composite={s['composite']:.4f}  "
                    f"distinct={s['distinctness']:.4f}  "
                    f"bg_ratio_max={s['bg_ratio_max']:.4f}  ({per_m})\n")

        rejected = [s for s in scored if s['bg_ratio_max'] > args.max_bg_ratio]
        if rejected:
            f.write(f"\nREJECTED for bg_ratio_max > {args.max_bg_ratio}\n")
            f.write("-" * 100 + "\n")
            for s in sorted(rejected, key=lambda x: x['bg_ratio_max']):
                f.write(f"  {s['subject_id']}  bg_ratio_max={s['bg_ratio_max']:.4f}  "
                        f"distinct={s['distinctness']:.4f}\n")

    np.savez(
        npz_path,
        subject_ids=np.array([s['subject_id'] for s in scored]),
        composite=np.array([s['composite'] for s in scored]),
        distinctness=np.array([s['distinctness'] for s in scored]),
        bg_ratio_max=np.array([s['bg_ratio_max'] for s in scored]),
        fg_unc_max=np.array([s['fg_unc_max'] for s in scored]),
        n_frames=np.array([s['n_frames'] for s in scored]),
        keeper_subject_ids=np.array([s['subject_id'] for s in keepers]),
    )
    return txt_path, npz_path


def render_montage(top_keepers, methods, per_method_subjects, save_path):
    """Save a (n_methods x top_k) grid: frame-mean ud_map per (method, subject)."""
    if not top_keepers:
        return
    K = len(top_keepers)
    M = len(methods)
    fig, axes = plt.subplots(M, K, figsize=(2.0 * K, 2.0 * M), squeeze=False)

    for col, s in enumerate(top_keepers):
        sid = s['subject_id']
        for row, m in enumerate(methods):
            frames = per_method_subjects[m].get(sid, {})
            if not frames:
                axes[row, col].set_axis_off()
                continue
            stack = np.stack(
                [frames[fi]['ud_map'] for fi in sorted(frames.keys())],
                axis=0,
            )
            mean_ud = stack.mean(axis=0)
            ax = axes[row, col]
            im = ax.imshow(mean_ud, cmap='magma')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(METHODS[m]['label'], fontsize=10)
            if row == 0:
                ax.set_title(f"{sid}\nc={s['composite']:.3f}", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Frame-averaged ud_map per method (top-K most distinct subjects)",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def render_per_subject_grid(subject_id, methods, per_method_subjects,
                            save_path, composite=None, share_scale=True,
                            cardiac_contour=False, target_series=None):
    """Save a (M methods x T frames) grid of per-frame ud_maps for one subject.

    Visual style mirrors visualization.py:246 — `cmap='jet'`, percentile-based
    var_vmin/var_vmax computed across all frames+methods for the subject,
    `ax.axis('off')`.

    share_scale: if True, all panels share a single (var_vmin, var_vmax) so
        cross-method magnitude differences are visible. If False, each panel
        auto-scales independently.
    cardiac_contour: optional white cardiac-mask boundary overlay.
    """
    frame_sets = [set(per_method_subjects[m].get(subject_id, {}).keys())
                  for m in methods]
    if not all(frame_sets):
        return
    frames = sorted(set.intersection(*frame_sets))
    # Drop the trailing t=14 frame (bridge endpoint -- always near-zero unc).
    frames = [f for f in frames if f != 14]
    if not frames:
        return

    M = len(methods)
    T = len(frames)
    has_target = target_series is not None
    rows_total = M + (1 if has_target else 0)

    # var_vmin / var_vmax — same recipe as visualization.py:159-162.
    if share_scale:
        all_var_vals = np.concatenate([
            per_method_subjects[m][subject_id][fi]['ud_map'].ravel()
            for m in methods for fi in frames
        ])
        var_vmin = float(np.percentile(all_var_vals, 1))
        var_vmax = float(np.percentile(all_var_vals, 100))
        var_vmax = max(var_vmax, 1e-6)

    fig, axes = plt.subplots(rows_total, T,
                             figsize=(1.0 * T, 1.0 * rows_total),
                             squeeze=False)

    # Optional target image row (frames 1..T match the ud_map columns).
    if has_target:
        for col, fi in enumerate(frames):
            ax = axes[0, col]
            if 0 <= fi < target_series.shape[0]:
                ax.imshow(target_series[fi], cmap='gray', aspect='auto')
            ax.axis('off')
            if col == 0:
                ax.text(-0.08, 0.5, 'Target',
                        transform=ax.transAxes, fontsize=10,
                        ha='center', va='center', rotation=90)
    row_offset = 1 if has_target else 0

    for row, m in enumerate(methods):
        for col, fi in enumerate(frames):
            d = per_method_subjects[m][subject_id][fi]
            ax = axes[row + row_offset, col]
            kwargs = dict(cmap='jet', aspect='auto')
            if share_scale:
                kwargs['vmin'] = var_vmin
                kwargs['vmax'] = var_vmax
            im = ax.imshow(d['ud_map'], **kwargs)
            if cardiac_contour and d['cardiac_mask'].any():
                try:
                    ax.contour(d['cardiac_mask'].astype(float),
                               levels=[0.5], colors='white',
                               linewidths=0.6, alpha=0.85)
                except Exception as e:
                    cardiac_contour = False
                    print(f"  [warn] contour overlay disabled: {e}")
            ax.axis('off')
            if col == 0:
                ax.text(-0.08, 0.5, METHODS[m]['label'],
                        transform=ax.transAxes, fontsize=10,
                        ha='center', va='center', rotation=90)

    fig.subplots_adjust(left=0.04, right=0.93, top=0.99, bottom=0.01,
                        wspace=0.02, hspace=0.02)

    if share_scale:
        # Vertical colorbar on the right, spanning only the method rows so
        # the (grayscale) target row keeps its own implicit scale.
        method_axes = axes[row_offset:, -1]
        # Compute the y-extent of the method rows in figure coords.
        top_pos = method_axes[0].get_position()
        bot_pos = method_axes[-1].get_position()
        cb_y0 = bot_pos.y0
        cb_h  = top_pos.y1 - bot_pos.y0
        cax = fig.add_axes([0.945, cb_y0, 0.012, cb_h])
        fig.colorbar(im, cax=cax)

    fig.savefig(save_path, dpi=180, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    args = parse_args()

    per_method_subjects = {}
    for m in args.methods:
        path = METHODS[m]['test_results']
        per_method_subjects[m] = collect_subjects(path)
        print(f"  [{m}] subjects with per-frame ud_maps: "
              f"{len(per_method_subjects[m])}  ({path})")

    common_ids = set.intersection(
        *(set(d.keys()) for d in per_method_subjects.values())
    )
    print(f"\nSubjects shared across all {len(args.methods)} methods: {len(common_ids)}")
    if not common_ids:
        print("ERROR: no overlapping subjects.")
        return

    scored = []
    for sid in sorted(common_ids):
        per_method_maps = {m: per_method_subjects[m][sid] for m in args.methods}
        s = score_subject(per_method_maps, args.bg_dilate, args.min_cardiac_pixels)
        if s is None:
            continue
        s['subject_id'] = sid
        scored.append(s)

    if not scored:
        print("ERROR: no subjects could be scored.")
        return

    keepers = [s for s in scored if s['bg_ratio_max'] <= args.max_bg_ratio]
    keepers.sort(key=lambda x: x['composite'], reverse=True)

    txt_path, npz_path = write_report(scored, keepers, args, args.report_dir)
    print(f"\nReport: {txt_path}")
    print(f"NPZ:    {npz_path}")
    print(f"\nTop {min(args.top_k, len(keepers))}:")
    for r, s in enumerate(keepers[:args.top_k], 1):
        print(f"  {r:>2}. {s['subject_id']}   composite={s['composite']:.4f}   "
              f"distinct={s['distinctness']:.4f}   "
              f"bg_ratio={s['bg_ratio_max']:.4f}")

    if not args.no_montage and keepers:
        montage_path = os.path.join(
            args.report_dir, f"montage_top{min(args.top_k, len(keepers))}.png"
        )
        render_montage(keepers[:args.top_k], args.methods,
                       per_method_subjects, montage_path)
        print(f"\nMontage: {montage_path}")

    if not args.no_per_subject_grids and keepers:
        per_subj_dir = os.path.join(args.report_dir, "per_subject")
        os.makedirs(per_subj_dir, exist_ok=True)

        target_dict = {} if args.no_target_row else load_target_images()
        if not args.no_target_row:
            print(f"Loaded target images for {len(target_dict)} subjects")

        for s in keepers[:args.top_k]:
            sid = s['subject_id']
            out_path = os.path.join(per_subj_dir, f"{sid}.png")
            render_per_subject_grid(
                sid, args.methods, per_method_subjects,
                out_path, composite=s['composite'],
                share_scale=not args.independent_scale,
                cardiac_contour=args.contour,
                target_series=target_dict.get(sid),
            )
        print(f"Per-subject grids: {per_subj_dir}/  ({min(args.top_k, len(keepers))} files)")


if __name__ == "__main__":
    main()
