"""
Run uncertainty_sde_combined_acdc_v2/test_fast.py for every requested
registration backbone (TLRN, LTMA, VoxelMorph, TM) over every test subject,
then rank subjects whose per-pixel uncertainty maps are most distinguishable
across methods AND have low background noise (uncertainty concentrated on
cardiac structures, not bleeding into the empty/air parts of the image).

Why a separate file: each backbone monkey-patches sys.path and shadows
`utils/`, `models/`, `dataload/` etc., so only one backbone may be loaded per
process (see backbones.py). We therefore dispatch test_fast.py once per
method via subprocess, then aggregate from the npz dumps it writes per
subject.

Per-subject uncertainty data is read from
    {workdir}/visualization/uncertainty_sde_combined_acdc_v2/test_results/
        {batch{i}_subject_{j}}/per_frame_curves/correlation_frame_{NN}.npz
written by test_fast.py when launched with --save_ud_maps (ud_map +
cardiac_mask). test_fast.py shares a single per-batch sampling pass across
all subjects in the batch -- much faster than test.py's per-subject pass.

Usage (quick check, num_runs=100):
    cd /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2
    python -m uncertainty_sde_combined_acdc_v2.run_all_methods_select_subjects \
        --num_runs 100 --top_k 8

Skip the heavy test.py runs and just re-score from existing dumps:
    python -m uncertainty_sde_combined_acdc_v2.run_all_methods_select_subjects \
        --skip_run --top_k 8
"""

import os
import sys
import glob
import argparse
import subprocess
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation


# ---------------------------------------------------------------------------
# Paths -- mirror backbones.load_backbone_and_data so we know where test.py
# will write each backbone's per-subject results.
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"

_BACKBONE_WORKDIRS = {
    'tlrn':       f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/TLRN/basic_MSE_Penp_img1200Reg0.03_plus_ACDC",
    'ltma':       f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/TLMA_TGrad/config_plus_ACDC",
    'tm':         f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/TransMorph/basic_plus_ACDC",
    'voxelmorph': f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/VoxelMorph/voxelmorph",
    'vmplus':     f"{PROJECT_ROOT}/2026Experiments/CINE/outputs/VoxelMorph/vmplus",
}

_RESULTS_SUBPATH = "visualization/uncertainty_sde_combined_acdc_v2/test_results"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run all-method UQ test, then pick subjects with the most "
                    "distinguishable, lowest-background uncertainty maps.")
    p.add_argument("--num_runs", type=int, default=100,
                   help="UQ runs per subject (default 100 for quick check).")
    p.add_argument("--methods", nargs='+',
                   default=['tlrn', 'ltma', 'voxelmorph', 'tm'],
                   choices=list(_BACKBONE_WORKDIRS.keys()),
                   help="Backbones to evaluate (default: tlrn ltma voxelmorph tm).")
    p.add_argument("--loss", default='mse',
                   help="Similarity-loss suffix passed through to test.py.")
    p.add_argument("--use_train", action="store_true",
                   help="Pass --use_train to test.py (rare).")
    p.add_argument("--top_k", type=int, default=8,
                   help="How many top subjects to print/save.")
    p.add_argument("--bg_dilate", type=int, default=6,
                   help="Pixels to dilate the cardiac mask before treating "
                        "the rest as background. Larger = stricter background.")
    p.add_argument("--max_bg_ratio", type=float, default=0.6,
                   help="Reject subjects whose mean BG/FG uncertainty ratio "
                        "(max over methods) exceeds this. 0.6 is a permissive "
                        "default; lower it (e.g. 0.4) to be stricter.")
    p.add_argument("--min_cardiac_pixels", type=int, default=20,
                   help="Skip frames with fewer than this many cardiac pixels.")
    p.add_argument("--skip_run", action="store_true",
                   help="Don't (re-)run test.py; just analyze existing dumps.")
    p.add_argument("--report_dir", default=None,
                   help="Where to write the ranking. Default: "
                        "{repo}/uncertainty_sde_combined_acdc_v2/selected_subjects/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: drive test.py once per backbone
# ---------------------------------------------------------------------------
def run_test_for_method(method, args):
    # Use test_fast.py: shares one sampling pass across all subjects in a
    # batch and skips visualization/correlation overhead, while still dumping
    # per-subject ud_map + cardiac_mask via --save_ud_maps for the scorer.
    cmd = [
        sys.executable, "-m", "uncertainty_sde_combined_acdc_v2.test_fast",
        "--backbone", method,
        "--num_runs", str(args.num_runs),
        "--loss", args.loss,
        "--save_ud_maps",
    ]
    if args.use_train:
        cmd.append("--use_train")

    print(f"\n{'='*72}\n[run] {method}\n{'='*72}")
    print(f"  cmd: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    if rc != 0:
        print(f"WARN: test.py for {method} exited rc={rc}")
    return rc


# ---------------------------------------------------------------------------
# Step 2: collect per-subject uncertainty dumps written by test.py
# ---------------------------------------------------------------------------
def collect_subjects(method):
    """Return dict subject_id -> sorted list of per-frame npz paths."""
    base = os.path.join(_BACKBONE_WORKDIRS[method], _RESULTS_SUBPATH)
    if not os.path.isdir(base):
        print(f"  [{method}] WARN: missing results dir {base}")
        return {}

    subjects = {}
    for entry in sorted(os.listdir(base)):
        sub_dir = os.path.join(base, entry)
        per_frame = os.path.join(sub_dir, 'per_frame_curves')
        if not os.path.isdir(per_frame):
            continue
        npzs = sorted(glob.glob(os.path.join(per_frame,
                                             'correlation_frame_*.npz')))
        if npzs:
            subjects[entry] = npzs
    return subjects


def load_subject_maps(npz_paths):
    """frame_idx -> {ud_map[H,W], cardiac_mask[H,W]}."""
    out = {}
    for p in npz_paths:
        name = os.path.basename(p)
        try:
            fi = int(name.replace('correlation_frame_', '').replace('.npz', ''))
        except ValueError:
            continue
        with np.load(p) as d:
            out[fi] = {
                'ud_map':       d['ud_map'].astype(np.float32),
                'cardiac_mask': d['cardiac_mask'].astype(bool),
            }
    return out


# ---------------------------------------------------------------------------
# Step 3: scoring
# ---------------------------------------------------------------------------
def _scale_normalize(m):
    """RMS-normalize a map so cross-method differences aren't dominated by
    overall magnitude (each method may produce uncertainty on a different
    absolute scale)."""
    s = float(np.sqrt(np.mean(m * m)))
    return m / s if s > 1e-12 else m


def score_subject(per_method_maps, bg_dilate, min_cardiac_pixels):
    """
    per_method_maps: method_name -> { frame_idx -> {ud_map, cardiac_mask} }

    Returns None when the subject can't be scored (no shared frames, no usable
    cardiac mask, etc.). Otherwise:

        distinctness  -- mean over frames of the mean cross-method per-pixel
                         std of RMS-normalized uncertainty inside the
                         (dilated) cardiac region. Higher => methods disagree.
        bg_ratio_max  -- worst (largest) mean-BG / mean-FG uncertainty ratio
                         across methods. Lower => uncertainty stays on the
                         heart instead of bleeding into the background.
        composite     -- distinctness / (bg_ratio_max + 0.05). Higher = better.
    """
    methods = list(per_method_maps.keys())
    if len(methods) < 2:
        return None

    # frames present for every method
    common_frames = set.intersection(
        *(set(per_method_maps[m].keys()) for m in methods))
    if not common_frames:
        return None

    distinctness_per_frame = []
    bg_ratio_per_method = defaultdict(list)
    fg_unc_per_method   = defaultdict(list)

    for fi in sorted(common_frames):
        # cardiac mask is from the data, identical across methods for the
        # same subject -- pick any one.
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
            ud_norm = _scale_normalize(ud)
            normalized_fg_stack.append(ud_norm[fg_mask])

            fg_mean = float(ud[fg_mask].mean())
            bg_mean = float(ud[bg_mask].mean())
            fg_unc_per_method[m].append(fg_mean)
            bg_ratio_per_method[m].append(bg_mean / (fg_mean + 1e-12))

        # cross-method per-pixel std on FG, averaged
        stack = np.stack(normalized_fg_stack, axis=0)   # [M, N_fg]
        distinctness_per_frame.append(float(stack.std(axis=0).mean()))

    if not distinctness_per_frame:
        return None

    distinctness  = float(np.mean(distinctness_per_frame))
    bg_ratio_max  = max(float(np.mean(v)) for v in bg_ratio_per_method.values())
    fg_unc_max    = max(float(np.mean(v)) for v in fg_unc_per_method.values())
    bg_ratio_perm = {m: float(np.mean(v)) for m, v in bg_ratio_per_method.items()}

    composite = distinctness / (bg_ratio_max + 0.05)

    return {
        'distinctness':        distinctness,
        'bg_ratio_max':        bg_ratio_max,
        'bg_ratio_per_method': bg_ratio_perm,
        'fg_unc_max':          fg_unc_max,
        'composite':           composite,
        'n_frames':            len(distinctness_per_frame),
    }


# ---------------------------------------------------------------------------
# Step 4: report
# ---------------------------------------------------------------------------
def write_report(scored, keepers, args, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    txt_path = os.path.join(report_dir, "ranked_subjects.txt")
    npz_path = os.path.join(report_dir, "ranked_subjects.npz")

    with open(txt_path, 'w') as f:
        f.write("Selected subjects: distinguishable across methods + low background noise\n")
        f.write("=" * 100 + "\n")
        f.write(f"Methods evaluated:  {args.methods}\n")
        f.write(f"UQ runs per subject: {args.num_runs}\n")
        f.write(f"Background dilation: {args.bg_dilate} px\n")
        f.write(f"Max accepted BG/FG ratio (per subject, max over methods): {args.max_bg_ratio}\n")
        f.write(f"Min cardiac pixels per frame: {args.min_cardiac_pixels}\n\n")

        f.write(f"TOP {min(args.top_k, len(keepers))} (passed BG filter, sorted by composite)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'rank':>4}  {'subject_id':<40} {'composite':>10} "
                f"{'distinct':>10} {'bg_ratio':>10} {'fg_unc':>10} {'frames':>7}\n")
        for rank, s in enumerate(keepers[:args.top_k], 1):
            f.write(f"{rank:>4}  {s['subject_id']:<40} {s['composite']:>10.4f} "
                    f"{s['distinctness']:>10.4f} {s['bg_ratio_max']:>10.4f} "
                    f"{s['fg_unc_max']:>10.4f} {s['n_frames']:>7d}\n")

        f.write("\n\nFULL ranking (passed BG filter)\n")
        f.write("-" * 100 + "\n")
        for rank, s in enumerate(keepers, 1):
            per_m = ", ".join(f"{m}={r:.3f}" for m, r in s['bg_ratio_per_method'].items())
            f.write(f"{rank:>4}. {s['subject_id']}  composite={s['composite']:.4f}  "
                    f"distinct={s['distinctness']:.4f}  bg_ratio_max={s['bg_ratio_max']:.4f}  "
                    f"({per_m})\n")

        rejected = [s for s in scored if s['bg_ratio_max'] > args.max_bg_ratio]
        if rejected:
            f.write(f"\n\nREJECTED for high background noise (bg_ratio_max > {args.max_bg_ratio})\n")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # 1. run test.py per method
    if not args.skip_run:
        for m in args.methods:
            run_test_for_method(m, args)

    # 2. gather per-subject npz dumps
    per_method_subjects = {}
    for m in args.methods:
        per_method_subjects[m] = collect_subjects(m)
        print(f"  [{m}] subjects with per-frame uncertainty dumps: "
              f"{len(per_method_subjects[m])}")

    # 3. intersect subject_ids across methods
    subject_ids = set.intersection(
        *(set(d.keys()) for d in per_method_subjects.values()))
    print(f"\nSubjects common to ALL methods: {len(subject_ids)}")
    if not subject_ids:
        print("ERROR: no overlapping subjects -- did test.py run successfully "
              "for every method? Re-run without --skip_run, or pass --methods "
              "to limit comparison.")
        return

    # 4. score each subject
    scored = []
    for sid in sorted(subject_ids):
        per_method_maps = {
            m: load_subject_maps(per_method_subjects[m][sid])
            for m in args.methods
        }
        s = score_subject(per_method_maps, args.bg_dilate,
                          args.min_cardiac_pixels)
        if s is None:
            continue
        s['subject_id'] = sid
        scored.append(s)

    if not scored:
        print("ERROR: no subjects could be scored -- likely missing cardiac masks.")
        return

    # 5. filter on background noise; rank by composite
    keepers = [s for s in scored if s['bg_ratio_max'] <= args.max_bg_ratio]
    keepers.sort(key=lambda x: x['composite'], reverse=True)

    # 6. report
    report_dir = args.report_dir or os.path.join(
        PROJECT_ROOT, "uncertainty_sde_combined_acdc_v2", "selected_subjects")
    txt_path, npz_path = write_report(scored, keepers, args, report_dir)

    print(f"\nReport: {txt_path}")
    print(f"NPZ:    {npz_path}")
    print(f"\nTop {min(args.top_k, len(keepers))} selected subjects:")
    for rank, s in enumerate(keepers[:args.top_k], 1):
        print(f"  {rank:>2}. {s['subject_id']}   "
              f"composite={s['composite']:.4f}   "
              f"distinct={s['distinctness']:.4f}   "
              f"bg_ratio={s['bg_ratio_max']:.4f}")


if __name__ == "__main__":
    main()
