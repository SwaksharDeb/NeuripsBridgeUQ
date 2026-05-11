#!/usr/bin/env python3
"""
Run all 3 model test scripts for a single subject and collect visualizations.

Runs:
  1. TLRN   — runs uncertainty_inv_gamma_prior/test.py
  2. VxM    — runs voxelmorph_prob/test_cine.py
  3. PULPo  — runs pulbo/test.py

All outputs collected into one folder for side-by-side comparison.

Usage:
    python uncertainty_inv_gamma_prior/run_subject_visualization.py --subject_idx 10
"""

import os
import sys
import argparse
import subprocess
import glob

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"
VXM_DIR = "/scratch/swd9tc/Uncertanity_quantification/voxelmorph_prob"
PULPO_DIR = "/scratch/swd9tc/Uncertanity_quantification/pulbo"

TLRN_SCRIPT = os.path.join(PROJECT_ROOT, "uncertainty_inv_gamma_prior", "test.py")

DEFAULT_OUTPUT = f"{PROJECT_ROOT}/uncertainty_inv_gamma_prior/subject_visualizations"


def run_tlrn(subject_idx, out_dir, num_runs=100, device="cuda", save_individual=False):
    """Run TLRN test.py for a single subject."""
    tlrn_out = os.path.join(out_dir, "tlrn")
    os.makedirs(tlrn_out, exist_ok=True)

    cmd = [
        sys.executable,
        TLRN_SCRIPT,
        "--subject_idx", str(subject_idx),
        "--num_runs", str(num_runs),
        "--output_dir", tlrn_out,
        "--device", device,
    ]
    if save_individual:
        cmd.append("--save_individual")
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: TLRN test exited with code {result.returncode}")
        return False
    return True


def run_vxm(subject_idx, out_dir, num_samples=100, device="cuda", save_individual=False):
    """Run VoxelMorph Prob test_cine.py for a single subject."""
    vxm_out = os.path.join(out_dir, "vxm")
    os.makedirs(vxm_out, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(VXM_DIR, "test_cine.py"),
        "--sequence_idx", str(subject_idx),
        "--num_samples", str(num_samples),
        "--output_dir", vxm_out,
        "--device", device,
    ]
    if save_individual:
        cmd.append("--save_individual")
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=VXM_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: VxM test exited with code {result.returncode}")
        return False
    return True


def run_pulpo(subject_idx, out_dir, num_samples=100, device="cuda", save_individual=False):
    """Run PULPo test.py for a single subject."""
    pulpo_out = os.path.join(out_dir, "pulpo")
    os.makedirs(pulpo_out, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(PULPO_DIR, "test.py"),
        "--no_eval_all",
        "--sequence_idx", str(subject_idx),
        "--num_samples", str(num_samples),
        "--output_dir", pulpo_out,
        "--device", device,
    ]
    if save_individual:
        cmd.append("--save_individual")
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PULPO_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: PULPo test exited with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run 3-model visualizations for a single test subject"
    )
    parser.add_argument("--subject_idx", type=int, default=10,
                        help="Subject index in test file list (0-indexed)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help="Base output directory")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of MC samples for VxM and PULPo")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of UQ runs for TLRN (default: 100)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_tlrn", action="store_true",
                        help="Skip TLRN")
    parser.add_argument("--skip_vxm", action="store_true",
                        help="Skip VoxelMorph")
    parser.add_argument("--skip_pulpo", action="store_true",
                        help="Skip PULPo")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save individual images (source, target, warped, uncertainty)")
    args = parser.parse_args()

    subj_dir = os.path.join(args.output_dir, f"subject_{args.subject_idx}")
    os.makedirs(subj_dir, exist_ok=True)

    print(f"Subject: {args.subject_idx}")
    print(f"Output:  {subj_dir}")
    print()

    # ============================================================
    # 1. TLRN
    # ============================================================
    if not args.skip_tlrn:
        print("=" * 60)
        print("TLRN — running test.py")
        print("=" * 60)
        run_tlrn(args.subject_idx, subj_dir,
                 num_runs=args.num_runs, device=args.device,
                 save_individual=args.save_individual)
    else:
        print("Skipping TLRN")

    # ============================================================
    # 2. VoxelMorph Probabilistic
    # ============================================================
    if not args.skip_vxm:
        print()
        print("=" * 60)
        print("VoxelMorph Probabilistic — running test_cine.py")
        print("=" * 60)
        run_vxm(args.subject_idx, subj_dir,
                num_samples=args.num_samples, device=args.device,
                save_individual=args.save_individual)
    else:
        print("Skipping VxM")

    # ============================================================
    # 3. PULPo
    # ============================================================
    if not args.skip_pulpo:
        print()
        print("=" * 60)
        print("PULPo — running test.py")
        print("=" * 60)
        run_pulpo(args.subject_idx, subj_dir,
                  num_samples=args.num_samples, device=args.device,
                  save_individual=args.save_individual)
    else:
        print("Skipping PULPo")

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 60)
    print("Done! Outputs:")
    print("=" * 60)
    for model_name in ["tlrn", "vxm", "pulpo"]:
        model_dir = os.path.join(subj_dir, model_name)
        if os.path.isdir(model_dir):
            pngs = glob.glob(os.path.join(model_dir, "*.png"))
            indiv_dir = os.path.join(model_dir, "individual_images")
            indiv_pngs = glob.glob(os.path.join(indiv_dir, "*.png")) if os.path.isdir(indiv_dir) else []
            print(f"  {model_name}: {len(pngs)} panel images in {model_dir}")
            if indiv_pngs:
                print(f"          {len(indiv_pngs)} individual images in {indiv_dir}")
        else:
            print(f"  {model_name}: (not available)")

    print(f"\nAll results: {subj_dir}")


if __name__ == "__main__":
    main()
