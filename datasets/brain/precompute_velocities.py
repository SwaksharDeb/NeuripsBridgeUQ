"""
Precompute TLRN velocity fields for the brain SDE UQ pipeline.

Iterates over ALL raw brain .pt files (same source as VxM), builds
cyclic image sequences, runs the 2D TLRN registration model, and saves
precomputed data in the same format as VxM.

Output structure:
    dataset/precomputed_brain_z64_tlrn/{split}/{class}/{filename}_precomputed.pt
    Each file contains:
        - 'series': [7, 128, 128] cyclic 2D axial images
        - 'v_series': list of 7 velocity fields, each [2, 128, 128]
        - 'label': disease class (0=CN, 1=AD, 2=MCI)

Run with:
    cd /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2
    python -m uncertainty_brain_sde.precompute_velocities
"""

import os
import sys

PROJECT_ROOT = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"
RAW_DATA_PATH = "/scratch/swd9tc/4D_brain_v2_multigpu/dataset"

sys.path.insert(0, PROJECT_ROOT)

import argparse
import importlib.util
import torch
from tqdm import tqdm


def std_img(tens):
    """Standardize image to [0, 1]."""
    return (tens - tens.min()) / (tens.max() - tens.min() + 1e-8)


def build_cyclic_sequence(brain_4d, axial_idx=64):
    """
    Build a cyclic 2D image sequence by mirroring the 4 brain timepoints.
    [t0, t1, t2, t3, t2, t1, t0] -> 7 frames
    """
    slices = [brain_4d[t, :, :, axial_idx] for t in range(4)]
    cyclic = [slices[0], slices[1], slices[2], slices[3],
              slices[2], slices[1], slices[0]]
    return torch.stack(cyclic, dim=0)  # [7, 128, 128]


def extract_velocities(model, series_hw):
    """
    Run TLRN registration on a 7-frame cyclic sequence and return
    velocity fields with zero endpoints matching VxM format.

    TLRN expects (W,H) images, so we transpose before/after.

    Returns: list of 7 velocity tensors [1, 2, H, W] with zero at both endpoints.
    """
    series_wh = series_hw.transpose(-2, -1)
    lv_segs = torch.zeros_like(series_wh[:, 0:1])

    with torch.no_grad():
        _, v_series_wh, _, _, _ = model.sequence_register_no_avg_lowf_addlatentf(
            series_wh, lv_segs
        )

    # Model returns 6 velocities (0->1 through 0->6).
    # Drop the last one (0->6, return frame) to match VxM's 5 interior velocities.
    # Then transpose back to (H,W) and swap vx/vy.
    v_series_hw = []
    for v in v_series_wh[:-1]:
        v_t = v.transpose(-2, -1)
        v_swapped = torch.stack([v_t[:, 1], v_t[:, 0]], dim=1)
        v_series_hw.append(v_swapped)

    zero = torch.zeros_like(v_series_hw[0])
    return [zero] + v_series_hw + [zero.clone()]  # 7 total


def main():
    parser = argparse.ArgumentParser(description="Precompute TLRN velocity fields")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'dataset', 'precomputed_brain_z64_tlrn'))
    parser.add_argument("--raw_data", type=str, default=RAW_DATA_PATH,
                        help="Path to raw 3D brain .pt files")
    parser.add_argument("--axial_idx", type=int, default=64)
    args = parser.parse_args()

    out_dir = args.output_dir

    # Load TLRN registration model
    print("\nLoading TLRN registration model...")
    _own_spec = importlib.util.spec_from_file_location(
        "own_data_utils", os.path.join(PROJECT_ROOT, "uncertainty_brain_sde", "data_utils.py"))
    _own_du = importlib.util.module_from_spec(_own_spec)
    _own_spec.loader.exec_module(_own_du)
    model, _, _, _, config = _own_du.load_model_and_data()

    class_map = {
        'ad_split': ('ad', 1),
        'cn_split': ('cn', 0),
        'mci_split': ('mci', 2),
    }

    total_processed = 0
    total_skipped = 0

    for class_dir, (class_name, label) in class_map.items():
        for split in ['train', 'val', 'test']:
            src_dir = os.path.join(args.raw_data, class_dir, split)
            if not os.path.isdir(src_dir):
                print(f"Skipping (not found): {src_dir}")
                continue

            dst_dir = os.path.join(out_dir, split, class_name)
            os.makedirs(dst_dir, exist_ok=True)

            files = sorted([f for f in os.listdir(src_dir) if f.endswith('.pt')])
            print(f"\nProcessing {class_name}/{split}: {len(files)} files")

            for fname in tqdm(files, desc=f"{class_name}/{split}"):
                out_fname = fname.replace('.pt', '_precomputed.pt')
                out_path = os.path.join(dst_dir, out_fname)
                if os.path.exists(out_path):
                    total_skipped += 1
                    continue

                try:
                    # Load raw 3D brain data [4, 128, 128, 128]
                    brain_4d = torch.load(os.path.join(src_dir, fname), weights_only=False)
                    brain_4d = std_img(brain_4d)

                    # Build cyclic 2D image sequence (same as VxM)
                    series = build_cyclic_sequence(brain_4d, axial_idx=args.axial_idx)
                    # series: [7, 128, 128]

                    # Run TLRN registration on the cyclic sequence
                    series_batch = series.unsqueeze(0).cuda()  # [1, 7, 128, 128]
                    v_series_bb = extract_velocities(model, series_batch)
                    # v_series_bb: list of 7, each [1, 2, 128, 128]

                    torch.save({
                        'series': series.cpu(),
                        'v_series': [v[0].cpu() for v in v_series_bb],
                        'label': label,
                    }, out_path)

                    total_processed += 1
                    del brain_4d, series, series_batch, v_series_bb
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n  Error processing {fname}: {e}")
                    continue

    print(f"\nDone! Processed: {total_processed}, Skipped (already exist): {total_skipped}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
