"""
Single-ptid variant of uncertainty_brain_sde_v2/test.py.

Given a patient ID via --ptid, locate that subject in the BrainDataset
enumeration order used by test.py (ConcatDataset(train+val+test) with
batch_size=4), look up its precomputed s^K row by filename, and run UQ +
visualization for just that one subject. Output directory is named
"batch{batch_idx}_{ptid}" so it lines up with test.py's per-subject dirs.

Run:
    python -m uncertainty_brain_sde_v2.test_single_ptid \
        --ptid 002_S_4262_0 --backbone tlrn
"""

import os
import sys

sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import torch
import gc
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from uncertainty_brain_sde_v2.losses import LTMALoss
from uncertainty_brain_sde_v2.brownian_bridge import BrownianBridgeLearnedScaling
from uncertainty_brain_sde_v2.trainer import ScalingFactorTrainer
from uncertainty_brain_sde.data_utils import BrainDataset
from uncertainty_brain_sde_v2.test_fast import (
    _BACKBONE_DATA_PATHS,
    _backbone_workdir,
    _load_s_K_lookup,
    _assemble_scaling_factors_bb,
)
from uncertainty_brain_sde_v2 import test as _test_module
from uncertainty_brain_sde_v2.test import test_single_subject


BATCH_SIZE = 4  # must match test.py so batch_idx maps to the same index used there
LAMBDA_SCALE = 3
ALPHA_SCALE = 0.0001


def _find_subject_by_ptid(datasets, ptid):
    """Walk ConcatDataset(train+val+test) in order, return (global_idx, sample).

    Mirrors test.py's iteration order so the resulting batch_idx matches the
    on-disk batch{N}_{ptid} naming scheme.
    """
    global_idx = 0
    for ds in datasets:
        for local_idx in range(len(ds)):
            sample = ds[local_idx]
            if sample['ptid'] == ptid:
                return global_idx, sample
            global_idx += 1
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Run Brain SDE v2 UQ test for a single patient (by ptid)."
    )
    parser.add_argument("--ptid", type=str, default='002_S_4262_0',
                        help="Patient ID to test, e.g. 002_S_4262_0")
    parser.add_argument("--num_runs", type=int, default=3000,
                        help="Number of UQ runs")
    parser.add_argument("--backbone",
                        choices=list(_BACKBONE_DATA_PATHS.keys()),
                        default='tlrn',
                        help="Pretrained registration backbone")
    parser.add_argument("--loss", type=str, default='mse',
                        help="Loss tag used to construct the workdir")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Override workdir whose visualization/uncertainty_brain_sde_v2/s_K is loaded")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override the auto-resolved precomputed data path")
    parser.add_argument("--dataset", type=str, default="mni88",
                        choices=["mni88", "full"],
                        help="Dataset selection")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: "
                             "{workdir}/visualization/uncertainty_brain_sde_v2/test_results_2)")
    parser.add_argument("--uncertainty_space", type=str, default="velocity",
                        choices=["velocity", "image"])
    parser.add_argument("--save_individual", action="store_true")
    args = parser.parse_args()

    workdir = args.workdir or _backbone_workdir(args.backbone, args.loss)
    data_path = args.data_path or _BACKBONE_DATA_PATHS[args.backbone]
    print(f"Backbone:   {args.backbone.upper()}")
    print(f"Data path:  {data_path}")
    print(f"Workdir:    {workdir}")
    print(f"Target ptid: {args.ptid}")

    # ----- Locate the subject in the same iteration order as test.py -----
    print(f"\nLoading BrainDataset (dataset={args.dataset})...")
    train_dataset = BrainDataset(data_path, split='all', mode='train', dataset=args.dataset)
    val_dataset   = BrainDataset(data_path, split='all', mode='val',   dataset=args.dataset)
    test_dataset  = BrainDataset(data_path, split='all', mode='test',  dataset=args.dataset)

    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"ConcatDataset size: {total}")

    global_idx, sample = _find_subject_by_ptid(
        [train_dataset, val_dataset, test_dataset], args.ptid
    )
    if sample is None:
        raise SystemExit(
            f"ptid '{args.ptid}' not found in ConcatDataset(train+val+test). "
            f"Check that --dataset ({args.dataset}) and --backbone ({args.backbone}) match "
            f"the run that produced the s_K folder."
        )
    batch_idx = global_idx // BATCH_SIZE
    subj_in_batch = global_idx % BATCH_SIZE
    print(f"Found ptid at global index {global_idx} -> batch_idx={batch_idx}, "
          f"subj_in_batch={subj_in_batch}")
    print(f"Filename: {sample['filename']}")

    # ----- Load s^K lookup and find this subject's row -----
    print("\nLoading s^K lookup...")
    s_K_lookup, s_K_meta = _load_s_K_lookup(workdir)
    NUM_DIFFUSION_STEPS = s_K_meta['num_diffusion_steps']

    raw = s_K_lookup.get(sample['filename'])
    if raw is None:
        # Fallback: maybe stored under the global-index sentinel from main.py
        fallback_key = f"subj_{global_idx:05d}"
        raw = s_K_lookup.get(fallback_key)
        if raw is None:
            available = list(s_K_lookup.keys())[:5]
            raise SystemExit(
                f"No saved s^K for filename '{sample['filename']}' or fallback "
                f"'{fallback_key}'. Re-run main.py first to populate s_K. "
                f"Example saved keys: {available}"
            )
        print(f"  matched via fallback global-index key '{fallback_key}'")
    else:
        print(f"  matched via filename key '{sample['filename']}'")

    # ----- Build a 1-element batch -----
    series = sample['series'].unsqueeze(0).cuda()                    # [1, T, H, W]
    v_series = [v.unsqueeze(0).cuda() for v in sample['v_series']]   # list of [1, 2, H, W]
    img_size = series.shape[-1]
    num_time_steps = len(v_series) - 1

    source_img = series[:, 0:1]
    target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]
    v_series_bb = v_series

    scaling_factors_bb = _assemble_scaling_factors_bb(
        [raw], s_K_meta['num_cardiac_frames'], device='cuda'
    )

    # ----- Trainer / brownian bridge (same as test.py) -----
    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=NUM_DIFFUSION_STEPS, img_size=img_size
    )
    loss_fn = LTMALoss(
        lambda_sim=1.0, lambda_reg=0.0,
        lambda_scale=LAMBDA_SCALE, alpha_scale=ALPHA_SCALE,
    )
    trainer = ScalingFactorTrainer(
        scaling_network=None, brownian_bridge=brownian_bridge,
        loss_fn=loss_fn, lr=1e-4, device='cuda', img_size=img_size,
        gamma=s_K_meta['gamma'], guidance_scale=s_K_meta['guidance_scale'],
    )

    # ----- UQ sampling -----
    print(f"\nRunning UQ ({args.num_runs} runs)...")
    num_frames_bb = len(v_series_bb)
    sampled_velocities = {i: [] for i in range(num_frames_bb)}

    for run in tqdm(range(args.num_runs), desc="UQ runs"):
        v_t_list, _ = trainer.compute_brownian_bridge_velocities(
            v_series_bb, scaling_factors_bb,
            source_img=source_img, target_imgs=target_imgs,
        )
        for frame_idx in range(num_frames_bb):
            sampled_velocities[frame_idx].append(v_t_list[frame_idx].detach().clone())
        if (run + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # ----- Resolve output dir + run per-subject visualization -----
    OUTPUT_DIR = args.output_dir or os.path.join(
        workdir, "visualization", "uncertainty_brain_sde_v2"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # test_single_subject defaults save_folder to {test_module.OUTPUT_DIR}/test_results_2/{subject_id}
    # when no output_dir is passed; setting it here makes the layout match test.py.
    _test_module.OUTPUT_DIR = OUTPUT_DIR
    print(f"Output dir base: {OUTPUT_DIR}")

    ncc_vx, std, subject_id, reg_metrics, _, _ = test_single_subject(
        subject_idx=0,
        sampled_velocities=sampled_velocities,
        v_series_bb=v_series_bb,
        series=series,
        scaling_factors_bb=scaling_factors_bb,
        brownian_bridge=brownian_bridge,
        training_history=[],
        ckpt_config={},
        trained_epochs=s_K_meta.get('inner_iterations', 0) if isinstance(s_K_meta, dict) else 0,
        num_time_steps=num_time_steps,
        NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS,
        NUM_RUNS=args.num_runs,
        img_size=img_size,
        batch_idx=batch_idx,
        output_dir=args.output_dir,  # None -> falls back to {OUTPUT_DIR}/test_results_2/<subject_id>
        ptid=sample['ptid'],
        age_list=sample['age_list'],
        save_individual=args.save_individual,
        filename=sample['filename'],
        uncertainty_space=args.uncertainty_space,
    )

    print("\n" + "=" * 60)
    print(f"Done: subject_id={subject_id}")
    print(f"  NCC_VX: {ncc_vx:.6f} +/- {std:.6f}")
    if reg_metrics is not None:
        r = reg_metrics['registration']
        m = reg_metrics['mean_velocity']
        print(f"  Reg RMSE: {r['avg_rmse']:.6f}, Neg Jac %: {r['avg_neg_jac_pct']:.4f}%")
        print(f"  MV  RMSE: {m['avg_rmse']:.6f}, Neg Jac %: {m['avg_neg_jac_pct']:.4f}%")

    del series, v_series, v_series_bb, source_img, target_imgs
    del scaling_factors_bb, sampled_velocities
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
