#!/usr/bin/env python3
"""
Compare brain UQ across all four registration backbones:
TLRN, LTMA (TGrad), TransMorph, and VoxelMorph.

All methods use BrainDataset with precomputed velocities (same pipeline).

Usage:
    cd /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2
    python -m uncertainty_brain_sde.compare_four_methods --ptid 009_S_4324_0 --num_runs 1000
"""

import os
import sys
import re
import argparse
import glob as globmod

import torch
import numpy as np
import gc
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

from uncertainty_brain_sde.networks import ScalingFactorNetwork
from uncertainty_brain_sde.brownian_bridge import BrownianBridgeLearnedScaling
from uncertainty_brain_sde.losses import LTMALoss
from uncertainty_brain_sde.trainer import ScalingFactorTrainer
from uncertainty_brain_sde.data_utils import BrainDataset
from torch.utils.data import DataLoader, ConcatDataset

FRAME_LABELS = ["t0", "t1", "t2", "t3 (peak)", "t2'", "t1'", "t0 (return)"]

# Per-method config: data_path and UQ checkpoint directory
METHOD_CONFIG = {
    'TLRN': {
        'data_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/dataset/precomputed_brain_z64_tlrn',
        'ckpt_dir': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/2026Experiments/BRAIN/outputs/TLRN/basic_brain/visualization/uncertainty_brain_sde/checkpoints',
    },
    'LTMA': {
        'data_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/dataset/precomputed_brain_z64_ltma',
        'ckpt_dir': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/2026Experiments/BRAIN/outputs/TLMA_TGrad/basic_brain/visualization/uncertainty_brain_sde/checkpoints',
    },
    'TM': {
        'data_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/TM/dataset/precomputed_brain_z64_tm',
        'ckpt_dir': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/TM/2026Experiments/BRAIN/outputs/TransMorph/basic_brain/visualization/uncertainty_brain_sde/checkpoints',
    },
    'VxM': {
        'data_path': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty/dataset/precomputed_brain_z64',
        'ckpt_dir': '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty/outputs/uncertainty_brain_sde/checkpoints',
    },
}


def _find_uq_checkpoint(ckpt_dir):
    ckpt_files = globmod.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pth'))
    if not ckpt_files:
        raise FileNotFoundError(f"No UQ checkpoints in {ckpt_dir}")
    def _epoch_num(p):
        m = re.search(r'checkpoint_epoch_(\d+)\.pth', p)
        return int(m.group(1)) if m else 0
    return max(ckpt_files, key=_epoch_num)


def collate_brain_batch(batch):
    series = torch.stack([b['series'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch])
    filenames = [b['filename'] for b in batch]
    ptids = [b['ptid'] for b in batch]
    age_lists = [b['age_list'] for b in batch]
    num_frames = len(batch[0]['v_series'])
    v_series = [torch.stack([b['v_series'][t] for b in batch]) for t in range(num_frames)]
    return {'series': series, 'v_series': v_series, 'label': labels,
            'filename': filenames, 'ptid': ptids, 'age_list': age_lists}


def run_method_pipeline(method, target_ptid, subject_idx_fallback, num_runs, dataset='mni88'):
    """Run UQ pipeline for one method using BrainDataset."""
    print(f"\n{'='*60}")
    print(f"{method} Pipeline")
    print(f"{'='*60}")

    cfg = METHOD_CONFIG[method]
    data_path = cfg['data_path']
    ckpt_dir = cfg['ckpt_dir']

    # Load data
    train_ds = BrainDataset(data_path, split='all', mode='train', dataset=dataset)
    val_ds = BrainDataset(data_path, split='all', mode='val', dataset=dataset)
    test_ds = BrainDataset(data_path, split='all', mode='test', dataset=dataset)
    eval_ds = ConcatDataset([train_ds, val_ds, test_ds])
    loader = DataLoader(eval_ds, batch_size=4, shuffle=False,
                        collate_fn=collate_brain_batch, num_workers=0, pin_memory=True)
    print(f"  Loaded {len(eval_ds)} subjects")

    # Find target subject
    series_found = v_bb_found = si_found = ptid_found = None
    for batch_idx, batch in enumerate(loader):
        if series_found is not None:
            break
        ptids = batch['ptid']
        for si in range(len(ptids)):
            match = (target_ptid is not None and ptids[si] == target_ptid) or \
                    (target_ptid is None and batch_idx == 0 and si == subject_idx_fallback)
            if match:
                series_found = batch['series'].cuda()
                v_bb_found = [v.cuda() for v in batch['v_series']]
                si_found = si
                ptid_found = ptids[si]
                print(f"  Found: ptid={ptid_found}, batch={batch_idx}, idx={si}")
                break

    if series_found is None:
        raise ValueError(f"ptid={target_ptid} not found in {method} data")

    # Velocity diagnostics
    print(f"  [{method}] Velocity diagnostics (subject {si_found}):")
    for fi, v in enumerate(v_bb_found):
        vs = v[si_found].cpu().numpy()
        vm = np.sqrt(vs[0]**2 + vs[1]**2)
        print(f"    frame {fi}: ||v|| mean={np.mean(vm):.6f}, std={np.std(vm):.6f}, max={np.max(vm):.6f}")

    # Load UQ checkpoint
    ckpt_path = _find_uq_checkpoint(ckpt_dir)
    print(f"    \033[91mUQ checkpoint: {ckpt_path}\033[0m")

    num_time_steps = len(v_bb_found) - 1
    img_size = series_found.shape[-1]

    scaling_net = ScalingFactorNetwork(num_time_steps=num_time_steps, img_size=img_size, num_heads=8)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    scaling_net.load_state_dict(checkpoint['model_state_dict'])
    scaling_net.cuda().eval()

    ckpt_config = checkpoint.get('config', {})
    epoch = checkpoint.get('epoch', 0)
    print(f"    epoch {epoch}, loss {checkpoint.get('best_loss', 0):.6f}")

    bb = BrownianBridgeLearnedScaling(num_diffusion_steps=7, img_size=img_size)
    loss_fn = LTMALoss(lambda_sim=1.0, lambda_reg=0.0,
                       lambda_scale=ckpt_config.get('lambda_scale', 3),
                       alpha_scale=ckpt_config.get('alpha_scale', 0.0001), use_ncc=False)
    trainer = ScalingFactorTrainer(scaling_network=scaling_net, brownian_bridge=bb,
                                   loss_fn=loss_fn, lr=1e-4, device='cuda', img_size=img_size,
                                   gamma=ckpt_config.get('gamma', 0.00005),
                                   guidance_scale=ckpt_config.get('guidance_scale', 1.0))

    # Run UQ
    source_img = series_found[:, 0:1]
    target_imgs = [series_found[:, t:t+1] for t in range(1, series_found.shape[1])]

    print(f"  [{method}] Computing scaling factors...")
    _, scaling_factors_bb = trainer.validate(source_img, target_imgs, v_bb_found)

    num_frames_bb = len(v_bb_found)
    sampled_velocities = {i: [] for i in range(num_frames_bb)}

    print(f"  [{method}] Running {num_runs} UQ samples...")
    for run in tqdm(range(num_runs), desc=f"  {method} UQ"):
        sampled = bb.sample_bridge_transition(v_bb_found, scaling_factors_bb, device='cuda')
        for fi in range(num_frames_bb):
            sampled_velocities[fi].append(sampled[fi].clone())
        del sampled
        if (run + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Extract results
    frame_imgs = [series_found[si_found, t].cpu().numpy() for t in range(series_found.shape[1])]
    var_mags, vel_mags = [], []
    for fi in range(num_frames_bb):
        vs = v_bb_found[fi][si_found].cpu().numpy()
        vel_mags.append(np.sqrt(vs[0]**2 + vs[1]**2))
        if fi in sampled_velocities and len(sampled_velocities[fi]) > 0:
            vels = np.stack([v[si_found].cpu().numpy() for v in sampled_velocities[fi]], axis=0)
            variance = np.var(vels, axis=0)
            var_mags.append(np.sqrt(variance[0] + variance[1]))
        else:
            var_mags.append(np.zeros_like(frame_imgs[0]))

    del scaling_net, bb, trainer, series_found, v_bb_found, sampled_velocities, scaling_factors_bb
    torch.cuda.empty_cache()
    gc.collect()
    return ptid_found, frame_imgs, var_mags, vel_mags


# ========================================================================
# Visualization
# ========================================================================

def make_combined_figure(image_sequence, method_data, vel_data, save_path, ptid):
    methods = list(method_data.keys())
    num_frames = len(image_sequence)
    nrows = 1 + len(methods)

    # Velocity statistics
    print(f"\n{'='*100}")
    print(f"Velocity Field Statistics (||v||) — ptid={ptid}")
    print(f"{'='*100}")
    header = f"{'Frame':>6}"
    for m in methods:
        header += f" | {m+' mean':>14} {m+' std':>12} {m+' max':>12}"
    print(header)
    print("-" * len(header))
    for fi in range(num_frames):
        line = f"{fi:>6}"
        for m in methods:
            v = vel_data[m][fi]
            line += f" | {np.mean(v):>14.6f} {np.std(v):>12.6f} {np.max(v):>12.6f}"
        print(line)
    line = f"{'ALL':>6}"
    for m in methods:
        all_v = np.concatenate([v.ravel() for v in vel_data[m]])
        line += f" | {np.mean(all_v):>14.6f} {np.std(all_v):>12.6f} {np.max(all_v):>12.6f}"
    print(line)

    # Uncertainty statistics
    print(f"\n{'='*100}")
    print(f"Uncertainty Statistics (var mag) — ptid={ptid}")
    print(f"{'='*100}")
    header = f"{'Frame':>6}"
    for m in methods:
        header += f" | {m+' mean':>14} {m+' std':>12} {m+' max':>12}"
    print(header)
    print("-" * len(header))
    for fi in range(num_frames):
        line = f"{fi:>6}"
        for m in methods:
            v = method_data[m][fi]
            line += f" | {np.mean(v):>14.6f} {np.std(v):>12.6f} {np.max(v):>12.6f}"
        print(line)
    line = f"{'ALL':>6}"
    overall_means = {}
    for m in methods:
        all_v = np.concatenate([v.ravel() for v in method_data[m]])
        overall_means[m] = np.mean(all_v)
        line += f" | {np.mean(all_v):>14.6f} {np.std(all_v):>12.6f} {np.max(all_v):>12.6f}"
    print(line)

    ref = methods[0]
    ref_mean = overall_means[ref]
    print(f"\nRatio (relative to {ref}):")
    for m in methods:
        ratio = overall_means[m] / ref_mean if ref_mean > 1e-10 else float('inf')
        print(f"  {m}: {ratio:.4f}x")

    # Per-method var_vmax
    method_var_vmax = {}
    for m in methods:
        vmax = 0.0
        for fi in range(1, num_frames - 1):
            vmax = max(vmax, np.max(method_data[m][fi]))
        method_var_vmax[m] = max(vmax, 0.1)
        print(f"  {m} var_vmax = {method_var_vmax[m]:.6f}")

    # Main figure
    fig, axes = plt.subplots(nrows, num_frames, figsize=(2.2 * num_frames, 2.2 * nrows + 0.5))
    for fi in range(num_frames):
        axes[0, fi].imshow(image_sequence[fi], cmap='gray', vmin=0, vmax=1)
        label = FRAME_LABELS[fi] if fi < len(FRAME_LABELS) else f'Frame {fi}'
        axes[0, fi].set_title(label, fontsize=9)
        axes[0, fi].set_xticks([]); axes[0, fi].set_yticks([])
        for mi, m in enumerate(methods):
            im = axes[1+mi, fi].imshow(method_data[m][fi], cmap='jet',
                                        vmin=0.3, vmax=method_var_vmax[m], alpha=1.0)
            axes[1+mi, fi].set_xticks([]); axes[1+mi, fi].set_yticks([])
            if fi == num_frames - 1:
                plt.colorbar(im, ax=axes[1+mi, fi], fraction=0.046)
    axes[0, 0].set_ylabel('Brain Images', fontsize=11, fontweight='bold')
    for mi, m in enumerate(methods):
        axes[1+mi, 0].set_ylabel(f'{m}', fontsize=11, fontweight='bold')
    plt.suptitle(f'Brain UQ Comparison — {ptid}', fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison: {save_path}")
    plt.close()

    # Bar chart
    bar_path = save_path.replace('.png', '_bar.png')
    fig3, ax = plt.subplots(figsize=(6, 4))
    method_names = list(overall_means.keys())
    means = [overall_means[m] for m in method_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(method_names)]
    bars = ax.bar(method_names, means, color=colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean Velocity Variance Magnitude')
    ax.set_title(f'Overall Uncertainty Level — {ptid}')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar chart: {bar_path}")
    plt.close()


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare brain UQ across TLRN, LTMA, TransMorph, and VoxelMorph"
    )
    parser.add_argument("--ptid", type=str, default='009_S_4324_1')
    parser.add_argument("--subject_idx", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs='+',
                        default=['TLRN', 'LTMA', 'TM', 'VxM'],
                        choices=['TLRN', 'LTMA', 'TM', 'VxM'])
    parser.add_argument("--dataset", type=str, default='mni88',
                        choices=['mni88', 'full'])
    args = parser.parse_args()

    results = {}
    vel_results = {}
    image_sequence = None
    found_ptid = None

    for method in args.methods:
        target_ptid = args.ptid or found_ptid
        ptid, imgs, vmags, velmags = run_method_pipeline(
            method, target_ptid, args.subject_idx, args.num_runs, args.dataset
        )
        if found_ptid is None:
            found_ptid = ptid
        if image_sequence is None:
            image_sequence = imgs
        results[method] = vmags
        vel_results[method] = velmags
        print(f"  [{method}] Done")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(
            "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2",
            "2026Experiments/BRAIN/outputs/TLRN/basic_brain/visualization",
            "uncertainty_brain_sde/comparison_four_methods"
        )
    save_path = os.path.join(out_dir, f"comparison_{found_ptid}.png")
    make_combined_figure(image_sequence, results, vel_results, save_path, found_ptid)
    print("\nAll done.")


if __name__ == "__main__":
    main()
