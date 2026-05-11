"""
Data utility functions for brain uncertainty quantification with TLRN backbone.

Loads the brain-trained TLRN model and brain NIfTI data loaders.
"""

import os
import sys
import glob as globmod
import random

project_root = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from box import Box

from utils.utils import instantiate_from_config, merge_yaml_files
from utils.utils_train import load_checkpoints


def load_model_and_data():
    """
    Load the brain-trained TLRN model and brain data loaders.

    Returns:
        model: Loaded TLRN model (trained on brain data)
        train_loader: Training data loader (brain)
        val_loader: Validation data loader (brain)
        test_loader: Test data loader (brain)
        config: Configuration object
    """
    setobj = Box({
        "name": "BRAIN",
        "model": "TLRN",
        "editfile": "basic_brain"
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_path = os.path.join(project_root, "2026Experiments", setobj.name)

    config_file = f"{target_path}/settings/{setobj.model}/{setobj.editfile}.yaml"

    seed_everything(20250101)
    config = OmegaConf.load(config_file)

    # Auto-detect latest epoch checkpoint
    import re
    ckpt_dir = f"{target_path}/outputs/{setobj.model}/{setobj.editfile}/checkpoints"

    def _extract_epoch(path):
        m = re.search(r'epoch=(\d+)', path)
        return int(m.group(1)) if m else 0

    all_ckpts = globmod.glob(f"{ckpt_dir}/epoch=*.ckpt")
    if all_ckpts:
        best_ckpt = max(all_ckpts, key=_extract_epoch)
        config.model.pretrained_checkpoint = best_ckpt
        print(f"Auto-detected checkpoint (epoch {_extract_epoch(best_ckpt)}): {best_ckpt}")
    else:
        print(f"WARNING: No checkpoints found in {ckpt_dir}")

    config.model.params.workdir = f"{target_path}/outputs/{setobj.model}/{setobj.editfile}"
    config.model.params.setobj_info = f"{setobj.name}_{setobj.model}_{setobj.editfile}"
    config.model.params.test_split = config.data.params.test.params.split
    config.model.params.Start_Subject_ID = 0

    print("Loading model...")
    model = instantiate_from_config(config.model)
    load_checkpoints(model, config.model)
    model.eval()
    model.cuda()

    print("Loading data...")
    data = instantiate_from_config(config.data)
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Val loader: {len(val_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")

    return model, train_loader, val_loader, test_loader, config


class BrainDataset(Dataset):
    """
    PyTorch Dataset for precomputed brain data.

    Loads precomputed .pt files containing cyclic image sequences and velocity fields.
    Each file contains:
        - 'series': [7, 128, 128] cyclic 2D axial images
        - 'v_series': list of 7 velocity fields, each [2, 128, 128]
        - 'label': disease class (0=CN, 1=AD, 2=MCI)
    """

    def __init__(self, data_path, split='all', mode='train',
                 csv_dir='/scratch/swd9tc/4D_brain_v2_multigpu/dataset',
                 dataset='mni88'):
        self.data_path = data_path
        self.mode = mode
        self.files = []

        df1 = pd.read_csv(os.path.join(csv_dir, 'MNI_88.csv'))[['ptid', 'age_list']]
        if dataset == 'full':
            df2 = pd.read_csv(os.path.join(csv_dir, 'MNI_data_DX_4f.csv'))[['ptid', 'age_list']]
            self.df = pd.concat([df1, df2], axis=0).drop_duplicates(subset='ptid').reset_index(drop=True)
        else:
            self.df = df1.drop_duplicates(subset='ptid').reset_index(drop=True)
        ptid_list = list(self.df['ptid'])

        classes = ['ad', 'cn', 'mci'] if split == 'all' else [split]
        all_files = []
        for cls in classes:
            class_dir = os.path.join(data_path, mode, cls)
            if os.path.isdir(class_dir):
                for f in sorted(os.listdir(class_dir)):
                    if f.endswith('.pt'):
                        all_files.append(os.path.join(class_dir, f))

        self.files = [f for f in all_files if any(ptid in f for ptid in ptid_list)]
        print(f"BrainDataset [{mode}]: {len(self.files)}/{len(all_files)} files (filtered by CSV)")

    def __len__(self):
        return len(self.files)

    def _extract_ptid(self, filename):
        name = os.path.basename(filename).replace('_precomputed.pt', '').replace('.pt', '')
        for prefix in ('ad_', 'cn_', 'mci_'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name

    def _get_age_list(self, ptid):
        row = self.df[self.df['ptid'] == ptid]
        if len(row) == 0:
            return None
        age_str = row.iloc[0]['age_list']
        try:
            ages = np.fromstring(str(age_str).strip('[]'), sep=' ')
            return torch.from_numpy(ages).float()
        except Exception:
            return None

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        series = data['series']
        v_series = data['v_series']
        label = data['label']
        ptid = self._extract_ptid(self.files[idx])
        age_list = self._get_age_list(ptid)
        return {
            'series': series, 'v_series': v_series, 'label': label,
            'filename': os.path.basename(self.files[idx]),
            'ptid': ptid, 'age_list': age_list,
        }


def ncc(a, v, zero_norm=True):
    """Normalized Cross-Correlation between two arrays."""
    a = a.flatten()
    v = v.flatten()
    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) + 1e-5)
        v = (v - np.mean(v)) / (np.std(v) + 1e-5)
    return np.mean(a * v)


def compute_ncc_vx_calibration(sampled_velocities, v_reg_list, subject_idx=0):
    """Compute NCC_VX calibration metric."""
    ncc_vx_per_frame = []
    for frame_idx in range(len(v_reg_list)):
        if frame_idx not in sampled_velocities or len(sampled_velocities[frame_idx]) == 0:
            ncc_vx_per_frame.append(0.0)
            continue
        velocities = np.stack([v[subject_idx].cpu().numpy() for v in sampled_velocities[frame_idx]], axis=0)
        variance = np.var(velocities, axis=0)
        var_mag = np.sqrt(variance[0] + variance[1])
        v_reg = v_reg_list[frame_idx][subject_idx].cpu().numpy()
        v_reg_mag = np.sqrt(v_reg[0]**2 + v_reg[1]**2)
        ncc_val = ncc(var_mag, v_reg_mag)
        ncc_vx_per_frame.append(ncc_val)

    overall_ncc_vx = np.mean(ncc_vx_per_frame)
    overall_std = np.std(ncc_vx_per_frame)
    return ncc_vx_per_frame, overall_ncc_vx, overall_std
