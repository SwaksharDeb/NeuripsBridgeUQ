"""Backbone → precomputed-velocity-directory map for the brain pipeline.

Each entry points to the output of ``precompute_velocities.py`` for one
registration backbone. ``_BACKBONE_TAGS`` is the folder tag used inside
the centralized output path under
``LightingTemplate_2/2026Experiments/BRAIN/outputs/<TAG>``.
"""

import torch


_BACKBONE_DATA_PATHS = {
    'tlrn':       '/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/dataset/precomputed_brain_z64_tlrn',
    'ltma':       '/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA/dataset/precomputed_brain_z64_ltma',
    'tm':         '/scratch/swd9tc/Uncertanity_quantification/TM/dataset/precomputed_brain_z64_tm',
    'voxelmorph': '/scratch/swd9tc/Uncertanity_quantification/brain_uncertainty/dataset/precomputed_brain_z64',
}

_BACKBONE_TAGS = {
    'tlrn':       'TLRN',
    'ltma':       'TLMA_TGrad',
    'tm':         'TransMorph',
    'voxelmorph': 'VoxelMorph',
}


def collate_brain_batch(batch):
    """Custom collate that handles a per-sample list of velocity tensors."""
    series = torch.stack([b['series'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch])
    filenames = [b['filename'] for b in batch]

    num_frames = len(batch[0]['v_series'])
    v_series = []
    for t in range(num_frames):
        v_t = torch.stack([b['v_series'][t] for b in batch])
        v_series.append(v_t)

    return {
        'series': series,
        'v_series': v_series,
        'label': labels,
        'filename': filenames,
    }
