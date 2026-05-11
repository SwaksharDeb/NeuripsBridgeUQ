"""
Pretrained-registration-backbone dispatcher for v2 main.py.

Selects between TLRN (in-tree), LTMA, TransMorph, and VoxelMorph
(/vmplus) external projects. Returns the (model, train_loader,
test_loader, config) tuple identical in shape to each project's
own load_model_and_data(), with config.model.params.workdir
rewritten to a centralized LightingTemplate_2 path.

NOTE: only one backbone may be loaded per process - switching
requires a fresh Python interpreter (each external project has
its own models/, dataload/, utils/ packages that collide via
sys.path).
"""
import os
import sys
import importlib.util


LT2_ROOT  = "/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2"
LTMA_ROOT = "/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_LTMA"
TM_ROOT   = "/scratch/swd9tc/Uncertanity_quantification/TM"
VXM_ROOT  = "/scratch/swd9tc/Uncertanity_quantification/voxel_and_R2R/voxelmorph"


_BACKBONE_TAGS = {
    'tlrn':       ('TLRN',       'basic_MSE_Penp_img1200Reg0.03_plus_ACDC'),
    'ltma':       ('TLMA_TGrad', 'config_plus_ACDC'),
    'tm':         ('TransMorph', 'basic_plus_ACDC'),
    'voxelmorph': ('VoxelMorph', 'voxelmorph'),
    'vmplus':     ('VoxelMorph', 'vmplus'),
}


# Top-level package names that exist (with different content) inside multiple
# project trees. They get cached in sys.modules once any LT2 module imports
# them, which then shadows the external project's versions when an external
# loader is exec'd. We evict them before each external load.
_SHADOW_NAMES = ('utils', 'models', 'dataload', 'networks')


def _evict_shadowed_modules():
    for name in list(sys.modules):
        if name in _SHADOW_NAMES or any(name.startswith(p + '.') for p in _SHADOW_NAMES):
            del sys.modules[name]


def _import_external_loader(unique_name, abs_path, project_root):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    _evict_shadowed_modules()
    spec = importlib.util.spec_from_file_location(unique_name, abs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_backbone_and_data(backbone: str, combine_acdc: bool):
    backbone = backbone.lower()
    if backbone == 'tlrn':
        from uncertainty_sde_combined_acdc.data_utils import load_model_and_data
        out = load_model_and_data(combine_acdc=combine_acdc)

    elif backbone == 'ltma':
        du = _import_external_loader(
            '_ltma_data_utils',
            os.path.join(LTMA_ROOT, 'uncertainty_sde_combined_acdc/data_utils.py'),
            LTMA_ROOT,
        )
        out = du.load_model_and_data(combine_acdc=combine_acdc)

    elif backbone == 'tm':
        du = _import_external_loader(
            '_tm_data_utils',
            os.path.join(TM_ROOT, 'uncertainty_sde_combined_acdc/data_utils.py'),
            TM_ROOT,
        )
        out = du.load_model_and_data(combine_acdc=combine_acdc)

    elif backbone in ('voxelmorph', 'vmplus'):
        du = _import_external_loader(
            '_vxm_data_utils',
            os.path.join(VXM_ROOT, 'uncertainty_sde_combined_acdc/data_utils.py'),
            VXM_ROOT,
        )
        out = du.load_model_and_data(resmode=backbone, combine_acdc=combine_acdc)

    else:
        raise ValueError(f"Unknown backbone {backbone!r}")

    model, train_loader, test_loader, config = out
    btag, etag = _BACKBONE_TAGS[backbone]
    config.model.params.workdir = (
        f"{LT2_ROOT}/2026Experiments/CINE/outputs/{btag}/{etag}"
    )
    return model, train_loader, test_loader, config
