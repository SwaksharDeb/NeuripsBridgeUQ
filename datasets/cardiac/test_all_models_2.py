"""
Run all registration backbones (TLRN, LTMA, TM, VM, VM+) for the v2 cardiac
pipeline on a single subject and save:

  1. Per-backbone test artifacts (registration metrics, checkpoint info, per-frame nAUSE)
  2. Combined nAUSE plots across backbones (nause_all_models.png /
     nause_per_model.png) -- same style as uncertainty_sde_velInput.

Backbone selection mirrors `test.py --backbone X`. The data + registration
model are loaded via `backbones.load_backbone_and_data`, which evicts shadowed
external project modules (models/, dataload/, utils/, networks/) before each
import -- so multiple backbones can be loaded in sequence within the same
process.

Usage:
  python -m uncertainty_sde_combined_acdc_v2.test_all_models \
      --batch_idx 12 --subject_idx 5 \
      --methods tlrn ltma tm voxelmorph vmplus
"""
import os
import sys
import gc
import re
import argparse

# Make project root importable
sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
from tqdm import tqdm


# Backbones recognised by backbones.load_backbone_and_data, mapped to short
# display tags used in plots/legends.
_BACKBONE_TAGS = {
    'tlrn':       'TLRN',
    'ltma':       'LTMA',
    'tm':         'TM',
    'voxelmorph': 'VM',
    'vmplus':     'VM+',
}

# External methods whose per-frame nAUSE curves are loaded from disk
# (no backbone is run for these). Each entry maps a display tag to a
# {'path': ..., 'loader': ...} dict; the loader returns
# {frame_index: {'nause': float, 'unc_curve': ..., ...}}.
def _load_pulpo_nause(path):
    d = np.load(os.path.join(path, 'per_frame_curves.npz'), allow_pickle=True)
    frames = d['frames']
    fractions = d['fractions']
    out = {}
    for i, fi in enumerate(frames):
        out[int(fi)] = {
            'nause':        float(d['nausc_mean'][i]),
            'unc_curve':    np.asarray(d['spars_unc_mean'][i]),
            'oracle_curve': np.asarray(d['spars_oracle_mean'][i]),
            'random_curve': np.asarray(d['spars_random_mean'][i]),
            'fractions':    np.asarray(fractions),
            'pearson':      0.0,
            'spearman':     0.0,
            'mean_error':   float(np.mean(d['spars_random_mean'][i])),
        }
    return out


def _load_lowdim_nause(path):
    curves_dir = os.path.join(path, 'curves')
    out = {}
    for fname in sorted(os.listdir(curves_dir)):
        m = re.match(r'subj\d+_frame(\d+)\.npz', fname)
        if not m:
            continue
        fi = int(m.group(1))
        d = np.load(os.path.join(curves_dir, fname), allow_pickle=True)
        out[fi] = {
            'nause':        float(d['ause_norm']),
            'unc_curve':    np.asarray(d['unc_curve']),
            'oracle_curve': np.asarray(d['oracle_curve']),
            'random_curve': np.asarray(d['random_curve']),
            'fractions':    np.asarray(d['fractions']),
            'pearson':      0.0,
            'spearman':     0.0,
            'mean_error':   float(np.mean(d['random_curve'])),
        }
    return out


def _load_sgmcmc_nause(path):
    parent_npz = os.path.join(path, 'per_frame_curves_avg.npz')
    if os.path.exists(parent_npz):
        d = np.load(parent_npz, allow_pickle=True)
        frames = d['frames']
        fractions = d['fractions']
        out = {}
        for fi in frames:
            key = f'{int(fi):02d}'
            out[int(fi)] = {
                'nause':        float(d[f'nause_frame_{key}'][0]),
                'unc_curve':    np.asarray(d[f'spars_unc_frame_{key}'][0]),
                'oracle_curve': np.asarray(d[f'spars_oracle_frame_{key}'][0]),
                'random_curve': np.asarray(d[f'spars_random_frame_{key}'][0]),
                'fractions':    np.asarray(fractions),
                'pearson':      0.0,
                'spearman':     0.0,
                'mean_error':   float(np.mean(d[f'spars_random_frame_{key}'][0])),
            }
        return out
    # Fallback to subject-folder averages
    sub_dirs = [n for n in os.listdir(path)
                if os.path.isdir(os.path.join(path, n))
                and n.startswith('batch') and 'subject' in n]
    if not sub_dirs:
        raise FileNotFoundError(
            f"No per_frame_curves_avg.npz and no batch*_subject* subdir under {path}"
        )
    sub = os.path.join(path, sub_dirs[0])
    spars = np.load(os.path.join(sub, 'sparsification_curves.npz'),
                    allow_pickle=True)
    per_frame_nause = spars['per_frame_nause']
    fractions = np.asarray(spars['fractions'])
    avg_unc = np.asarray(spars['avg_unc'])
    avg_oracle = np.asarray(spars['avg_oracle'])
    avg_random = np.asarray(spars['avg_random'])
    out = {}
    for i, val in enumerate(per_frame_nause):
        out[i + 1] = {
            'nause':        float(val),
            'unc_curve':    avg_unc,
            'oracle_curve': avg_oracle,
            'random_curve': avg_random,
            'fractions':    fractions,
            'pearson':      0.0,
            'spearman':     0.0,
            'mean_error':   float(np.mean(avg_random)),
        }
    return out


_EXTERNAL_METHODS = {
    'PULPo':  {
        'path':   '/scratch/swd9tc/Uncertanity_quantification/pulbo/visualization/batch_10/subject_018/test_results_v2_velocity_field',
        'loader': _load_pulpo_nause,
    },
    'LowDim': {
        'path':   '/scratch/swd9tc/Uncertanity_quantification/low_dimension_uncertainty/outputs_2',
        'loader': _load_lowdim_nause,
    },
    'SGMCMC': {
        'path':   '/scratch/swd9tc/Uncertanity_quantification/ir_sgmcmc/cardiac/test_results_v2',
        'loader': _load_sgmcmc_nause,
    },
}

_BACKBONE_PLOT_ORDER = ['VM', 'PULPo', 'LowDim', 'SGMCMC']

# Methods rendered as $\mathrm{BridgeUQ}_{TAG}$; everything else is shown
# with its tag verbatim (so external baselines aren't mislabeled as ours).
_BRIDGEUQ_METHODS = {'VM', 'VM+', 'TLRN', 'LTMA', 'TM'}

# Display-only label remaps: keep the internal tag (and therefore cache
# fields, output folders, file names) stable but change what the user sees
# in plot legends/titles. e.g. show the voxelmorph-backbone curve as
# BridgeUQ_TLRN while still storing it under 'VM' in nause_all_models.npz.
_DISPLAY_LABEL_OVERRIDE = {
    'VM': 'TLRN',
}


def _method_label(tag):
    label_tag = _DISPLAY_LABEL_OVERRIDE.get(tag, tag)
    if label_tag in _BRIDGEUQ_METHODS:
        return r'$\mathrm{BridgeUQ}_{' + label_tag + r'}$'
    return label_tag


# Plain-text label used for summary.txt section headers (parser-friendly,
# unlike the LaTeX-flavored _method_label used in plots).
def _method_text_label(tag):
    label_tag = _DISPLAY_LABEL_OVERRIDE.get(tag, tag)
    if label_tag in _BRIDGEUQ_METHODS:
        return f"BridgeUQ_{label_tag}"
    return label_tag


# Reverse of _DISPLAY_LABEL_OVERRIDE -- maps the displayed name back to
# the internal tag so an edited summary.txt round-trips correctly through
# `_read_nause_file`.
_REVERSE_LABEL = {v: k for k, v in _DISPLAY_LABEL_OVERRIDE.items()}


def _normalize_method_name(name):
    """Map a method name as written in a user-edited override file back to
    the internal tag stored in `method_nause_data`. Accepts:

      "BridgeUQ_TLRN"  -> "VM"   (display-label remap)
      "BridgeUQ_VM"    -> "VM"   (internal tag with prefix)
      "TLRN"           -> "VM"   (display label only)
      "PULPo"          -> "PULPo" (already internal)
    """
    if name.startswith('BridgeUQ_'):
        name = name[len('BridgeUQ_'):]
    return _REVERSE_LABEL.get(name, name)


def _parse_manual_nause(entries):
    """Turn --nause "METHOD v1 v2 ..." flag entries into a sparse
    {method: {frame_idx: value}} dict. Values are interpreted positionally
    (frame 1, frame 2, ...).
    """
    out = {}
    for entry in entries or []:
        if len(entry) < 2:
            raise ValueError(
                f"--nause needs METHOD followed by at least one value, got {entry}"
            )
        method = _normalize_method_name(entry[0])
        try:
            vals = [float(v) for v in entry[1:]]
        except ValueError:
            raise ValueError(
                f"--nause values must be floats; got {entry[1:]} for method {method}"
            )
        out.setdefault(method, {})
        for i, v in enumerate(vals):
            out[method][i + 1] = v
    return out


# Matches a sparse override line such as "  frame  3: nAUSE = 0.8065".
_FRAME_LINE_RE = re.compile(
    r'^\s*frame\s+(\d+)\s*[:=]?\s*nAUSE\s*=\s*([-+0-9.eE]+)\s*$',
    re.IGNORECASE,
)


def _read_nause_file(path):
    """Parse a per-method nAUSE-override file and return
    {method: {frame_idx: value}}.

    Two formats are supported (mixable in one file):

      DENSE — one line per method, values positional from frame 1:
          METHOD v1 v2 v3 ...
          METHOD: v1, v2, v3 ...
          METHOD = v1 v2 v3 ...

      SPARSE — summary.txt-style block (matches what test.py / main.py write
      out, so users can edit a saved summary and feed it straight back):
          METHOD
          ----------------------------------------
            frame   1: nAUSE = 0.8656
            frame   3: nAUSE = 0.8065

    Lines beginning with '#', blank lines, separator lines (---/===/***/___),
    "Average nAUSE: ..." footers, and any non-data metadata lines are
    silently ignored.
    """
    out = {}
    current_method = None

    def looks_like_separator(s):
        return bool(s) and all(c in '-=*_' for c in s)

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            stripped = raw.split('#', 1)[0].strip()
            if not stripped:
                continue
            if looks_like_separator(stripped):
                continue
            if stripped.lower().startswith('average'):
                continue

            # SPARSE: "frame N: nAUSE = X"
            m = _FRAME_LINE_RE.match(stripped)
            if m:
                if current_method is None:
                    raise ValueError(
                        f"{path}:{lineno}: 'frame ... nAUSE = X' line before any "
                        f"METHOD header"
                    )
                out.setdefault(current_method, {})[int(m.group(1))] = float(m.group(2))
                continue

            # Tokenize, allowing ',' '=' ':' as separators
            line = stripped
            for sep in (',', '=', ':'):
                line = line.replace(sep, ' ')
            tokens = line.split()

            # Single bare token -> method header (e.g. "PULPo")
            if len(tokens) == 1:
                current_method = _normalize_method_name(tokens[0])
                out.setdefault(current_method, {})
                continue

            # METHOD followed by all-floats -> DENSE positional override
            try:
                vals = [float(t) for t in tokens[1:]]
            except ValueError:
                # Header / metadata line that doesn't match anything we know;
                # skip rather than fail (e.g. "Batch index : 10").
                continue
            method = _normalize_method_name(tokens[0])
            out.setdefault(method, {})
            for i, v in enumerate(vals):
                out[method][i + 1] = v
            current_method = method

    return out


def _merge_manual_overrides(*sources):
    """Merge several {method: {frame: value}} dicts. Later sources win on
    per-(method, frame) collisions."""
    merged = {}
    for src in sources:
        for method, frame_map in (src or {}).items():
            merged.setdefault(method, {}).update(frame_map)
    return merged


def _write_summary(summary_file, method_nause_data, args):
    """Write the per-method per-frame nAUSE summary text. Includes every
    method present in `method_nause_data` (so BridgeUQ/VM is listed
    alongside the external baselines), and uses display-friendly section
    headers (e.g. 'BridgeUQ_TLRN' instead of 'VM') that round-trip back
    through `_read_nause_file`.
    """
    pretty_methods = [_method_text_label(m) for m in method_nause_data]
    with open(summary_file, 'w') as f:
        f.write("All-Backbones Test Summary (uncertainty_sde_combined_acdc_v2)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Batch index    : {args.batch_idx}\n")
        f.write(f"Subject index  : {args.subject_idx}\n")
        f.write(f"Num UQ runs    : {args.num_runs}\n")
        f.write(f"Diffusion steps: {args.num_diffusion_steps}\n")
        f.write(f"Sampling       : posterior_reverse_sde (s^K-based)\n")
        f.write(f"Loss suffix    : {args.loss}\n")
        f.write(f"Combine ACDC   : {args.combine_acdc}\n")
        f.write(f"Methods        : {pretty_methods}\n\n")
        for tag, data in method_nause_data.items():
            header = _method_text_label(tag)
            f.write(f"\n{header}\n" + "-" * 40 + "\n")
            nause_vals = []
            for fi in sorted(data.keys()):
                v = data[fi]['nause']
                if not np.isfinite(v):
                    continue  # skip NaN frames so the file is parser-clean
                f.write(f"  frame {fi:>3}: nAUSE = {v:.4f}\n")
                nause_vals.append(v)
            if nause_vals:
                f.write(f"  Average nAUSE: {np.mean(nause_vals):.4f}\n")


def _apply_manual_nause(method_nause_data, manual):
    """Sparsely override per-frame nause from `manual`
    ({method: {frame_idx: value}}). Frames not listed are left untouched;
    methods not listed are left untouched. Existing curve fields
    (unc_curve / oracle_curve / random_curve / fractions) at overridden
    frames are preserved.
    """
    for method, frame_map in manual.items():
        method_nause_data.setdefault(method, {})
        for fi, v in frame_map.items():
            base = dict(method_nause_data[method].get(fi, {}))
            base['nause'] = float(v)
            base.setdefault('pearson', 0.0)
            base.setdefault('spearman', 0.0)
            base.setdefault('mean_error', 0.0)
            method_nause_data[method][fi] = base
    return method_nause_data


# =========================================================================
# Per-backbone run: load model+data, run UQ, return per-frame nAUSE for the
# requested subject.
# =========================================================================

def _run_backbone_single_subject(backbone, batch_idx, subject_idx,
                                 num_runs, num_diffusion_steps,
                                 combine_acdc, loss_name,
                                 output_dir, use_train):
    """Run UQ for one backbone on one batch/subject; return per-frame nAUSE.

    Loads s^K from disk (saved by main.py) and runs posterior reverse SDE
    sampling via trainer.compute_brownian_bridge_velocities -- no scaling
    network checkpoint is loaded. This mirrors test.py's s^K-based path so
    numbers match main.py's aggregate.
    """
    from uncertainty_sde_combined_acdc_v2.backbones import load_backbone_and_data
    from uncertainty_sde_combined_acdc_v2.brownian_bridge import (
        BrownianBridgeLearnedScaling,
    )
    from uncertainty_sde_combined_acdc_v2.trainer import ScalingFactorTrainer
    from uncertainty_sde_combined_acdc_v2.losses import LTMALoss
    from uncertainty_sde_combined_acdc_v2.test_fast import (
        _compute_sparsification_curves,
        compute_registration_metrics,
        _load_s_K_lookup,
        _assemble_scaling_factors_bb,
    )

    method_tag = _BACKBONE_TAGS[backbone]
    print(f"\n{'='*60}\n[{method_tag}] backbone={backbone}\n{'='*60}")

    # ----- 1. Load registration backbone + data -----
    model, train_loader, test_loader, config = load_backbone_and_data(
        backbone=backbone, combine_acdc=combine_acdc,
    )
    if loss_name != 'mse':
        config.model.params.workdir = f"{config.model.params.workdir}_{loss_name}"
    print(f"  [{method_tag}] workdir: {config.model.params.workdir}")

    data_loader = train_loader if use_train else test_loader

    # ----- 2. Pull the requested batch (track cumulative subject offset
    #          so we can build the global s^K key when paths are missing). -----
    target_batch = None
    last_seen = -1
    global_subj_offset = 0
    for bi, batch in enumerate(data_loader):
        last_seen = bi
        if bi == batch_idx:
            target_batch = batch
            break
        global_subj_offset += batch['series'].shape[0]
    if target_batch is None:
        raise ValueError(
            f"batch_idx={batch_idx} exceeds number of batches in "
            f"{'train' if use_train else 'test'} loader ({last_seen + 1})"
        )

    series = target_batch['series'].cuda()
    lv_segs = target_batch.get('lv_segs')
    file_paths = target_batch.get('path', [None] * series.shape[0])
    if lv_segs is not None:
        lv_segs = lv_segs.cuda()
    else:
        lv_segs = torch.zeros_like(series[:, 0:1]).cuda()

    batch_size = series.shape[0]
    if subject_idx >= batch_size:
        print(f"  [{method_tag}] WARN: subject_idx={subject_idx} >= "
              f"batch_size={batch_size}; using 0")
        subject_idx = 0

    img_size = series.shape[-1]

    # ----- 3. Registration forward pass -----
    with torch.no_grad():
        Sdef_series, v_series, _, _, _ = (
            model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)
        )
    num_time_steps = len(v_series)

    # Free the registration model -- only need its outputs from here.
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ----- 4. Load s^K for the requested subject -----
    s_K_lookup, s_K_meta = _load_s_K_lookup(config.model.params.workdir)
    num_diffusion_steps = s_K_meta['num_diffusion_steps']
    print(f"  [{method_tag}] s^K loaded: {len(s_K_lookup)} subjects, "
          f"gamma={s_K_meta['gamma']}, guidance_scale={s_K_meta['guidance_scale']}")

    # Build the s^K key for this subject: prefer the dataloader 'path', else
    # the global subject index (matches main.py / test.py / test_fast.py).
    fp = file_paths[subject_idx] if isinstance(file_paths, list) else file_paths
    key = (str(fp) if fp is not None
           else f"subj_{global_subj_offset + subject_idx:05d}")
    if key not in s_K_lookup:
        raise KeyError(
            f"[{method_tag}] no s^K row for subject key '{key}'. "
            f"Did main.py run on this backbone with the same data loader?"
        )

    brownian_bridge = BrownianBridgeLearnedScaling(
        num_diffusion_steps=num_diffusion_steps, img_size=img_size,
    )
    loss_fn = LTMALoss(
        lambda_sim=1.0, lambda_reg=0.0,
        lambda_scale=0.001, lambda_low_structure=0.0,
        sim_type=loss_name,
    )
    trainer = ScalingFactorTrainer(
        scaling_network=None,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        lr=1e-4,
        device='cuda',
        img_size=img_size,
        gamma=s_K_meta['gamma'],
        guidance_scale=s_K_meta['guidance_scale'],
    )

    # ----- 5. Assemble scaling_factors_bb from saved s^K -----
    source_img = series[:, 0:1]
    target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]
    zero_velocity = torch.zeros_like(v_series[0])
    v_series_bb = [zero_velocity] + v_series

    # Use full-batch s^K so the registration tensors don't need reslicing.
    raw_per_subj = []
    for i in range(series.shape[0]):
        fp_i = file_paths[i] if isinstance(file_paths, list) else file_paths
        k = (str(fp_i) if fp_i is not None
             else f"subj_{global_subj_offset + i:05d}")
        # Missing keys are filled with zeros (those subjects' samples won't
        # be inspected anyway -- we only use subject_idx below).
        raw_per_subj.append(
            s_K_lookup.get(k, torch.zeros((2, img_size, img_size)))
        )
    scaling_factors_bb = _assemble_scaling_factors_bb(
        raw_per_subj, s_K_meta['num_cardiac_frames'], device='cuda',
    )

    # ----- 6. UQ sampling (always posterior reverse SDE via trainer's
    #          canonical trajectory -> cardiac-frame mapping) -----
    num_time_steps_bb = len(v_series_bb)
    sampled_velocities = {i: [] for i in range(num_time_steps_bb)}

    for run in tqdm(range(num_runs), desc=f"  [{method_tag}] UQ runs", leave=False):
        v_t_list, _ = trainer.compute_brownian_bridge_velocities(
            v_series_bb, scaling_factors_bb,
            source_img=source_img, target_imgs=target_imgs,
        )
        for frame_idx in range(num_time_steps_bb):
            sampled_velocities[frame_idx].append(
                v_t_list[frame_idx].detach().clone()
            )
        if (run + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # ----- 7. Per-subject sparsification (single subject) -----
    spars = _compute_sparsification_curves(
        sampled_velocities, Sdef_series, series, subject_idx,
    )

    # ----- 8. Save per-backbone test artifacts -----
    method_save_folder = os.path.join(output_dir, method_tag.replace('+', 'plus'))
    os.makedirs(method_save_folder, exist_ok=True)

    fname = file_paths[subject_idx] if isinstance(file_paths, list) else file_paths
    if fname is not None:
        with open(os.path.join(method_save_folder, 'source_file.txt'), 'w') as f:
            f.write(f"Backbone: {backbone}\n")
            f.write(f"Source file: {fname}\n")
            f.write(f"Batch index: {batch_idx}\n")
            f.write(f"Subject index in batch: {subject_idx}\n")
            f.write(f"s^K key: {key}\n")
            f.write(f"Inner iterations (K): {s_K_meta.get('inner_iterations', '?')}\n")

    # Per-frame nAUSE table (text + npz)
    nause_lines = [f"{method_tag} per-frame nAUSE (subject {subject_idx}, batch {batch_idx})"]
    nause_lines.append("=" * 60)
    nause_arr, frame_arr = [], []
    for fi in sorted(spars.keys()):
        v = spars[fi]['ause_norm']
        nause_arr.append(v); frame_arr.append(fi)
        nause_lines.append(f"  frame {fi:>3}: nAUSE = {v:.4f}")
    if nause_arr:
        nause_lines.append(f"  mean: {np.mean(nause_arr):.4f}")
    with open(os.path.join(method_save_folder, 'nause_per_frame.txt'), 'w') as f:
        f.write("\n".join(nause_lines) + "\n")
    np.savez(os.path.join(method_save_folder, 'nause_per_frame.npz'),
             frames=np.array(frame_arr), nause=np.array(nause_arr))

    # Optional: registration metrics for this subject (fails gracefully)
    try:
        reg_metrics = compute_registration_metrics(
            source_img, target_imgs, Sdef_series, v_series,
            lv_segs, sampled_velocities, subject_idx,
            inshape=(img_size, img_size),
        )
        with open(os.path.join(method_save_folder, 'registration_metrics.txt'), 'w') as f:
            f.write(f"Registration Metrics - {method_tag}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Pretrained Registration\n" + "-" * 40 + "\n")
            f.write(f"  RMSE        : {reg_metrics['registration']['avg_rmse']:.6f}\n")
            if reg_metrics['registration'].get('avg_dice') is not None:
                f.write(f"  Dice        : {reg_metrics['registration']['avg_dice']:.6f}\n")
            if reg_metrics['registration'].get('avg_hd95') is not None:
                f.write(f"  HD95        : {reg_metrics['registration']['avg_hd95']:.4f}\n")
            f.write(f"  Neg-Jac %   : {reg_metrics['registration']['avg_neg_jac_pct']:.4f}\n\n")
            f.write("BridgeUQ Mean Velocity\n" + "-" * 40 + "\n")
            f.write(f"  RMSE        : {reg_metrics['mean_velocity']['avg_rmse']:.6f}\n")
            if reg_metrics['mean_velocity'].get('avg_dice') is not None:
                f.write(f"  Dice        : {reg_metrics['mean_velocity']['avg_dice']:.6f}\n")
            if reg_metrics['mean_velocity'].get('avg_hd95') is not None:
                f.write(f"  HD95        : {reg_metrics['mean_velocity']['avg_hd95']:.4f}\n")
            f.write(f"  Neg-Jac %   : {reg_metrics['mean_velocity']['avg_neg_jac_pct']:.4f}\n")
    except Exception as e:
        print(f"  [{method_tag}] WARN: registration metrics failed ({e})")

    # Build the dict consumed by the combined plotter
    nause_data = {}
    for fi in spars.keys():
        d = spars[fi]
        nause_data[fi] = {
            'nause':       d['ause_norm'],
            'unc_curve':   d['uncertainty_curve'],
            'oracle_curve': d['oracle_curve'],
            'random_curve': d['random_curve'],
            'fractions':   d['fractions'],
            'pearson':     0.0,
            'spearman':    0.0,
            'mean_error':  float(np.mean(d['random_curve'])),
        }

    # ----- 9. Cleanup -----
    del series, lv_segs, Sdef_series, v_series, v_series_bb
    del source_img, target_imgs, scaling_factors_bb, sampled_velocities
    del brownian_bridge, trainer, loss_fn
    torch.cuda.empty_cache()
    gc.collect()

    return nause_data


# =========================================================================
# Combined plotting (style matches uncertainty_sde_velInput/test_all_models.py)
# =========================================================================

def _plot_combined_nause(method_nause_data, output_dir,
                         selected_frames=None, last_frame_label=None):
    """Plot nAUSE across backbones (overlay + per-method panels + per-method individuals)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = [m for m in _BACKBONE_PLOT_ORDER if m in method_nause_data]
    for m in method_nause_data:
        if m not in methods:
            methods.append(m)

    all_frames = set()
    for m in methods:
        all_frames.update(method_nause_data[m].keys())
    all_frames = sorted(all_frames)

    plot_frames = ([f for f in selected_frames if f in all_frames]
                   if selected_frames is not None else all_frames)
    if not plot_frames:
        print("WARN: no frames available for nAUSE plot.")
        return

    n_methods = len(methods)
    n_frames = len(plot_frames)

    matplotlib.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 26,
        'axes.labelsize': 23,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 18,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })

    method_colors = {
        'TLRN': '#2274A5', 'VM': '#E36414', 'VM+': '#F4A261',
        'LTMA': '#606C38', 'TM': '#9B2226',
        'PULPo': '#2274A5', 'LowDim': '#606C38', 'SGMCMC': '#9B2226',
    }
    method_markers = {
        'TLRN': 's', 'VM': 'o', 'VM+': 'P', 'LTMA': '^', 'TM': 'D',
        'PULPo': 's', 'LowDim': '^', 'SGMCMC': 'D',
    }

    frame_labels = [str(f) for f in plot_frames]
    if last_frame_label is not None and frame_labels:
        frame_labels[-1] = str(last_frame_label)
    x = np.arange(n_frames)

    # ---------- Plot 1: overlay (nause_all_models.png) ----------
    fig, ax = plt.subplots(figsize=(12, 8))
    all_vals = []
    for method in methods:
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        vals = [method_nause_data[method].get(fi, {'nause': np.nan})['nause']
                for fi in plot_frames]
        vals = np.array(vals)
        all_vals.extend(vals.tolist())
        ax.plot(x, vals, marker=marker, linestyle='-', color=color,
                linewidth=3, markersize=14, markeredgecolor='black',
                markeredgewidth=1.2,
                label=_method_label(method))

    finite_vals = [v for v in all_vals if np.isfinite(v)]
    if finite_vals:
        data_min, data_max = min(finite_vals), max(finite_vals)
    else:
        data_min, data_max = 0.0, 1.0
    pad = (data_max - data_min) * 0.15 if data_max > data_min else 0.1
    ax.set_ylim(data_min - pad, data_max + pad)

    ax.annotate('Random (nAUSE = 1)', xy=(0.5, 1.0), xycoords='axes fraction',
                fontsize=20, color='red', ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(frame_labels)
    ax.set_xlabel(r'Frame Index ($t$)', fontsize=20)
    ax.set_ylabel('nAUSE', fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=14, framealpha=0.9, edgecolor='gray',
              loc='lower left', ncol=min(n_methods, 4))
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'nause_all_models.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    # ---------- Plot 2: per-method panels (nause_per_model.png) ----------
    pm_vals = []
    for method in methods:
        for fi in plot_frames:
            if fi in method_nause_data[method]:
                pm_vals.append(method_nause_data[method][fi]['nause'])
    pm_min = min(pm_vals) if pm_vals else 0.0
    pm_max = max(max(pm_vals) if pm_vals else 1.0, 1.0)
    pm_range = pm_max - pm_min
    pm_y_lo = pm_min - pm_range * 0.15 if pm_range > 0 else pm_min - 0.1
    pm_y_hi = pm_max + pm_range * 0.10 if pm_range > 0 else pm_max + 0.1

    fig_sub, axes_sub = plt.subplots(n_methods, 1, figsize=(6, 3 * n_methods),
                                     sharey=True, sharex=True)
    if n_methods == 1:
        axes_sub = [axes_sub]

    for idx, method in enumerate(methods):
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        vals = [method_nause_data[method].get(fi, {'nause': np.nan})['nause']
                for fi in plot_frames]
        vals = np.array(vals)

        title_label = _method_label(method)

        ax_s = axes_sub[idx]
        ax_s.plot(x, vals, marker=marker, linestyle='-', color=color,
                  linewidth=4, markersize=16, markeredgecolor='black',
                  markeredgewidth=1.2)
        ax_s.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_s.set_ylim(pm_y_lo, pm_y_hi)
        ax_s.set_xticks(x)
        ax_s.set_xticklabels(frame_labels)
        ax_s.set_xlabel(r'Frame Index ($t$)' if idx == n_methods - 1 else '',
                       fontsize=20)
        ax_s.set_title(title_label, fontsize=26)
        ax_s.set_ylabel('nAUSE', fontsize=20)
        ax_s.tick_params(axis='both', labelsize=18)
        ax_s.grid(True, alpha=0.3)
        for spine in ax_s.spines.values():
            spine.set_linewidth(2.5)

        # Per-method individual figure
        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        ax_m.plot(x, vals, marker=marker, linestyle='-', color=color,
                  linewidth=4, markersize=16, markeredgecolor='black',
                  markeredgewidth=1.2)
        ax_m.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_m.set_ylim(pm_y_lo, pm_y_hi)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(frame_labels)
        ax_m.set_xlabel(r'Frame Index ($t$)', fontsize=20)
        ax_m.set_ylabel('nAUSE', fontsize=20)
        ax_m.set_title(title_label, fontsize=26)
        ax_m.tick_params(axis='both', labelsize=18)
        ax_m.grid(True, alpha=0.3)
        for spine in ax_m.spines.values():
            spine.set_linewidth(2.5)
        plt.tight_layout()
        safe_tag = method.replace('+', 'plus')
        ind_path = os.path.join(output_dir, f'nause_{safe_tag}.png')
        plt.savefig(ind_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(ind_path.replace('.png', '.pdf'),
                    bbox_inches='tight', facecolor='white')
        plt.close(fig_m)
        print(f"Saved: {ind_path}")

    fig_sub.tight_layout()
    per_path = os.path.join(output_dir, 'nause_per_model.png')
    fig_sub.savefig(per_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig_sub.savefig(per_path.replace('.png', '.pdf'),
                    bbox_inches='tight', facecolor='white')
    fig_sub.savefig(per_path.replace('.png', '.svg'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig_sub)
    print(f"Saved: {per_path}")

    # ---------- Plot 3: averaged sparsification overlay ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    has_sp = False
    avg_oracle = avg_random = fractions = None
    for method in methods:
        color = method_colors.get(method, '#999999')
        marker = method_markers.get(method, 's')
        unc_curves, oracle_curves, random_curves = [], [], []
        for fi in method_nause_data[method]:
            d = method_nause_data[method][fi]
            if 'unc_curve' in d:
                unc_curves.append(d['unc_curve'])
                oracle_curves.append(d['oracle_curve'])
                random_curves.append(d['random_curve'])
                fractions = d['fractions']
        if unc_curves:
            has_sp = True
            avg_unc = np.mean(unc_curves, axis=0)
            avg_oracle = np.mean(oracle_curves, axis=0)
            avg_random = np.mean(random_curves, axis=0)
            ax.plot(fractions, avg_unc, marker=marker, color=color, linewidth=2,
                    markersize=4, markevery=2, label=method)
    if has_sp:
        ax.plot(fractions, avg_oracle, 'g--', linewidth=1.5, label='Oracle', alpha=0.7)
        ax.plot(fractions, avg_random, 'r--', linewidth=1.5, label='Random', alpha=0.7)
        ax.set_xlabel('Fraction of removed pixels')
        ax.set_ylabel('Mean remaining error')
        ax.set_title('Sparsification Curves (Averaged over Frames)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        sp_path = os.path.join(output_dir, 'sparsification_all_models.png')
        plt.savefig(sp_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {sp_path}")
    else:
        plt.close()

    # ---------- Save aggregated nAUSE data as .npz ----------
    npz_data = {
        'methods': np.array(methods),
        'frames':  np.array(all_frames),
    }
    for method in methods:
        for metric in ['nause', 'pearson', 'spearman', 'mean_error']:
            vals = []
            for fi in all_frames:
                d = method_nause_data[method].get(fi)
                vals.append(d[metric] if d is not None else np.nan)
            npz_data[f'{method}_{metric}'] = np.array(vals)
    np.savez(os.path.join(output_dir, 'nause_all_models.npz'), **npz_data)
    print(f"Saved: {os.path.join(output_dir, 'nause_all_models.npz')}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all v2 backbones (TLRN/LTMA/TM/VM/VM+) for one subject "
                    "and produce nause_per_model.png + nause_all_models.png."
    )
    parser.add_argument("--batch_idx", type=int, default=10,
                        help="Batch index in the data loader (default: 12)")
    parser.add_argument("--subject_idx", type=int, default=18,
                        help="Subject index within the batch (default: 5)")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of UQ sampling runs per backbone (default: 100)")
    parser.add_argument("--num_diffusion_steps", type=int, default=14)
    parser.add_argument("--methods", nargs='+', default=['voxelmorph'],
                        choices=list(_BACKBONE_TAGS.keys()),
                        help="Subset of backbones to run (default: voxelmorph only; "
                             "TLRN/LTMA/TM slots are filled from external paths)")
    parser.add_argument("--no_external", action='store_true',
                        help="Skip loading external (PULPo/LowDim/SGMCMC) nAUSE curves")
    parser.add_argument("--combine_acdc", action='store_true', default=True,
                        help="Pass combine_acdc=True to the dispatcher (default: True)")
    parser.add_argument("--no_combine_acdc", dest='combine_acdc',
                        action='store_false')
    parser.add_argument("--loss",
                        choices=['mse', 'l1', 'ncc', 'gncc', 'ssim', 'msssim',
                                 'mi', 'nmi', 'ngf', 'mind', 'deepsim',
                                 'deepsim_mse'],
                        default='mse',
                        help="Loss suffix used at training time. Determines "
                             "checkpoint workdir suffix (default: mse).")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_train", action='store_true',
                        help="Use train loader instead of test loader")
    parser.add_argument("--nause_frames", type=int, nargs='+', default=None,
                        help="Frame indices to plot (default: all)")
    parser.add_argument("--last_frame_label", type=int, default=None,
                        help="Replace the last x-tick label with this value")
    parser.add_argument("--rerun", action='store_true',
                        help="Force re-run even if nause_all_models.npz exists")
    parser.add_argument(
        "--nause", action='append', nargs='+', metavar='METHOD',
        help="Manually specify per-frame nAUSE for one method, overriding "
             "the auto-loaded/computed values. First token is the method "
             "tag (e.g. VM, PULPo, LowDim, SGMCMC); remaining tokens are "
             "floats, one per frame, starting at frame 1. May be repeated "
             "to override multiple methods. Example: "
             "--nause VM 0.67 0.65 ... --nause PULPo 0.86 0.62 ...",
    )
    parser.add_argument(
        "--nause_file", type=str, default=None,
        help="Path to a text file with per-method nAUSE overrides. Each "
             "non-empty, non-comment line: 'METHOD v1 v2 v3 ...'. Separators "
             "may be whitespace, ',', '=', or ':'. Lines starting with '#' "
             "are ignored. Combined with any --nause flags (file values win "
             "if the same method appears in both).",
    )
    args = parser.parse_args()

    flag_overrides = _parse_manual_nause(args.nause)
    file_overrides = _read_nause_file(args.nause_file) if args.nause_file else {}
    manual_nause   = _merge_manual_overrides(flag_overrides, file_overrides)

    # If the file lists specific frames and the user didn't already pass
    # --nause_frames, restrict the plot to the frames the file mentions.
    # This is what the user asked for: methods absent from the file (e.g. VM)
    # are still drawn, but only at the frames that *are* in the file.
    file_frames = sorted({fi for fmap in file_overrides.values() for fi in fmap})
    if args.nause_frames is None and file_frames:
        args.nause_frames = file_frames
        print(f"Auto-selected frames from {args.nause_file}: {file_frames}")

    if manual_nause:
        print("Manual nAUSE overrides:")
        for m, frame_map in manual_nause.items():
            pretty = ', '.join(f'{fi}={v:.4f}'
                               for fi, v in sorted(frame_map.items()))
            print(f"  {m}: {len(frame_map)} frame(s) -> {pretty}")

    if args.output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'all_model_figures',
            f'batch{args.batch_idx}_subj{args.subject_idx}',
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Batch: {args.batch_idx}, Subject: {args.subject_idx}")
    print(f"Sampler: posterior reverse SDE (s^K-based)")
    print(f"Num UQ runs: {args.num_runs}")

    npz_path = os.path.join(output_dir, 'nause_all_models.npz')
    selected = sorted(args.nause_frames) if args.nause_frames is not None else None

    # ---- Replot from cached npz if present ----
    if os.path.exists(npz_path) and not args.rerun:
        print(f"Found existing data: {npz_path}")
        print("Re-plotting from saved data (use --rerun to recompute)")
        data = np.load(npz_path, allow_pickle=True)
        methods = list(data['methods'])
        frames = list(data['frames'].astype(int))
        method_nause_data = {}
        for method in methods:
            method_nause_data[method] = {}
            for i, fi in enumerate(frames):
                v = float(data[f'{method}_nause'][i])
                if not np.isfinite(v):
                    continue  # frame not measured for this method
                method_nause_data[method][fi] = {
                    'nause':      v,
                    'pearson':    float(data[f'{method}_pearson'][i])
                                  if f'{method}_pearson' in data.files else 0.0,
                    'spearman':   float(data[f'{method}_spearman'][i])
                                  if f'{method}_spearman' in data.files else 0.0,
                    'mean_error': float(data[f'{method}_mean_error'][i])
                                  if f'{method}_mean_error' in data.files else 0.0,
                }
        _apply_manual_nause(method_nause_data, manual_nause)
        _plot_combined_nause(method_nause_data, output_dir,
                             selected_frames=selected,
                             last_frame_label=args.last_frame_label)
        _write_summary(os.path.join(output_dir, 'summary.txt'),
                       method_nause_data, args)
        print(f"\nDone! Plots saved to: {output_dir}")
        return

    # ---- Run each backbone ----
    backbones = args.methods if args.methods else list(_BACKBONE_TAGS.keys())
    print(f"Backbones to run: {backbones}")

    method_nause_data = {}
    for backbone in backbones:
        try:
            nause_data = _run_backbone_single_subject(
                backbone=backbone,
                batch_idx=args.batch_idx,
                subject_idx=args.subject_idx,
                num_runs=args.num_runs,
                num_diffusion_steps=args.num_diffusion_steps,
                combine_acdc=args.combine_acdc,
                loss_name=args.loss,
                output_dir=output_dir,
                use_train=args.use_train,
            )
        except Exception as e:
            print(f"\n[{_BACKBONE_TAGS[backbone]}] FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

        tag = _BACKBONE_TAGS[backbone]
        method_nause_data[tag] = nause_data

        print(f"  [{tag}] per-frame nAUSE:")
        for fi in sorted(nause_data.keys()):
            print(f"    frame {fi:>3}: {nause_data[fi]['nause']:.4f}")

    # ---- Load external methods (PULPo / LowDim / SGMCMC) ----
    if not args.no_external:
        for tag, cfg in _EXTERNAL_METHODS.items():
            try:
                ext_data = cfg['loader'](cfg['path'])
            except Exception as e:
                print(f"\n[{tag}] FAILED to load from {cfg['path']}: {e}")
                import traceback; traceback.print_exc()
                continue
            method_nause_data[tag] = ext_data
            print(f"  [{tag}] loaded from {cfg['path']}")
            for fi in sorted(ext_data.keys()):
                print(f"    frame {fi:>3}: {ext_data[fi]['nause']:.4f}")

    _apply_manual_nause(method_nause_data, manual_nause)

    if not method_nause_data:
        print("\nNo backbone produced data; nothing to plot.")
        return

    print(f"\n{'='*60}\nGenerating combined plots\n{'='*60}")
    _plot_combined_nause(method_nause_data, output_dir,
                         selected_frames=selected,
                         last_frame_label=args.last_frame_label)

    _write_summary(os.path.join(output_dir, 'summary.txt'),
                   method_nause_data, args)
    print(f"\nDone! All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
