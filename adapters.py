"""Dataset adapters — the only seam between the shared training core
(``BridgeUQ.trainer``, ``BridgeUQ.brownian_bridge`` etc.) and the
dataset-specific data loaders / metric assemblies.

Each adapter implements:

  - ``add_dataset_args(parser)``         — register dataset-specific CLI flags.
  - ``default_hyperparameters()``        — return per-dataset defaults
                                           (NUM_DIFFUSION_STEPS, INNER_ITERATIONS, ...).
  - ``build_loader(args)``               — return a ``DataLoader`` and any extras
                                           (e.g. live registration model for cardiac).
  - ``get_dimensions(loader, ctx)``      — probe the first batch and return
                                           ``(num_time_steps, img_size)``.
  - ``get_output_dir(args)``             — resolve the per-run output directory.
  - ``process_batch(batch, args, ctx)``  — convert one DataLoader batch into
                                           the canonical training inputs and
                                           cache anything per-batch that the
                                           metrics functions need (lv_segs,
                                           Sdef_series, ...).
  - ``per_subject_metrics(...)``         — return ``(subject_id, spars, rc, reg_metrics)``
                                           via the dataset's metrics module.
  - ``write_per_batch_summary(...)``     — dump ``per_frame_nAURC_nAUSE_summary.txt``
                                           and ``registration_metrics.npz/.txt`` for
                                           one batch (cardiac adds HD95 columns).
  - ``write_run_summary(...)``           — same for the full run.

Per-batch shape conventions (returned by ``process_batch``):

  ``{
      'series':       Tensor [B, T+1, 1, H, W],
      'source_img':   Tensor [B, 1, H, W],
      'target_imgs':  list of T  Tensor [B, 1, H, W],
      'v_series_bb':  list of T+1 Tensor [B, 2, H, W]  (zero at index 0),
      'lv_segs':      Tensor [B, ...] | None,
      'file_paths':   list[str | None] of length B,
      'extras':       dict — adapter-private bag (Sdef_series, etc.).
  }``
"""

import os
import gc
import copy

import numpy as np
import torch


# ============================================================================
# Base interface
# ============================================================================
class DatasetAdapter:
    name: str = ""

    @staticmethod
    def add_dataset_args(parser):
        raise NotImplementedError

    def default_hyperparameters(self) -> dict:
        # Common defaults shared by both datasets. Subclasses override the
        # ones that differ.
        return {
            'NUM_RUNS':        100,
            'LEARNING_RATE':   1e-4,
            'INNER_ITERATIONS': 500,
            'NUM_DIFFUSION_STEPS': 14,
            'LAMBDA_SIM':      1.0,
            'LAMBDA_REG':      0.0,
            'LAMBDA_SCALE':    3,
            'ALPHA_SCALE':     0.0001,
            'LAMBDA_LOW_STRUCTURE': 0.0,
            'GAMMA':           0.0001 / 2,
            'GUIDANCE_SCALE':  1.0,
            'REG_WEIGHT':      0.1,
        }

    def build_loader(self, args):
        raise NotImplementedError

    def get_dimensions(self, loader, ctx):
        raise NotImplementedError

    def get_output_dir(self, args):
        raise NotImplementedError

    def process_batch(self, batch, args, ctx):
        raise NotImplementedError

    def per_subject_metrics(self, batch_idx, subj_idx, batch_data, sampled_velocities, img_size):
        raise NotImplementedError

    def write_per_batch_summary(self, batch_idx, output_dir,
                                 batch_subject_ids,
                                 batch_sparsification_results,
                                 batch_rc_results,
                                 batch_reg_metrics):
        raise NotImplementedError

    def write_run_summary(self, output_dir, all_subject_ids,
                          all_sparsification_results, all_rc_results,
                          all_reg_metrics):
        raise NotImplementedError


# ============================================================================
# Shared aggregation helpers (identical text/npz dump format in both legacy
# main.py files, modulo the optional HD95 column for cardiac).
# ============================================================================
def _write_naurc_nause_summary(out_path, sparsification_results, rc_results, n_total, header_label):
    if not (sparsification_results and rc_results):
        return
    frame_indices = sorted(
        {fi for sr in sparsification_results for fi in sr.keys()}
        | {fi for rr in rc_results for fi in rr.keys()}
    )
    blines = ["=" * 70,
              header_label.center(70),
              "=" * 70,
              f"  {'Frame':>5}  |  {'nAURC mean':>10}  {'nAURC std':>10}  |  "
              f"{'nAUSE mean':>10}  {'nAUSE std':>10}  |  {'N':>4}",
              "  " + "-" * 65]
    for fi in frame_indices:
        naurc_vals = [rr[fi]['aurc_norm'] for rr in rc_results if fi in rr]
        nause_vals = [sr[fi]['ause_norm'] for sr in sparsification_results if fi in sr]
        n = len(naurc_vals)
        blines.append(
            f"  {fi:>5}  |  {np.mean(naurc_vals):>10.4f}  {np.std(naurc_vals):>10.4f}  |  "
            f"{np.mean(nause_vals):>10.4f}  {np.std(nause_vals):>10.4f}  |  {n:>4}"
        )
    subj_mean_naurc = [np.mean([rr[fi]['aurc_norm'] for fi in rr]) for rr in rc_results if rr]
    subj_mean_nause = [np.mean([sr[fi]['ause_norm'] for fi in sr]) for sr in sparsification_results if sr]
    blines += ["  " + "-" * 65,
               f"  {'All':>5}  |  {np.mean(subj_mean_naurc):>10.4f}  {np.std(subj_mean_naurc):>10.4f}  |  "
               f"{np.mean(subj_mean_nause):>10.4f}  {np.std(subj_mean_nause):>10.4f}  |  {n_total:>4}",
               "=" * 70]
    with open(out_path, "w") as f:
        f.write("\n".join(blines) + "\n")


def _write_reg_metrics(folder, reg_metrics, *, with_hd95, header):
    """Common .npz + .txt dump for both modalities. Cardiac sets with_hd95=True."""
    reg_rmse   = [m['registration']['avg_rmse']         for m in reg_metrics]
    reg_dice   = [m['registration']['avg_dice']         for m in reg_metrics if m['registration']['avg_dice']  is not None]
    reg_negjac = [m['registration']['avg_neg_jac_pct']  for m in reg_metrics]
    mv_rmse    = [m['mean_velocity']['avg_rmse']        for m in reg_metrics]
    mv_dice    = [m['mean_velocity']['avg_dice']        for m in reg_metrics if m['mean_velocity']['avg_dice'] is not None]
    mv_negjac  = [m['mean_velocity']['avg_neg_jac_pct'] for m in reg_metrics]

    save_data = {
        'reg_rmse':        np.array(reg_rmse),
        'reg_neg_jac_pct': np.array(reg_negjac),
        'mv_rmse':         np.array(mv_rmse),
        'mv_neg_jac_pct':  np.array(mv_negjac),
    }
    if reg_dice: save_data['reg_dice'] = np.array(reg_dice)
    if mv_dice:  save_data['mv_dice']  = np.array(mv_dice)

    reg_hd95 = mv_hd95 = None
    if with_hd95:
        reg_hd95 = [m['registration']['avg_hd95']  for m in reg_metrics if m['registration']['avg_hd95']  is not None]
        mv_hd95  = [m['mean_velocity']['avg_hd95'] for m in reg_metrics if m['mean_velocity']['avg_hd95'] is not None]
        if reg_hd95: save_data['reg_hd95'] = np.array(reg_hd95)
        if mv_hd95:  save_data['mv_hd95']  = np.array(mv_hd95)

    np.savez(os.path.join(folder, 'registration_metrics.npz'), **save_data)

    with open(os.path.join(folder, 'registration_metrics_summary.txt'), "w") as f:
        f.write(f"{header} ({len(reg_metrics)} subjects)\n")
        f.write(f"Pretrained Registration:\n")
        f.write(f"  RMSE: {np.mean(reg_rmse):.6f} +/- {np.std(reg_rmse):.6f}\n")
        if reg_dice: f.write(f"  Dice: {np.mean(reg_dice):.6f} +/- {np.std(reg_dice):.6f}\n")
        if with_hd95 and reg_hd95: f.write(f"  HD95: {np.mean(reg_hd95):.4f} +/- {np.std(reg_hd95):.4f}\n")
        f.write(f"  Neg Jac %: {np.mean(reg_negjac):.4f} +/- {np.std(reg_negjac):.4f}\n")
        f.write(f"BridgeUQ Mean Velocity:\n")
        f.write(f"  RMSE: {np.mean(mv_rmse):.6f} +/- {np.std(mv_rmse):.6f}\n")
        if mv_dice: f.write(f"  Dice: {np.mean(mv_dice):.6f} +/- {np.std(mv_dice):.6f}\n")
        if with_hd95 and mv_hd95: f.write(f"  HD95: {np.mean(mv_hd95):.4f} +/- {np.std(mv_hd95):.4f}\n")
        f.write(f"  Neg Jac %: {np.mean(mv_negjac):.4f} +/- {np.std(mv_negjac):.4f}\n")


# ============================================================================
# Brain
# ============================================================================
class BrainAdapter(DatasetAdapter):
    name = "brain"

    @staticmethod
    def add_dataset_args(parser):
        from BridgeUQ.datasets.brain._backbone_paths import _BACKBONE_DATA_PATHS
        parser.add_argument('--backbone',
                            choices=list(_BACKBONE_DATA_PATHS.keys()),
                            default='tlrn',
                            help='Pretrained registration backbone (selects precomputed dir)')
        parser.add_argument('--data_path', type=str, default=None,
                            help='Override the auto-resolved precomputed data path')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Override the auto-resolved output directory')
        parser.add_argument('--use_train', action='store_true', default=True,
                            help='Iterate over train+val+test (ConcatDataset)')
        parser.add_argument('--brain_subset', dest='brain_subset',
                            type=str, default='mni88',
                            choices=['mni88', 'full'],
                            help="ADNI subset selection: 'mni88' for MNI_88.csv only, "
                                 "'full' for MNI_88.csv + MNI_data_DX_4f.csv")

    def default_hyperparameters(self) -> dict:
        hp = super().default_hyperparameters()
        hp.update({
            'INNER_ITERATIONS':    1000,
            'NUM_DIFFUSION_STEPS': 7,
        })
        return hp

    def build_loader(self, args):
        from torch.utils.data import DataLoader, ConcatDataset
        from BridgeUQ.datasets.brain._backbone_paths import _BACKBONE_DATA_PATHS, collate_brain_batch
        from uncertainty_brain_sde.data_utils import BrainDataset

        data_path = args.data_path or _BACKBONE_DATA_PATHS[args.backbone]
        if not os.path.isdir(data_path) or not any(
            f.endswith('.pt') for d, _, fs in os.walk(data_path) for f in fs
        ):
            raise FileNotFoundError(
                f"Precomputed brain data not found at: {data_path}\n"
                f"Run BridgeUQ.datasets.brain.precompute_velocities first."
            )

        train_ds = BrainDataset(data_path, split='all', mode='train', dataset=args.brain_subset)
        val_ds   = BrainDataset(data_path, split='all', mode='val',   dataset=args.brain_subset)
        test_ds  = BrainDataset(data_path, split='all', mode='test',  dataset=args.brain_subset)

        if args.use_train:
            active = ConcatDataset([train_ds, val_ds, test_ds])
        else:
            active = test_ds

        loader = DataLoader(
            active,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_brain_batch,
            num_workers=2,
            pin_memory=True,
        )
        return loader, {}  # no extra context for brain

    def get_dimensions(self, loader, ctx):
        first = next(iter(loader))
        num_time_steps = len(first['v_series']) - 1
        img_size = first['series'].shape[-1]
        return num_time_steps, img_size

    def get_output_dir(self, args):
        from BridgeUQ.datasets.brain._backbone_paths import _BACKBONE_TAGS
        if args.output_dir:
            return args.output_dir
        tag = _BACKBONE_TAGS[args.backbone]
        base = "basic_brain" if args.loss == 'mse' else f"basic_brain_{args.loss}"
        out_subdir = getattr(args, 'out_subdir', None) or 'uncertainty_brain_sde_v3'
        return ("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/"
                f"2026Experiments/BRAIN/outputs/{tag}/{base}/visualization/{out_subdir}")

    def process_batch(self, batch, args, ctx):
        series   = batch['series'].cuda()
        v_series = [v.cuda() for v in batch['v_series']]
        filenames = batch['filename']

        source_img  = series[:, 0:1]
        target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]
        v_series_bb = v_series   # zero endpoint at t=0 already present

        return {
            'series':      series,
            'source_img':  source_img,
            'target_imgs': target_imgs,
            'v_series_bb': v_series_bb,
            'lv_segs':     None,
            'file_paths':  filenames,
            'extras':      {},
        }

    def per_subject_metrics(self, batch_idx, subj_idx, batch_data, sampled_velocities, img_size):
        from BridgeUQ.datasets.brain.metrics import (
            _compute_sparsification_curves,
            _compute_risk_coverage_curves,
            compute_registration_metrics,
        )
        filenames = batch_data['file_paths']
        fname = filenames[subj_idx] if subj_idx < len(filenames) else f"subj{subj_idx}"
        subject_id = f"batch{batch_idx}_{fname}"

        spars = _compute_sparsification_curves(
            sampled_velocities, batch_data['v_series_bb'], batch_data['series'],
            batch_data['source_img'], subj_idx, img_size,
        )
        rc = _compute_risk_coverage_curves(
            sampled_velocities, batch_data['v_series_bb'], batch_data['series'],
            batch_data['source_img'], subj_idx, img_size,
        )
        reg_metrics = compute_registration_metrics(
            batch_data['source_img'], batch_data['target_imgs'], batch_data['v_series_bb'],
            sampled_velocities, subj_idx,
            inshape=(img_size, img_size),
            filename=fname,
        )
        return subject_id, spars, rc, reg_metrics

    def write_per_batch_summary(self, batch_idx, output_dir,
                                 batch_subject_ids,
                                 batch_sparsification_results,
                                 batch_rc_results,
                                 batch_reg_metrics):
        from BridgeUQ.datasets.brain.metrics import _plot_avg_sparsification, _plot_avg_risk_coverage
        folder = f"{output_dir}/aggregate/batch_{batch_idx:03d}"
        os.makedirs(folder, exist_ok=True)
        if batch_sparsification_results:
            _plot_avg_sparsification(batch_sparsification_results, batch_subject_ids, folder)
        if batch_rc_results:
            _plot_avg_risk_coverage(batch_rc_results, batch_subject_ids, folder)
        _write_naurc_nause_summary(
            os.path.join(folder, "per_frame_nAURC_nAUSE_summary.txt"),
            batch_sparsification_results, batch_rc_results,
            n_total=len(batch_rc_results),
            header_label=f"nAURC & nAUSE PER TIME FRAME (batch {batch_idx})",
        )
        if batch_reg_metrics:
            _write_reg_metrics(folder, batch_reg_metrics,
                               with_hd95=False,
                               header=f"Registration metrics — batch {batch_idx}")
        print(f"  [batch {batch_idx}] per-batch results saved to: {folder}")

    def write_run_summary(self, output_dir, all_subject_ids,
                          all_sparsification_results, all_rc_results,
                          all_reg_metrics):
        from BridgeUQ.datasets.brain.metrics import _plot_avg_sparsification, _plot_avg_risk_coverage
        folder = f"{output_dir}/aggregate"
        os.makedirs(folder, exist_ok=True)
        if all_sparsification_results:
            _plot_avg_sparsification(all_sparsification_results, all_subject_ids, folder)
        if all_rc_results:
            _plot_avg_risk_coverage(all_rc_results, all_subject_ids, folder)
        _write_naurc_nause_summary(
            os.path.join(folder, "per_frame_nAURC_nAUSE_summary.txt"),
            all_sparsification_results, all_rc_results,
            n_total=len(all_rc_results),
            header_label="nAURC & nAUSE PER TIME FRAME (averaged across subjects)",
        )
        if all_reg_metrics:
            _write_reg_metrics(folder, all_reg_metrics,
                               with_hd95=False,
                               header=f"Registration metrics averaged over {len(all_reg_metrics)} subjects")


# ============================================================================
# Cardiac
# ============================================================================
class CardiacAdapter(DatasetAdapter):
    name = "cardiac"

    @staticmethod
    def add_dataset_args(parser):
        parser.add_argument('--backbone',
                            choices=['tlrn', 'ltma', 'tm', 'voxelmorph', 'vmplus'],
                            default='tlrn',
                            help='Pretrained registration backbone')
        parser.add_argument('--combine_acdc', action='store_true', default=True,
                            help='Combine original 1200 + ACDC 862 training samples (default: True)')
        parser.add_argument('--no_combine_acdc', dest='combine_acdc', action='store_false',
                            help='Use only the original 1200 training samples')
        parser.add_argument('--use_train', action='store_true', default=False,
                            help='Iterate over the training set instead of test')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Override the workdir resolved by load_backbone_and_data')

    def default_hyperparameters(self) -> dict:
        hp = super().default_hyperparameters()
        hp.update({
            'INNER_ITERATIONS':    500,
            'NUM_DIFFUSION_STEPS': 14,
        })
        return hp

    def build_loader(self, args):
        from torch.utils.data import DataLoader
        from BridgeUQ.datasets.cardiac.backbones import load_backbone_and_data

        model, train_loader, test_loader, config = load_backbone_and_data(
            backbone=args.backbone, combine_acdc=args.combine_acdc,
        )
        if args.output_dir is not None:
            config.model.params.workdir = args.output_dir

        # Append loss-type suffix to workdir so non-MSE runs don't overwrite MSE outputs
        if args.loss != 'mse':
            config.model.params.workdir = f"{config.model.params.workdir}_{args.loss}"

        # Rebuild loaders with the CLI batch size
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=getattr(train_loader, 'num_workers', 4),
            pin_memory=getattr(train_loader, 'pin_memory', True),
            collate_fn=getattr(train_loader, 'collate_fn', None),
        )
        test_loader = DataLoader(
            test_loader.dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=getattr(test_loader, 'num_workers', 4),
            pin_memory=getattr(test_loader, 'pin_memory', True),
            collate_fn=getattr(test_loader, 'collate_fn', None),
        )
        loader = train_loader if args.use_train else test_loader
        ctx = {'model': model, 'config': config}
        return loader, ctx

    def get_dimensions(self, loader, ctx):
        model = ctx['model']
        first = next(iter(loader))
        first_series = first['series'].cuda()
        first_lv_segs = first.get('lv_segs')
        first_lv_segs = (first_lv_segs.cuda() if first_lv_segs is not None
                         else torch.zeros_like(first_series[:, 0:1]).cuda())
        with torch.no_grad():
            _, first_v_series, _, _, _ = model.sequence_register_no_avg_lowf_addlatentf(
                first_series, first_lv_segs
            )
        num_time_steps = len(first_v_series)
        img_size = first_series.shape[-1]
        del first_series, first_lv_segs, first_v_series
        torch.cuda.empty_cache()
        return num_time_steps, img_size

    def get_output_dir(self, args):
        # Cardiac's output dir is set by load_backbone_and_data via config.model.params.workdir.
        # We store ctx in a class-level attribute when build_loader is called from main.py;
        # but for clean dispatch, the unified main.py pulls it from ctx directly.
        raise RuntimeError(
            "CardiacAdapter.get_output_dir should not be called directly — "
            "the unified main.py reads <ctx['config'].model.params.workdir> after build_loader."
        )

    def resolve_output_dir(self, ctx, args):
        """Cardiac variant — call after build_loader so config is populated."""
        out_subdir = getattr(args, 'out_subdir', None) or 'uncertainty_sde_combined_acdc_v3'
        return f"{ctx['config'].model.params.workdir}/visualization/{out_subdir}"

    def process_batch(self, batch, args, ctx):
        model = ctx['model']
        series = batch['series'].cuda()
        lv_segs = batch.get('lv_segs')
        if lv_segs is not None:
            lv_segs = lv_segs.cuda()
        else:
            lv_segs = torch.zeros_like(series[:, 0:1]).cuda()

        file_paths = batch.get('path', [None] * series.shape[0])

        with torch.no_grad():
            Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = \
                model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)

        source_img = series[:, 0:1]
        target_imgs = [series[:, t:t+1] for t in range(1, series.shape[1])]

        zero_vel_train = torch.zeros_like(v_series[0])
        v_series_bb_train = [zero_vel_train] + v_series

        return {
            'series':      series,
            'source_img':  source_img,
            'target_imgs': target_imgs,
            'v_series_bb': v_series_bb_train,
            'lv_segs':     lv_segs,
            'file_paths':  file_paths,
            'extras': {
                'Sdef_series':      Sdef_series,
                'v_series':         v_series,
                'u_series':         u_series,
                'Sdef_mask_series': Sdef_mask_series,
                'ui_series':        ui_series,
            },
        }

    def per_subject_metrics(self, batch_idx, subj_idx, batch_data, sampled_velocities, img_size):
        from BridgeUQ.datasets.cardiac.metrics import (
            _compute_sparsification_curves,
            _compute_risk_coverage_curves,
            compute_registration_metrics,
        )
        file_paths = batch_data['file_paths']
        fname = file_paths[subj_idx] if isinstance(file_paths, list) else file_paths
        if fname is not None:
            base = os.path.basename(str(fname)).replace('.npy', '').replace('.npz', '')
            subject_id = f"batch{batch_idx}_{base}"
        else:
            subject_id = f"batch{batch_idx}_subj{subj_idx}"

        Sdef_series = batch_data['extras']['Sdef_series']
        v_series    = batch_data['extras']['v_series']

        spars = _compute_sparsification_curves(
            sampled_velocities, Sdef_series, batch_data['series'], subj_idx
        )
        rc = _compute_risk_coverage_curves(
            sampled_velocities, Sdef_series, batch_data['series'], subj_idx
        )
        reg_metrics = compute_registration_metrics(
            batch_data['source_img'], batch_data['target_imgs'],
            Sdef_series, v_series,
            batch_data['lv_segs'], sampled_velocities, subj_idx,
            inshape=(img_size, img_size),
        )
        return subject_id, spars, rc, reg_metrics

    def write_per_batch_summary(self, batch_idx, output_dir,
                                 batch_subject_ids,
                                 batch_sparsification_results,
                                 batch_rc_results,
                                 batch_reg_metrics):
        from BridgeUQ.datasets.cardiac.metrics import _plot_avg_sparsification, _plot_avg_risk_coverage
        folder = f"{output_dir}/aggregate/batch_{batch_idx:03d}"
        os.makedirs(folder, exist_ok=True)
        if batch_sparsification_results:
            _plot_avg_sparsification(batch_sparsification_results, batch_subject_ids, folder)
        if batch_rc_results:
            _plot_avg_risk_coverage(batch_rc_results, batch_subject_ids, folder)
        _write_naurc_nause_summary(
            os.path.join(folder, "per_frame_nAURC_nAUSE_summary.txt"),
            batch_sparsification_results, batch_rc_results,
            n_total=len(batch_rc_results),
            header_label=f"nAURC & nAUSE PER TIME FRAME (batch {batch_idx})",
        )
        if batch_reg_metrics:
            _write_reg_metrics(folder, batch_reg_metrics,
                               with_hd95=True,
                               header=f"Registration metrics — batch {batch_idx}")
        print(f"  [batch {batch_idx}] per-batch results saved to: {folder}")

    def write_run_summary(self, output_dir, all_subject_ids,
                          all_sparsification_results, all_rc_results,
                          all_reg_metrics):
        from BridgeUQ.datasets.cardiac.metrics import _plot_avg_sparsification, _plot_avg_risk_coverage
        folder = f"{output_dir}/aggregate"
        os.makedirs(folder, exist_ok=True)
        if all_sparsification_results:
            _plot_avg_sparsification(all_sparsification_results, all_subject_ids, folder)
        if all_rc_results:
            _plot_avg_risk_coverage(all_rc_results, all_subject_ids, folder)
        _write_naurc_nause_summary(
            os.path.join(folder, "per_frame_nAURC_nAUSE_summary.txt"),
            all_sparsification_results, all_rc_results,
            n_total=len(all_rc_results),
            header_label="nAURC & nAUSE PER TIME FRAME (averaged across subjects)",
        )
        if all_reg_metrics:
            _write_reg_metrics(folder, all_reg_metrics,
                               with_hd95=True,
                               header=f"Registration metrics averaged over {len(all_reg_metrics)} subjects")


# ============================================================================
# Dispatch
# ============================================================================
_REGISTRY = {
    'brain':   BrainAdapter,
    'cardiac': CardiacAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(_REGISTRY)}.")
    return _REGISTRY[name]()
