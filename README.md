# BridgeUQ — Brain (longitudinal MRI)

Aleatoric uncertainty quantification for **deformable brain registration** on longitudinal MRI sequences. Plug-and-play module for any pretrained registration backbone — no retraining required.

This directory implements the brain pipeline of *BridgeUQ: Hierarchical Gaussian Process Brownian Bridge for Spatiotemporal Registration Uncertainty* (NeurIPS 2026). The cardiac counterpart lives in `../uncertainty_sde_combined_acdc_v3/`; both share the same algorithm and trainer.

---

## What it does

Given a pretrained registration network `R` that maps a brain MRI sequence `I = {I_t}` to a sequence of deterministic velocity fields `v̂ = {v̂_t}`, BridgeUQ produces:

- **`v̄_t`** — a refined mean velocity at each time point,
- **`u_t`** — a per-voxel, per-frame uncertainty map (epistemic over the registration solution).

The model treats velocity trajectories as random variables drawn from a **learnable Brownian bridge** posterior, conditioned on the image sequence and the deterministic prediction. A single network `f_θ` predicts the bridge's diffusion variance σ from `(I, v̂, v)`; samples are drawn via a posterior reverse-time SDE.

---

## Method (one-paragraph version)

We define a hierarchical GP prior in spatiotemporal velocity space — a Brownian bridge anchored at `v̂_0` and `v̂_T` with learnable variance schedule σ — combined with an energy-based likelihood derived from the registration objective and an inverse-Gamma hyper-prior on σ. Inference alternates two amortized MAP subproblems:

1. **Predict σ** from `(I, v̂, v^(r))` via a 3-branch UNet (`networks.py`).
2. **Sample N trajectories** `v^(n)` from the posterior reverse SDE conditioned on σ (`brownian_bridge.py`), then take a gradient step on the Monte-Carlo expected log-posterior (`trainer.py`).

After K inner iterations on each minibatch, the final σ ("s^K") is persisted to disk; downstream test scripts redraw N samples from it to compute `v̄_t` and `u_t`. See Algorithm 1 in the paper.

---

## Directory contents

| File | Purpose |
|------|---------|
| `main.py` | Entry point. Loads precomputed velocities for the chosen backbone, runs K-iteration test-time training per minibatch, persists `s^K`, draws `NUM_RUNS` samples per minibatch for UQ, writes per-batch + run-level aggregates. |
| `trainer.py` | `ScalingFactorTrainer.train_step` — Algorithm 1 inner loop. Predict σ → sample N reverse-SDE trajectories with σ in the autograd graph → mean-of-loss MC estimator of the expected log-posterior → backprop → update v_current. |
| `brownian_bridge.py` | Brownian-bridge formulae: prior reverse process (`run_reverse_process`) and posterior reverse process (`run_posterior_reverse_process`) with energy-gradient correction. |
| `networks.py` | `ScalingFactorNetwork` — 3-branch UNet that takes the image sequence, the registration velocities `v̂`, and the current iterate `v^(r)` and predicts diagonal σ at each voxel. Softplus output for positivity. |
| `losses.py` | `LTMALoss.compute_loss` — assembles similarity + bridge transition log-density + inverse-Gamma prior into the optimization objective. |
| `warping.py` | Velocity-field warping via `grid_sample` (vectorized, autograd-friendly). |
| `precompute_velocities.py` | One-time offline pass: runs the chosen backbone over the dataset and saves `(image_sequence, v̂)` pairs as `.pt` files for fast loading. |
| `data_utils.py` | `BrainDataset` — reads precomputed `.pt` files, applies CSV-based filtering (MNI_88, full ADNI). |
| `test.py` / `test_fast.py` | Test-time inference: loads `s^K`, draws `NUM_RUNS` posterior samples, computes registration metrics + uncertainty curves (sparsification, risk-coverage, nAUSC, nAURC). |
| `test_dice_boxplot.py` / `plot_combined_dice.py` | Per-frame Dice statistics + box-plot rendering across backbones. |
| `compare_four_methods.py` | Side-by-side qualitative + quantitative comparison vs. PULPo / DIF-VM / SGMCMC baselines. |
| `visualization.py` | Per-iteration training visualization (target / warped / `v̂` / σ / std_tilde / `v_t` / residual). |
| `slurm_train_bridgeuq.sh` | SLURM submission script for the cluster. |

---

## Quick start

### 1. Precompute backbone velocities (one-time, per backbone)

```bash
python -m uncertainty_brain_sde_v3.precompute_velocities --backbone tlrn
```

Supported backbones: `tlrn`, `ltma`, `tm`, `voxelmorph`. Output goes to one of the directories listed in `_BACKBONE_DATA_PATHS` at the top of `main.py`.

### 2. Run BridgeUQ training + UQ in one pass

```bash
python -m uncertainty_brain_sde_v3.main --backbone tlrn
```

Common flags:

```
--backbone {tlrn,ltma,tm,voxelmorph}    pretrained registration model (default: tlrn)
--batch_size INT                         minibatch size
--num_train_samples INT                  N reverse-SDE trajectories per inner iter (default: 100)
--loss {mse,ncc,ssim,...}                similarity term inside the energy
--mixed_precision / --no_mixed_precision fp16 autocast for activations (SDE coeffs stay fp32)
--dataset {mni88,full}                   ADNI subset
```

The script writes:

```
<output>/training_vis/batch_NNN/training_vis_iterKKKK.png   # per-iter visualization
<output>/s_K/batch_NNN.pt                                    # σ artifact
<output>/aggregate/batch_NNN/                                # per-batch metrics
<output>/aggregate/                                          # run-level metrics
```

### 3. Test from a saved `s^K` (no retraining)

```bash
python -m uncertainty_brain_sde_v3.test --backbone tlrn --subject_idx 0
python -m uncertainty_brain_sde_v3.test_fast --backbone tlrn      # fast aggregate sweep
```

`test.py` / `test_fast.py` load `s_K/batch_NNN.pt` and reproduce `main.py`'s posterior reverse-SDE samples without redoing the K-iteration optimization.

---

## Outputs

For every minibatch and at the end of the run:

- **Sparsification curve** + **nAUSC** — does removing high-uncertainty voxels reduce error monotonically?
- **Risk-coverage curve** + **nAURC** — registration error vs. retained-coverage fraction.
- **Registration RMSE / Dice / Neg-Jacobian %** — both for the deterministic backbone and BridgeUQ's mean velocity, so you can verify uncertainty quantification doesn't sacrifice accuracy.
- Per-frame breakdown (since the bridge variance peaks at the midpoint of the trajectory).

All saved as `.svg` / `.png` plots and `.npz` / `.txt` summaries.

---

## Key hyperparameters (top of `main.py`)

| Name | Default | Meaning |
|------|---------|---------|
| `INNER_ITERATIONS` | 1000 | K test-time training iterations per minibatch |
| `NUM_DIFFUSION_STEPS` | 7 | discretization of the bridge SDE |
| `NUM_RUNS` | 100 | number of UQ samples per subject |
| `--num_train_samples` | 100 | N i.i.d. trajectories per inner iter (mean-of-loss MC estimator) |
| `GAMMA` | 5e-5 | likelihood temperature in `p(I|v) ∝ exp(−E(v)/γ)` |
| `GUIDANCE_SCALE` | 1.0 | strength of the energy-gradient correction in the reverse SDE |
| `LAMBDA_SCALE` / `ALPHA_SCALE` | 3 / 1e-4 | inverse-Gamma prior on σ |

---

## Algorithm — what runs per inner iteration

```
1.  σ        ← f_θ( I, v̂, v_current )                          # network forward
2.  v^(1..N) ← reverse-SDE( σ, v̂, I )                           # N samples, σ in the graph
3.  L̂(σ)    ← (1/N) Σ_n  L_one( v^(n), σ )                     # mean-of-loss MC estimator
4.  θ        ← θ + α ∇_θ L̂                                     # gradient step
5.  v̄_t     ← (1/N) Σ_n v^(n);   v_current ← v̄.detach()         # update iterate
```

Step 2 is reparameterized: every `randn_like` in the SDE is a fixed Gaussian noise `ε^(n)`, the rest is a deterministic function of σ. That makes step 4's chain rule walk through the SDE back to θ — including the data-fit gradient on σ that comes from the Jensen gap between `(1/N) Σ L(v^(n))` and `L(mean(v))`. The mean-of-loss form (rather than loss-of-mean) is what preserves this gradient, and is the difference between "uniform σ" and "anatomically structured σ" in practice.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{deb2026bridgeuq,
  title     = {BridgeUQ: Hierarchical Gaussian Process Brownian Bridge for
               Spatiotemporal Registration Uncertainty},
  author    = {Deb, Swakshar and Zhang, Miaomiao and ...},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## License

Research / academic use only.
