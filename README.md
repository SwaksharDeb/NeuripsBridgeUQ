# BridgeUQ: Bayesian Aleatoric Uncertainty Quantification For Time-Series Medical Image Registration via Brownian Bridge Prior

Plug-and-play module for any pretrained registration backbone to quantify aleatoric uncertainty for time series medical images.

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
| `data_utils.py` | `BrainDataset` — reads precomputed `.pt` files, applies CSV-based filtering (MNI_88, full ADNI). |
| `visualization.py` | Per-iteration training visualization (target / warped / `v̂` / σ / std_tilde / `v_t` / residual). |

---

## Quick start

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

## License

Research / academic use only.
