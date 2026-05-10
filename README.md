# BridgeUQ: Bayesian Aleatoric Uncertainty Quantification For Time-Series Medical Image Registration via Brownian Bridge Prior

Plug-and-play module for any pretrained registration backbone to quantify aleatoric uncertainty for time series medical images.

---

## Our Approach

![MFD-V2D Architecture](figures/overall_fig.png)

## Directory contents

| File | Purpose |
|------|---------|
| `main.py` | Contain the uncertainty estimatio code for the BridgeUQ. Given the backbone, this compute the uncertainty. |
| `trainer.py` | Contain code for the variance learning. |
| `brownian_bridge.py` | Contain code realated to Brownian-bridge. |
| `networks.py` | Variance learning network. |
| `losses.py` | Contain code for the optimization objective. |
| `warping.py` | Velocity-field warping via grid_sample. |
| `visualization.py` | Contain visualization code. |

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
--dataset {mni88,full}                   
```


## Key hyperparameters (top of `main.py`)

| Name | Default | Meaning |
|------|---------|---------|
| `INNER_ITERATIONS` | 1000 | K test-time training iterations per minibatch |
| `NUM_RUNS` | 100 | number of UQ samples per subject |
| `LAMBDA` / `ALPHA` | 3 / 1e-4 | inverse-Gamma prior on σ |

---

## License

Research / academic use only.
