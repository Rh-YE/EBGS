# EBGS — Euclid BGS Image Synthesis via Schrödinger Bridge

**EBGS** translates ground-based [DESI Legacy Survey](https://www.legacysurvey.org/) galaxy images into synthetic [Euclid](https://www.esa.int/Science_Exploration/Space_Science/Euclid) VIS-band images using an Image-to-Image Schrödinger Bridge (I2SB) diffusion model.

The model is trained on matched DESI–Euclid Q1 image pairs and learns the stochastic transport from the DESI *grz* photometry domain to the Euclid VIS domain, enabling high-fidelity morphology reconstruction for galaxies in the Bright Galaxy Sample (BGS).

---

## Method overview

The forward diffusion bridges the DESI mean image (x₁) to the Euclid VIS image (x₀) via a Gaussian Schrödinger Bridge:

```
q(x_t | x_0, x_1) = N(μ_x0·x_0 + μ_x1·x_1,  σ_sb²)
```

A UNet backbone is trained to predict the score `(x_t − x_0) / σ_fwd` at each timestep.  
Conditioning uses adjacent DESI band *difference* maps `[g−r, r−z]`, which encode colour gradients without redundancy with x₁.

Key extensions over the vanilla I2SB:

| Extension | Description |
|-----------|-------------|
| **BandwiseArcsinhTransform** | Per-band arcsinh stretch + linear normalisation to `[−1, 1]`; parameters measured from the training set |
| **χ² loss** | MSE weighted by `1/σ_euclid²` (heteroscedastic noise model) |
| **Forked sampling** | Multiple trajectories share the first `split_ratio` of diffusion steps, then diverge — ~47% fewer UNet calls for ensemble outputs |
| **Async FITS writer** | Multiprocess `ProcessPoolExecutor` + back-pressure semaphore for high-throughput inference |

---

## Repository structure

```
EBGS/
├── main.py                          # Training entry point (PyTorch Lightning)
├── i2sb_infrence.py                 # Inference with async FITS writing
├── fits_writer.py                   # FITS output worker
├── configs/
│   ├── generation/astroIR_SBn.yaml  # Training config
│   └── inference/astroIR_SBn.yaml   # Inference config (update data paths)
├── sgm/                             # Core package
│   ├── models/astreIR_SBn.py        # MultiModalSBDiffusion (primary model)
│   ├── modules/
│   │   ├── diffusionmodules/
│   │   │   └── openaimodel.py       # UNet backbone
│   │   ├── attention.py
│   │   └── ema.py
│   ├── data/
│   │   ├── dataset.py               # Unified FITS dataset
│   │   ├── base.py                  # BaseDataset (crop, normalise, reproject)
│   │   ├── datamodule.py            # LightningDataModule
│   │   └── utils.py
│   ├── transforms/
│   │   └── pixel_stretch.py         # BandwiseArcsinhTransform
│   ├── lr_scheduler.py
│   └── util.py
└── tests/
    ├── compare_forward.py
    └── test_numerical_equivalence.py
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-org>/EBGS.git
cd EBGS

# 2. Create conda environment
conda env create -f environment.yaml
conda activate ai4galaxy

# 3. (Optional) Install as editable package
pip install -e .
```

---

## Data preparation

Training expects matched DESI and Euclid cutouts organised as:

```
<root>/
├── train/
│   ├── DESI_reproj/
│   │   ├── BGSUB/<galaxy_id>.fits   # grz image (3, H, W)
│   │   └── RMS/<galaxy_id>.fits     # grz RMS error (3, H, W)
│   └── Euclid/
│       ├── BGSUB/<galaxy_id>.fits   # VIS image (1, H, W)
│       └── RMS/<galaxy_id>.fits     # VIS RMS error (1, H, W)
└── valid/  (same structure)
```

Update the `data.params.train.dataset.paths` entries in `configs/generation/astroIR_SBn.yaml` to point to your dataset root.

Pixel statistics for `BandwiseArcsinhTransform` (the `lower_bound`, `upper_bound`, `bg_median`, `bg_std` fields in the config) should be measured from your training set.  See `sgm/transforms/pixel_stretch.py` for the transform definition.

---

## Training

```bash
python main.py \
  -b configs/generation/astroIR_SBn.yaml \
  -t True \
  --wandb True \
  -n my_run
```

Key flags:

| Flag | Description |
|------|-------------|
| `-b` | One or more YAML config files (merged left-to-right) |
| `-t True` | Enable training mode |
| `--wandb False` | Disable WandB logging |
| `-r <logdir>` | Resume from a log directory or checkpoint path |
| `--resume_from_checkpoint <path.ckpt>` | Resume from a single checkpoint |
| `-n <name>` | Name postfix appended to the auto-generated log directory |
| `--scale_lr` | Scale `base_lr` by `ngpu × batch_size × accum_grad_batches` |

Logs and checkpoints are written to `logs/<timestamp>_<config_name>/`.  
Send `SIGUSR1` to the training process to save a checkpoint immediately.

---

## Inference

```bash
python i2sb_infrence.py \
  --config configs/inference/astroIR_SBn.yaml \
  --ckpt_path logs/<run>/checkpoints/last.ckpt \
  --output_dir /path/to/output \
  --seed 1024
```

Performance options:

| Flag | Default | Description |
|------|---------|-------------|
| `--bf16` | `True` | bf16 autocast (A100 tensor-core utilisation) |
| `--compile` | `True` | `torch.compile` the UNet |
| `--num_samples` | `3` | Repeated samples per input galaxy |
| `--split_ratio` | `0.7` | Shared-prefix fraction for forked sampling |
| `--save_workers` | `4` | FITS writer worker processes |
| `--max_inflight` | `64` | Max pending write tasks (back-pressure) |

### Output format

Each input galaxy produces one FITS file:

| HDU | Name | Shape | Description |
|-----|------|-------|-------------|
| 0 | Primary | `(num_samples, 1, H, W)` | Predicted Euclid VIS images |
| 1 | `DESI_IMG` | `(C, H, W)` | Input DESI bands |
| 2 | `DESI_ERROR` | `(C, H, W)` | DESI RMS error (optional) |
| 3 | `EUCLID_IMG` | `(1, H, W)` | Ground-truth Euclid image (optional) |
| 4 | `EUCLID_ERROR` | `(1, H, W)` | Euclid RMS error (optional) |

Images are in physical flux units (same zero-point as Euclid VIS, ZP = 24.6 AB mag).

---

## Tests

```bash
cd EBGS
python -m pytest tests/ -v
```

`tests/test_numerical_equivalence.py` verifies numerical consistency of the SBDiffusion process and model forward pass.  Tests that require a checkpoint are automatically skipped if `logs/` is absent.

---

## Configuration system

All model, dataset, and trainer components are specified as `target: dotted.class.Path` + `params:` dicts in YAML, instantiated via `sgm.util.instantiate_from_config` (OmegaConf-based).  Config values may reference other values with `${path.to.key}` interpolation.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ren2026ebgs,
  title  = {EBGS: Euclid BGS Image Synthesis via Schr\"{o}dinger Bridge Diffusion},
  author = {Anonymous},
  year   = {2026},
  url    = {https://github.com/<your-org>/EBGS}
}
```

---

## Acknowledgements

The UNet backbone and diffusion utilities build on [Stability AI / generative-models](https://github.com/Stability-AI/generative-models) and [NVlabs/I2SB](https://github.com/NVlabs/I2SB).  
This work uses data from the [DESI Legacy Imaging Surveys](https://www.legacysurvey.org/) and [Euclid Q1 public release](https://www.esa.int/Science_Exploration/Space_Science/Euclid).

---

## License

[MIT](LICENSE)
