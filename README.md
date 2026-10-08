# EBGS — Euclid BGS Image Synthesis via Schrödinger Bridge

EBGS translates ground-based [DESI Legacy Survey](https://www.legacysurvey.org/) galaxy images into synthetic [Euclid](https://www.esa.int/Science_Exploration/Space_Science/Euclid) VIS-band images using an Image-to-Image Schrödinger Bridge (I2SB) diffusion model.


---

## Single-galaxy demo (CPU or GPU)

The self-contained [single-galaxy notebook](single_galaxy_demo/single_galaxy_demo.ipynb) downloads a DESI cutout from RA/Dec or reads a local `rz` / `grz` FITS, then generates one central **12.8 arcsec, 128 × 128 pixel** EBGS image without stitching. CPU inference is the default; no error-map input is needed, and the output FITS contains only the generated image and its WCS.

The folder includes inference source code, the final EMA weights, example FITS, dependency installation, and an executed notebook. See the [demo README](single_galaxy_demo/README.md) for setup, input conventions, and validation records. The weights use [Git LFS](https://git-lfs.com/):

```bash
git lfs install
git clone https://github.com/Rh-YE/EBGS.git
cd EBGS
git lfs pull
cd single_galaxy_demo
```

Then follow the demo README to create a Python environment, run `python install.py`, and open the notebook. Set `RA`, `DEC`, and `INPUT_BANDS` (`"grz"` or `"rz"`) in its first code cell. New coordinates require internet access; cached cutouts and local FITS can run offline after installation.

The demo was checked in an isolated Linux CPU environment using real downloaded cutouts, including identical results for matching r/z data through both input modes. These checks establish software consistency, not independent validation of the generated galaxy structure.

---

## Method

The model implements I2SB (Liu et al., 2023), bridging the DESI r-band image (x₁) to the Euclid VIS image (x₀):

```
q(x_t | x_0, x_1) = N(μ_x0·x_0 + μ_x1·x_1,  σ_sb²)
```

The UNet is trained to predict the score `(x_t − x_0) / σ_fwd`.

**Inputs and conditioning** (from `build_x1` / `build_cond` in `astreIR_SBn.py`):

- `x1` (bridge source): DESI **r-band** (index 1)
- `cond` (UNet conditioning): DESI **z-band** (index 2)
- UNet input: `cat([x_t, cond], dim=1)` — **2 channels** in, 1 channel out

**Pixel transform** (`BandwiseArcsinhTransform` in `pixel_stretch.py`): each band is independently clipped, background-subtracted, arcsinh-stretched, and linearly mapped to `[−1, 1]`. Parameters (lower/upper bound, bg_median, bg_std) are measured from the training set and stored in the config.

**Loss**: plain MSE on the score prediction:

```
L = mean( (pred − label)² )
```

where `label = (x_t − x_0) / σ_fwd`. The monitoring metric `val/reduced_chi2` is computed separately using the Euclid error map and is used for checkpoint selection (`monitor: val/reduced_chi2, mode: min`).

---

## Repository structure

```
EBGS/
├── main.py                          # Training entry point (PyTorch Lightning)
├── i2sb_infrence.py                 # Inference with async FITS writing
├── fits_writer.py                   # FITS writer (runs in separate worker processes)
├── configs/
│   ├── generation/astroIR_SBn.yaml  # Training config
│   └── inference/astroIR_SBn.yaml   # Inference config
├── sgm/
│   ├── models/astreIR_SBn.py        # MultiModalSBDiffusion — primary model
│   ├── modules/
│   │   ├── diffusionmodules/openaimodel.py  # UNet backbone
│   │   ├── attention.py
│   │   └── ema.py
│   ├── data/
│   │   ├── base.py                  # BaseDataset (crop, rotate, error maps, pixel masks)
│   │   ├── dataset.py               # Dataset (single / multi-folder / matched-dict paths)
│   │   ├── datamodule.py            # LightningDataModule
│   │   └── utils.py
│   ├── transforms/
│   │   └── pixel_stretch.py         # BandwiseArcsinhTransform, IdentityTransform
│   ├── lr_scheduler.py              # LambdaLinearScheduler
│   └── util.py
└── tests/
    ├── compare_forward.py
    └── test_numerical_equivalence.py
```

---

## Installation

```bash
git clone https://github.com/Rh-YE/EBGS.git
cd EBGS
conda env create -f environment.yaml
conda activate ai4galaxy
pip install -e .
```

---

## Data preparation

Training expects matched DESI and Euclid cutouts with the following layout:

```
<root>/
├── train/
│   ├── DESI_reproj/
│   │   ├── BGSUB/<id>.fits    # grz image (3, H, W), background-subtracted
│   │   └── RMS/<id>.fits      # grz RMS error (3, H, W)
│   └── Euclid/
│       ├── BGSUB/<id>.fits    # VIS image (1, H, W), background-subtracted
│       └── RMS/<id>.fits      # VIS RMS error (1, H, W)
└── valid/  (same structure)
```

The dataset is loaded as a matched dict (`match_files: true`), taking the filename intersection of the DESI and Euclid folders.

Update `data.params.train.dataset.paths` in `configs/generation/astroIR_SBn.yaml` to point to your data root.

**Measuring pixel-transform parameters**: the `lower_bound`, `upper_bound`, `bg_median`, and `bg_std` entries in `pixel_transform_config` must be measured from your training set. See `sgm/transforms/pixel_stretch.py` for the transform definition.

---

## Training

```bash
python main.py \
  -b configs/generation/astroIR_SBn.yaml \
  -t True \
  --wandb True \
  -n <run_name>
```

Key CLI flags:

| Flag | Description |
|------|-------------|
| `-b` | One or more YAML config files (merged left-to-right) |
| `-t True` | Enable training mode |
| `-n <name>` | Name postfix appended to the auto-generated log directory |
| `-r <path>` | Resume from a log directory or checkpoint path |
| `--resume_from_checkpoint <path>` | Resume from a single `.ckpt` file |
| `--wandb False` | Disable WandB logging |
| `--scale_lr` | Scale `base_lr` by `ngpu × batch_size × accum_grad_batches` |

Logs and checkpoints are written to `logs/<timestamp>_<config_name>/`.

Send `SIGUSR1` to the training process to save a checkpoint immediately.

**Training hyperparameters** (from `configs/generation/astroIR_SBn.yaml`):

| Parameter | Value |
|-----------|-------|
| Diffusion interval | 500 |
| β_max | 0.15 |
| OT-ODE | false |
| NFE (inference) | 10 |
| UNet model_channels | 64 |
| UNet channel_mult | [1, 2, 4, 4] |
| Attention resolutions | [16, 8] |
| num_res_blocks | 2 |
| num_head_channels | 32 |
| Attention type | softmax-xformers |
| Optimizer | AdamW (lr=1e-4, β=[0.9,0.999], wd=1e-5) |
| LR schedule | warm-up 1000 steps → linear decay |
| Training precision | bf16-mixed |
| Batch size | 256 |
| Patch size | 128 × 128 |
| Monitored metric | `val/reduced_chi2` (min) |

---

## Inference

```bash
python i2sb_infrence.py \
  --config configs/inference/astroIR_SBn.yaml \
  --ckpt_path logs/<run>/checkpoints/last.ckpt \
  --output_dir /path/to/output \
  --seed 1024
```

**All CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/inference/astroIR_SBn.yaml` | Config file |
| `--ckpt_path` | — | Checkpoint path (required) |
| `--output_dir` | — | Output directory (required) |
| `--seed` | `1024` | Random seed |
| `--device_id` | `cuda:0` | CUDA device |
| `--num_samples` | `3` | Repeated samples per input |
| `--split_ratio` | `0.7` | Shared-prefix fraction for forked sampling |
| `--bf16` | `True` | bf16 autocast (recommended on A100) |
| `--compile` | `True` | `torch.compile` the UNet |
| `--channels_last` | `True` | channels-last memory format for conv2d |
| `--save_workers` | `32` | FITS writer worker processes |
| `--max_inflight` | `128` | Max pending write tasks (back-pressure) |
| `--catalog` | — | BGS catalog FITS path; filters inputs by `TARGET_RA`/`TARGET_DEC` |
| `--shape_r_max` | — | Only process galaxies with `SHAPE_R < shape_r_max` (arcsec) |
| `--footprint` | — | HEALPix footprint mask; keeps only files inside the valid sky area |

**Throughput optimisations:**

- **bf16 autocast**: UNet runs in bf16; diffusion posterior ops stay in fp32
- **Forked sampling**: `num_samples` trajectories share the first `split_ratio` fraction of reverse steps, then diverge. With `num_samples=3, split_ratio=0.7, NFE=10`: 7 shared steps (batch B) + 3 forked steps (batch 3B), saving ~47% UNet forwards vs. 3 independent runs
- **One-shot EMA swap**: EMA weights copied into the model once before the loop, eliminating per-batch overhead
- **`torch.compile`**: UNet compiled with `mode='default'`; first run triggers 5–15 min compilation, subsequent runs use cache
- **Channels-last**: maximises A100 utilisation for conv2d

### Output format

Each input produces one FITS file:

| HDU | Name | Shape | Description |
|-----|------|-------|-------------|
| 0 | Primary | `(1, H, W)` | Predicted Euclid VIS (mean over `num_samples`) |
| 1 | `DESI_IMG` | `(C, H, W)` | Input DESI bands (raw physical units) |
| 2 | `DESI_ERROR` | `(C, H, W)` | DESI RMS error (optional) |
| 3 | `EUCLID_IMG` | `(1, H, W)` | Ground-truth Euclid image (optional) |
| 4 | `EUCLID_ERROR` | `(1, H, W)` | Euclid RMS error (optional) |
| 5 | `PREDICTION_STD` | `(1, H, W)` | Pixel-wise std over `num_samples` (present when `num_samples > 1`) |

The Primary HDU header contains `GENTYPE = 'SB_DESI_to_Euclid'`. All images are in raw physical flux units (same zero-point as the input data). The output is written atomically via a `.tmp` rename to avoid partial files on crash.

---

## Tests

```bash
cd EBGS
python -m pytest tests/ -v
```

`tests/test_numerical_equivalence.py` verifies:

1. `SBDiffusion` internal arrays (`std_fwd`, `mu_x0`, `mu_x1`, `std_sb`) match the analytic formulas
2. `q_sample` OT-ODE mode equals `μ_x0·x0 + μ_x1·x1` exactly
3. Stochastic `q_sample` is reproducible under same seed
4. `compute_pred_x0(step, xt, compute_label(step, x0, xt)) == x0` (score round-trip)
5. `run_network` output shape is `(B, 1, H, W)`
6. `sb_loss` formula matches `mean((pred − label)²)`

Tests requiring a checkpoint are automatically skipped if `logs/` is absent.

---

## Configuration system

All components (model, dataset, optimizer, callbacks) are specified as `target: dotted.class.Path` + `params:` dicts in YAML, instantiated via `sgm.util.instantiate_from_config` (OmegaConf). Config values support `${path.to.key}` interpolation. Multiple `-b` flags merge configs left-to-right.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ye2026desi2euclid,
  title   = {From DESI to Euclid: A Generative Bridge to Unbiased Galaxy Structures},
  author  = {Ye, Renhao and Shen, Shiyin},
  journal = {arXiv preprint arXiv:2607.06891},
  year    = {2026},
  eprint  = {2607.06891},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.GA},
  url     = {https://arxiv.org/abs/2607.06891}
}
```

The predicted Euclid-resolution BGS dataset (E-BGS) covering the Euclid DR1 footprint is released on Zenodo: [10.5281/zenodo.21032414](https://doi.org/10.5281/zenodo.21032414).

---

## Acknowledgements

The UNet backbone and diffusion utilities build on [Stability AI / generative-models](https://github.com/Stability-AI/generative-models) and [NVlabs/I2SB](https://github.com/NVlabs/I2SB). This work uses data from the [DESI Legacy Imaging Surveys](https://www.legacysurvey.org/) and the Euclid Q1 public release.

---

## License

[MIT](LICENSE)
