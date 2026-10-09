# Single-galaxy DESI → EBGS notebook

Open **`single_galaxy_demo.ipynb`**, enter RA and Dec, and run the cells in order. The tool downloads a DESI cutout, passes it to CPU inference, generates an EBGS / synthetic Euclid VIS image, and displays a comparison. The default coordinates correspond to example galaxy `1-10166`; local FITS input is also supported.

Inputs are sky-subtracted, linear-flux DESI FITS images with `r,z` or `g,r,z` channels. Supply coordinates or a local image; the interface does not accept, read, or write error maps. Production preprocessing resamples the image and selects the central **128×128 pixels, or 12.8″×12.8″**, before I2SB inference. No tiling or stitching is performed. The output sampling is 0.1″/pixel, which is not a measurement of the model's actual resolution. Large galaxies may extend beyond this central field; generated images are not actual Euclid observations.

## Copyright and required citation

**Copyright (c) 2026 EBGS Authors.** Authors: Renhao Ye and Shiyin Shen. The code is distributed under the repository's [MIT License](https://github.com/Rh-YE/EBGS/blob/main/LICENSE); retain its copyright and license notices when redistributing it. Third-party components retain their respective notices.

**Citation requirement:** Cite Renhao Ye and Shiyin Shen (2026), *From DESI to Euclid: A Generative Bridge to Unbiased Galaxy Structures*, [arXiv:2607.06891](https://arxiv.org/abs/2607.06891), in any research or publication that uses this notebook, the EBGS model, or images generated with it. A BibTeX entry is provided at the beginning of the notebook.

## Folder contents

| File / directory | Purpose |
| --- | --- |
| `single_galaxy_demo.ipynb` | Complete example: coordinate download or local input, inference, display, and saving |
| `download_cutout.py`, `downloads/` | RA/Dec download tool and validated cutout cache |
| `inference.py` | CPU / CUDA single-galaxy interface; CPU float32 by default |
| `sgm/` | Required project source code, independent of the original checkout |
| `resampling.py`, `display_utils.py` | Production resampling and default `arcsinh_rgb` display |
| `configs/model.yaml` | Inference configuration with native PyTorch attention |
| `configs/production_model.yaml` | Original model configuration for provenance |
| `weights/best_ema.safetensors` | Complete float32 EMA inference weights, approximately 124 MiB |
| `weights/provenance.json` | Source checkpoint and exported-weight hashes and metadata |
| `example/BGSUB/` | grz and rz input FITS images of the same galaxy |
| `requirements.txt`, `install.py` | Third-party dependency versions and installation entry point |
| `requirements-tested-lock.txt` | Complete pip version list for the tested environment, including indirect dependencies |
| `validate_demo.py`, `validate_download.py`, `validation/` | CPU / download checks, production references, and measured results |
| `outputs/` | Generated FITS, JSON, and comparison PNG files from completed runs; each new run creates a subdirectory |
| `source_manifest.json`, `SHA256SUMS` | Source provenance and checksums for the delivered files |

Copy the complete folder. The original `DESI2Euclid` repository, `/datapool`, PSF data, and additional model weights are not required. Project source dependencies, weights, and example data are included; the installer obtains third-party Python packages. **Initial installation and downloading new coordinates require internet access; local files and cached cutouts support offline inference.** The folder does not include a preinstalled virtual environment, a Python interpreter, or offline wheels for every operating system.

## Install and open

Use **Python 3.11–3.13**. Run the environment and installation commands from this folder.

If cloning from GitHub, install [Git LFS](https://git-lfs.com/) first and retrieve the complete weights:

```bash
git lfs install
git clone https://github.com/Rh-YE/EBGS.git
cd EBGS
git lfs pull
cd single_galaxy_demo
```

On GitHub, `weights/best_ema.safetensors` is stored using Git LFS. Its actual size is **129,668,236 bytes**. A file containing only a few lines of text is an LFS pointer; run `git lfs pull` from the repository to obtain the weights. The complete standalone ZIP already includes the actual weights.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python install.py
python -m jupyterlab single_galaxy_demo.ipynb
```

Windows PowerShell, without changing the script execution policy:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe install.py
.\.venv\Scripts\python.exe -m jupyterlab single_galaxy_demo.ipynb
```

`install.py` installs the official CPU builds of torch / torchvision on Linux and Windows, and the platform PyTorch packages on macOS. The tested platform and measured runtime are recorded in `validation/cpu_validation.json`; Windows and macOS have not been tested on this server.

Select this virtual environment's Python kernel in Jupyter. Launching Jupyter with the commands above uses the environment. To open the notebook in an existing Jupyter or VS Code installation, register the kernel first:

```bash
python -m ipykernel install --user --name desi2euclid-demo --display-name "DESI2Euclid demo"
```

## CPU and GPU

The notebook defaults to `DEVICE = "cpu"` and `CPU_THREADS = 4`, using float32, native attention, and a single spatial crop. It does not use autocast, xformers, torch.compile, or a multiprocess DataLoader.

The defaults `NUM_SAMPLES = 3`, `NFE = 10`, and `SPLIT_RATIO = 0.7` match the MaNGA inference configuration. All three samples represent the same central image and are averaged after transforming back to physical values. Setting `NUM_SAMPLES = 1` reduces computation but changes the generated result.

For an NVIDIA GPU, create a separate environment and follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to install torch 2.8.0 / torchvision 0.23.0 CUDA builds compatible with your driver. Then run `python install.py --keep-torch` and set `DEVICE = "cuda:0"`. `DEVICE = "auto"` selects CUDA when available and otherwise falls back to CPU. Apple MPS is not enabled in this demo; Macs can use CPU inference.

## Download from RA and Dec and run inference

Set the following in the first code cell:

```python
INPUT_MODE = "coordinates"
RA = 199.2955574
DEC = 0.913561334
INPUT_BANDS = "grz"  # or "rz"
```

RA and Dec are ICRS decimal degrees, in `[0,360)` and `[-90,90]`, respectively. Running the download cell automatically points `IMAGE_PATH` to the downloaded FITS. Continue through the remaining cells to generate EBGS.

The tool uses the [official Legacy Surveys FITS cutout endpoint](https://www.legacysurvey.org/dr10/description/#obtaining-images-and-raw-data), with `layer="ls-dr10"` by default: DR9 north combined with DR10 south. It requests only the coadd `IMAGE`, at a fixed size of 128×128 pixels and 0.262″/pixel, covering approximately 33.5″. The grz or rz channels and celestial WCS are retained. The original `prepare_input` then prepares the central 12.8″ field at 0.1″/pixel; it does not shrink the entire 33.5″ image into the model input.

The official coadd images are already sky-subtracted. The tool preserves linear flux without additional background subtraction or display stretching. The viewer resamples to the requested WCS, so downloaded pixels need not be bitwise identical to a direct cutout from a survey brick.

The downloader can also be called independently, without loading the model:

```python
from download_cutout import download_cutout
from inference import prepare_input

image_path = download_cutout(199.2955574, 0.913561334, bands="rz")
desi, header, info = prepare_input(image_path)
```

Or download from the terminal:

```bash
python download_cutout.py --ra 199.2955574 --dec 0.913561334 --bands grz
```

Files are saved under `downloads/`. An accompanying JSON records request parameters, the official URL, download time, and SHA-256. A complete, validated cache with identical parameters is reused without network access or overwriting. Format, completeness, bands, shape, WCS, and pixel values are validated before temporary files are published as the final FITS. HTML responses, damaged files, and images that fail the coverage checks below are rejected before inference.

Coverage checks use the images alone: NaN/Inf values, constant bands, and any exact-zero pixels are rejected to avoid using common zero-filled gaps. This conservative check can also reject legitimate zeros; it is not a complete exposure-coverage audit. If a cutout is rejected, check the coordinates and survey coverage rather than filling missing pixels with zeros to continue inference.

## Use your own local FITS

Set `INPUT_MODE = "file"`, then set `IMAGE_PATH`, `INPUT_BANDS`, and `INPUT_PIXEL_SCALE` in the first code cell. The primary HDU must have shape `(2,H,W)` in `r,z` order or `(3,H,W)` in `g,r,z` order. Units and preprocessing must match the DESI BGSUB training inputs. RGB/JPEG images and previously stretched arrays are not suitable inputs.

`INPUT_BANDS = "auto"` first reads the FITS `BANDS` keyword. If absent, two channels are interpreted as `rz` and three as `grz`. You can also specify `"rz"` or `"grz"` explicitly. Conflicting channel counts, declared orders, or header metadata raise an error to prevent band mismatches.

The folder includes `example/BGSUB/1-10166.fits` (grz) and `example/BGSUB/1-10166_rz.fits` (rz). The latter contains an exact extraction of the former's r/z channels, with the same WCS. In file mode, switch `IMAGE_PATH` and use `INPUT_BANDS="auto"` to try either format.

Preprocessing preserves the production area-average resampling without additional zero-point or pixel-area flux conversion. If WCS is present, the pixel scale is checked and the coordinate mapping is carried into the output. This fast path does not support distorted WCS. Input coverage must include the central 12.8″; NaN/Inf pixels raise an error rather than representing unobserved sky as valid data.

The model uses the r band as the bridge source and the z band as conditioning; g is used only for the DESI color display. The two-channel path retains the original r/z pixel-transform parameters. With identical r/z data and random seeds, both input formats generate identical EBGS images. No additional channel is required in user inputs or saved results.

The grz display uses the original `main.py` defaults for `arcsinh_rgb(..., clip=True)`. For rz input, r and z are shown separately as arcsinh grayscale images; no missing g band is synthesized. EBGS is displayed with an arcsinh stretch over the full data range, without percentile clipping. Training-time pixel transforms remain part of the model; display settings do not change saved arrays.

## Weights and outputs

Source checkpoint: `logs/2026-06-01T20-48-48_generation-astroIR_SBn/checkpoints/best.ckpt`, epoch **332**, global step **86580**.

SHA-256: `788dbab951dd7f127aaada1f142191deaa52151708cb69c99da5ea456bd7bcb8`.

Export removes only optimizer state, training state, and duplicate non-EMA weights. Every inference tensor was checked against the original EMA values; no quantization was applied. Runtime `use_ema=False` means the EMA values are already folded into the weights and no second EMA buffer is needed. Loading uses `strict=True` and verifies the weight hash.

Each notebook run creates `outputs/<target>_<device>_<unique-suffix>/` containing:

- `*_ebgs.fits`: a single primary HDU with the two-dimensional float32 generated image and WCS, without input or error images; includes FITS CHECKSUM / DATASUM.
- `*_ebgs.json`: input and weight provenance, device, random seed, sampling parameters, and timing.
- `*_comparison.png`: DESI / EBGS comparison over the same central field.

## Validation

```bash
python validate_demo.py
python validate_download.py
```

These checks compare preprocessing of a real galaxy against the original Dataset, compare exported-EMA predictions against an original-checkpoint CPU float32 reference, verify pixelwise rz/grz equivalence, test implicit and explicit band selection and invalid declarations, check WCS mapping, and confirm that all project modules are imported from this folder. Reference inference uses the original model source and checkpoint, with only the attention backend changed to the native CPU-compatible implementation. The reference prediction is not Euclid ground truth.

`validate_download.py` uses actual downloaded caches to check integration, HTML and truncated responses, incorrect bands or coordinates, zero-filled regions, and cache reuse. It does not access the internet. Results are recorded in `validation/download_validation.json`.

`validation/notebook_execution.json` and `validation/rz_notebook_execution.json` record complete coordinate-mode grz/rz runs; file-mode execution is recorded in `validation/file_notebook_execution.json`. CPU, production GPU bf16, different operating systems, and different numerical libraries need not produce bitwise-identical results. Software execution checks are distinct from scientific validation of generated structure.

The demo was tested on 2026-10-08 in an isolated Linux environment with Python 3.13.9 and `torch 2.8.0+cpu`, without xformers or CUDA. Current rz/grz equivalence, runtime, and memory measurements are recorded in `validation/cpu_validation.json`. Both notebook input modes were executed. Server measurements do not predict performance on an ordinary laptop.

All bundled documentation, comments, docstrings, and runtime messages are in English. Source provenance distinguishes the original source hashes from this language revision; numerical operations and model weights are unchanged. `validation/english_validation.json` records the text scan and code comparison for the translation.
