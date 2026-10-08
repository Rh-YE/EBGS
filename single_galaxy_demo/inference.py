"""Single DESI galaxy -> synthetic Euclid VIS, using packaged EMA weights.

No tiles, no CUDA requirement, no downloads, no production pipeline imports.
"""
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import torch
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from omegaconf import OmegaConf
from safetensors.torch import load_file

from resampling import _reproj_fast
from sgm.util import instantiate_from_config
from sgm.transforms.pixel_stretch import BandwiseArcsinhTransform

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(device="cpu", cpu_threads=4):
    """Load float32 EMA weights; native torch attention works on CPU/CUDA."""
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This demo supports device='cpu', 'cuda:0', or 'auto'.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use DEVICE = 'cpu'.")
    if int(cpu_threads) < 1:
        raise ValueError("cpu_threads must be positive.")
    torch.set_num_threads(min(int(cpu_threads), os.cpu_count() or 1))
    config = OmegaConf.load(ROOT / "configs/model.yaml")
    model = instantiate_from_config(config.model)
    weights = ROOT / "weights/best_ema.safetensors"
    provenance = json.loads((ROOT / "weights/provenance.json").read_text())
    if sha256(weights) != provenance["inference_weights_sha256"]:
        raise RuntimeError("Packaged model checksum mismatch; copy the weights again.")
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    return model


def resolve_bands(channels, bands="auto", header_bands=None):
    """Accept only ordered rz or grz; check declarations against array shape."""
    def canonical(value):
        return str(value).lower().replace(",", "").replace(" ", "").strip()

    if channels not in (2, 3):
        raise ValueError(f"Expected 2 (r,z) or 3 (g,r,z) channels, got {channels}.")
    requested = canonical(bands)
    if requested not in {"auto", "rz", "grz"}:
        raise ValueError("bands must be 'auto', 'rz', or 'grz'.")
    declared = canonical(header_bands) if header_bands is not None else None
    if declared is not None and declared not in {"rz", "grz"}:
        raise ValueError(f"Unsupported FITS BANDS={header_bands!r}; use ordered r,z or g,r,z.")
    if requested != "auto" and declared is not None and requested != declared:
        raise ValueError(f"bands={requested!r} conflicts with FITS BANDS={declared!r}.")
    result = declared if requested == "auto" else requested
    result = result or ("rz" if channels == 2 else "grz")
    if len(result) != channels:
        raise ValueError(f"bands={result!r} requires {len(result)} channels, got {channels}.")
    return result


def prepare_input(image_path, *, bands="auto", input_pixel_scale=0.262):
    """Production fast resampling, then one central 128x128 crop.

    Input is background-subtracted linear flux, channels [r,z] or [g,r,z].
    The area-average convention and zero-point treatment match production.
    This resampling itself is not the learned super-resolution operation.
    """
    image_path = Path(image_path)
    with fits.open(image_path, memmap=False) as hdul:
        raw = np.array(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    if raw.ndim != 3:
        raise ValueError(f"Expected (2,H,W) rz or (3,H,W) grz, got {raw.shape}.")
    bands = resolve_bands(raw.shape[0], bands, header.get("BANDS"))
    if not np.isfinite(raw).all():
        raise ValueError("Input contains NaN/Inf; supply a fully covered cutout.")
    if not np.isfinite(input_pixel_scale) or input_pixel_scale <= 0:
        raise ValueError("input_pixel_scale must be a positive arcsec/pixel value.")

    native_wcs = WCS(header, naxis=2).celestial
    has_wcs = native_wcs.has_celestial
    if has_wcs:
        scales = proj_plane_pixel_scales(native_wcs) * 3600
        if native_wcs.has_distortion:
            raise ValueError("Distorted WCS is unsupported by this fast resampling demo.")
        if not np.allclose(scales, input_pixel_scale, rtol=1e-3, atol=1e-6):
            raise ValueError(f"WCS pixel scales {scales} disagree with input_pixel_scale.")

    def resample(array):
        return _reproj_fast(array, False, 30, 22.5, False,
                            float(input_pixel_scale), 0.1)

    full = resample(raw)
    h, w = full.shape[-2:]
    if min(h, w) < 128:
        raise ValueError("Input does not cover the required 12.8 arcsec field.")
    center_xy = ((w - 1) / 2, (h - 1) / 2)
    geometry = Cutout2D(full[0], center_xy, (128, 128), mode="strict")
    sy, sx = geometry.slices_original
    cropped = np.ascontiguousarray(full[:, sy, sx], dtype=np.float32)

    out_header = fits.Header()
    if has_wcs:
        scaled_wcs = native_wcs.deepcopy()
        ratio = 0.1 / input_pixel_scale
        native_center = (np.array(raw.shape[-2:][::-1]) - 1) / 2
        new_center = (np.array([w, h]) - 1) / 2
        scaled_wcs.wcs.crpix = (
            (native_wcs.wcs.crpix - 1 - native_center) / ratio + new_center + 1
        )
        if scaled_wcs.wcs.has_cd():
            scaled_wcs.wcs.cd *= ratio
        else:
            scaled_wcs.wcs.cdelt *= ratio
        scaled_wcs.wcs.crpix -= [sx.start, sy.start]
        out_header = scaled_wcs.to_header()
    for key in ("MANGAID", "RA", "DEC"):
        if key in header:
            out_header[key] = header[key]
    out_header["PIXSCALE"] = (0.1, "arcsec/pixel; sampling only")
    out_header["BAND"] = "VIS"
    out_header["SYNTHET"] = True
    out_header["STITCH"] = False
    out_header["INBANDS"] = (bands, "Input bands in channel order")
    info = {
        "input_file": image_path.name,
        "input_sha256": sha256(image_path),
        "input_bands": bands,
        "input_shape": list(raw.shape), "resampled_shape": list(full.shape),
        "crop_yx": [sy.start, sx.start], "output_shape": [128, 128],
        "input_pixel_scale_arcsec": float(input_pixel_scale),
        "output_pixel_scale_arcsec": 0.1, "field_arcsec": 12.8,
        "wcs_available": has_wcs,
        "preprocessing": "production fast area-average; center crop; apply_zp_conv=False",
    }
    return cropped, out_header, info


def generate(model, desi, *, bands="auto", seed=1024, num_samples=3, nfe=10,
             split_ratio=0.7, verbose=True):
    """One spatial crop, K stochastic samples averaged in physical space."""
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")
    if not isinstance(nfe, int) or not 1 <= nfe < model.interval:
        raise ValueError(f"nfe must be an integer in [1,{model.interval - 1}].")
    if not 0 <= split_ratio <= 1:
        raise ValueError("split_ratio must be in [0,1].")
    if desi.ndim != 3 or desi.shape[1:] != (128, 128) or not np.isfinite(desi).all():
        raise ValueError("Prepared input must be finite (2,128,128) rz or (3,128,128) grz.")
    bands = resolve_bands(desi.shape[0], bands)
    if (model.x1_mode != "rz" or model.heteroscedastic or model.hetero_cond_channel
            or not isinstance(model.pixel_transform, BandwiseArcsinhTransform)):
        raise ValueError("This demo requires the established rz checkpoint and bandwise transform.")
    device = model.device
    tensor = torch.as_tensor(np.ascontiguousarray(desi), dtype=torch.float32, device=device).unsqueeze(0)
    if bands == "rz":
        # The saved per-band transform has grz slots. Its channels transform
        # independently; the unused g slot cannot affect r or z. Do not shift
        # r/z into the wrong normalization slots. This slot is never displayed.
        tensor = torch.cat([torch.zeros_like(tensor[:, :1]), tensor], dim=1)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        desi_t = model.pixel_transform.normalize(
            model.pixel_transform.forward(tensor, source="desi"), source="desi"
        )
        reference = torch.zeros((1, 1, 128, 128), device=device)
        stack = model.sample_forked(
            model.build_x1(reference, desi_t), model.build_cond(desi_t),
            num_samples=num_samples, nfe=nfe,
            split_ratio=split_ratio, verbose=verbose,
        )
        flat = stack.reshape(num_samples, 1, 128, 128).float()
        physical = model.pixel_transform.inverse(
            model.pixel_transform.denormalize(flat, source="euclid"), source="euclid"
        )
        prediction = physical.mean(dim=0)[0].cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if prediction.shape != (128, 128) or not np.isfinite(prediction).all():
        raise RuntimeError("Model output has invalid shape or nonfinite pixels.")
    info = {
        "device": str(device), "dtype": "float32", "attention": "torch softmax SDPA",
        "seed": int(seed), "num_samples": num_samples, "nfe": nfe,
        "split_ratio": split_ratio, "ema": True, "stitch": False,
        "input_bands": bands,
        "cpu_threads": torch.get_num_threads(), "inference_seconds": elapsed,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "weights": json.loads((ROOT / "weights/provenance.json").read_text()),
    }
    return prediction, info


def save_result(path, prediction, header, metadata):
    """Only the generated image goes into FITS; provenance is a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Result already exists: {path}; choose a new output directory.")
    header = header.copy()
    for key, value in {
        "SEED": metadata["seed"], "NSAMPLE": metadata["num_samples"],
        "NFE": metadata["nfe"], "SPLIT": metadata["split_ratio"],
        "DEVICE": metadata["device"], "EMA": True,
        "CKPTSHA": metadata["weights"]["source_checkpoint_sha256"],
    }.items():
        header[key] = value
    header.add_history("Single central crop; no stitching. EMA I2SB prediction.")
    header.add_history("Inverse model pixel transform, then mean in physical space.")
    fits.PrimaryHDU(np.asarray(prediction, dtype=np.float32), header).writeto(
        path, checksum=True, overwrite=False
    )
    with fits.open(path, checksum=True) as hdul:
        assert len(hdul) == 1 and hdul[0].data.shape == (128, 128)
        assert hdul[0].verify_checksum() == hdul[0].verify_datasum() == 1
    metadata = dict(metadata, output_sha256=sha256(path))
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
