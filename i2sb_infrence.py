"""
I2SB DESI -> Euclid inference script (async FITS writing + optimised throughput).

Key speedups over a naive implementation:
  1. bf16 autocast around model.sample_forked():
     UNet forward runs in bf16 (full A100 tensor-core utilisation);
     small diffusion ops (q_sample / p_posterior) stay in fp32 automatically.
     Model weights remain fp32 throughout.

  2. torch.set_float32_matmul_precision('high'):
     TF32 acceleration for fp32 matmuls, ~10% free speedup.

  3. One-shot EMA swap:
     model_ema.copy_to(model) before the loop; entire run uses EMA weights,
     eliminating per-batch ema_scope overhead.

  4. torch.compile(model.model, mode='default'):
     UNet has fixed shapes (batch B for shared prefix, num_samples*B for forked
     suffix), so the compile cache is friendly. First run compiles for 2-5 min;
     worthwhile at large scale.

  5. sample_forked(num_samples=3, split_ratio=0.7):
     3 sampling trajectories share the first 70% of steps, then diverge.
     A single call yields (num_samples, B, 1, H, W) without a Python for-loop.
     Saves ~47% UNet forward calls (NFE=50, K=35).

  6. Channels-last memory format:
     conv2d with bf16 + channels_last maximises A100 utilisation (~10-20% extra).

CLI options:
    --bf16            : default True; disable with --bf16 false
    --compile         : default True; disable with --compile false to skip JIT
    --split_ratio     : shared-prefix fraction, default 0.7
                        0.0 = fully independent baseline (for std calibration)
                        0.7 = recommended production value
    --num_samples     : repeated samples per input, default 3
    --save_workers    : FITS writer worker processes
    --max_inflight    : max in-flight (pending) write tasks (back-pressure)
"""

import argparse
import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from threading import Semaphore, Lock

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

from sgm.util import instantiate_from_config
from sgm.data.dataset import *

from fits_writer import save_fits_worker

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")


# ============================================================
# CLI arguments
# ============================================================

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--config", type=str,
        default="configs/inference/astroIR_SBn.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default=None,
        help="Path to the checkpoint file",
    )
    parser.add_argument("--seed", type=int, default=1024, help="Seed for seed_everything")
    parser.add_argument(
        "--output_dir", type=str,
        default=None,
        help="Directory to save the generated FITS files",
    )

    parser.add_argument("--save_workers", type=int, default=32,
                        help="Number of FITS writer worker processes (CPU + astropy)")
    parser.add_argument("--max_inflight", type=int, default=128,
                        help="Max in-flight (pending) write tasks")

    parser.add_argument("--num_samples", type=int, default=3,
                        help="Repeated samples per input; output is mean (Primary HDU) + std (HDU)")
    parser.add_argument("--split_ratio", type=float, default=0.7,
                        help="Shared-prefix fraction in [0,1]; 0=fully independent baseline, 0.7=recommended")
    parser.add_argument("--bf16", type=str2bool, default=True,
                        help="bf16 autocast (strongly recommended on A100)")
    parser.add_argument("--compile", type=str2bool, default=True,
                        help="torch.compile(model.model) for UNet acceleration")
    parser.add_argument("--channels_last", type=str2bool, default=True,
                        help="channels_last memory format for conv2d")
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="CUDA device, e.g. cuda:0 / cuda:1")

    parser.add_argument("--catalog", type=str, default=None,
                        help="BGS catalog FITS path (e.g. BGS.fits); if set, only matching files are processed")
    parser.add_argument("--shape_r_max", type=float, default=None,
                        help="Filter condition: SHAPE_R < shape_r_max (arcsec); None = no filter")

    parser.add_argument("--footprint", type=str, default="",
                        help="HEALPix footprint mask FITS path; non-zero pixels = valid sky; empty = skip")
    return parser


# ============================================================
# Catalog filtering: read BGS.fits -> build allowed filename set
# ============================================================

def build_catalog_whitelist(catalog_path: str, shape_r_max=None) -> set:
    """
    Filter targets from a BGS catalog and return the set of allowed filenames.

    Filename format: '{TARGET_RA}_{TARGET_DEC}.fits' where ra/dec are Python
    float reprs, exactly matching the on-disk filenames.

    Add extra filter conditions by extending the `mask` variable below, e.g.:
        mask &= (np.array(tbl['Z']) > 0.05) & (np.array(tbl['Z']) < 0.3)
        mask &= np.array(tbl['ZCAT_PRIMARY']) == True
    """
    try:
        from astropy.table import Table
    except ImportError:
        raise ImportError("astropy is required: pip install astropy")

    print(f"[catalog] Reading {catalog_path} ...")
    tbl = Table.read(catalog_path)
    print(f"[catalog] Total rows: {len(tbl)}")

    mask = np.ones(len(tbl), dtype=bool)

    if shape_r_max is not None:
        mask &= np.array(tbl['SHAPE_R']) < shape_r_max
        print(f"[catalog] SHAPE_R < {shape_r_max}: {mask.sum()} rows remaining")

    filtered = tbl[mask]
    print(f"[catalog] After filtering: {len(filtered)} targets")

    whitelist = set()
    for row in filtered:
        ra = float(row['TARGET_RA'])
        dec = float(row['TARGET_DEC'])
        whitelist.add(f"{ra}_{dec}.fits")

    print(f"[catalog] Whitelist size: {len(whitelist)}")
    return whitelist


def filter_by_footprint(file_list: list, footprint_path: str) -> set:
    """
    Filter a file list using a HEALPix footprint mask; keep only files whose
    coordinates fall inside the valid sky area.

    Filename format: '{ra}_{dec}.fits' (equatorial coordinates, degrees).
    Footprint mask: non-zero pixels = valid coverage (NESTED ordering, equatorial).
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError("healpy is required: pip install healpy")

    print(f"[footprint] Reading {footprint_path} ...")
    mask_map = hp.read_map(footprint_path, nest=None, verbose=False)
    nside = hp.get_nside(mask_map)
    print(f"[footprint] NSIDE={nside}, valid pixels={(mask_map > 0).sum()}")

    ras, decs, fnames = [], [], []
    for fname in file_list:
        stem = fname.replace(".fits", "")
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            ras.append(float(parts[0]))
            decs.append(float(parts[1]))
            fnames.append(fname)
        except ValueError:
            continue

    if not fnames:
        print("[footprint] Could not parse coordinates from filenames; skipping footprint filter")
        return set(file_list)

    ras = np.array(ras)
    decs = np.array(decs)

    # (ra, dec) in degrees -> HEALPix (theta, phi) in radians
    theta = np.radians(90.0 - decs)
    phi = np.radians(ras)

    pix_idx = hp.ang2pix(nside, theta, phi, nest=False)
    in_footprint = mask_map[pix_idx] > 0

    whitelist = set(np.array(fnames)[in_footprint])
    print(f"[footprint] File filter: {len(file_list)} -> {len(whitelist)} (inside footprint)")
    return whitelist


def apply_whitelist_to_dataset(dataset, whitelist: set):
    """
    In-place filter of dataset.file_list to only keep files in the whitelist.
    Compatible with both matched-mode (str) and multi-folder mode ((idx, str) tuple).
    """
    original = len(dataset.file_list)

    if dataset.file_list and isinstance(dataset.file_list[0], tuple):
        dataset.file_list = [item for item in dataset.file_list if item[1] in whitelist]
    else:
        dataset.file_list = [f for f in dataset.file_list if f in whitelist]

    dataset.sampled_indices = list(range(len(dataset.file_list)))
    print(f"[catalog] Dataset filtered: {original} -> {len(dataset.file_list)} samples")


# ============================================================
# Main inference loop
# ============================================================

def generate_images(model, dataloader, device, output_dir, executor, max_inflight,
                    num_samples=1, split_ratio=0.7, autocast_dtype=None,
                    channels_last=False):
    """
    Run inference over the dataloader and write FITS outputs asynchronously.

    Speedups:
      - num_samples parallelised inside sample_forked (no Python for-loop)
      - bf16 autocast wraps sample_forked
      - No ema_scope (weights already swapped before loop)
      - Whole-batch D2H transfer avoids per-sample synchronisation
      - FITS writing is async; GPU proceeds to next batch immediately
    """
    model.eval()

    inflight_sem = Semaphore(max_inflight)
    stats_lock = Lock()
    stats = {"submitted": 0, "done": 0, "failed": 0}

    def _on_done(future):
        inflight_sem.release()
        with stats_lock:
            try:
                future.result()
                stats["done"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(f"[write failed] {e}")

    from contextlib import nullcontext
    if autocast_dtype is not None:
        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
    else:
        autocast_ctx = nullcontext

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")
        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            try:
                euclid, desi, euclid_err, pixel_mask = model.get_input(batch)

                desi_raw_t = batch.get(model.input_key_desi)
                if desi_raw_t is None:
                    desi_raw_t = batch.get("images")
                desi_err_raw_t = batch.get(model.input_key_desi_error)
                euclid_raw_t = batch.get(model.input_key_euclid)
                euclid_err_raw_t = batch.get(model.input_key_euclid_error)

                if euclid is not None:
                    x0_ref = euclid
                else:
                    x0_ref = torch.zeros(
                        desi.shape[0], 1, desi.shape[2], desi.shape[3], device=device,
                    )

                x1 = model.build_x1(x0_ref, desi)
                cond = model.build_cond(desi)

                if channels_last:
                    x1 = x1.contiguous(memory_format=torch.channels_last)
                    cond = cond.contiguous(memory_format=torch.channels_last)

                with autocast_ctx():
                    accum_stack = model.sample_forked(
                        x1=x1,
                        cond=cond,
                        num_samples=num_samples,
                        split_ratio=split_ratio,
                        verbose=False,
                    )  # (N, B, 1, H, W) in transform domain

                N, B, C, H, W = accum_stack.shape
                flat = accum_stack.reshape(N * B, C, H, W)
                flat_fp32 = flat.float()
                flat_phys = model.pixel_transform.inverse(
                    model.pixel_transform.denormalize(flat_fp32, source="euclid"),
                    source="euclid",
                )
                accum_phys = flat_phys.reshape(N, B, C, H, W)

                if num_samples == 1:
                    generated_np = accum_phys[0].cpu().numpy()
                    generated_std_np = None
                else:
                    generated_mean = accum_phys.mean(dim=0)
                    generated_std = accum_phys.std(dim=0, unbiased=False)
                    generated_np = generated_mean.cpu().numpy()
                    generated_std_np = generated_std.cpu().numpy()

                desi_np = desi_raw_t.to(torch.float32).cpu().numpy() if desi_raw_t is not None else None
                desi_err_np = desi_err_raw_t.to(torch.float32).cpu().numpy() if desi_err_raw_t is not None else None
                euclid_np = euclid_raw_t.to(torch.float32).cpu().numpy() if euclid_raw_t is not None else None
                euclid_err_np = euclid_err_raw_t.to(torch.float32).cpu().numpy() if euclid_err_raw_t is not None else None

                filenames = batch["filename"]

                Bn = generated_np.shape[0]
                for i in range(Bn):
                    fname = filenames[i]
                    out_path = os.path.join(output_dir, fname)

                    if os.path.exists(out_path):
                        continue

                    inflight_sem.acquire()

                    future = executor.submit(
                        save_fits_worker,
                        out_path,
                        generated_np[i],
                        desi_np[i] if desi_np is not None else None,
                        desi_err_np[i] if desi_err_np is not None else None,
                        euclid_np[i] if euclid_np is not None else None,
                        euclid_err_np[i] if euclid_err_np is not None else None,
                        generated_std_np[i] if generated_std_np is not None else None,
                    )
                    future.add_done_callback(_on_done)

                    with stats_lock:
                        stats["submitted"] += 1

                with stats_lock:
                    pbar.set_postfix(
                        submitted=stats["submitted"],
                        done=stats["done"],
                        failed=stats["failed"],
                        refresh=False,
                    )

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"[inference done] submitted={stats['submitted']}, done={stats['done']}, failed={stats['failed']}")
    print("Waiting for remaining FITS writes to complete...")


# ============================================================
# One-shot EMA weight swap (call before inference loop)
# ============================================================

def swap_in_ema_weights(model):
    """Copy EMA weights into model once before inference; avoids per-batch ema_scope overhead."""
    if not getattr(model, "use_ema", False):
        print("[EMA] Model has no EMA; skipping swap")
        return False
    if not hasattr(model, "model_ema"):
        print("[EMA] model_ema attribute not found; skipping")
        return False
    model.model_ema.copy_to(model.model)
    print("[EMA] EMA weights swapped into model; inference will use EMA throughout")
    return True


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.output_dir is None:
        raise ValueError("--output_dir is required")
    if opt.ckpt_path is None:
        raise ValueError("--ckpt_path is required")

    config = OmegaConf.load(opt.config)
    model_config = config.model

    os.makedirs(opt.output_dir, exist_ok=True)
    print(f"Output directory: {opt.output_dir}")

    print(f"Starting {opt.save_workers} FITS writer worker processes...")
    executor = ProcessPoolExecutor(max_workers=opt.save_workers)

    try:
        device = torch.device(opt.device_id if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        seed_everything(opt.seed, workers=True)

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("[SPEEDUP] cudnn.benchmark=True, matmul_precision='high'")

        autocast_dtype = torch.bfloat16 if opt.bf16 else None
        if opt.bf16:
            print("[SPEEDUP] bf16 autocast enabled")
        else:
            print("[WARNING] bf16 disabled; inference will be slower (not recommended on A100)")

        print("Loading MultiModalSBDiffusion model...")
        model = instantiate_from_config(model_config)
        model.to(device)
        model.eval()

        print(f"Loading checkpoint: {opt.ckpt_path}")
        checkpoint = torch.load(opt.ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        print("Model loaded")

        swap_in_ema_weights(model)

        if opt.channels_last:
            model.model = model.model.to(memory_format=torch.channels_last)
            print("[SPEEDUP] UNet converted to channels_last memory_format")

        if opt.compile:
            _cache_dir = os.path.join(opt.output_dir, ".torch_compile_cache")
            os.makedirs(_cache_dir, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir
            print(f"[SPEEDUP] torch.compile cache dir: {_cache_dir}")

            print("[SPEEDUP] Enabling torch.compile(model.model, mode='default')")
            try:
                model.model = torch.compile(model.model, mode="default",
                                            fullgraph=False, dynamic=False)
            except Exception as e:
                print(f"[WARNING] torch.compile failed; falling back to eager: {e}")

            try:
                _unet = model.model
                _orig = getattr(_unet, "_orig_mod", _unet)
                _in_ch = getattr(_orig, "in_channels", 2)
                _img_sz = 128
                _bs = 1
                _bs_fork = opt.num_samples

                print(f"[SPEEDUP] warmup compile: in_ch={_in_ch}, img={_img_sz}, "
                      f"batch={_bs}/{_bs_fork} (first run compiles 5-15 min)...")

                with torch.no_grad():
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.bfloat16 if opt.bf16 else torch.float32,
                                        enabled=opt.bf16):
                        _x1 = torch.zeros(_bs, _in_ch, _img_sz, _img_sz, device=device)
                        _t1 = torch.zeros(_bs, device=device, dtype=torch.long)
                        if opt.channels_last:
                            _x1 = _x1.to(memory_format=torch.channels_last)
                        _unet(_x1, _t1)

                        if _bs_fork > 1:
                            _x2 = torch.zeros(_bs * _bs_fork, _in_ch, _img_sz, _img_sz, device=device)
                            _t2 = torch.zeros(_bs * _bs_fork, device=device, dtype=torch.long)
                            if opt.channels_last:
                                _x2 = _x2.to(memory_format=torch.channels_last)
                            _unet(_x2, _t2)

                print("[SPEEDUP] warmup complete; compile cache ready")
            except Exception as e:
                print(f"[WARNING] warmup failed (inference unaffected): {e}")

        if "data" not in config:
            raise ValueError("No data config found in configuration file")

        print("Loading data...")
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup("test")

        if opt.catalog:
            whitelist = build_catalog_whitelist(
                catalog_path=opt.catalog,
                shape_r_max=opt.shape_r_max,
            )
            apply_whitelist_to_dataset(data.test_dataset, whitelist)

        if opt.footprint:
            raw_file_list = data.test_dataset.file_list
            if raw_file_list and isinstance(raw_file_list[0], tuple):
                names = [item[1] for item in raw_file_list]
            else:
                names = list(raw_file_list)
            fp_whitelist = filter_by_footprint(names, opt.footprint)
            apply_whitelist_to_dataset(data.test_dataset, fp_whitelist)

        dataloader = data.test_dataloader()
        print(f"Data loaded; batch size: {dataloader.batch_size}")
        print(f"[SPEEDUP] num_samples={opt.num_samples}, split_ratio={opt.split_ratio}")
        if opt.num_samples > 1 and opt.split_ratio > 0:
            saved = 1.0 - (opt.split_ratio + (1 - opt.split_ratio) * opt.num_samples) / opt.num_samples
            print(f"         Shared prefix saves ~{saved * 100:.1f}% UNet forwards vs. {opt.num_samples} independent trajectories")

        generate_images(
            model, dataloader, device, opt.output_dir,
            executor=executor, max_inflight=opt.max_inflight,
            num_samples=opt.num_samples,
            split_ratio=opt.split_ratio,
            autocast_dtype=autocast_dtype,
            channels_last=opt.channels_last,
        )

    finally:
        print("Shutting down writer pool and waiting for all writes to complete...")
        executor.shutdown(wait=True)
        print("Done!")
