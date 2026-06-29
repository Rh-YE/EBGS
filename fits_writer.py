"""
Standalone FITS writing module for use with ProcessPoolExecutor worker processes.

Design:
  - Does not import torch; workers are lighter, start faster, and use less memory.
  - Maintains the same HDU structure and ordering as the main inference pipeline.
  - Receives ndarrays that have already been converted to CPU numpy in the main process.
"""

import os
import numpy as np
from astropy.io import fits


def save_fits_worker(
    path: str,
    prediction: np.ndarray,
    desi_img: np.ndarray,
    desi_error: np.ndarray = None,
    euclid_img: np.ndarray = None,
    euclid_error: np.ndarray = None,
    prediction_std: np.ndarray = None,
) -> str:
    """
    FITS write function executed in a worker process.

    HDU layout:
      HDU 0 (Primary) : prediction mean   (1, H, W)
      HDU 1 (IMAGE)   : DESI_IMG          (C, H, W)
      HDU 2 (IMAGE)   : DESI_ERROR        (C, H, W)  [optional]
      HDU 3 (IMAGE)   : EUCLID_IMG        (1, H, W)  [optional]
      HDU 4 (IMAGE)   : EUCLID_ERROR      (1, H, W)  [optional]
      HDU N (IMAGE)   : PREDICTION_STD    (1, H, W)  [optional, present when num_samples > 1]

    Returns the written file path.
    """
    try:
        if os.path.exists(path):
            return path

        primary_hdu = fits.PrimaryHDU(prediction.astype(np.float32, copy=False))
        primary_hdu.header["GENTYPE"] = ("SB_DESI_to_Euclid", "Generation type")

        desi_hdu = fits.ImageHDU(desi_img.astype(np.float32, copy=False), name="DESI_IMG")

        hdul = fits.HDUList([primary_hdu, desi_hdu])
        if desi_error is not None:
            hdul.append(fits.ImageHDU(desi_error.astype(np.float32, copy=False), name="DESI_ERROR"))
        if euclid_img is not None:
            hdul.append(fits.ImageHDU(euclid_img.astype(np.float32, copy=False), name="EUCLID_IMG"))
        if euclid_error is not None:
            hdul.append(fits.ImageHDU(euclid_error.astype(np.float32, copy=False), name="EUCLID_ERROR"))
        if prediction_std is not None:
            hdul.append(fits.ImageHDU(prediction_std.astype(np.float32, copy=False), name="PREDICTION_STD"))

        # Write to a temp file first, then atomically rename to avoid partial files on crash.
        tmp_path = path + ".tmp"
        hdul.writeto(tmp_path, overwrite=True)
        os.replace(tmp_path, path)

        return path

    except Exception as e:
        tmp_path = path + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise RuntimeError(f"Failed to save {path}: {e}") from e
