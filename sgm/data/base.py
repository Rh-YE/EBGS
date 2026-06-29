import numpy as np
import os
import json
from typing import Optional
from astropy.io import fits
from sgm.util import get_obj_from_str
from .utils import (
    ensure_native_byteorder,
    normalize_image,
    center_crop,
    standardize_image,
    del_target_channel,
    get_crop_position,
    rotate_image_with_wcs,
    invvar_to_sigma,
    rms_to_sigma,
    reproj
)


class BaseDataset:
    """Base dataset class encapsulating core image processing logic."""

    def __init__(
        self,
        crop=False,
        normalize=False,
        standardize=False,
        mean=None,
        std=None,
        target_h=256,
        target_w=256,
        random_crop=False,
        rotation_angle=0,
        random_rotation=False,
        pixel_scale=0.262,
        error_type='rms',
        custom_fits_reader=None,
        del_channel=None,
        reproj_wcs=False,
        reproj_mode='exact',
        apply_zp_conv=False,
        from_zp=30,
        to_zp=22.5,
        psf_folder='PSF',
        # --- Pixel validity masks ---
        # Each entry format: [lo, hi], None means no bound.
        # e.g. [0, null] keeps only pixels >= 0.
        # desi_img_valid_range  : list[list[lo|None, hi|None]], length = desi_bands
        # desi_err_valid_range  : list[list[lo|None, hi|None]], length = desi_bands
        # euclid_img_valid_range: list[list[lo|None, hi|None]], length = euclid_bands
        # euclid_err_valid_range: list[list[lo|None, hi|None]], length = euclid_bands
        desi_img_valid_range=None,
        desi_err_valid_range=None,
        euclid_img_valid_range=None,
        euclid_err_valid_range=None,
    ):
        """
        Initialize base class.

        Mask logic
        ----------
        Per-channel valid range [lo, hi] (None = no bound) for image and error maps
        of each source (desi/euclid). Per-channel conditions are AND-ed together;
        then euclid and desi sides are AND-ed to produce the final pixel_mask (1,H,W) bool.
        True = pixel participates in loss; False = excluded.

        Args:
            crop: whether to crop
            normalize: whether to normalize
            standardize: whether to standardize
            mean: standardization mean
            std: standardization std
            target_h: target height
            target_w: target width
            random_crop: whether to use random crop
            rotation_angle: fixed rotation angle in degrees (arbitrary angle)
            random_rotation: whether to use random rotation (uniform 0-360 when True)
            pixel_scale: pixel scale in arcsec/pixel (for WCS rotation)
            error_type: error map type ('rms' or 'invvar'), default 'rms'
            custom_fits_reader: custom FITS reader callable
            del_channel: list of channels to delete
            reproj_wcs: whether to apply WCS reprojection
            apply_zp_conv: whether to apply zero-point conversion
            from_zp: source zero-point
            to_zp: target zero-point
            desi_img_valid_range:   per-channel valid range list for DESI image
            desi_err_valid_range:   per-channel valid range list for DESI error map
            euclid_img_valid_range: per-channel valid range list for Euclid image
            euclid_err_valid_range: per-channel valid range list for Euclid error map
        """
        self.crop = crop
        self.normalize = normalize
        self.standardize = standardize
        self.mean = np.array(mean) if mean is not None else None
        self.std = np.array(std) if std is not None else None
        self.target_h = target_h
        self.target_w = target_w
        self.target_size = (target_h, target_w)  # fix: should be a tuple, not a single int
        self.random_crop = random_crop
        self.rotation_angle = rotation_angle
        self.random_rotation = random_rotation
        self.pixel_scale = pixel_scale
        self.error_type = error_type
        self.custom_fits_reader = custom_fits_reader
        self.del_channel = del_channel
        self.reproj_wcs = reproj_wcs
        self.reproj_mode = reproj_mode
        self.apply_zp_conv = apply_zp_conv
        self.from_zp = from_zp
        self.to_zp = to_zp
        self.psf_folder = psf_folder

        # --- valid range configs (list of [lo|None, hi|None] per channel) ---
        self.desi_img_valid_range   = desi_img_valid_range
        self.desi_err_valid_range   = desi_err_valid_range
        self.euclid_img_valid_range = euclid_img_valid_range
        self.euclid_err_valid_range = euclid_err_valid_range

        self._crop_positions = {}
        self._current_rotation_angle = None

    def _get_crop_position(self, image, key='default'):
        """Get crop position (supports independent positions for multiple data sources)."""
        if key not in self._crop_positions:
            self._crop_positions[key] = None

        pos = get_crop_position(
            image,
            self.target_h,
            self.target_w,
            self.random_crop,
            self._crop_positions[key]
        )
        self._crop_positions[key] = pos
        return pos

    def reset_crop_positions(self):
        """Reset all crop positions."""
        self._crop_positions.clear()

    def _get_rotation_angle(self):
        """Get rotation angle (supports random rotation)."""
        if not self.random_rotation:
            return self.rotation_angle

        if self._current_rotation_angle is not None:
            return self._current_rotation_angle

        # Random rotation: sample uniformly in [0, 360)
        angle = np.random.uniform(0, 360)
        self._current_rotation_angle = angle
        return angle

    def reset_rotation_angle(self):
        """Reset rotation angle (called per sample)."""
        self._current_rotation_angle = None

    def process_image(self, image, data_key='default', is_error_map=False):
        """
        Image processing pipeline: rotate -> crop -> normalize -> standardize.

        Args:
            image: input image
            data_key: data source identifier (for multi-source scenarios)
            is_error_map: if True, skip normalization and standardization
        """
        image = ensure_native_byteorder(image)

        if image.ndim == 2:
            image = image.reshape(1, image.shape[0], image.shape[1])

        # 1. Rotation (on raw data)
        angle = self._get_rotation_angle()
        if angle != 0:
            image = rotate_image_with_wcs(image, angle, self.pixel_scale)

        # 2. Crop
        if self.crop:
            pos = self._get_crop_position(image, data_key)
            image = center_crop(image, pos, self.target_size)

        # 3. Normalize (images only, not error maps)
        if not is_error_map and self.normalize:
            image = normalize_image(image)

        # 4. Standardize (images only, not error maps)
        if not is_error_map and self.standardize and self.mean is not None and self.std is not None:
            image = standardize_image(image, self.mean, self.std)

        return image.astype(np.float32)

    def read_fits(self, file_path, is_invvar=False):
        """
        Read a FITS file.

        Args:
            file_path: path to the FITS file
            is_invvar: whether the data is inverse variance
        """
        if self.custom_fits_reader is not None:
            data = self.custom_fits_reader(file_path)
            data = ensure_native_byteorder(data)

            if self.reproj_wcs:
                header = fits.getheader(file_path)
                data = reproj(
                    data, header,
                    apply_zp_conv=self.apply_zp_conv,
                    from_zp=self.from_zp,
                    to_zp=self.to_zp,
                    is_invvar=is_invvar,
                    mode=self.reproj_mode,
                )

            if self.del_channel is not None:
                data = del_target_channel(data, self.del_channel)

            return data

        with fits.open(file_path, memmap=False) as hdul:
            data = hdul[0].data.copy()

        if self.reproj_wcs:
            header = fits.getheader(file_path)
            data = reproj(
                data, header,
                apply_zp_conv=self.apply_zp_conv,
                from_zp=self.from_zp,
                to_zp=self.to_zp,
                is_invvar=is_invvar,
                mode=self.reproj_mode,
            )

        if self.del_channel is not None:
            data = del_target_channel(data, self.del_channel)

        return data

    def read_error_map(self, error_path, data_key='default'):
        """
        Read an error map and convert to sigma (with the same augmentation as the image).
        Sigma is clipped to [1e-10, 1]; out-of-range pixels are excluded from loss.

        Args:
            error_path: path to the error map file
            data_key: data source identifier

        Returns:
            Error map data (sigma format, clipped to [1e-10, 1])
        """
        if not os.path.exists(error_path):
            return None

        data = self.read_fits(error_path, is_invvar=(self.error_type == 'invvar'))
        data = ensure_native_byteorder(data)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Convert to sigma (before augmentation)
        if self.error_type == 'invvar':
            # invvar -> sigma, clipped to [1e-10, 1]
            data = invvar_to_sigma(data, clip_min=1e-10, clip_max=1.0)
        else:
            # rms -> sigma (clip only), clipped to [1e-10, 1]
            data = rms_to_sigma(data, clip_min=1e-10, clip_max=1.0)

        # Apply the same augmentation as the image (rotate + crop), no normalize/standardize
        data = self.process_image(data, data_key, is_error_map=True)
        return data

    def _apply_range_mask(
        self,
        data: np.ndarray,          # (C, H, W)
        valid_ranges,              # list of [lo|None, hi|None], length C; or None
    ) -> np.ndarray:               # (1, H, W) bool, True = valid
        """
        Given a (C,H,W) array and per-channel range list, return the logical AND of
        per-channel validity masks.

        valid_ranges examples:
            [[0, null], [0, null], [0, null]]   -- three channels all require >= 0
            [[null, 1.0], [null, 1.0]]           -- two channels all require <= 1.0
            [[0, null], [null, null]]            -- only first channel has lower bound

        Returns all-True mask if valid_ranges is None or empty.
        """
        H, W = data.shape[1], data.shape[2]
        mask = np.ones((H, W), dtype=bool)

        if not valid_ranges:
            return mask[np.newaxis]   # (1, H, W)

        for c, rng in enumerate(valid_ranges):
            if c >= data.shape[0]:
                break
            if rng is None:
                continue
            lo, hi = rng[0], rng[1]
            ch = data[c]
            if lo is not None:
                mask &= (ch >= lo)
            if hi is not None:
                mask &= (ch <= hi)

        return mask[np.newaxis].astype(bool)   # (1, H, W)

    def build_pixel_mask(
        self,
        desi_img:    np.ndarray,                    # (C_d, H, W) processed DESI image
        euclid_img:  np.ndarray,                    # (C_e, H, W) processed Euclid image
        desi_err:    Optional[np.ndarray] = None,   # (C_d, H, W) or None
        euclid_err:  Optional[np.ndarray] = None,   # (C_e, H, W) or None
    ) -> np.ndarray:
        """
        Merge four-way range masks and return the final pixel_mask (1, H, W) bool.
        True = pixel is valid across all four sources and participates in loss.

        If any range is not set (None), that source defaults to all-valid.
        """
        H, W = desi_img.shape[1], desi_img.shape[2]
        mask = np.ones((1, H, W), dtype=bool)

        mask &= self._apply_range_mask(desi_img,   self.desi_img_valid_range)
        mask &= self._apply_range_mask(euclid_img, self.euclid_img_valid_range)

        if desi_err is not None:
            mask &= self._apply_range_mask(desi_err, self.desi_err_valid_range)
        if euclid_err is not None:
            mask &= self._apply_range_mask(euclid_err, self.euclid_err_valid_range)

        return mask   # (1, H, W) bool

    def read_psf(self, psf_path: str) -> np.ndarray:
        """
        Read a PSF FITS file and return (C, H, W) float32.
        File is already normalized; single HDU.
        DESI PSF: (3, H, W); Euclid PSF: (1, H, W).
        No rotation/crop/augmentation applied.
        """
        with fits.open(psf_path, memmap=False) as hdul:
            data = hdul[0].data.copy()
        data = ensure_native_byteorder(data)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        return data.astype(np.float32)

    def read_json(self, json_path):
        """Read a JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)
