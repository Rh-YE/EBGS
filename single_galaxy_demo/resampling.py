"""Unmodified functions extracted from sgm/data/utils.py for fast resampling."""
import numpy as np
from astropy.nddata import Cutout2D
_OVERLAP_CACHE = {}

def center_crop(image, position, target_size):
    """
    Crop using the floating-point position returned by get_crop_position.
    """
    # position is supplied in (y, x) order.
    if image.ndim == 2:
        # Convert the supplied coordinate order for Cutout2D.
        # Cutout2D(data, position, size) expects position in (x, y) order.
        # Reverse a (h_pos, w_pos) result from get_crop_position here.
        cutout_pos = (position[1], position[0]) 
        
        cutout = Cutout2D(image, cutout_pos, target_size, mode='partial', fill_value=0)
        return cutout.data
    
    # Process multiple channels.
    cutout_data = []
    cutout_pos = (position[1], position[0])
    for i in range(image.shape[0]):
        cutout = Cutout2D(image[i], cutout_pos, target_size, mode='partial', fill_value=0)
        cutout_data.append(cutout.data)
    return np.array(cutout_data)

def get_crop_position(image, target_h, target_w, random_crop=False, current_pos=None):
    """
    Get the crop position.

    Args:
        image: Input image.
        target_h: Target height.
        target_w: Target width.
        random_crop: Whether to draw a random crop from a Gaussian distribution.
        current_pos: Existing position, returned directly when provided.
    """
    if current_pos is not None:
        return current_pos
    
    # Use the geometric center as floating-point coordinates.
    img_h, img_w = image.shape[-2], image.shape[-1]
    center_h = (img_h - 1) / 2.0
    center_w = (img_w - 1) / 2.0
    
    if not random_crop:
        # Return the exact floating-point center consistently for sizes 128 and 91.
        return (center_h, center_w)
    
    # Random crop selection.
    min_h, max_h = target_h / 2.0, img_h - target_h / 2.0
    min_w, max_w = target_w / 2.0, img_w - target_w / 2.0
    
    if min_h >= max_h or min_w >= max_w:
        return (center_h, center_w)
    
    sigma_h, sigma_w = target_h * 3 / 8.0, target_w * 3 / 8.0
    h_offset = np.random.normal(0, sigma_h)
    w_offset = np.random.normal(0, sigma_w)
    
    h_pos = np.clip(center_h + h_offset, min_h, max_h)
    w_pos = np.clip(center_w + w_offset, min_w, max_w)
    
    return (h_pos, w_pos)


def convert_flux_zeropoint(flux_data, from_zp=30, to_zp=22.5):
    factor = 10 ** (0.4 * (to_zp - from_zp))
    return flux_data * factor

def convert_invvar_zeropoint(invvar_data, from_zp=30, to_zp=22.5):
    factor = 10 ** (0.4 * (to_zp - from_zp))
    return invvar_data / (factor ** 2)

def _overlap_matrix(n_in, n_out, scale):
    """
    Area-overlap weights along rows/columns, shape (n_out, n_in).
    Row normalization gives an area average (surface brightness), matching
    the default semantics of reproject_exact.

    Alignment: both grids share a physical center, consistent with the
    synthetic WCS CRPIX=(N+1)/2. A zero-based FITS pixel i is centered at i
    and covers [i-0.5, i+0.5].

    scale = to_pixel_scale / from_pixel_scale: the width of an output pixel
    in input-pixel units; less than 1 when oversampling.
    """
    center_in = (n_in - 1) / 2.0
    center_out = (n_out - 1) / 2.0
    j = np.arange(n_out)
    x_center = center_in + (j - center_out) * scale
    out_lo = (x_center - scale / 2.0)[:, None]
    out_hi = (x_center + scale / 2.0)[:, None]
    i = np.arange(n_in)[None, :]
    in_lo = i - 0.5
    in_hi = i + 0.5
    overlap = np.clip(np.minimum(out_hi, in_hi) - np.maximum(out_lo, in_lo), 0.0, None)
    rowsum = overlap.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return (overlap / rowsum).astype(np.float64)

def _reproj_fast(img_data, apply_zp_conv, from_zp, to_zp, is_invvar,
                 from_pixel_scale, to_pixel_scale):
    """
    Separable area-weighted resampling; see above. img_data has shape (C,H,W).

    Weights are constructed in f64 for clipping/normalization precision,
    then cached in f32. Two BLAS sgemm operations (np.matmul) resample the image:
        out = Wr @ x @ WcT    # (Ho,Hi)@(C,Hi,Wi)@(Wi,Wo) -> (C,Ho,Wo)
    The original implementation reports f32 matmul as about 6x faster than
    f64 einsum, with rounding differences of about 5e-8, much smaller than
    the approximately 6e-4 residual relative to reproject_exact.
    """
    c, h_in, w_in = img_data.shape
    sf = from_pixel_scale / to_pixel_scale
    h_out, w_out = int(h_in * sf), int(w_in * sf)
    scale = to_pixel_scale / from_pixel_scale

    key = (h_in, h_out, w_in, w_out, scale)
    wmats = _OVERLAP_CACHE.get(key)
    if wmats is None:
        # Construct in f64 and cache in f32; transpose column weights to (w_in, w_out) for right multiplication.
        Wr = _overlap_matrix(h_in, h_out, scale).astype(np.float32)      # (Ho, Hi)
        WcT = _overlap_matrix(w_in, w_out, scale).T.copy().astype(np.float32)  # (Wi, Wo)
        _OVERLAP_CACHE[key] = wmats = (Wr, WcT)
    Wr, WcT = wmats

    x = np.ascontiguousarray(img_data, dtype=np.float32)
    out = Wr @ x       # (Ho,Hi) @ (C,Hi,Wi) -> (C,Ho,Wi)  (batched sgemm)
    out = out @ WcT    # (C,Ho,Wi) @ (Wi,Wo) -> (C,Ho,Wo)

    if apply_zp_conv:
        out = (convert_invvar_zeropoint(out, from_zp, to_zp) if is_invvar
               else convert_flux_zeropoint(out, from_zp, to_zp))
    return out.astype(np.float32, copy=False)

