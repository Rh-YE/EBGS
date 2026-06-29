import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.io.fits import Header
from reproject import reproject_interp, reproject_exact


def ensure_native_byteorder(image: np.ndarray) -> np.ndarray:
    if image.dtype.byteorder not in ('=', '|'):
        image = image.byteswap().view(image.dtype.newbyteorder())
    return image

def normalize_image(image):
    max_v = np.max(image, axis=(1,2), keepdims=True)
    min_v = np.min(image, axis=(1,2), keepdims=True)
    return (image - min_v) / (max_v - min_v)

def standardize_image(image, mean, std):
    return (image - mean) / std

def del_target_channel(image, target_channel: list):
    """Delete specified channels."""
    all_channels = list(range(image.shape[0]))
    keep_channels = [i for i in all_channels if i not in target_channel]
    return image[keep_channels]


def rotate_image_with_wcs(image, angle, pixel_scale=1.0):
    """
    Rotate an image using WCS and reproject.

    Args:
        image: input image (C, H, W) or (H, W)
        angle: rotation angle in degrees (arbitrary)
        pixel_scale: pixel scale in arcsec/pixel

    Returns:
        Rotated image
    """
    if angle == 0:
        return image

    from reproject import reproject_interp

    is_2d = image.ndim == 2
    if is_2d:
        image = image[np.newaxis, ...]

    c, h, w = image.shape
    center_x = w / 2
    center_y = h / 2

    # Create input WCS (unrotated)
    input_header = Header()
    input_header['NAXIS'] = 2
    input_header['NAXIS1'] = w
    input_header['NAXIS2'] = h
    input_header['CRPIX1'] = center_x
    input_header['CRPIX2'] = center_y
    input_header['CRVAL1'] = 0.0
    input_header['CRVAL2'] = 0.0
    input_header['CTYPE1'] = 'RA---TAN'
    input_header['CTYPE2'] = 'DEC--TAN'
    cd_scale = pixel_scale / 3600.0
    input_header['CD1_1'] = -cd_scale
    input_header['CD1_2'] = 0.0
    input_header['CD2_1'] = 0.0
    input_header['CD2_2'] = cd_scale
    input_wcs = WCS(input_header)

    # Create output WCS (rotated)
    output_header = Header()
    output_header['NAXIS'] = 2
    output_header['NAXIS1'] = w
    output_header['NAXIS2'] = h
    output_header['CRPIX1'] = center_x
    output_header['CRPIX2'] = center_y
    output_header['CRVAL1'] = 0.0
    output_header['CRVAL2'] = 0.0
    output_header['CTYPE1'] = 'RA---TAN'
    output_header['CTYPE2'] = 'DEC--TAN'

    # Apply rotation matrix
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    output_header['CD1_1'] = -cd_scale * cos_a
    output_header['CD1_2'] = cd_scale * sin_a
    output_header['CD2_1'] = cd_scale * sin_a
    output_header['CD2_2'] = cd_scale * cos_a
    output_wcs = WCS(output_header)

    # Reproject (rotate) each channel
    rotated = []
    for i in range(c):
        array, _ = reproject_exact((image[i], input_wcs), output_wcs,
                                    shape_out=(h, w))
        # Replace NaN with 0
        array = np.nan_to_num(array, nan=0.0, copy=False)
        rotated.append(array)

    result = np.array(rotated)
    return result[0] if is_2d else result



def center_crop(image, position, target_size):
    """
    Crop function; position must be the float value returned by get_crop_position.
    """
    # position is in (y, x) format
    if image.ndim == 2:
        # Cutout2D(data, position, size): position defaults to (x, y)
        # If get_crop_position returns (h_pos, w_pos), reverse here
        cutout_pos = (position[1], position[0])

        cutout = Cutout2D(image, cutout_pos, target_size, mode='partial', fill_value=0)
        return cutout.data

    # Multi-channel processing
    cutout_data = []
    cutout_pos = (position[1], position[0])
    for i in range(image.shape[0]):
        cutout = Cutout2D(image[i], cutout_pos, target_size, mode='partial', fill_value=0)
        cutout_data.append(cutout.data)
    return np.array(cutout_data)

def get_crop_position(image, target_h, target_w, random_crop=False, current_pos=None):
    """
    Get crop position.

    Args:
        image: input image
        target_h: target height
        target_w: target width
        random_crop: whether to use random crop (Gaussian distribution)
        current_pos: existing position (returned directly if provided)
    """
    if current_pos is not None:
        return current_pos

    # Get physical geometric center (float)
    img_h, img_w = image.shape[-2], image.shape[-1]
    center_h = (img_h - 1) / 2.0
    center_w = (img_w - 1) / 2.0

    if not random_crop:
        # Return exact float center for consistent behavior across sizes
        return (center_h, center_w)

    # Random crop logic
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


def invvar_to_sigma(invvar_data, eps=1e-5, clip_min=1e-5, clip_max=1.0):
    """
    Convert invvar to sigma: sigma = 1/sqrt(invvar), clipped to a valid range.

    Safe handling:
    1. Set negative and zero invvar to eps (corresponding to maximum sigma)
    2. Compute sigma = 1/sqrt(invvar)
    3. Clip to [clip_min, clip_max]

    Args:
        invvar_data: inverse variance data
        eps: small value to prevent division by zero (default 1e-5)
        clip_min: minimum sigma (default 1e-5)
        clip_max: maximum sigma (default 1.0)

    Returns:
        Sigma data clipped to [clip_min, clip_max]
    """
    # Copy to avoid modifying original data
    invvar_safe = np.array(invvar_data, dtype=np.float64)

    # Set non-positive and invalid values to eps
    invvar_safe[invvar_safe <= 0] = eps
    invvar_safe[~np.isfinite(invvar_safe)] = eps

    # Compute sigma (suppress warnings)
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma = 1.0 / np.sqrt(invvar_safe)

    # Handle NaN and Inf
    sigma = np.nan_to_num(sigma, nan=clip_max, posinf=clip_max, neginf=clip_max)

    # Clip to valid range
    sigma = np.clip(sigma, clip_min, clip_max)

    return sigma.astype(np.float32)


def rms_to_sigma(rms_data, clip_min=1e-5, clip_max=1.0):
    """
    Convert RMS to sigma (essentially clip with outlier handling).

    Safe handling:
    1. Replace NaN and Inf with clip_max
    2. Replace negative values with clip_max
    3. Clip to [clip_min, clip_max]

    Args:
        rms_data: RMS data (standard deviation)
        clip_min: minimum sigma (default 1e-5)
        clip_max: maximum sigma (default 1.0)

    Returns:
        Sigma data clipped to [clip_min, clip_max]
    """
    # Copy
    sigma = np.array(rms_data, dtype=np.float32)

    # Handle NaN and Inf
    sigma = np.nan_to_num(sigma, nan=clip_max, posinf=clip_max, neginf=clip_max)

    # Handle negatives
    sigma[sigma < 0] = clip_max

    # Clip to valid range
    sigma = np.clip(sigma, clip_min, clip_max)

    return sigma


def convert_flux_zeropoint(flux_data, from_zp=30, to_zp=22.5):
    factor = 10 ** (0.4 * (to_zp - from_zp))
    return flux_data * factor


def convert_invvar_zeropoint(invvar_data, from_zp=30, to_zp=22.5):
    factor = 10 ** (0.4 * (to_zp - from_zp))
    return invvar_data / (factor ** 2)


# ===================================================================
# fast mode: separable axis-aligned area-weighted resampling
#
# When input/output WCS are synthetic pure-geometric scaling (no rotation,
# no real astrometric WCS, reference point center-aligned — as is the case
# for all data in this project), reproject_exact's area-weighted integration
# reduces to separable axis-aligned rectangular overlap integrals:
#     out = W_row @ img @ W_col.T
# W_row/W_col are (out, in) overlap weight matrices per row/column direction,
# depending only on image shape + pixel scale ratio; cached and reused across
# all channels.
#
# Consistency with reproject_exact (measured on real DESI BGSUB/RMS data):
#   - Total flux ratio max|dev| ~1.3e-4
#   - Pixel relative error P99 ~7e-5, P99.99 ~6e-4
#   - Only ~0.015% of pixels (bright galaxy steep cores) exceed 5e-4, worst ~1.1%
#   - Speedup ~70x
# Residuals arise from bright-core sub-pixel integration details, far below
# model/photon noise; morphology and photometry are unaffected.
# ===================================================================

_OVERLAP_CACHE = {}


def _overlap_matrix(n_in, n_out, scale):
    """
    Area overlap weight matrix (n_out, n_in) along one axis, row-normalized
    to area average (surface brightness), replicating reproject_exact semantics.

    Alignment convention: both grids share the same physical center
    (consistent with synthetic WCS CRPIX=(N+1)/2).
    FITS pixel i (0-based) is centered at coordinate i, covering [i-0.5, i+0.5].

    scale = to_pixel_scale / from_pixel_scale (output pixels span how many input
    pixels; < 1 when oversampling).
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
    Separable area-weighted resampling; see module comment above. img_data: (C,H,W).

    Weight matrices are built in f64 (clip/normalize precision), cached as f32,
    and applied via BLAS sgemm (np.matmul) with two passes:
        out = Wr @ x @ WcT    # (Ho,Hi)@(C,Hi,Wi)@(Wi,Wo) -> (C,Ho,Wo)
    f32 matmul is ~6x faster than f64 einsum; numerical difference ~5e-8
    (pure rounding, far below ~6e-4 residual vs reproject_exact).
    """
    c, h_in, w_in = img_data.shape
    sf = from_pixel_scale / to_pixel_scale
    h_out, w_out = int(h_in * sf), int(w_in * sf)
    scale = to_pixel_scale / from_pixel_scale

    key = (h_in, h_out, w_in, w_out, scale)
    wmats = _OVERLAP_CACHE.get(key)
    if wmats is None:
        # Build in f64, cache as f32; column matrix pre-transposed to (w_in, w_out) for right multiply
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


def reproj(img_data, img_header, apply_zp_conv=False, from_zp=30, to_zp=22.5, is_invvar=False,
           from_pixel_scale=0.262, to_pixel_scale=0.1, mode='exact'):
    """
    Image reprojection/resampling (resolution conversion, e.g. 0.262" -> 0.1").

    Pure resolution resampling independent of FITS WCS; uses synthetic WCS
    with the reproject package for high-quality resampling.

    Args:
        img_data: input image data (C, H, W)
        img_header: FITS header (optional; real WCS used if present)
        apply_zp_conv: whether to apply zero-point conversion
        from_zp: source zero-point
        to_zp: target zero-point
        is_invvar: whether data is inverse variance
        from_pixel_scale: source pixel scale in arcsec/pixel (default 0.262)
        to_pixel_scale: target pixel scale in arcsec/pixel (default 0.1)
        mode: 'exact' (reproject_exact, precise polygon area integration, default)
              or 'fast' (separable area-weighted matrix, same flux semantics as exact, ~70x faster;
              see _reproj_fast consistency notes)
    """
    if mode == 'fast':
        return _reproj_fast(img_data, apply_zp_conv, from_zp, to_zp, is_invvar,
                            from_pixel_scale, to_pixel_scale)
    if mode != 'exact':
        raise ValueError(f"reproj mode must be 'exact' or 'fast', got {mode!r}")

    # Input data shape
    c, h_in, w_in = img_data.shape

    # Compute output size from pixel scale ratio
    scale_factor = from_pixel_scale / to_pixel_scale
    h_out = int(h_in * scale_factor)
    w_out = int(w_in * scale_factor)

    # Create input WCS (synthetic, unrelated to real astrometric coordinates)
    input_header = Header()
    input_header['NAXIS'] = 2
    input_header['NAXIS1'] = w_in
    input_header['NAXIS2'] = h_in
    input_header['CRPIX1'] = (w_in + 1) / 2.0  # image center
    input_header['CRPIX2'] = (h_in + 1) / 2.0
    input_header['CRVAL1'] = 0.0  # arbitrary reference point
    input_header['CRVAL2'] = 0.0
    input_header['CTYPE1'] = 'RA---TAN'
    input_header['CTYPE2'] = 'DEC--TAN'
    input_header['CDELT1'] = -from_pixel_scale / 3600.0  # convert to degrees
    input_header['CDELT2'] = from_pixel_scale / 3600.0

    # Attempt to use real astrometric info from original header if available
    if img_header is not None:
        try:
            wcs_original = WCS(img_header)
            if wcs_original.has_celestial:
                wcs_2d = wcs_original.celestial
                # Use real WCS info
                for key in ['CRVAL1', 'CRVAL2']:
                    if key in img_header:
                        input_header[key] = img_header[key]
        except:
            pass  # Fall back to synthetic WCS if parsing fails

    wcs_in = WCS(input_header)

    # Create output WCS (higher resolution)
    output_header = Header()
    output_header['NAXIS'] = 2
    output_header['NAXIS1'] = w_out
    output_header['NAXIS2'] = h_out
    output_header['CRPIX1'] = (w_out + 1) / 2.0
    output_header['CRPIX2'] = (h_out + 1) / 2.0
    output_header['CRVAL1'] = input_header['CRVAL1']
    output_header['CRVAL2'] = input_header['CRVAL2']
    output_header['CTYPE1'] = 'RA---TAN'
    output_header['CTYPE2'] = 'DEC--TAN'
    output_header['CDELT1'] = -to_pixel_scale / 3600.0
    output_header['CDELT2'] = to_pixel_scale / 3600.0

    wcs_out = WCS(output_header)
    shape_out = (h_out, w_out)

    # Reproject each channel
    resampled = []
    for i in range(c):
        array, _ = reproject_exact(
            (img_data[i], wcs_in),
            wcs_out,
            shape_out=shape_out,
        )

        # Handle NaN values (fill edge NaNs with 0)
        if np.isnan(array).any():
            array = np.nan_to_num(array, nan=0.0)

        # Apply zero-point conversion if needed
        if apply_zp_conv:
            array = convert_invvar_zeropoint(array, from_zp, to_zp) if is_invvar else convert_flux_zeropoint(array, from_zp, to_zp)

        resampled.append(array)

    return np.stack(resampled, axis=0)


def euclid_fits_reader(file_path):
    """Euclid FITS reader."""
    if fits.getheader(file_path, 5).get("BAND") != "VIS":
        data = fits.getdata(file_path, 6)
    else:
        data = fits.getdata(file_path, 5)
    if data.ndim == 2:
        data = data.reshape(1, data.shape[0], data.shape[1])
    return ensure_native_byteorder(data)
