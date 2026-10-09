"""RA/Dec (ICRS degrees) -> Legacy Surveys FITS accepted by prepare_input.

Official service documentation:
https://www.legacysurvey.org/dr10/description/#obtaining-images-and-raw-data
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import math
import os
import tempfile

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://www.legacysurvey.org/viewer/fits-cutout"
PIXEL_SCALE = 0.262
SIZE = 128
LAYERS = ("ls-dr10", "ls-dr9-north", "ls-dr10-south")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _request(ra, dec, bands, layer):
    try:
        ra, dec = float(ra), float(dec)
    except (TypeError, ValueError) as exc:
        raise ValueError("RA and Dec must be ICRS coordinates in decimal degrees.") from exc
    if not math.isfinite(ra) or not 0 <= ra < 360:
        raise ValueError("RA must satisfy 0 <= RA < 360 degrees.")
    if not math.isfinite(dec) or not -90 <= dec <= 90:
        raise ValueError("Dec must be in [-90, 90] degrees.")
    bands = str(bands).lower().replace(",", "").replace(" ", "")
    if bands not in {"rz", "grz"}:
        raise ValueError("bands must be 'rz' or 'grz'.")
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}.")
    return dict(ra=ra, dec=dec, layer=layer, bands=bands,
                pixscale=PIXEL_SCALE, size=SIZE)


def _inspect(hdul, request):
    hdul.verify("exception")
    if len(hdul) != 1 or hdul[0].data is None:
        raise ValueError("The service did not return a single image primary HDU.")
    data = np.array(hdul[0].data, dtype=np.float32)
    header = hdul[0].header.copy()
    expected = (len(request["bands"]), SIZE, SIZE)
    if data.shape != expected:
        raise ValueError(f"Cutout shape {data.shape} does not match {expected}; bands may be missing.")
    if header.get("IMAGETYP", "").strip().upper() != "IMAGE":
        raise ValueError("The response is not a coadd IMAGE.")
    declared = str(header.get("BANDS", "")).replace(",", "").replace(" ", "").lower()
    if declared != request["bands"]:
        raise ValueError(f"Returned bands {declared!r} do not match the request.")
    for i, band in enumerate(declared):
        if str(header.get(f"BAND{i}", "")).strip().lower() != band:
            raise ValueError("The FITS per-channel band metadata are inconsistent.")
    for band, plane in zip(declared, data):
        if not np.isfinite(plane).all():
            raise ValueError(f"Band {band} contains NaN/Inf and cannot provide a complete inference input.")
        # The image-only viewer may encode unobserved regions as exact zeros.
        # Be conservative: do not turn zero-filled gaps into inferred sources.
        if np.any(plane == 0):
            raise ValueError(f"Band {band} contains zeros; unobserved regions cannot be ruled out. This cutout will not be used for inference.")
        if float(np.ptp(plane)) == 0:
            raise ValueError(f"Band {band} is constant and may lack valid observations.")
    wcs = WCS(header, naxis=2).celestial
    if not wcs.has_celestial or wcs.has_distortion:
        raise ValueError("The cutout must have celestial WCS without distortion.")
    scales = proj_plane_pixel_scales(wcs) * 3600
    if not np.allclose(scales, PIXEL_SCALE, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Unexpected cutout pixel scales: {scales}.")
    center = wcs.pixel_to_world((SIZE - 1) / 2, (SIZE - 1) / 2)
    wanted = SkyCoord(request["ra"], request["dec"], unit="deg", frame="icrs")
    separation = float(center.separation(wanted).arcsec)
    if separation > 1e-4:
        raise ValueError(f"The cutout center is offset from the requested coordinates by {separation:.6g} arcsec.")
    for hdu in hdul:
        if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
            raise ValueError("FITS CHECKSUM verification failed.")
        if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
            raise ValueError("FITS DATASUM verification failed.")
    return data, header, separation


def _decode(payload, request):
    if not payload.startswith(b"SIMPLE  ="):
        raise ValueError("The service did not return FITS; the target may lack coverage or the response may be an HTML error page.")
    if len(payload) % 2880:
        raise ValueError("Incomplete FITS download: the byte count is not a multiple of 2880.")
    # Check the declared length before checksum verification reads the pixels.
    # _inspect verifies any CHECKSUM/DATASUM after this truncation guard.
    with fits.open(io.BytesIO(payload), memmap=False, checksum=False) as hdul:
        # Astropy may warn and still expose a truncated file: reject explicitly.
        end = hdul.fileinfo(0)["datLoc"] + hdul.fileinfo(0)["datSpan"]
        if len(payload) < end:
            raise ValueError("The FITS image data are truncated.")
        return _inspect(hdul, request)


def download_cutout(ra, dec, *, bands="grz", output_dir=None, layer="ls-dr10", timeout=90):
    """Download an image-only 128x128 cube at 0.262 arcsec/pixel.

    Returns a Path directly usable by prepare_input(path, bands='auto').
    An identical validated local download is reused without network access.
    Existing unrelated/incomplete files are never overwritten.
    """
    request = _request(ra, dec, bands, layer)
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive, in seconds.")
    directory = ROOT / "downloads" if output_dir is None else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    path = directory / (f"desi_{request['ra']:.6f}_{request['dec']:+.6f}_"
                        f"{request['bands']}_{identity}.fits")
    sidecar = path.with_suffix(".json")
    lock = path.with_suffix(".lock")

    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise FileExistsError(f"The local download record is incomplete; use another output_dir: {path}")
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if info.get("request") != request or info.get("file_sha256") != _sha256(path):
            raise ValueError(f"Cache parameters or checksum do not match; the file will not be reused or overwritten: {path}")
        with fits.open(path, memmap=False, checksum=True) as hdul:
            _inspect(hdul, request)
        return path

    # This lock also prevents two notebook kernels from publishing partial pairs.
    try:
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise FileExistsError(f"The same cutout is being downloaded, or a stale lock remains: {lock}") from exc
    staging = []
    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "DESI2Euclid-single-galaxy-demo/1.0"})
            retry = Retry(total=2, backoff_factor=1, status_forcelist=(429, 502, 503, 504),
                          allowed_methods=frozenset({"GET"}), respect_retry_after_header=False)
            session.mount("https://", HTTPAdapter(max_retries=retry))
            with session.get(ENDPOINT, params=request, timeout=(15, timeout), stream=True) as response:
                response.raise_for_status()
                blocks, total = [], 0
                for block in response.iter_content(65536):
                    total += len(block)
                    if total > 4 * 1024 * 1024:
                        raise ValueError("The response is unexpectedly large for a 128x128 cutout.")
                    blocks.append(block)
                payload = b"".join(blocks)
                size_header = response.headers.get("Content-Length")
                if size_header and not response.headers.get("Content-Encoding"):
                    if len(payload) != int(size_header):
                        raise ValueError("The downloaded byte count does not match HTTP Content-Length.")
                source_url = response.url
        data, header, center_error = _decode(payload, request)
        upstream_sha = hashlib.sha256(payload).hexdigest()
        # Coadd IMAGE is already sky-subtracted and calibrated; no extra sky,
        # zero-point, pixel-area, or display transform is applied here.
        header["PIXSCALE"] = (PIXEL_SCALE, "arcsec/pixel")
        header["RA"] = request["ra"]
        header["DEC"] = request["dec"]
        header["BUNIT"] = ("nanomaggy", "Legacy Surveys coadd IMAGE flux unit")
        header["LSLAYER"] = layer
        header["SRCURL"] = source_url
        header["SRCSHA"] = upstream_sha
        header.add_history("Downloaded coadd IMAGE; no additional background subtraction.")
        header.add_history("Native 0.262 arcsec/pixel input for DESI2Euclid prepare_input.")
        with tempfile.NamedTemporaryFile(dir=directory, prefix=path.stem+"_", suffix=".part", delete=False) as f:
            temp_fits = Path(f.name)
        staging.append(temp_fits)
        fits.PrimaryHDU(data, header).writeto(temp_fits, checksum=True, overwrite=True)
        with fits.open(temp_fits, memmap=False, checksum=True) as hdul:
            _inspect(hdul, request)
        info = {
            "request": request, "url": source_url, "upstream_sha256": upstream_sha,
            "upstream_bytes": len(payload), "file_sha256": _sha256(temp_fits),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "shape": list(data.shape), "center_error_arcsec": center_error,
            "preprocessing": "Official sky-subtracted coadd IMAGE; flux pixels unchanged.",
            "coverage_check": "Finite, nonconstant images; reject any exact-zero pixel conservatively.",
            "documentation": "https://www.legacysurvey.org/dr10/description/",
        }
        with tempfile.NamedTemporaryFile(dir=directory, prefix=path.stem+"_", suffix=".part", delete=False,
                                         mode="w", encoding="utf-8") as f:
            temp_json = Path(f.name)
            staging.append(temp_json)
            json.dump(info, f, indent=2, ensure_ascii=False)
            f.write("\n")
        if path.exists() or sidecar.exists():
            raise FileExistsError(f"A file with the same name appeared during download; refusing to overwrite: {path}")
        os.replace(temp_fits, path)
        os.replace(temp_json, sidecar)
        return path
    except requests.RequestException as exc:
        raise RuntimeError("The official cutout download failed. Check the network, service status, and target coverage.") from exc
    finally:
        for file in staging:
            file.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="RA/Dec (ICRS degrees) -> DESI2Euclid input FITS")
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--bands", choices=("rz", "grz"), default="grz")
    parser.add_argument("--layer", choices=LAYERS, default="ls-dr10")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"downloads")
    args = parser.parse_args()
    print(download_cutout(args.ra, args.dec, bands=args.bands,
                          layer=args.layer, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
