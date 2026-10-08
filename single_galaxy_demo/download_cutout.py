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
        raise ValueError("RA、Dec 必须是 ICRS 十进制度数。") from exc
    if not math.isfinite(ra) or not 0 <= ra < 360:
        raise ValueError("RA 必须满足 0 <= RA < 360 度。")
    if not math.isfinite(dec) or not -90 <= dec <= 90:
        raise ValueError("Dec 必须在 [-90, 90] 度内。")
    bands = str(bands).lower().replace(",", "").replace(" ", "")
    if bands not in {"rz", "grz"}:
        raise ValueError("bands 必须为 'rz' 或 'grz'。")
    if layer not in LAYERS:
        raise ValueError(f"layer 必须为 {LAYERS} 之一。")
    return dict(ra=ra, dec=dec, layer=layer, bands=bands,
                pixscale=PIXEL_SCALE, size=SIZE)


def _inspect(hdul, request):
    hdul.verify("exception")
    if len(hdul) != 1 or hdul[0].data is None:
        raise ValueError("服务没有返回单个图像主 HDU。")
    data = np.array(hdul[0].data, dtype=np.float32)
    header = hdul[0].header.copy()
    expected = (len(request["bands"]), SIZE, SIZE)
    if data.shape != expected:
        raise ValueError(f"裁图形状 {data.shape} 与所需 {expected} 不一致，可能缺少波段。")
    if header.get("IMAGETYP", "").strip().upper() != "IMAGE":
        raise ValueError("响应不是观测图像 IMAGE。")
    declared = str(header.get("BANDS", "")).replace(",", "").replace(" ", "").lower()
    if declared != request["bands"]:
        raise ValueError(f"返回波段 {declared!r} 与请求不一致。")
    for i, band in enumerate(declared):
        if str(header.get(f"BAND{i}", "")).strip().lower() != band:
            raise ValueError("FITS 的逐通道波段信息不一致。")
    for band, plane in zip(declared, data):
        if not np.isfinite(plane).all():
            raise ValueError(f"{band} 波段包含 NaN/Inf，不能作为完整推理输入。")
        # The image-only viewer may encode unobserved regions as exact zeros.
        # Be conservative: do not turn zero-filled gaps into inferred sources.
        if np.any(plane == 0):
            raise ValueError(f"{band} 波段含零值，无法排除无覆盖区域；此裁图不送入推理。")
        if float(np.ptp(plane)) == 0:
            raise ValueError(f"{band} 波段为常数图像，可能没有有效观测。")
    wcs = WCS(header, naxis=2).celestial
    if not wcs.has_celestial or wcs.has_distortion:
        raise ValueError("裁图需要无畸变的天球 WCS。")
    scales = proj_plane_pixel_scales(wcs) * 3600
    if not np.allclose(scales, PIXEL_SCALE, rtol=1e-6, atol=1e-8):
        raise ValueError(f"裁图像素尺度异常：{scales}。")
    center = wcs.pixel_to_world((SIZE - 1) / 2, (SIZE - 1) / 2)
    wanted = SkyCoord(request["ra"], request["dec"], unit="deg", frame="icrs")
    separation = float(center.separation(wanted).arcsec)
    if separation > 1e-4:
        raise ValueError(f"裁图中心偏离请求坐标 {separation:.6g} arcsec。")
    for hdu in hdul:
        if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
            raise ValueError("FITS CHECKSUM 校验失败。")
        if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
            raise ValueError("FITS DATASUM 校验失败。")
    return data, header, separation


def _decode(payload, request):
    if not payload.startswith(b"SIMPLE  ="):
        raise ValueError("服务返回的不是 FITS（可能无覆盖或返回 HTML 错误页）。")
    if len(payload) % 2880:
        raise ValueError("FITS 下载不完整：字节数不是 2880 的整数倍。")
    # Check the declared length before checksum verification reads the pixels.
    # _inspect verifies any CHECKSUM/DATASUM after this truncation guard.
    with fits.open(io.BytesIO(payload), memmap=False, checksum=False) as hdul:
        # Astropy may warn and still expose a truncated file: reject explicitly.
        end = hdul.fileinfo(0)["datLoc"] + hdul.fileinfo(0)["datSpan"]
        if len(payload) < end:
            raise ValueError("FITS 图像数据被截断。")
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
        raise ValueError("timeout 必须是正数（秒）。")
    directory = ROOT / "downloads" if output_dir is None else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    path = directory / (f"desi_{request['ra']:.6f}_{request['dec']:+.6f}_"
                        f"{request['bands']}_{identity}.fits")
    sidecar = path.with_suffix(".json")
    lock = path.with_suffix(".lock")

    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise FileExistsError(f"本地下载记录不完整，请使用另一个 output_dir：{path}")
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if info.get("request") != request or info.get("file_sha256") != _sha256(path):
            raise ValueError(f"缓存参数或校验值不符，不复用也不覆盖：{path}")
        with fits.open(path, memmap=False, checksum=True) as hdul:
            _inspect(hdul, request)
        return path

    # This lock also prevents two notebook kernels from publishing partial pairs.
    try:
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise FileExistsError(f"相同裁图正在下载（或遗留锁文件）：{lock}") from exc
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
                        raise ValueError("128x128 裁图响应异常过大。")
                    blocks.append(block)
                payload = b"".join(blocks)
                size_header = response.headers.get("Content-Length")
                if size_header and not response.headers.get("Content-Encoding"):
                    if len(payload) != int(size_header):
                        raise ValueError("下载字节数与 HTTP Content-Length 不一致。")
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
            raise FileExistsError(f"下载期间出现同名文件，不覆盖：{path}")
        os.replace(temp_fits, path)
        os.replace(temp_json, sidecar)
        return path
    except requests.RequestException as exc:
        raise RuntimeError("官方裁图下载失败，请检查网络、服务状态或目标覆盖。") from exc
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
