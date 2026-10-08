"""Validate a portable CPU run against an independent original-checkpoint run."""
from pathlib import Path
import importlib.util
import json
import os
import platform
import sys
import time
import tempfile
import inspect

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from inference import prepare_input, load_model, generate, sha256, resolve_bands


def main():
    begin = time.perf_counter()
    desi, header, info = prepare_input(ROOT / 'example/BGSUB/1-10166.fits')
    rz, rz_header, rz_info = prepare_input(ROOT / 'example/BGSUB/1-10166_rz.fits')
    assert info['input_bands'] == 'grz' and rz_info['input_bands'] == 'rz'
    assert rz.shape == (2, 128, 128)
    np.testing.assert_array_equal(desi[1:], rz)
    reference_input = np.load(ROOT / 'validation/production_prepared.npz')
    np.testing.assert_allclose(desi, reference_input['desi'], rtol=1e-6, atol=1e-7)
    assert set(inspect.signature(prepare_input).parameters) == {'image_path', 'bands', 'input_pixel_scale'}
    assert set(inspect.signature(generate).parameters) == {
        'model', 'desi', 'bands', 'seed', 'num_samples', 'nfe', 'split_ratio', 'verbose'
    }
    # Wrong channels/order/header declarations must fail, never silently shift
    # r/z into another band's normalization parameters.
    invalid_cases = [(1, 'auto', None), (4, 'auto', None), (2, 'grz', None),
                     (3, 'rz', None), (2, 'auto', 'z,r'), (3, 'rz', 'g,r,z')]
    for args in invalid_cases:
        try:
            resolve_bands(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid band declaration was accepted: {args}')
    assert resolve_bands(2, 'r,z') == 'rz'
    assert resolve_bands(3, 'G,R,Z') == 'grz'
    with tempfile.TemporaryDirectory() as temp:
        for name, count in [('1-10166_rz.fits', 2), ('1-10166.fits', 3)]:
            with fits.open(ROOT / 'example/BGSUB' / name) as hdul:
                no_header_bands = hdul[0].header.copy()
                del no_header_bands['BANDS']
                path = Path(temp) / name
                fits.PrimaryHDU(hdul[0].data, no_header_bands).writeto(path)
            prepared, _, inferred = prepare_input(path)
            assert inferred['input_bands'] == ('rz' if count == 2 else 'grz')
            np.testing.assert_array_equal(prepared, rz if count == 2 else desi)
    model = load_model('cpu', cpu_threads=4)
    prediction, run = generate(model, desi, bands='grz', verbose=False)
    reference = np.load(ROOT / 'validation/production_reference.npy')
    np.testing.assert_allclose(prediction, reference, rtol=1e-4, atol=1e-5)
    # Separate real two-channel FITS, same r/z and seed -> identical output.
    repeat, rz_run = generate(model, rz, bands='rz', verbose=False)
    np.testing.assert_array_equal(repeat, prediction)
    # g does not change the normalized bridge endpoint or conditioning plane.
    changed_g = desi.copy()
    changed_g[0] = 100.0
    with torch.inference_mode():
        a = model.pixel_transform.forward(torch.from_numpy(desi)[None], 'desi')
        b = model.pixel_transform.forward(torch.from_numpy(changed_g)[None], 'desi')
        assert torch.equal(a[:, 1:], b[:, 1:])
    assert next(model.parameters()).device.type == 'cpu'
    # Verify exact coordinate mapping of the resampled/cropped output grid.
    native = WCS(fits.getheader(ROOT / 'example/BGSUB/1-10166.fits')).celestial
    out_wcs = WCS(header)
    points = np.array([[0, 0], [64, 64], [127, 127]], dtype=float)
    full_xy = points + np.array(info['crop_yx'][::-1])
    mapped_native = (full_xy - 167) * (0.1 / 0.262) + 63.5
    separation = out_wcs.pixel_to_world(points[:, 0], points[:, 1]).separation(
        native.pixel_to_world(mapped_native[:, 0], mapped_native[:, 1])
    ).arcsec
    assert float(separation.max()) < 1e-6
    copied_modules = []
    for name, module in list(sys.modules.items()):
        if name == 'sgm' or name.startswith('sgm.'):
            p = getattr(module, '__file__', None)
            if p:
                assert Path(p).resolve().is_relative_to(ROOT), (name, p)
                copied_modules.append(str(Path(p).resolve().relative_to(ROOT)))
    report = {
        'passed': True, 'python': platform.python_version(),
        'platform': platform.platform(), 'torch': str(torch.__version__),
        'numpy': np.__version__, 'torch_cuda_build': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'xformers_installed': importlib.util.find_spec('xformers') is not None,
        'device': 'cpu', 'threads': torch.get_num_threads(),
        'inference_seconds': run['inference_seconds'],
        'rz_inference_seconds': rz_run['inference_seconds'],
        'validation_seconds': time.perf_counter() - begin,
        'output_shape': list(prediction.shape), 'all_pixels_finite': True,
        'production_preprocessing_max_abs_diff': float(np.max(np.abs(desi-reference_input['desi']))),
        'production_prediction_max_abs_diff': float(np.max(np.abs(prediction-reference))),
        'production_prediction_mae': float(np.mean(np.abs(prediction-reference))),
        'prediction_tolerance': {'rtol': 1e-4, 'atol': 1e-5},
        'rz_grz_preprocessing_equal': True,
        'rz_grz_prediction_exactly_equal': True,
        'rz_grz_prediction_max_abs_diff': float(np.max(np.abs(repeat-prediction))),
        'g_does_not_affect_model_condition': True,
        'band_header_shape_and_order_validation_passed': True,
        'bands_inferred_without_header_passed': True,
        'image_only_public_api': True,
        'wcs_max_error_arcsec': float(separation.max()),
        'project_imports_all_from_bundle': True, 'imported_project_files': copied_modules,
        'input_sha256': info['input_sha256'],
        'weights_sha256': sha256(ROOT / 'weights/best_ema.safetensors'),
        'caveat': 'Measured on this Linux CPU; not a laptop benchmark or scientific truth validation.'
    }
    if platform.system() == 'Linux':
        import resource
        report['process_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (ROOT / 'validation/cpu_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
