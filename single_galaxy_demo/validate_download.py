"""Offline checks using the bundled, genuinely downloaded coordinate cutouts."""
from pathlib import Path
import io
import json
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
from astropy.io import fits
from download_cutout import download_cutout, _decode, _request, ENDPOINT

ROOT = Path(__file__).resolve().parent


def must_fail(fn, error=(ValueError, OSError)):
    try:
        fn()
    except error:
        return
    raise AssertionError('Expected rejection did not occur.')


def payload_variant(payload, change):
    with fits.open(io.BytesIO(payload), memmap=False) as h:
        data=h[0].data.copy();header=h[0].header.copy()
    for key in ['CHECKSUM', 'DATASUM']:
        header.pop(key, None)
    data, header=change(data,header)
    out=io.BytesIO();fits.PrimaryHDU(data,header).writeto(out)
    return out.getvalue()


def mock_session(payload):
    response=MagicMock()
    response.headers={'Content-Type':'image/fits','Content-Length':str(len(payload))}
    response.url=ENDPOINT
    response.iter_content.return_value=iter([payload])
    session=MagicMock()
    session.__enter__.return_value=session
    session.get.return_value.__enter__.return_value=response
    return session


def main():
    # No network needed for this check: absent/incomplete cache fails here.
    paths={}
    with patch('download_cutout.requests.Session',side_effect=AssertionError('Network access during cache validation')):
        for bands in ['grz','rz']:
            paths[bands]=download_cutout(199.2955574,0.913561334,bands=bands)
    infos={b:json.loads(p.with_suffix('.json').read_text()) for b,p in paths.items()}
    payload=paths['grz'].read_bytes();request=infos['grz']['request']
    data,_,_= _decode(payload,request)
    rz,_,_= _decode(paths['rz'].read_bytes(),infos['rz']['request'])
    np.testing.assert_array_equal(data[1:],rz)
    for ra,dec in [(360,0),(-1,0),(0,91),(0,-91),(float('nan'),0),(0,float('inf'))]:
        must_fail(lambda: _request(ra,dec,'grz','ls-dr10'))
    must_fail(lambda: _request(1,1,'zrg','ls-dr10'))
    must_fail(lambda: _decode(b'<html>no coverage</html>',request))
    must_fail(lambda: _decode(payload[:-1],request))
    must_fail(lambda: _decode(payload[:-2880],request))
    def mutate_header(data,header,key,value):
        header[key]=value
        return data,header
    must_fail(lambda: _decode(payload_variant(payload,lambda d,h: mutate_header(d,h,'BANDS','rz')),request))
    must_fail(lambda: _decode(payload_variant(payload,lambda d,h: mutate_header(d,h,'CRVAL1',request['ra']+1)),request))
    def blank_pixel(data,header):
        data[0,0,0]=0
        return data,header
    must_fail(lambda: _decode(payload_variant(payload,blank_pixel),request))
    def nonfinite(data,header):
        data[0,0,0]=np.nan
        return data,header
    must_fail(lambda: _decode(payload_variant(payload,nonfinite),request))
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp)
        # Successful validated publication, followed by offline reuse.
        with patch('download_cutout.requests.Session',return_value=mock_session(payload)):
            p=download_cutout(request['ra'],request['dec'],output_dir=directory)
        first=p.read_bytes()
        with patch('download_cutout.requests.Session',side_effect=AssertionError('Cache tried to download')):
            assert download_cutout(request['ra'],request['dec'],output_dir=directory)==p
        assert p.read_bytes()==first
        assert not list(directory.glob('*.part')) and not list(directory.glob('*.lock'))
        # Corrupt cache must not be overwritten or silently reused.
        p.write_bytes(first[:-1])
        must_fail(lambda: download_cutout(request['ra'],request['dec'],output_dir=directory))
        assert p.read_bytes()==first[:-1]
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp)
        with patch('download_cutout.requests.Session',return_value=mock_session(b'<html>Error</html>')):
            must_fail(lambda: download_cutout(request['ra'],request['dec'],output_dir=directory))
        assert not list(directory.iterdir()),'Failed download left a published product or staging file.'
    report={
        'passed':True,'live_downloads':{b:{'file':p.name,'upstream_url':infos[b]['url'],
        'upstream_sha256':infos[b]['upstream_sha256'],'file_sha256':infos[b]['file_sha256'],
        'shape':infos[b]['shape']} for b,p in paths.items()},
        'downloaded_rz_equals_grz_rz_channels':True,
        'html_and_truncated_response_rejected':True,'wrong_bands_or_center_rejected':True,
        'invalid_coordinates_rejected':True,'nonfinite_and_zero_filled_images_rejected':True,
        'valid_cache_reused_without_network':True,'corrupt_cache_rejected_without_overwrite':True,
        'failure_leaves_no_fits_or_staging_files':True,
        'all_checks_offline_against_actual_downloads_and_mock_responses':True,
    }
    (ROOT/'validation/download_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
