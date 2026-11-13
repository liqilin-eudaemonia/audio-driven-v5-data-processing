"""Fetch official GPAvatar release assets with resumable HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import time
import urllib.request

WORK = Path(__file__).resolve().parent
ASSETS = [
    ('one_model.ckpt', 'https://github.com/xg-chu/GPAvatar/releases/download/v1.0.0/one_model.ckpt', 429486995, '28bbec93d503562477d9468c50524978'),
    ('resources.tar', 'https://github.com/xg-chu/GPAvatar/releases/download/Resources/resources.tar', 1181726720, '1d386517baa670307243c3fd45494a53'),
]


def download(spec):
    name, url, size, md5 = spec
    folder = WORK / 'assets' / (name + '.parts')
    folder.mkdir(parents=True, exist_ok=True)
    chunk = 4 * 1024**2
    starts = list(range(0, size, chunk))

    def part(start):
        end = min(start + chunk, size) - 1
        path = folder / str(start)
        if path.exists() and path.stat().st_size == end-start+1:
            return
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={'Range':f'bytes={start}-{end}', 'User-Agent':'GPAvatar-research-download'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{size}', response.headers
                    temporary = path.with_suffix('.partial')
                    with temporary.open('wb') as output:
                        while data := response.read(256*1024):
                            output.write(data)
                assert temporary.stat().st_size == end-start+1
                temporary.replace(path)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 8))

    with ThreadPoolExecutor(max_workers=20) as pool:
        for index,_ in enumerate(pool.map(part, starts), 1):
            print(name, index, '/', len(starts), flush=True)
    output = WORK / 'assets' / name
    temporary = output.with_suffix(output.suffix+'.complete')
    md5sum, sha = hashlib.md5(), hashlib.sha256()
    with temporary.open('wb') as dest:
        for start in starts:
            with (folder / str(start)).open('rb') as source:
                while data := source.read(1024**2):
                    dest.write(data); md5sum.update(data); sha.update(data)
    assert temporary.stat().st_size == size
    assert md5sum.hexdigest() == md5, (name, md5sum.hexdigest(), md5)
    temporary.replace(output)
    return dict(path=str(output), url=url, size=size, official_md5=md5, sha256=sha.hexdigest(), passed=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts=list(pool.map(download, ASSETS))
    (WORK/'DOWNLOADS.json').write_text(json.dumps(dict(at=time.time(),receipts=receipts),indent=2)+'\n')
