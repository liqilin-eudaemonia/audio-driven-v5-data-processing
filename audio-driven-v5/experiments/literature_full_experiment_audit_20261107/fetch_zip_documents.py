"""Read CVF ZIP directories/range-fetch documents, without downloading videos."""
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

BASE = Path(__file__).resolve().parent


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.transferred = url, 0, 0
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            self.size = int(response.headers['Content-Length'])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else self.size) + offset
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.pos
        count = min(count, self.size - self.pos)
        if count <= 0:
            return b''
        if count > 25_000_000:
            raise ValueError('Unexpected large document read')
        request = urllib.request.Request(self.url, headers={
            'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={self.pos}-{self.pos + count - 1}'})
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 206:
                raise ValueError(f'Range response required, got {response.status}')
            result = response.read(count)
        self.pos += len(result)
        self.transferred += len(result)
        return result


def main():
    results = []
    for source in json.loads((BASE / 'web/CVF_SUPPLEMENT_LINKS.json').read_text()):
        for link in source.get('supp_links', []):
            row = dict(name=source['name'], url='https://openaccess.thecvf.com' + link,
                       accessed_utc=datetime.now(timezone.utc).isoformat())
            try:
                stream = RemoteZip(row['url'])
                with zipfile.ZipFile(stream) as archive:
                    row['members'] = [dict(name=i.filename, bytes=i.file_size, compressed=i.compress_size)
                                      for i in archive.infolist()]
                    row['documents'] = []
                    for info in archive.infolist():
                        if info.filename.startswith('__MACOSX/') or Path(info.filename).name.startswith('._'):
                            continue
                        if Path(info.filename).suffix.lower() not in ['.pdf', '.html', '.htm', '.txt']:
                            continue
                        if info.file_size > 25_000_000:
                            continue
                        data = archive.read(info)
                        name = source['name'] + '_supp_' + Path(info.filename).name
                        output = BASE / 'web' / name
                        output.write_bytes(data)
                        doc = dict(member=info.filename, local=name, bytes=len(data),
                                   sha256=hashlib.sha256(data).hexdigest())
                        if output.suffix.lower() == '.pdf':
                            pdf = pymupdf.open(output)
                            text = '\f'.join(page.get_text(sort=True) for page in pdf)
                            (BASE / 'texts' / (output.stem + '.txt')).write_text(text)
                            doc['pages'] = len(pdf)
                        row['documents'].append(doc)
                row.update(ok=True, zip_total_bytes=stream.size, transferred_bytes=stream.transferred)
            except Exception as error:
                row.update(ok=False, error=str(error))
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'members'}, ensure_ascii=False), flush=True)
    (BASE / 'web/CVF_ZIP_DOCUMENTS.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
