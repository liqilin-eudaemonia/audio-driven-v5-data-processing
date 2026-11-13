"""Serve only the review directory, with single-range video seeking support."""
import argparse
import functools
import http.server
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        route = posixpath.normpath('/' + unquote(urlsplit(self.path).path).lstrip('/'))
        retired = ('/index_v1.html', '/rule_turns_v2', '/speaker_end', '/media', '/manifest.json')
        if any(route == prefix or route.startswith(prefix + '/') for prefix in retired):
            self.send_error(410, 'This legacy preview is retired; use /listen_qc3/')
            return None
        path = Path(self.translate_path(self.path))
        value = self.headers.get('Range')
        if not value or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
        size = path.stat().st_size
        if not match or not size:
            self.send_error(416); return None
        a,b=match.groups()
        if not a:
            start=max(0,size-int(b or 0));end=size-1
        else:
            start=int(a);end=min(int(b) if b else size-1,size-1)
        if start>end or start>=size:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return None
        f=path.open('rb');f.seek(start)
        self.send_response(206)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Accept-Ranges','bytes')
        self.end_headers();self.remaining=end-start+1
        return f

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            try:super().copyfile(source,outputfile)
            except (BrokenPipeError,ConnectionResetError):pass
            return
        try:
            while self.remaining:
                block=source.read(min(1024*1024,self.remaining))
                if not block:break
                outputfile.write(block);self.remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8875)
    args=parser.parse_args()
    directory=Path(__file__).resolve().parents[1]/'data_previews/seamless_face_listening_v1'
    http.server.ThreadingHTTPServer(('0.0.0.0',args.port),functools.partial(Handler,directory=str(directory))).serve_forever()
