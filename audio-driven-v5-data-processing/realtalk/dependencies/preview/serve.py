#!/usr/bin/env python3
"""Serve the RealTalk explorer with byte-range support for large MP3 files."""

from __future__ import annotations

import email.utils
import http.server
import os
import re
from functools import partial
from pathlib import Path

ROOT = Path("/public/0_DATA/2_Avatar/realtalk")
PORT = 8791


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    byte_range: tuple[int, int] | None = None

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        requested = self.headers.get("Range")
        if not requested:
            self.byte_range = None
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            self.byte_range = None
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(source.fileno()).st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        if not match:
            source.close()
            self.send_error(416, "Invalid byte range")
            return None
        first, last = match.groups()
        if first:
            start = int(first)
            end = int(last) if last else size - 1
        else:
            suffix = int(last)
            start, end = max(size - suffix, 0), size - 1
        if start >= size or start > end:
            source.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        end = min(end, size - 1)
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", email.utils.formatdate(os.path.getmtime(path), usegmt=True))
        self.end_headers()
        source.seek(start)
        return source

    def copyfile(self, source, outputfile) -> None:
        if self.byte_range is None:
            return super().copyfile(source, outputfile)
        start, end = self.byte_range
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__ == "__main__":
    handler = partial(RangeRequestHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), handler)
    print(f"RealTalk explorer: http://0.0.0.0:{PORT}/preview/", flush=True)
    server.serve_forever()
