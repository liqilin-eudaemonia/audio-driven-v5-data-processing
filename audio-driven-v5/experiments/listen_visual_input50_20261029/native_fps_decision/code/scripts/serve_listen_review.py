#!/usr/bin/env python3
"""Serve existing review pages and participant media with HTTP range support."""
import argparse
import functools
from http.server import ThreadingHTTPServer
from pathlib import Path

from serve_seamless_preview import Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8876)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'data_previews'
    ThreadingHTTPServer(('0.0.0.0',args.port),functools.partial(Handler,directory=str(root))).serve_forever()
