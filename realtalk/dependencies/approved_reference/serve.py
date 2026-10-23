"""Local pilot review server; reuse existing byte-range handler."""
import importlib.util
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('realtalk_existing_server','/public/0_DATA/2_Avatar/realtalk/preview/serve.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':
    server=ThreadingHTTPServer(('0.0.0.0',8797),partial(module.RangeRequestHandler,directory=str(ROOT/'review')))
    print('RealTalk pilot: http://127.0.0.1:8797/',flush=True)
    server.serve_forever()
