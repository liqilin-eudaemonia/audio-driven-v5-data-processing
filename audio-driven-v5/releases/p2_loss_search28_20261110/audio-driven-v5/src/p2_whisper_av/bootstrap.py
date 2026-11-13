"""Select a private copy of LTX before importing any renderer/trainer modules."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'external_sources/p2_whisper_av'


def activate_ltx():
    for name, module in tuple(sys.modules.items()):
        if name == 'ltx_core' or name.startswith('ltx_core.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(CORE):
                raise RuntimeError('Whisper W1 requires its private LTX core; start a fresh process')
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    return CORE
