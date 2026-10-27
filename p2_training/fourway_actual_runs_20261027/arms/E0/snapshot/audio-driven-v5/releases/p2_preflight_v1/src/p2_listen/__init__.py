"""Versioned preflight overlay; frozen model/cache modules retain their original paths."""
from pathlib import Path
import sys
BASE_ROOT=Path(__file__).resolve().parents[4]
__path__.append(str(BASE_ROOT/'src/p2_listen'))
if str(BASE_ROOT/'src') not in sys.path:sys.path.append(str(BASE_ROOT/'src'))
VERSION = 'p2-listen-event-v2-visual6-audio1'
