#!/usr/bin/env python3
"""Frozen original training semantics, separately audited faster execution."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    if '--config' not in sys.argv:raise ValueError('Explicit original-run config required')
    config=Path(sys.argv[sys.argv.index('--config')+1]).resolve()
    from p2_legacy_execution import install
    install(config.parent.parent/'execution.json')
    import runpy
    runpy.run_path(str(ROOT/'scripts/p2_legacy_qwen_lora.py'),run_name='__main__')
