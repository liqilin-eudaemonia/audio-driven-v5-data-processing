#!/usr/bin/env python3
"""Run a command on authorized node8 using the existing cluster SSH configuration."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
c=json.loads((root/'configs/p2_j_qwen_whisper_av_train32_cluster_20261104.json').read_text())
raise SystemExit(subprocess.call(['ssh','-i',c['ssh_key'],'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new','root@172.29.104.237',*sys.argv[1:]]))
