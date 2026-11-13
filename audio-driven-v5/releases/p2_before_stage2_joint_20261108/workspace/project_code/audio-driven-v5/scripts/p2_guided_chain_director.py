#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p2_guided_chain.director import supervise
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();supervise(a.root)
