#!/usr/bin/env python3
import os,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.j_visual_gate import main
if __name__=='__main__':main()
