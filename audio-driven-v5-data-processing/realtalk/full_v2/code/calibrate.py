from pathlib import Path
import json,numpy as np,soundfile as sf
R=Path(__file__).resolve().parents[1];SR=16000;HOP=160;WIN=320

def rms(a):
 centers=np.arange(0,len(a),HOP);left=np.maximum(0,centers-160);right=np.minimum(len(a),centers+160);cs=np.r_[0,np.cumsum(a.astype('float64')**2)];return np.sqrt(np.maximum(0,(cs[right]-cs[left])/(right-left)))
def db(a):return 20*np.log10(np.maximum(a,1e-12))
def scale(a):return max(float(np.quantile(rms(a),.90)),1e-10)
