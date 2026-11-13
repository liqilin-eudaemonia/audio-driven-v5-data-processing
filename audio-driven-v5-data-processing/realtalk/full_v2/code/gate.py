import numpy as np
SR=16000;HOP=160;WIN=320
def runs(mask):
 edges=np.diff(np.r_[False,mask,False].astype('int8'));return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
def make_gain(a,setting):
 threshold=setting['threshold_dbfs'];bridge=round(setting['bridge_low_gap_ms']*SR/1000);pre=round(setting['pre_roll_ms']*SR/1000);post=round(setting['post_roll_ms']*SR/1000)
 n=len(a);centers=np.arange(0,n,HOP);left=np.maximum(0,centers-WIN//2);right=np.minimum(n,centers+WIN//2);cs=np.r_[0,np.cumsum(a.astype('float64')**2)];rms=np.sqrt((cs[right]-cs[left])/(right-left));hot=rms>=10**(threshold/20)
 # Close only short internal low-energy gaps; never discard a short positive response.
 for start,end in runs(~hot):
  if start>0 and end<len(hot) and (end-start)*HOP<=bridge:hot[start:end]=True
 keep=np.zeros(n,dtype=bool)
 for start,end in runs(hot):keep[max(0,start*HOP-pre):min(n,end*HOP+post)]=True
 gain=keep.astype('float64');fade=320
 for start,end in runs(keep):
  lo=max(0,start-fade);hi=min(n,end+fade)
  if start>lo:gain[lo:start]=np.maximum(gain[lo:start],.5-.5*np.cos(np.pi*(np.arange(lo,start)-(start-fade))/fade))
  if hi>end:gain[end:hi]=np.maximum(gain[end:hi],.5+.5*np.cos(np.pi*(np.arange(end,hi)-end)/fade))
 return gain,rms
