from boundary_core import *
from pathlib import Path
import tempfile

def run():
 raw=np.zeros(250,int);scores=np.tile([2.,-1.],(250,1));scores[150:,1]=3.;scores[180:,0]=-1
 r,e,_=refine_roles(raw,scores,[0,250],np.zeros(250,bool),250)
 assert (r[:150]==0).all() and (r[150:180]==-3).all() and (r[180:]==1).all(),spans(r)
 raw=np.zeros(250,int);raw[100:105]=1;scores=np.tile([2.,-1.],(250,1));scores[100:105,1]=3
 r,e,_=refine_roles(raw,scores,[0,250],np.zeros(250,bool),250);assert (r==0).all();assert any(x['type']=='short_return_backchannel_candidate' for x in e)
 raw=np.zeros(250,int);raw[100:120]=-1;scores=np.tile([2.,-1.],(250,1));scores[100:120]=np.nan;r,_,_=refine_roles(raw,scores,[0,250],np.zeros(250,bool),250);assert (r[100:120]==-1).all()
 raw=np.zeros(250,int);raw[100:110]=-1;scores=np.tile([2.,-1.],(250,1));r,e,_=refine_roles(raw,scores,[0,250],np.zeros(250,bool),250);assert (r==0).all()
 black=np.r_[np.zeros(25,bool),np.ones(25,bool)];r,_,_=refine_roles(np.zeros(50,int),np.tile([2.,-1.],(50,1)),[0,25,50],black,49);assert (r[25:]==-2).all()
 r,_,_=refine_roles(np.zeros(50,int),np.tile([2.,-1.],(50,1)),[0,50],np.zeros(50,bool),49);assert r[-1]==-4
 boxes=np.tile(np.array([[10.,10.,30.,30.],[60.,10.,80.,30.]]),(20,1,1));boxes[10,0]+=40;gray=np.zeros((20,10,10));h,rev=reconcile_geometry(boxes,[10,11],[],gray,100,100);assert h==[]
 boxes=np.tile(np.array([[10.,10.,30.,30.],[60.,10.,80.,30.]]),(20,1,1));boxes[10:,0]+=40;gray[10:]=255;h,rev=reconcile_geometry(boxes,[10],[],gray,100,100);assert h==[10]
 h,rev=reconcile_geometry(boxes,[10],[11],gray,100,100);assert h==[11]
 words=[{'start':0,'end':10,'text':'Sentence.'},{'start':10.1,'end':20,'text':'Next.'}];bounds,meta=partition(0,500,words,np.ones(500));assert bounds==[0,250,500],bounds
 bounds,meta=partition(0,900,[],np.ones(900));assert bounds[0]==0 and bounds[-1]==900 and max(np.diff(bounds))<=275
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'code';p.write_text('a');a=fingerprint([p],{'input':'x'});p.write_text('b');assert fingerprint([p],{'input':'x'})!=a;assert fingerprint([p],{'input':'y'})!=fingerprint([p],{'input':'x'})
 print('12 boundary contract checks passed')
if __name__=='__main__':run()
