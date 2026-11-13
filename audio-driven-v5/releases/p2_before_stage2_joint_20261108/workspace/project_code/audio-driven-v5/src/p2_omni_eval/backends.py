"""Original metric functions/models with explicit inputs and device selection."""
from pathlib import Path
import math
import os
import random
import subprocess
import sys
import numpy as np
from .common import module,read,atomic_json,subprocess_env,ROOT
from .media import mux_listener


def ok(**values):return dict(status='success',**values)
def unavailable(reason,**values):return dict(status='not_applicable',reason=reason,**values)


class Backends:
    def __init__(self,protocol,device):
        self.p=protocol;self.runtime=protocol['runtime'];self.a=self.runtime['paths'];self.options=protocol['options'];self.device=device;self.models={}
        os.environ.update(subprocess_env(self.runtime,device))
        import torch
        torch.set_num_threads(2)
        self.torch=torch
        root=Path(self.a['reference_root'])
        self.fvd=module('p2_reference_fvd',root/'video/fvd/frechet_video_distance.py')
        self.fd=module('p2_reference_fd',root/'video/fd/compute_fd_features.py')

    def seed(self):
        seed=self.options['metric_seed'];random.seed(seed);np.random.seed(seed);self.torch.manual_seed(seed)

    def model(self,name):
        if name not in self.models:
            if name=='i3d':self.models[name]=self.fvd.load_torchscript_detector(self.a['i3d'],self.torch.device(self.device))
            elif name=='utmos':
                sys.path.insert(0,self.a['utmos_code']);import utmosv2
                self.models[name]=utmosv2.create_model(pretrained=True,checkpoint_path=Path(self.a['utmos_checkpoint']),device=self.device)
            elif name=='bert':
                from bert_score import BERTScorer
                self.models[name]=BERTScorer(model_type=self.a['bert_model'],num_layers=self.options['bert_layers'],device=self.device,
                    rescale_with_baseline=False,batch_size=8,lang='en')
        return self.models[name]

    def fvd_features(self,path,indices,dest):
        import cv2
        if indices is None:return unavailable('GT_shorter_than_FVD_window')
        capture=cv2.VideoCapture(str(path));frames=[];wanted=set(indices);i=0
        try:
            while True:
                success,bgr=capture.read()
                if not success:break
                if i in wanted:frames.append(self.torch.from_numpy(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
                i+=1
        finally:capture.release()
        if len(frames)!=len(indices):raise ValueError('Generated clip cannot cover frozen GT window')
        x=self.fvd.batch_to_bcthw_uint8([self.torch.stack(frames)],resize_hw=(224,224)).to(self.device)
        with self.torch.inference_mode():features=self.model('i3d')(x,rescale=True,resize=False,return_features=True)
        values=features.detach().cpu().numpy()
        if values.shape!=(1,400) and (values.ndim!=2 or values.shape[0]!=1):raise ValueError('Unexpected I3D feature shape')
        if not np.isfinite(values).all():raise ValueError('Nonfinite I3D features')
        np.save(dest,values[0]);return ok(features=str(dest),dimension=values.shape[1],indices=indices)

    def fd_features(self,path,dest):
        a=np.load(path,allow_pickle=False)
        if a.ndim!=2 or len(a)<1 or a.shape[1]<1 or not np.isfinite(a).all():raise ValueError('FD needs finite nonempty T x C features')
        np.save(dest,a);return ok(features=str(dest),frames=len(a),dimension=a.shape[1])

    def mos(self,audio):
        if audio['rms']<self.options['silence_rms']:return unavailable('silent_listener_audio')
        self.seed();model=self.model('utmos');self.seed()
        value=float(model.predict(input_path=audio['path'],device=self.device,verbose=False))
        if not math.isfinite(value):raise ValueError('Nonfinite UTMOSv2')
        return ok(value=value,model_preprocessing='original UTMOSv2 trim/tile/TTA; no adapter padding')

    def sync(self,video,audio,dest):
        if audio['rms']<self.options['silence_rms']:return unavailable('silent_listener_audio',tracks=[])
        dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
        combined=dest/'listener.mkv';mux_listener(video,audio['path'],combined)
        request=dict(video=str(combined),output=str(dest/'result.json'),runtime=self.runtime,options=self.options,device=self.device)
        atomic_json(dest/'request.json',request)
        with (dest/'worker.log').open('w') as log:
            subprocess.run([self.a['sync_python'],str(ROOT/'scripts/p2_omni_eval.py'),'sync-worker','--request',str(dest/'request.json')],
                env=subprocess_env(self.runtime,self.device),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        return read(dest/'result.json')

    def text(self,generated,gt):
        if not generated or not generated.strip() or not gt or not gt.strip():return unavailable('empty_or_missing_text')
        import nltk
        # Refuse the upstream tokenizer's implicit fallback to a different protocol.
        nltk.word_tokenize('resource check')
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        root=Path(self.a['reference_root'])/'text'
        meteor=module('p2_reference_meteor',root/'compute_meteor.py')
        rouge=module('p2_reference_rouge',root/'compute_rouge-l.py')
        m=meteor.meteor_score([meteor.safe_word_tokenize(gt)],meteor.safe_word_tokenize(generated))
        r=rouge.Rouge().pair_score(generated,gt)
        self.seed();precision,recall,f1=self.model('bert').score([generated],[gt])
        values=dict(meteor=float(m),rouge_l=float(r),bertscore_p=float(precision[0]),bertscore_r=float(recall[0]),bertscore_f1=float(f1[0]))
        if not all(math.isfinite(x) for x in values.values()):raise ValueError('Nonfinite text scores')
        return ok(**values)
