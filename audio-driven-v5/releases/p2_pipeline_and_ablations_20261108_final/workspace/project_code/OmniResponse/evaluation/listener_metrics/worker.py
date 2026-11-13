"""Persistent subprocess models in existing isolated environments; JSON protocol."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]/'audio-driven-v5'
sys.path.insert(0,str(ROOT/'src'))


class GlareWorker:
    def __init__(self):
        from p2_glare_rgb.detector import Engine
        self.engine=Engine()
    def run(self,req):
        from p2_glare_eval.common import sha,read
        from p2_glare_eval.upstream import metrics
        folder=Path(req['output']);self.engine.detect(req['video'],folder,sha(req['video']))
        completion=read(folder/'complete.json');quality=read(folder/'quality.json')
        result=dict(quality=quality,valid=completion.get('valid',False),identity=self.engine.identity(sha(req['video'])))
        if result['valid']:
            ann=metrics()['common'].read_annotation(folder/'scores.csv')
            result['events']=[dict(reaction=e.reaction,start=e.start,end=e.end) for e in ann.events]
        return result


class AudioWorker:
    def __init__(self,device="cpu"):
        self.device=device
        from p2_omni_eval.common import read,DEFAULTS
        from p2_omni_eval.backends import Backends
        self.runtime=read(ROOT/'configs/p2_omni_runtime.lock.json')
        self.backend=Backends(dict(runtime=self.runtime,options=DEFAULTS),device)
        import importlib.metadata
        self.identity=dict(device=device,runtime_lock=str(ROOT/'configs/p2_omni_runtime.lock.json'),
            versions={n:importlib.metadata.version(n) for n in ['torch','numpy','librosa','scipy']})
        self.verified=False
    def run(self,req):
        import numpy as np
        import soundfile as sf
        from p2_omni_eval.common import sha
        if not self.verified:
            # Check pinned weights used by these metrics, not unrelated text/I3D assets.
            paths=self.runtime['paths'];wanted=[paths['utmos_checkpoint'],str(Path(paths['sync_root'])/'data/syncnet_v2.model'),str(Path(paths['sync_root'])/'detectors/s3fd/weights/sfd_face.pth')]
            for p in wanted:
                if sha(p)!=self.runtime['models'][p]:raise ValueError('Pinned audio metric weight changed: '+p)
            self.identity['weights']={p:sha(p) for p in wanted}
            self.identity['source']={str(p):sha(p) for p in (ROOT/'src/p2_omni_eval').glob('*.py')}
            self.identity['utmos_source']={str(p):sha(p) for p in Path(paths['utmos_code']).joinpath('utmosv2').rglob('*.py')}
            self.verified=True
        kind=req['kind'];info={}
        for side in ('gt','generated'):
            wave,sr=sf.read(req[side+'_audio'],dtype='float32');info[side]=dict(path=req[side+'_audio'],rms=float(np.sqrt(np.mean(wave.astype(float)**2))))
        if kind=='utmos':
            import librosa
            for side in info:
                wave,sr=sf.read(info[side]['path'],dtype='float32')
                trimmed,_=librosa.effects.trim(wave,top_db=30)
                # Upstream SSL branch tiles when <=3 seconds. Do not invoke it then.
                if len(trimmed)<=3*sr:return dict(status='not_applicable',reason=side+'_too_short_after_30db_trim',trimmed_samples=len(trimmed))
                sf.write(info[side]['path'],trimmed,sr,subtype='FLOAT')
            scores={}
            model=self.backend.model('utmos')
            if float(model._cfg.dataset.ssl.duration)!=3:raise ValueError('UTMOS input duration contract changed')
            for side in info:
                self.backend.seed()
                value=float(model.predict(input_path=info[side]['path'],device=self.device,verbose=False,remove_silent_section=False))
                if not np.isfinite(value):raise ValueError('Nonfinite MOS')
                scores[side]=dict(status='success',value=value,preprocessing='30dB edge trim; >3s; upstream internal silence removal disabled; no tile')
            return dict(status='success',scores=scores,identity=self.identity)
        if kind!='sync':raise ValueError('Unknown audio worker task')
        scores={side:self.backend.sync(req[side+'_video'],info[side],Path(req['output'])/side) for side in info}
        valid=all(s['status']=='success' and not s.get('low_support',True) for s in scores.values())
        return dict(status='success' if valid else 'not_applicable',reason=None if valid else 'no_common_supported_face_tracks',scores=scores,identity=self.identity)


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['glare','audio'],required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
    engine=None
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if engine is None:engine=GlareWorker() if args.kind=='glare' else AudioWorker(args.device)
            result=engine.run(request)
            reply=dict(ok=True,result=result)
        except Exception as exc:
            traceback.print_exc();reply=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
        print('LISTENER_RESULT '+json.dumps(reply,allow_nan=False),flush=True)


if __name__=='__main__':main()
