"""Controlled evaluation-input change: official ViCo256 GT/speaker, same AF output."""
import argparse
import csv
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
PARENT=WORK.parent/'p2_avatarforcing_table4_reference_mean_20261110_v3'
sys.path[:0]=[str(PARENT),str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import read,save,sha,guard


def prepare():
    path=WORK/'PROTOCOL.json'
    code={str(p):sha(p) for p in [Path(__file__),PARENT/'official_spectre.py',PARENT/'official_spectre_workers.py',ROOT/'scripts/p2_avatarforcing_table4_score.py',ROOT/'compare/deployment/vico_literature_metrics.py',ROOT/'compare/Dyadic-Interaction-Modeling/code/metrics/eval_utils.py',ROOT/'compare/Dyadic-Interaction-Modeling/code/mymetrics.py']}
    if path.exists():
        p=read(path);assert p['code']==code
        return p
    canonical=Path('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv')
    metadata={r['uuid']:r for r in csv.DictReader(canonical.open()) if r['data_split']=='test'}
    generation=read(PARENT/'NATIVE_GENERATION_PROTOCOL.json');tasks=[]
    for sample in generation['samples']:
        uid=sample['id'];frames=math.ceil(round(sample['duration']*16000)/640)
        for role,key,person in [('gt','gt_video','listener'),('speaker','speaker_video','speaker')]:
            video=Path(sample[key]);assert video.resolve().stem==metadata[uid][person]
            expected=next(v for k,v in sample['source_sha256'].items() if Path(k).resolve()==video.resolve())
            assert sha(video)==expected
            tasks.append(dict(id=uid,role=role,video=str(video),video_sha256=expected,frames=frames,output=str(WORK/'features'/role/(uid+'.npz'))))
    assert len(tasks)==86 and len(metadata)==43
    p=dict(schema='AF-fixed-generation-canonical-ViCo256-evaluation-v1',created_unix=time.time(),code=code,tasks=tasks,
        parent_generation_protocol_sha256=sha(PARENT/'NATIVE_GENERATION_PROTOCOL.json'),parent_feature_protocol_sha256=sha(PARENT/'OFFICIAL_SPECTRE_PROTOCOL.json'),
        canonical_metadata_sha256=sha(canonical),geometry_evidence_sha256=sha(PARENT/'CANONICAL_CROP_GEOMETRY_AUDIT.json'),
        only_changed='GT and speaker source images use the released ViCo256 videos, instead of original-view512 mean square crops',
        generated_videos_unchanged=True,generated_native_resolution=512,generated_scoring_downsample=False,
        source_native_resolution=256,shared_decoder_upsample='Unchanged V3 decode25 resizes non512 frames to512 with OpenCV cubic before tracking; preserves the released aspect deformation',
        fps=25,full_interval=True,canonical_frames='Same nearest source timestamp rule; keep all43 and original duration',
        feature_frontend='Byte-identical V3 official SPECTRE RetinaFace/FAN2/JPEG/context46/FP32; raw pose6 then exp50',
        primary_recipe=dict(FD='standard Frechet equal-clip mean',rPCC='pooled flattened residual correlation',SID='K15/9 GT-fit pooled entropy bits',Var='pooled temporal feature-mean variance'),
        second_whole_recipe='Entire public DIM math: same FD/rPCC, SID40/20, flattened Var; not per-cell selection',
        selection_basis='Official benchmark supplies256 GT; independent image registration shows median1.317 anisotropic scaling,77of85 valid streams exceed1.10',
        target=generation['target'],no_target_based_formula_sample_seed_selection=True,
        author_AF_scored_GT_source_confirmed=False,private_cluster_centers_available=False,alignment_claim=False)
    save(path,p);return p


def feature_path(task):return Path(task['output'])


def checked(task):
    path=feature_path(task);receipt=path.with_suffix('.json')
    if not receipt.exists():return None
    r=read(receipt)
    assert r['canonical_protocol_sha256']==sha(WORK/'PROTOCOL.json')
    assert all(r[k]==task[k] for k in ['id','role','video','video_sha256','frames'])
    assert sha(path)==r['sha256']
    return r


def worker():
    import numpy as np
    import torch
    import cv2
    import official_spectre_workers as source_worker
    p=prepare();source,model,tracker,get_landmarks=source_worker.load_models()
    state=WORK/'workers';state.mkdir(exist_ok=True)
    for task in p['tasks']:
        if checked(task):continue
        with (state/(task['role']+'_'+task['id']+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if checked(task):continue
            assert sha(task['video'])==task['video_sha256'];started=time.time()
            cap=cv2.VideoCapture(task['video']);native_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));native_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
            assert (native_width,native_height)==(256,256)
            frames,grid=source.decode25(task['video'],task['frames'])
            assert frames.shape[1:3]==(512,512)  # Shared decoder's fixed cubic resize.
            grid['source_resolution']=[native_width,native_height]
            grid['tracking_input_resolution']=[512,512]
            images,landmarks,audit=source.official_crop(frames,tracker,get_landmarks);motion=model.chunks(images,batch=46)
            assert motion.shape==(task['frames'],56) and np.isfinite(motion).all()
            path=feature_path(task);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
            with temp.open('wb') as f:np.savez_compressed(f,motion=motion,landmarks=landmarks,fps=25,frame_indices=np.arange(len(motion)))
            temp.replace(path)
            r=dict(task,sha256=sha(path),canonical_protocol_sha256=sha(WORK/'PROTOCOL.json'),source_feature_protocol_sha256=p['parent_feature_protocol_sha256'],grid=grid,face_audit=audit,seconds=time.time()-started,worker_pid=os.getpid(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            save(path.with_suffix('.json'),r);save(state/f'{os.getpid()}.json',dict(pid=os.getpid(),last=task['role']+'/'+task['id'],updated_unix=time.time()))
            print('canonical',task['role'],task['id'],round(r['seconds'],2),flush=True)
            del frames,images,landmarks,motion


def extract():
    import official_spectre_workers as source_worker
    p=prepare();assert read(PARENT/'FINAL_AUDIT.json')['passed_execution_audit']
    assert read(PARENT/'official_tracker_audit/PARALLEL_PARITY.json')['exact_motion_bytes']
    children=[];state=WORK/'workers';state.mkdir(exist_ok=True)
    with (WORK/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                for child in children:
                    if child.poll() not in [None,0]:raise RuntimeError(f'Canonical worker{child.pid} failed{child.returncode}')
                n=sum(feature_path(t).with_suffix('.json').exists() for t in p['tasks'])
                live=[c for c in children if c.poll() is None]
                if n==86 and not live:
                    records=[checked(t) for t in p['tasks']];assert all(records)
                    save(WORK/'EXTRACTION_COMPLETE.json',dict(passed=True,streams=86,records=records,protocol_sha256=sha(WORK/'PROTOCOL.json')));return
                free,total,used=source_worker.gpu_state();budget=math.ceil(total*.025)+1024
                required=16384+sum(max(0,budget-used.get(c.pid,0)) for c in live)+budget
                if len(children)<4 and n<86 and free>=required:
                    with (state/f'launch_{len(children)}.log').open('a') as log:child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'worker'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                    children.append(child);live.append(child)
                save(WORK/'EXTRACTION_STATUS.json',dict(stage='canonical_extraction',pid=os.getpid(),completed=n,total=86,live_worker_pids=[c.pid for c in live],free_mib=free,required_to_add_mib=required,updated_unix=time.time()))
                time.sleep(5)
        except BaseException:
            for child in children:
                if child.poll() is None:child.terminate()
            raise


def score():
    import numpy as np
    from sklearn.cluster import KMeans
    from p2_avatarforcing_table4_score import metrics
    from vico_literature_metrics import evaluate,parity,load_author
    p=prepare();assert read(PARENT/'FINAL_AUDIT.json')['passed_execution_audit']
    complete=read(WORK/'EXTRACTION_COMPLETE.json');assert complete['passed'] and complete['streams']==86
    data={};records=[]
    for task in p['tasks']:
        r=checked(task);assert sha(task['video'])==task['video_sha256']
        with np.load(task['output']) as d:
            a=d['motion'].copy();assert float(d['fps'])==25 and np.array_equal(d['frame_indices'],np.arange(task['frames']))
            assert d['landmarks'].shape==(task['frames'],68,2) and np.isfinite(d['landmarks']).all()
        assert a.shape==(task['frames'],56) and np.isfinite(a).all()
        grid=r['grid'];expected=np.floor(np.arange(task['frames'])*grid['source_fps']/25+.5).astype(int).clip(0,grid['source_frames']-1);assert np.array_equal(grid['source_indices'],expected)
        data[task['id'],task['role']]=a;records.append(r)
    generation=read(PARENT/'NATIVE_GENERATION_PROTOCOL.json');ids=[s['id'] for s in generation['samples']]
    for uid in ids:
        path=PARENT/'official_features/avatarforcing'/(uid+'.npz');r=read(path.with_suffix('.json'));assert sha(path)==r['sha256'] and sha(r['video'])==r['video_sha256'];assert r['protocol_sha256']==p['parent_feature_protocol_sha256']
        with np.load(path) as d:data[uid,'avatarforcing']=d['motion'].copy()
    gt=[data[u,'gt'] for u in ids];pred=[data[u,'avatarforcing'] for u in ids];sp=[data[u,'speaker'] for u in ids]
    assert len(ids)==43 and len(data)==129
    for g,a,x in zip(gt,pred,sp):assert g.shape==a.shape==x.shape
    result=metrics(gt,pred,sp);author=load_author();independent={m:{} for m in ['FD','rPCC','SID','Var']}
    for group,sl,k in [('exp',slice(6,56),15),('pose',slice(0,6),9)]:
        g,a,x=[np.concatenate(v)[:,sl] for v in [gt,pred,sp]]
        independent['FD'][group]=float(np.mean([author.calculate_frechet_distance(*author.calculate_activation_statistics(gg[:,sl]),*author.calculate_activation_statistics(aa[:,sl])) for gg,aa in zip(gt,pred)]))
        def corr(v,w):
            v=v.ravel().astype(float);w=w.ravel().astype(float);v-=v.mean();w-=w.mean();return float(v@w/math.sqrt((v@v)*(w@w)))
        independent['rPCC'][group]=abs(corr(g,x)-corr(a,x));lab=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(g).predict(a);h=np.bincount(lab,minlength=k).astype(float);h/=h.sum();independent['SID'][group]=float(-h@np.log2(h+1e-6));independent['Var'][group]=float(np.var(a,axis=0).mean())
        for metric in independent:assert abs(independent[metric][group]-result['primary'][metric][group])<1e-6
    dim=evaluate(gt,pred,sp);dim_parity=parity(gt,pred,sp)
    parent_results=read(PARENT/'OFFICIAL_RESULTS.json');parent_dim=read(PARENT/'DIM_STYLE_RESULTS.json')
    for group in ['exp','pose']:
        assert result['primary']['Var'][group] == parent_results['primary']['Var'][group]
        assert dim['metrics']['Var'][group] == parent_dim['metrics']['Var'][group]
    assert sha('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv') == p['canonical_metadata_sha256']
    def cells(values):return [dict(metric=m,group=g,measured=values[m][g],published=p['target'][m][g],same_at_2dp=f'{values[m][g]:.2f}'==f'{p["target"][m][g]:.2f}') for m in ['FD','rPCC','SID','Var'] for g in ['exp','pose']]
    c1=cells(result['primary']);c2=cells(dim['metrics']);n1=sum(c['same_at_2dp'] for c in c1);n2=sum(c['same_at_2dp'] for c in c2)
    save(WORK/'RESULTS.json',dict(primary=result,primary_comparison=c1,primary_matched=n1,DIM_whole_recipe=dim,DIM_comparison=c2,DIM_matched=n2,DIM_original_print_parity=dim_parity,canonical_protocol_sha256=sha(WORK/'PROTOCOL.json'),parent_generation_unchanged=True))
    save(WORK/'FINAL_AUDIT.json',dict(passed_execution_audit=True,canonical_streams=86,reused_generated_streams=43,clips=43,frames=sum(map(len,gt)),independent_primary=independent,matched_cells_primary=n1,matched_cells_DIM=n2,all8_numerically_matched=n1==8 or n2==8,goal_complete_proven=False,remaining='Author GT source, private cluster centers and checkpoint/run binding remain unconfirmed; evaluate any numerical match against complete goal requirements',parent_final_audit_sha256=sha(PARENT/'FINAL_AUDIT.json'),results_sha256=sha(WORK/'RESULTS.json')))
    lines=['# 相同AF输出：使用ViCo官方256真值／speaker的完整对照','',f'43条，{sum(map(len,gt))}帧；主口径{n1}/8，完整DIM代码候选{n2}/8。生成视频和生成特征保持V3不变，未按单元格混合公式。','', '| 指标 | 分组 | 主口径 | DIM完整口径 | Table4 |','|---|---|---:|---:|---:|']
    for a,b in zip(c1,c2):lines.append(f"| {a['metric']} | {a['group']} | {a['measured']:.6f} | {b['measured']:.6f} | {a['published']:.2f} |")
    lines+=['','本对照只改变评测GT／speaker图像来源，不改变生成、生成图像分辨率或评分公式。官方GT256包含非等比缩放，不能仅当作宽视野512的等比缩小。AF是否实际采用这套GT输入仍未公开；不能仅因数值接近就认定作者设置。',''];(WORK/'RESULTS.md').write_text('\n'.join(lines));print('Canonical complete',n1,n2,'of8',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','extract','score']);a=parser.parse_args();globals()[a.action]()
