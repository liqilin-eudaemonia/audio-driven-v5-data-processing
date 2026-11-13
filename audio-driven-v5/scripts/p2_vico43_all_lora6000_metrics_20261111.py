"""Existing ViCo BFM and DIM/EMOCA metrics, prediction-only extraction on GPU7."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_testsets_eval.protocol import read,save,sha
from p2_vico43_all_lora6000_gpu7_20261111 import ARM,WORK,MPY,MORE,UUID,gpu,link

BFM=ROOT/'experiments/p2_vico_motion8000_20261106'
EMOCA=ROOT/'experiments/p2_vico_literature_protocol_20261108'
BFMPY=ROOT/'compare/.envs/vico/bin/python'


def prepare(work):
    dest=work/'METRICS_PROTOCOL.json'
    if dest.exists():
        p=read(dest)
        for path,value in p['code'].items():assert sha(path)==value,path
        return p
    assert read(work/'GENERATION_COMPLETE.json')['outputs']==43
    p=read(work/'protocol.json');items=read(work/'shards/00.json');byid={x['id']:x for x in items}
    reference=read(BFM/'protocol.json');bfm=work/'bfm';bfm.mkdir(exist_ok=True)
    from vico_unified_evaluate import bfm_source_versions
    bfm_sources,bfm_provenance=bfm_source_versions()
    for path,value in reference['assets'].items():assert sha(path)==value,path
    reference['source_sha256']=bfm_sources
    reference['prior_source_versions']=bfm_provenance
    tasks=[];reused=[]
    for sample in reference['samples']:
        uid=sample['id'];pred=work/'scoring'/ARM/(uid+'.listener.mp4')
        assert pred.exists() and sha(pred)==read(work/'conversion'/(uid+'.json'))['file_sha256']
        for role in ['gt','speaker']:
            parent=BFM/'features'/role/(uid+'.npz');receipt=read(parent.with_suffix('.json'))
            assert sha(parent)==receipt['feature_sha256']
            assert receipt['input_sha256']==sample['streams'][role]['sha256']==sha(sample['streams'][role]['path'])
            reused.append(dict(id=uid,role=role,feature=str(parent),feature_sha256=receipt['feature_sha256'],input_sha256=receipt['input_sha256']))
        sample['streams']['prediction']=dict(sample['streams']['prediction'],path=str(pred),sha256=sha(pred))
        assert sample['streams']['prediction']['frames']==byid[uid]['geometry']['gt_frames']
        tasks.append(dict(id=uid,role='prediction',video=str(pred),video_sha256=sha(pred),frames=byid[uid]['geometry']['gt_frames']))
    reference.update(parent_generation=str(work),parent_files={n:sha(work/n) for n in ['protocol.json','IMAGE_RESULTS.json']},
        model='All attention/FFN LoRA6000 + SFT2752',reused_reference_features_parent_protocol_sha256=sha(BFM/'protocol.json'))
    save(bfm/'protocol.json',reference)
    for role in ['gt','speaker']:link(BFM/'features'/role,bfm/'features'/role)
    save(bfm/'REUSED_REFERENCE_FEATURES.json',dict(passed=True,streams=86,records=reused))
    save(bfm/'SCORING_PROTOCOL.json',dict(extraction_protocol_sha256=sha(bfm/'protocol.json'),
        implementation_sha256={str(ROOT/'scripts/p2_vico_motion_metrics.py'):sha(ROOT/'scripts/p2_vico_motion_metrics.py')},
        shared_existing_parameters=reference['metric_parameters']))
    save(bfm/'tasks.json',tasks)
    original=read(EMOCA/'protocol.json');groups=[[] for _ in range(4)];loads=[0]*4
    for task in sorted(tasks,key=lambda x:(-x['frames'],x['id'])):
        i=min(range(4),key=lambda j:(loads[j],j));groups[i].append(task);loads[i]+=task['frames']
    for i,group in enumerate(groups):
        ework=work/'emoca'/f'shard_{i:02d}';ework.mkdir(parents=True,exist_ok=True)
        link(EMOCA/'assets',ework/'assets')
        (ework/'LICENSE_CONFIRMATION.json').write_bytes((EMOCA/'LICENSE_CONFIRMATION.json').read_bytes())
        ep=copy.deepcopy(original);ep.update(tasks=[dict(id=t['id'],role=ARM,video=t['video'],video_sha256=t['video_sha256'],expected_frames=t['frames']) for t in group],
            selected_methods=[ARM],resource='node11 physicalGPU7 only',parent_generation_protocol_sha256=sha(work/'protocol.json'),
            parent_extraction_protocol_sha256=sha(EMOCA/'EXTRACTION_PROTOCOL.json'))
        ep['source_files'][str(SELF)]=sha(SELF);save(ework/'protocol.json',ep)
    code=[SELF,ROOT/'scripts/p2_vico_motion_extract.py',ROOT/'scripts/p2_vico_motion_metrics.py',
        ROOT/'compare/deployment/vico_literature_emoca.py',ROOT/'compare/deployment/vico_literature_metrics.py']
    result=dict(schema='same-ViCo-BFM-and-DIM-EMOCA-on-new-predictions-v1',created_unix=time.time(),code={str(f):sha(f) for f in code},
        generation_protocol_sha256=sha(work/'protocol.json'),BFM_reference_protocol_sha256=sha(BFM/'protocol.json'),
        EMOCA_reference_protocol_sha256=sha(EMOCA/'EXTRACTION_PROTOCOL.json'),
        BFM_recipe='Same original BFM exp64/angle3/trans3; FD,RTLCC,RWTLCC,FID_delta_fm,SND,V-D; raw coefficients,lag60,window120,stride30,equal clips',
        DIM_recipe='Same EMOCA_v2_lr_mse_20,pose6+exp50; listener1:T and speaker0:T-1; source DIM FD/P-FD/MSE/SID/rPCC; requested coefficientwise pooled Var',
        DIM_shared_valid_mask='Finite GT/speaker/new prediction plus the two sealed joint/base and SFT2752 reference streams; report every original ID and exact frame indices',
        DIM_released_flattened_Var_also_retained=True,single_sample_seed=137,oracle_selection=False,
        official_test43=True,AF_SPECTRE_requested=False,author_private_protocol_alignment_claim=False)
    save(dest,result);return result


def worker(work,kind,slot):
    import numpy as np
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    prepare(work);free,total=gpu();assert free>=24*1024
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(8192/total)
    if kind=='emoca':
        import vico_literature_emoca as source
        source.WORK=work/'emoca'/f'shard_{slot:02d}'
        source.run();return
    assert kind=='bfm'
    import p2_vico_motion_extract as source
    bfm=work/'bfm';tasks=read(bfm/'tasks.json')[slot::4];extractor=source.Extractor()
    records=[]
    for task in tasks:
        target=bfm/'features/prediction'/(task['id']+'.npz');receipt=target.with_suffix('.json')
        if receipt.exists():
            r=read(receipt);assert r['input_sha256']==task['video_sha256'] and sha(target)==r['feature_sha256'];records.append(r);continue
        assert sha(task['video'])==task['video_sha256'];frames=source.load_frames(task['video']);assert len(frames)==task['frames']
        arrays,timing,faces=extractor.extract(frames)
        if not records:save(bfm/f'BATCH_PARITY_{slot:02d}.json',extractor.parity(frames,arrays,faces))
        for key in ['coeff','motion','landmarks','transform_params']:assert np.isfinite(arrays[key]).all()
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temp.replace(target)
        r=dict(id=task['id'],role='prediction',frames=len(frames),feature=str(target),feature_sha256=sha(target),input_sha256=task['video_sha256'],
            protocol_sha256=sha(bfm/'protocol.json'),extractor_source_sha256=sha(ROOT/'scripts/p2_vico_motion_extract.py'),
            missing_face_frames=int((~arrays['detected']).sum()),multi_face_frames=int((arrays['face_counts']>1).sum()),seconds=timing)
        save(receipt,r);records.append(r)
        save(bfm/f'worker_{slot:02d}.json',dict(pid=os.getpid(),completed=len(records),total=len(tasks),last_id=task['id'],at=time.time()))
    save(bfm/f'EXTRACTION_COMPLETE_{slot:02d}.json',dict(passed=True,streams=len(records),records=records))


def extract(work,kind):
    active={};pending=list(range(4));done=set()
    try:
        while active or pending:
            for slot,(proc,log) in list(active.items()):
                if proc.poll() is not None:
                    if proc.returncode:raise RuntimeError(f'{kind} shard{slot} failed: {log}')
                    done.add(slot);del active[slot]
            free,total=gpu();budget=9*1024
            text=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            used={int(a):int(b) for line in text.splitlines() if len(v:=line.replace(' ','').split(','))==2 and (a:=v[0]).isdigit() and (b:=v[1]).isdigit()}
            required=16384+budget+sum(max(0,budget-used.get(p.pid,0)) for p,_ in active.values())
            # The existing EMOCA guard requires32GiB visible on entry.
            if kind=='emoca':required=max(required,34*1024)
            if pending and free>=required:
                slot=pending.pop(0);log=work/'logs'/f'{kind}_{slot:02d}.log';log.parent.mkdir(exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONPATH='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
                with log.open('a') as stream:proc=subprocess.Popen([str(BFMPY if kind=='bfm' else MORE),'-u',str(SELF),'worker','--work',str(work),'--kind',kind,'--slot',str(slot)],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[slot]=(proc,log)
                save(work/'processes'/f'{kind}_{slot:02d}.json',dict(pid=proc.pid,stage=kind,log=str(log),GPU_UUID=UUID,physical_gpu=7))
            save(work/'MOTION_STATUS.json',dict(stage=kind,pid=os.getpid(),pending=pending,active_pids=[p.pid for p,_ in active.values()],finished_shards=sorted(done),at=time.time()))
            time.sleep(10)
    except BaseException:
        for proc,_ in active.values():
            if proc.poll() is None:proc.terminate()
        raise


def score_DIM(work):
    import numpy as np
    from vico_literature_metrics import evaluate,parity
    p=read(EMOCA/'protocol.json');ids=p['ids'];predictions={}
    for slot in range(4):
        ework=work/'emoca'/f'shard_{slot:02d}';done=read(ework/'EXTRACTION_COMPLETE.json');assert done['success']
        for record in done['records']:
            task=next(t for t in read(ework/'protocol.json')['tasks'] if t['id']==record['id'])
            assert sha(record['path'])==record['sha256']
            assert task['video_sha256']==record['video_sha256']==sha(task['video'])
            predictions[record['id']]=record
    assert set(predictions)==set(ids) and len(ids)==43
    streams={key:[] for key in ['gt','speaker','ours','previous_sft2752']};coverage=[]
    for uid in ids:
        values={}
        for role in ['gt','speaker','joint','joint_sft2752']:
            path=EMOCA/'features'/role/(uid+'.npz');receipt=read(path.with_suffix('.json'));assert sha(path)==receipt['sha256']
            with np.load(path) as data:values[role]=data['motion'].copy()
        with np.load(predictions[uid]['path']) as data:values['ours']=data['motion'].copy()
        assert len({a.shape for a in values.values()})==1
        listener={k:v[1:] for k,v in values.items() if k!='speaker'};speaker=values['speaker'][:-1]
        valid=np.isfinite(speaker).all(1)
        for v in listener.values():valid &= np.isfinite(v).all(1)
        assert valid.sum()>=2,uid
        streams['gt'].append(listener['gt'][valid]);streams['speaker'].append(speaker[valid])
        streams['ours'].append(listener['ours'][valid]);streams['previous_sft2752'].append(listener['joint_sft2752'][valid])
        coverage.append(dict(id=uid,candidate_frames=len(valid),valid_frames=int(valid.sum()),
            listener_frame_indices=(np.where(valid)[0]+1).tolist(),speaker_frame_indices=np.where(valid)[0].tolist(),excluded_listener_frames=(np.where(~valid)[0]+1).tolist()))
    outputs={}
    for name in ['ours','previous_sft2752']:
        source=evaluate(streams['gt'],streams[name],streams['speaker'])
        primary=copy.deepcopy(source['metrics']);a=np.concatenate(streams[name]);g=np.concatenate(streams['gt'])
        for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]:
            primary['Var'][group]=float(np.var(a[:,sl],axis=0,ddof=0).mean())
            primary['Var_GT'][group]=float(np.var(g[:,sl],axis=0,ddof=0).mean())
        outputs[name]=dict(primary=primary,published_code_formula=source,Var_definition='mean_j Var_t(x[t,j]),ddof0,all common valid frames pooled')
    proof=parity(streams['gt'],streams['ours'],streams['speaker'])
    result=dict(passed=True,clips=43,frames=sum(x['valid_frames'] for x in coverage),models=outputs,coverage=coverage,
        current_model=read(work/'protocol.json')['arms'][0],
        reference_model=dict(name='Joint8000 text-LoRA + SFT2752',step=8000,
            source_work=str(ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'),
            feature_protocol_sha256=sha(EMOCA/'EXTRACTION_PROTOCOL.json'),single_variable_ablation=False),
        original_code_parity=proof,single_seed=137,oracle_selection=False,strict_paper_reproduction=False,
        primary_variance_user_requested=True,protocol_sha256=sha(work/'METRICS_PROTOCOL.json'))
    save(work/'DIM_RESULTS.json',result)


def finish(work):
    import cv2
    import numpy as np
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json');items=read(work/'shards/00.json');audits=[]
    assert read(work/'MODEL_ACCEPTANCE.json')['targets']==1632
    for item in items:
        uid=item['id'];g=item['geometry'];folder=work/'generated'/ARM/item['key'];meta=read(folder/'metadata.json')
        assert ready(folder) and meta['step']==6000 and meta['seed']==137
        assert meta['checkpoint_sha256']==p['arms'][0]['original_sha256'] and meta['planning_adapter']==p['planner_adapter_identity']
        assert meta['generation']==p['generation'] and not meta['listener_future_input'] and not meta['target_description_access']
        conversion=read(work/'conversion'/(uid+'.json'));score=work/'scoring'/ARM/(uid+'.listener.mp4');assert sha(score)==conversion['file_sha256']
        counts=[]
        for path,size,fps,count in [(folder/'generated.mp4',512,25,g['frames']),(score,256,30,g['gt_frames'])]:
            cap=cv2.VideoCapture(str(path));assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-6
            n=0
            while True:
                okay,frame=cap.read()
                if not okay:break
                assert frame.shape==(size,size,3);n+=1
            cap.release();assert n==count,(uid,path,n,count);counts.append(n)
        r=read(work/'bfm/features/prediction'/(uid+'.json'))
        assert r['input_sha256']==sha(score) and sha(r['feature'])==r['feature_sha256']
        audits.append(dict(id=uid,native_frames=counts[0],scoring_frames=counts[1]))
    image=read(work/'IMAGE_RESULTS.json');motion=read(work/'bfm/RESULTS.json');dim=read(work/'DIM_RESULTS.json')
    assert image['clips']==motion['clips']==dim['clips']==43
    # The reused official wrapper has a historical8000-step display label.
    # Keep its numeric results and publish the corrected model identity.
    save(work/'RESULTS.json',image)
    result=dict(dataset='ViCo official test43',model='All attention/FFN LoRA6000 + SFT2752',checkpoint=p['arms'][0]['original_checkpoint'],
        checkpoint_sha256=p['arms'][0]['original_sha256'],seed=137,official_image_identity5=image,
        CustomListener_public_definitions=motion,DIM_EMOCA12=dim,protocol_sha256=sha(work/'protocol.json'),AF_SPECTRE_requested=False)
    save(work/'ALL_RESULTS.json',result)
    lines=['# 全attention／FFN LoRA6000：ViCo正式test43','',
        '43条已生成并完成官方画质／身份5项、CustomListener风格运动补测、DIM表情／姿态12项。',
        'SFT2752、seed137、Euler30、CFG3／7；原生512×512／25fps，抗混叠缩小到256×256并按最近时间戳转30fps评分。','',
        '| 官方画质／身份 | 实测 |','|---|---:|']
    for key,value in image['metrics'].items():lines.append(f'| {key} | {value:.6f} |')
    lines+=['','| 运动指标 | 分组 | 实测（原展示倍率） | 覆盖条数 |','|---|---|---:|---:|']
    for name in ['FD','RTLCC','RWTLCC','FID_delta_fm','SND','V_D']:
        for group,value in motion['metrics_table_scale'][name].items():lines.append(f"| {name} | {group} | {value:.6f} | {motion['coverage_clips'][name][group]} |")
    lines+=['','| DIM口径 | Exp | Pose |','|---|---:|---:|']
    for name in ['FD','P_FD','MSE','SID','Var','rPCC']:
        v=dim['models']['ours']['primary'][name];lines.append(f"| {name} | {v['exp']:.6f} | {v['pose']:.6f} |")
    lines+=['','DIM主Var沿用用户指定的逐系数总体方差平均，原发布代码拉平方差另存。所有指标使用单次seed137；DIM逐帧共同有效范围和旧SFT2752基线同覆盖重算结果见JSON。','',
        '运动补测属于已确认公开定义复现，DIM属于公开代码口径；这些结果不等同于严格复现论文私有执行配置，AF的SPECTRE口径不在本轮范围。','',
        '[完整JSON](ALL_RESULTS.json) · [图像结果](IMAGE_RESULTS.json) · [运动结果](bfm/RESULTS.json) · [DIM结果及有效帧](DIM_RESULTS.json)','']
    (work/'RESULTS.md').write_text('\n'.join(lines))
    save(work/'FINAL_AUDIT.json',dict(passed=True,clips=43,decoded_videos=86,rows=audits,strict_LoRA_targets=1632,
        source_only=True,DIM_common_coverage_frames=dim['frames'],metrics_protocol_sha256=sha(work/'METRICS_PROTOCOL.json'),
        all_results_sha256=sha(work/'ALL_RESULTS.json'),at=time.time()))


def run(work):
    prepare(work)
    for kind in ['bfm','emoca']:
        marker=work/(kind.upper()+'_EXTRACTION_COMPLETE.json')
        if marker.exists():continue
        extract(work,kind);save(marker,dict(passed=True,clips=43,at=time.time()))
    import p2_vico_motion_metrics as source
    motion=source.run(work/'bfm');motion['model']='All attention/FFN LoRA6000 + SFT2752';save(work/'bfm/RESULTS.json',motion)
    score_DIM(work);finish(work)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','worker','score','finish'])
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--kind',choices=['bfm','emoca']);parser.add_argument('--slot',type=int,default=0)
    a=parser.parse_args()
    if a.action=='run':run(a.work)
    elif a.action=='worker':worker(a.work,a.kind,a.slot)
    elif a.action=='score':score_DIM(a.work)
    else:finish(a.work)
