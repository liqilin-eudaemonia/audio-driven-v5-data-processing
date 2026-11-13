"""Freeze reusable full test inputs without changing any model or historical result."""
from pathlib import Path
import sys,json,time,hashlib
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'compare/deployment'))
from vico_more_common import read,sha,save

def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def prepare():
    quicks=dict(responsenet=R/'experiments/p2_responsenet_random200_twelveway_20261105/references/references.jsonl',seamless=R/'experiments/seamless_quick200_latest_20261105/prepared_200/references.jsonl')
    oldfolders=dict(responsenet=R/'experiments/p2_compare_responsenet_random200_20261106/dystream',seamless=R/'experiments/p2_compare_seamless200_20261106/dystream')
    response=R/'experiments/glare_final_vico43_responsenet586_20261112/protocol.json'
    seamless=R/'experiments/glare_final_seamless_test6624_20261112/protocol.json'
    srcs=dict(responsenet=response,seamless=seamless);samples=[];sources={};reused=[]
    for dataset,source in srcs.items():
        p=read(source);quick={r['id']:r for r in rows(quicks[dataset])};assert len(quick)==200
        specs=[s for s in p['samples'] if s['dataset']==dataset]
        assert len(specs)==(586 if dataset=='responsenet' else 6624)
        for s in specs:
            prepared=read(s['ready'])['sample'] if dataset=='seamless' else s
            for key in ['audio','reference']:assert sha(prepared[key])==prepared[key+'_sha256'],(s['key'],key)
            audit=read(prepared['source_input_audit']);assert audit['source_only'] and not audit['listener_future_media_in_model_input']
            ref=s['item']['reference'];evaluation=quick.get(s['id'],ref)
            if s['id'] in quick and dataset=='responsenet':
                for key in ['gt','speaker','media','duration_s','listening_intervals']:assert ref[key]==evaluation[key],(s['id'],key)
            if s['id'] in quick and dataset=='seamless':
                a,b=ref['gt_reference'],evaluation['gt_reference']
                for key in (['row_sha256','crop_config_sha256','reader_contract'] if dataset=='seamless' else []):assert a[key]==b[key]
            sample=dict(id=s['id'],key=s['key'],dataset=dataset,quick200=s['id'] in quick,reference=prepared['reference'],reference_sha256=prepared['reference_sha256'],audio=prepared['audio'],audio_sha256=prepared['audio_sha256'],duration=s['duration'],frames=s['frames'],fps=s['fps'],size=512,source_input_audit=prepared['source_input_audit'],source_input_audit_sha256=sha(prepared['source_input_audit']),evaluation_reference=evaluation,input_reference=ref)
            samples.append(sample)
        old=oldfolders[dataset];op=read(old/'protocol.json');assert len(op['samples'])==200
        current={s['id']:s for s in samples if s['dataset']==dataset}
        for s in op['samples']:
            x=current[s['id']];receipt=read(old/'generated'/s['id']/'complete.json');video=old/'scoring/dystream'/(s['id']+'.listener.mp4')
            assert sha(s['image'])==s['image_sha256']==x['reference_sha256']
            # Different WAV headers are allowed only after comparing actual float32 waveforms in input verification.
            assert sha(video)==receipt['score_sha256'] and sha(old/'generated'/s['id']/'generated.mp4')==receipt['native_sha256']
            assert s['frames']==x['frames'] and abs(s['fps']-x['fps'])<1e-8
            reused.append(dict(key=x['key'],dataset=dataset,id=s['id'],video=str(video),video_sha256=receipt['score_sha256'],old_audio=s['speaker'],new_audio=x['audio'],old_waveform_sha256=s['speaker_waveform_sha256'],receipt=str(old/'generated'/s['id']/'complete.json'),receipt_sha256=sha(old/'generated'/s['id']/'complete.json'),old_protocol=str(old/'protocol.json'),old_protocol_sha256=sha(old/'protocol.json')))
        sources[dataset]=dict(input_protocol=str(source),input_protocol_sha256=sha(source),quick200=str(quicks[dataset]),quick200_sha256=sha(quicks[dataset]),metric_assets=p['metric_assets'],metrics_gt_cache=p['metrics_gt_cache'],face_padding=64 if dataset=='responsenet' else 0,old_dystream=str(old),historical_results_sha256=sha(old/'RESULTS.json'))
    samples.sort(key=lambda s:(not s['quick200'],s['dataset'],s['id']))
    p=dict(schema='reference-methods-responsenet586-seamless6624-v1',seed=137,scope='full test with separate fixed200 subsets',methods=['dystream','avatarforcing','listenformer','l2l_conan','l2l_fallon','l2l_stephen','l2l_trevor','dualtalk'],sources=sources,samples=samples,resource=dict(host='172.29.189.119',gpu=7,uuid='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'),audio_metrics='N/A: no generated listener waveform')
    target=W/'protocol.json'
    if target.exists():assert read(target)==p
    else:save(target,p)
    save(W/'DY_REUSE_CANDIDATES.json',dict(records=reused,requires_waveform_verification=True))
    save(W/'INPUT_COMPLETE.json',dict(passed=True,total=len(samples),responsenet=586,seamless=6624,fixed200_each=True,protocol_sha256=sha(target),source_only=True,at=time.time()))
    print('Frozen',len(samples),'verified source-only inputs;',len(reused),'DyStream reuse candidates',flush=True)
if __name__=='__main__':prepare()
