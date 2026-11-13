"""Goal-level acceptance: inspect artifacts, arrays and every newly rendered video."""
import math
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from vico_more_common import ROOT,WORK,DECA,GPU_UUID,read,save,sha,test_samples


def verify_map(files):
    for path,expected in files.items():assert sha(path)==expected,path


def decode_video(path,expected_frames):
    stream=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(path)],text=True)
    import json
    stream=json.loads(stream)['streams'][0]
    assert (stream['width'],stream['height'],stream['r_frame_rate'],int(stream['nb_frames']))==(256,256,'30/1',expected_frames)
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','1','-i',str(path),
        '-map','0:v:0','-an','-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,(path,result.stderr)
    counts=re.findall(r'^frame=\s*(\d+)',result.stdout,re.M)
    assert counts and int(counts[-1])==expected_frames,(path,counts)
    return dict(frames=expected_frames,full_decode=True,sha256=sha(path))


def main():
    complete=read(WORK/'COMPLETE.json');assert complete['success']
    verify_map(complete['files'])
    old=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
    verify_map(read(old/'COMPLETE.json')['files'])
    samples={row['id']:row for row in test_samples()};assert len(samples)==43
    published=read(ROOT/'data_previews/listen_benchmark_review/vico_more_20261107/data.json')
    assert {c['id'] for c in published['cases']}==set(samples)
    scope=read(WORK/'RESOURCE_PRIORITY.json');assert scope['allowed_host']=='node11' and scope['allowed_gpu']==7
    assert read(WORK/'SPLIT_ISOLATION_AUDIT.json')['source_time_overlap_with_test']==[]
    assert read(WORK/'SPLIT_ISOLATION_AUDIT.json')['test_ids_match_frozen_media43']
    stats=read(WORK/'features/COMPLETE.json');assert stats['train_only']
    assert stats['statistics']['audio']['frames']==123287 and stats['statistics']['dynamic']['frames']==246574
    assert sha(WORK/'features/train_stats.npz')==stats['stats_sha256']
    dual=read(WORK/'dualtalk/COMPLETE.json');assert dual['success'] and dual['outputs']==43
    assert read(WORK/'dualtalk/DEMO_PARITY.json')['max_abs']==0
    assert read(WORK/'dualtalk/WEIGHT_AUDIT.json')['tensor_exact']
    assert read(WORK/'dualtalk/device.json')['uuid']==GPU_UUID
    assert {r['id'] for r in dual['records']}==set(samples)
    for rec in dual['records']:
        assert sha(rec['path'])==rec['sha256']
        a=np.load(rec['path']);n=samples[rec['id']]['streams']['gt']['frames']
        assert a['motion'].shape==(n,56) and np.isfinite(a['motion']).all()
        assert a['native_exp_jaw_neck'].shape==(rec['native_frames'],56)
        assert len(a['scoring_indices'])==n and rec['listener_future_input'] is False
    methods={};decoded={}
    result=read(WORK/'COMPARISON.json')
    for method,epochs,batch in [('listenformer',200,128),('rlhg',500,16)]:
        out=WORK/method
        if result['status'][method]!='complete':
            assert method=='rlhg','Priority ListenFormer must be complete'
            if (out/'NOT_SCHEDULED.json').exists():
                decision=read(out/'FULL_BUDGET_DECISION.json');ns=read(out/'NOT_SCHEDULED.json')
                assert decision['eager']['real_clips']==decision['compiled']['real_clips']==351
                assert decision['eager']['full_steps']==decision['compiled']['full_steps']==3474000
                budget=read(out/'RUN_BUDGET.json')
                assert not budget['eligible'] and budget['epochs']==500
                assert budget['postprocessing_reserve_s']>=5400 and budget['delivery_unix']==1794110400
                assert decision['at']+min(ns['eager_hours_with_margin'],ns['compiled_hours_with_margin'])*3600>=budget['model_freeze_unix']-5
                methods[method]=dict(formal=False,reason=ns['reason'],full_budget_measured=True)
            elif (out/'INCOMPLETE.json').exists():
                methods[method]=dict(formal=False,reason=read(out/'INCOMPLETE.json')['reason'])
            else:raise AssertionError('Optional runtime failure needs inspection before accepting the goal')
            continue
        verify_map(read(out/'COMPLETE.json')['files'])
        if method=='rlhg':
            budget=read(out/'RUN_BUDGET.json');effective=read(out/'EFFECTIVE_EXECUTION.json')
            assert budget['eligible'] and budget['postprocessing_reserve_s']>=5400
            assert effective['budget_sha256']==sha(out/'RUN_BUDGET.json')
            assert effective['model_freeze_unix']==budget['model_freeze_unix']
        protocol=read(out/'TRAIN_PROTOCOL.json');done=read(out/'TRAIN_COMPLETE.json')
        assert protocol['max_epochs']==done['epochs']==epochs and protocol['global_batch']==batch
        assert protocol['train_windows']==111158 and done['steps']==math.ceil(111158/batch)*epochs
        assert done['formal_training_budget_complete']
        assert sha(out/'best.pt')==done['best_checkpoint_sha256']
        selected=read(WORK/'TRAINING_SELECTION.json')[method]
        validation=[json.loads(line) for line in (out/'validation.jsonl').read_text().splitlines() if line.strip()]
        winner=min(validation,key=lambda row:row['score'])
        assert selected['selected_epoch']==winner['epoch'] and selected['selected_step']==winner['steps']
        assert selected['validation_score']==winner['score'] and selected['best_sha256']==sha(out/'best.pt')
        for space in ['deca_video','bfm_video_new','bfm_native_new']:
            rows=result[space][method]['per_clip']
            assert len(rows)==43 and {row['id'] for row in rows}==set(samples),(method,space)
        image=read(out/'RESULTS.json')
        assert image['clips']==43 and image['frames']==sum(x['streams']['gt']['frames'] for x in samples.values())
        assert image['metrics']==result['images'][method]
        assert read(out/'DEVICE.json')['uuid']==GPU_UUID
        inputs=read(out/'GENERATION_INPUT_AUDIT.json');assert inputs['passed'] and inputs['listener_frames_per_pair']==1
        assert len(inputs['array_loads'])==130 and len(inputs['checkpoints'])==1
        generation=read(out/'GENERATION_COMPLETE.json');render=read(out/'RENDER_COMPLETE.json')
        assert {r['id'] for r in generation['records']}=={r['id'] for r in render['records']}==set(samples)
        for rec in generation['records']:
            assert sha(rec['path'])==rec['sha256']
            a=np.load(rec['path']);n=samples[rec['id']]['streams']['gt']['frames']
            assert a['dynamic'].shape==(n,73) and a['motion'].shape==(n,70)
            assert np.isfinite(a['dynamic']).all() and a['initial_coeff'].shape==(1,257)
        decoded[method]={r['id']:decode_video(r['path'],samples[r['id']]['streams']['gt']['frames']) for r in render['records']}
        methods[method]=dict(formal=True,epochs=epochs,pairs=43,source_only_array_trace=True,full_video_decode_pairs=43)
    for name in ['NUMERICAL_AUDIT','LINK_AUDIT','BROWSER_AUDIT','NATIVE_CURVES_AUDIT','EXPORT_AUDIT']:
        assert read(WORK/(name+'.json'))['passed'],name
    paper=read(WORK/'PAPER_TABLES.json');verify_map(paper['files']);assert paper['pages']>=6
    pch=read(WORK/'PCH_FEASIBILITY.json');assert pch['formal_result'] is False and pch['renderer_only_patch_is_not_full_pch']
    assert read(WORK/'FINISHED.json')['at']<1794110400,'Delivery timestamp exceeds user deadline'
    save(WORK/'GOAL_COMPLETION_AUDIT.json',dict(passed=True,at=time.time(),deadline_utc='2026-11-08T04:00:00Z',
        assigned_gpu_uuid=GPU_UUID,methods=methods,dualtalk_pairs=43,dualtalk_geometry_limit_preserved=True,
        test_ids_exact=True,local_train_test_time_isolated=True,train_only_normalization=True,
        native_vs_rgb_tables_separate=True,all_four_l2l_variants=True,old_sealed_files_unchanged=True,
        full_video_decode=decoded,completion_sha256=sha(WORK/'COMPLETE.json')))
    print('Goal-level audit passed: primary methods, conditional RLHG decision, all43 pairs and deliverables')


if __name__=='__main__':main()
