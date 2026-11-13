import copy,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval import media,protocol,scoring
from p2_omni_eval.common import DEFAULTS,read,sha,atomic_json,artifact_valid,complete,subprocess_env,check_device


def manifest(tmp_path):
    f=tmp_path/'video';f.write_bytes(b'video')
    a=tmp_path/'audio';a.write_bytes(b'audio')
    return dict(schema='p2-omni-pairs-v1',samples=[dict(sample_id='a',sample_seed=s,gt_video='video',generated_video='video',gt_audio='audio',generated_audio='audio') for s in (137,251)])


def test_manifest_normalizes_same_ids_and_listener_audio(tmp_path):
    r=protocol.normalize(manifest(tmp_path),tmp_path,['fvd','utmos']);assert len(r)==2
    assert r[0]['hashes']['gt_audio']==sha(tmp_path/'audio')

@pytest.mark.parametrize('kind',['duplicate','seed_type','missing_gen','missing_audio','gt_mismatch','seed_ids','missing_fd_schema','missing_text_source'])
def test_invalid_manifests(tmp_path,kind):
    m=manifest(tmp_path);metrics=['utmos']
    if kind=='duplicate':m['samples'].append(dict(m['samples'][0]))
    if kind=='seed_type':m['samples'][0]['sample_seed']='137'
    if kind=='missing_gen':m['samples'][0].pop('generated_video')
    if kind=='missing_audio':m['samples'][0].pop('generated_audio')
    if kind=='gt_mismatch':m['samples'][0]['gt_text']='different'
    if kind=='seed_ids':m['samples'][0]['sample_id']='other'
    if kind=='missing_fd_schema':metrics=['fd'];[r.update(gt_features='video',generated_features='video') for r in m['samples']]
    if kind=='missing_text_source':metrics=['text']
    with pytest.raises(ValueError):protocol.normalize(m,tmp_path,metrics)


def test_generation_failure_retained(tmp_path):
    m=manifest(tmp_path);r=m['samples'][0];r.pop('generated_video');r.pop('generated_audio');r['generation_error']='OOM'
    rows=protocol.normalize(m,tmp_path,['utmos']);assert rows[0]['generation_error']=='OOM'


def test_fvd_window_shared_across_seed_and_order():
    a=media.window('one',88,DEFAULTS);b=media.window('one',88,dict(DEFAULTS))
    assert a==b and len(a)==32 and a==list(range(a[0],a[0]+32))
    assert media.window('one',31,DEFAULTS) is None
    assert media.window('one',32,DEFAULTS)==list(range(32))
    assert media.window('one',255,dict(DEFAULTS,fvd_frames=256)) is None


def test_geometry_mismatch_is_not_short_eligibility():
    with pytest.raises(ValueError):media.aligned(dict(frame_count=50),dict(frame_count=40))


def test_subset_denominator_not_full_mean():
    rows=[]
    for i,status in enumerate(['success','not_applicable','error']):
        rows.append(dict(sample_id=str(i),sample_seed=137,metrics=dict(utmos=dict(generated=dict(status=status,value=4)))))
    score=scoring.scalar_side(rows,'utmos','generated')
    assert score['full_value'] is None and score['supported_subset_value']==4
    assert score['coverage']==dict(planned=3,eligible=2,succeeded=1,inapplicable=1,failed=1,
        succeeded_ids=[dict(sample_id='0',sample_seed=137)],inapplicable_ids=[dict(sample_id='1',sample_seed=137)],failed_ids=[dict(sample_id='2',sample_seed=137)])


def test_sync_track_mean_differs_from_clip_mean():
    rows=[]
    for i,values in enumerate([[1,3],[8]]):
        tracks=[dict(status='success',lse_d=v,lse_c=v+1,low_support=False) for v in values]
        score=dict(status='success',tracks=tracks,lse_d=float(np.mean(values)),lse_c=float(np.mean(values))+1)
        rows.append(dict(sample_id=str(i),sample_seed=1,metrics=dict(sync=dict(generated=score))))
    result=scoring.scalar_side(rows,'sync','generated')
    assert result['full_value']['lse_d']==4 and result['clip_equal_mean']['lse_d']==5


def test_empty_support_has_null_score():
    rows=[dict(sample_id='x',sample_seed=1,metrics=dict(utmos=dict(generated=dict(status='not_applicable'))))]
    r=scoring.scalar_side(rows,'utmos','generated');assert r['full_value'] is None and r['supported_subset_value'] is None


def test_nonfinite_aggregate_rejected():
    with pytest.raises(ValueError):scoring.finish(float('nan'),dict(planned=1,succeeded=1))


def test_fd_math_matches_reference(tmp_path):
    from p2_omni_eval.common import module
    reference=ROOT.parent/'OmniResponse/evaluation';ref=module('test_fd_reference',reference/'video/fd/compute_fd_features.py')
    a=np.array([[0.,1],[2,3],[5,2]]);b=a+2
    rows=[]
    for i in range(2):
        sides={}
        for side,data in [('gt',a+i),('generated',b+i)]:
            p=tmp_path/f'{i}_{side}.npy';np.save(p,data);sides[side]=dict(status='success',features=str(p))
        rows.append(dict(sample_id=str(i),sample_seed=1,metrics=dict(fd=sides)))
    r=scoring.distributions(rows,'fd',reference,DEFAULTS)
    expected=ref.calculate_frechet_distance(*ref.calculate_activation_statistics(np.concatenate([a,a+1])),*ref.calculate_activation_statistics(np.concatenate([b,b+1])))
    assert r['full_value']==pytest.approx(expected) and r['frame_counts']==dict(gt=6,generated=6)


def test_fvd_math_and_one_sample_support(tmp_path):
    from p2_omni_eval.common import module
    root=ROOT.parent/'OmniResponse/evaluation';ref=module('test_fvd_reference',root/'video/fvd/frechet_video_distance.py')
    rows=[];arrays={s:[] for s in ('gt','generated')}
    for i in range(3):
        sides={}
        for side,delta in [('gt',0),('generated',2)]:
            a=np.array([i,i*i],np.float32)+delta;arrays[side].append(a);p=tmp_path/f'{side}{i}.npy';np.save(p,a)
            sides[side]=dict(status='success',features=str(p))
        rows.append(dict(sample_id=str(i),sample_seed=1,metrics=dict(fvd=sides)))
    stats=[]
    for side in ('gt','generated'):
        st=ref.FeatureStats(3);st.append(np.stack(arrays[side]));stats.extend(st.get_mean_cov())
    assert scoring.distributions(rows,'fvd',root,DEFAULTS)['full_value']==pytest.approx(ref.frechet_distance(*stats))
    assert scoring.distributions(rows[:1],'fvd',root,DEFAULTS)['full_value'] is None


def test_distinct_is_corpus_not_clip_average():
    root=ROOT.parent/'OmniResponse/evaluation';rows=[]
    for i,text in enumerate(['yes yes','yes no']):
        values={k:1 for k in ('meteor','rouge_l','bertscore_p','bertscore_r','bertscore_f1')}
        rows.append(dict(sample_id=str(i),sample_seed=1,texts=dict(generated=text,gt=text),metrics=dict(text=dict(status='success',**values))))
    value=scoring.text_score(rows,root)['full_value'];assert value['generated_distinct1']==.5 and value['generated_distinct2']==1


def test_cached_artifact_corruption_is_not_ignored(tmp_path):
    p=tmp_path/'result.json';atomic_json(p,dict(value=1));identity={'seed':137};complete(tmp_path,identity,[p]);assert artifact_valid(tmp_path,identity)
    p.write_text('{}')
    with pytest.raises(ValueError,match='corrupt'):artifact_valid(tmp_path,identity)


def test_cpu_workers_hide_all_gpus(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    r={'paths':dict(hf_home='/hf',nltk_data='/nltk')};assert subprocess_env(r,'cpu')['CUDA_VISIBLE_DEVICES']==''


def test_gpu_guard_rejects_context_even_if_memory_low(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(subprocess,'check_output',lambda args,**kw:'1\n' if '--query-gpu=memory.used' in args else '1234\n')
    with pytest.raises(RuntimeError,match='another compute process'):check_device('cuda:0')


def test_mux_selects_external_audio_and_has_no_shortest(monkeypatch,tmp_path):
    commands=[];monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:commands.append(cmd))
    media.mux_listener('video_with_wrong_audio.mp4','correct_listener.wav',tmp_path/'out.mkv')
    cmd=commands[0];assert ['-map','0:v:0','-map','1:a:0']==cmd[cmd.index('-map'):cmd.index('-map')+4]
    assert '-shortest' not in cmd


def test_generation_error_never_reads_gt_or_models(tmp_path):
    from p2_omni_eval.pipeline import compute_row
    row=dict(sample_id='x',sample_seed=1,generation_error='failed')
    result=compute_row(row,dict(metrics=['fvd','utmos','sync','text']),object(),tmp_path)
    assert result['metrics']['text']['status']=='error'
    assert result['metrics']['fvd']['generated']['status']=='error'


def test_fd_nonfinite_rejected(tmp_path):
    from p2_omni_eval.backends import Backends
    p=tmp_path/'input.npy';np.save(p,np.array([[float('nan'),1]]))
    with pytest.raises(ValueError):Backends.fd_features(None,p,tmp_path/'out.npy')


def test_audio_duration_mismatch_is_rejected(tmp_path):
    import soundfile as sf
    f=tmp_path/'short.wav';sf.write(f,np.ones(1600)*.1,16000)
    with pytest.raises(ValueError,match='duration mismatch'):media.audio(f,tmp_path/'normalized.wav',2.)


def test_external_listener_audio_overrides_wrong_embedded_track(tmp_path):
    import soundfile as sf
    rate=16000;t=np.arange(32000)/rate
    source=tmp_path/'wrong.wav';listener=tmp_path/'listener.wav'
    sf.write(source,np.sin(2*np.pi*1000*t)*.1,rate);sf.write(listener,np.sin(2*np.pi*220*t)*.1,rate)
    video=tmp_path/'input.mkv'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=blue:s=64x64:r=25:d=2','-i',str(source),
        '-map','0:v:0','-map','1:a:0','-c:v','libx264','-threads','1','-c:a','pcm_s16le',str(video)],check=True)
    output=tmp_path/'combined.mkv';media.mux_listener(video,listener,output)
    a=media.audio(output,tmp_path/'decoded.wav',2.);wave,_=sf.read(a['path']);freq=np.fft.rfftfreq(len(wave),1/rate)[np.argmax(abs(np.fft.rfft(wave)))]
    assert freq==pytest.approx(220) and media.geometry(video)==media.geometry(output)


def prepared_fixture(tmp_path,monkeypatch):
    m=manifest(tmp_path)
    # Two distinct samples for a well-defined distribution; one generation seed.
    m['samples'][1].update(sample_id='b',sample_seed=137)
    path=tmp_path/'pairs.json';atomic_json(path,m);runtime=tmp_path/'runtime.json'
    atomic_json(runtime,dict(paths=dict(reference_root=str(ROOT.parent/'OmniResponse/evaluation'))))
    monkeypatch.setattr(protocol,'check_runtime',lambda *a,**kw:None)
    protocol.prepare(path,tmp_path/'work',runtime,['fvd'])
    return tmp_path/'work'

@pytest.mark.parametrize('changed',['manifest','media','code'])
def test_frozen_inputs_and_code_rejected(tmp_path,monkeypatch,changed):
    from p2_omni_eval.pipeline import check_protocol
    work=prepared_fixture(tmp_path,monkeypatch);p=read(work/'protocol.json')
    if changed=='manifest':(tmp_path/'pairs.json').write_text('{}')
    elif changed=='media':(tmp_path/'video').write_bytes(b'changed')
    else:p['code']={}
    with pytest.raises(ValueError):check_protocol(p)


def test_worker_reuses_success_and_rejects_corrupt_cache(tmp_path,monkeypatch):
    from p2_omni_eval import pipeline,backends
    work=prepared_fixture(tmp_path,monkeypatch);calls=[]
    class Fake:
        def __init__(self,p,device):pass
        def fvd_features(self,path,indices,dest):
            calls.append(str(path));np.save(dest,np.array([1.,2.]));return dict(status='success',features=str(dest),indices=indices)
    monkeypatch.setattr(backends,'Backends',Fake)
    monkeypatch.setattr(media,'geometry',lambda path:dict(frame_count=50,fps=25,width=64,height=64,duration_s=2.,start_time=0))
    a=pipeline.worker(work,'cpu');b=pipeline.worker(work,'cpu');assert len(calls)==4 and a==b and a['status']=='complete'
    files=list(work.glob('rows/*/attempts/*/generated_i3d.npy'));assert files;files[0].write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):pipeline.worker(work,'cpu')


def test_errors_retry_and_are_not_marked_complete(tmp_path,monkeypatch):
    from p2_omni_eval import pipeline,backends
    work=prepared_fixture(tmp_path,monkeypatch)
    class Broken:
        def __init__(self,p,device):pass
        def fvd_features(self,*args):raise RuntimeError('deliberate feature failure')
    monkeypatch.setattr(backends,'Backends',Broken)
    monkeypatch.setattr(media,'geometry',lambda path:dict(frame_count=50,fps=25,width=64,height=64,duration_s=2.,start_time=0))
    result=pipeline.worker(work,'cpu');assert result['status']=='incomplete'
    assert not list(work.glob('rows/*/complete.json'))
    assert result['per_seed']['137']['fvd']['full_value'] is None


def test_two_seeds_form_separate_pools(tmp_path):
    root=ROOT.parent/'OmniResponse/evaluation';rows=[]
    for seed in (137,251):
        for i in range(2):
            sides={}
            for side in ('gt','generated'):
                f=tmp_path/f'{seed}_{i}_{side}.npy';np.save(f,np.array([i,i*i],np.float32));sides[side]=dict(status='success',features=str(f))
            rows.append(dict(sample_id=str(i),sample_seed=seed,metrics=dict(fvd=sides)))
    p=dict(metrics=['fvd'],options=DEFAULTS,seeds=[137,251],sample_count=2,runtime=dict(paths=dict(reference_root=str(root))))
    result=scoring.aggregate(p,rows)
    assert all(s['fvd']['coverage']['planned']==2 for s in result['per_seed'].values())


def test_compare_requires_same_gt_and_recomputes_common_support(tmp_path,monkeypatch):
    from p2_omni_eval import compare as comparison
    from p2_omni_eval.common import digest,job_key
    root=ROOT.parent/'OmniResponse/evaluation';base=dict(metrics=['utmos'],options=DEFAULTS,seeds=[137],sample_count=2,runtime=dict(paths=dict(reference_root=str(root))),code={})
    paths=[tmp_path/'left',tmp_path/'right']
    for j,path in enumerate(paths):
        path.mkdir();planned=[dict(sample_id=str(i),sample_seed=137,gt_video='same',gt_audio='same') for i in range(2)]
        p=dict(base,rows=planned);atomic_json(path/'protocol.json',p);records=[]
        for i in range(2):
            value=dict(status='success',value=3+j)
            if j==1 and i==1:value=dict(status='not_applicable',reason='silent')
            r=dict(sample_id=str(i),sample_seed=137,metrics=dict(utmos=dict(gt=dict(status='success',value=4),generated=value)))
            records.append(r);dest=path/'rows'/job_key(str(i),137);atomic_json(dest/'result.json',r)
            complete(dest,dict(protocol=digest(p),row=planned[i],device='cpu'),[dest/'result.json'])
        atomic_json(path/'results.json',dict(protocol_sha256=digest(p),device='cpu',rows=records))
    monkeypatch.setattr(comparison,'check_protocol',lambda p:None)
    monkeypatch.setattr(comparison,'check_runtime',lambda p:None)
    result=comparison.compare(*paths,tmp_path/'comparison.json');v=result['metrics']['utmos']['137']
    assert v['common_count']==1 and v['left']['generated']['full_value']==3 and v['right']['generated']['full_value']==4
    p=read(paths[1]/'protocol.json');p['rows'][0]['gt_audio']='other';atomic_json(paths[1]/'protocol.json',p)
    with pytest.raises(ValueError,match='GT/IDs'):comparison.compare(*paths,tmp_path/'comparison2.json')


def test_original_and_preflight_generation_bootstrap():
    script=ROOT/'scripts/p2_eval_generate.py'
    runs=[ROOT/'experiments/p2_e1_train50h_lora_tb_20261026/runs/20261023/E1',ROOT/'experiments/p2_preflight_fix_20261026/production/runs/20261023/E0']
    for run in runs:
        program="import runpy; n=runpy.run_path(%r); n['bootstrap'](['prepare-generation','--run',%r]); from p2_listen.config import source_identity; import json; r=json.load(open(%r)); assert source_identity(r['config'])==r['source']"%(str(script),str(run),str(run/'run.json'))
        subprocess.run([sys.executable,'-c',program],check=True)


def test_small_negative_frechet_keeps_original_and_clamps_only_roundoff():
    stats=(np.zeros(2),np.eye(2)*100,np.zeros(2),np.eye(2)*100)
    value,audit=scoring.numerical_distance(-.00015,stats)
    assert value==0 and audit['raw_reference_value']==-.00015 and audit['negative_roundoff_clamped']
    with pytest.raises(ValueError,match='Unstable'):scoring.numerical_distance(-1.,stats)


def test_export_listener_audio_checks_gt_video_association_and_unique_attempts(tmp_path,monkeypatch):
    from p2_glare_eval import generation
    gt=tmp_path/'gt.mp4';gt.write_bytes(b'gt');wrong=tmp_path/'wrong.mp4';wrong.write_bytes(b'other')
    wave=tmp_path/'listener.wav';wave.write_bytes(b'wave');calls=[]
    def export(work,dest):
        calls.append(dest)
        return dict(samples=[dict(sample_id='x',sample_seed=1,gt_video=str(gt),generated_video=str(tmp_path/'generated.mp4'))])
    monkeypatch.setattr(generation,'export_pairs',export)
    targets=tmp_path/'targets.json';atomic_json(targets,[dict(sample_id='x',target_audio=str(wave),target_video=str(gt))])
    for i in range(2):protocol.export_generation(tmp_path,targets,tmp_path/f'out{i}.json')
    assert len(set(calls))==2
    atomic_json(targets,[dict(sample_id='x',target_audio=str(wave),target_video=str(wrong))])
    with pytest.raises(ValueError,match='different GT video'):protocol.export_generation(tmp_path,targets,tmp_path/'bad.json')
