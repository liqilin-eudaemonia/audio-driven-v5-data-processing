"""Evaluation contract tests; no GPU and no trained event-head predictions."""
import csv
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_glare_eval.common import atomic_json, read, complete, artifact_valid, code_identity, digest, sha, job_key
from p2_glare_eval.upstream import metrics, REACTIONS
from p2_glare_eval.pipeline import prepare_pairs, evaluate, load_protocol
from p2_glare_eval.scoring import aggregate_seeds
from p2_glare_eval.media import probe


def csv_file(path, values):
    with path.open('w', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame_index',*REACTIONS])
        for i,value in enumerate(values):
            writer.writerow([i,*([value,0,0,0,0,0] if not isinstance(value,list) else value)])
    return path


def annotation(events=(), count=50):
    shared=metrics()['common']
    return shared.Annotation(count,tuple(shared.Event(*e) for e in events))


def test_threshold_and_neutral_gap(tmp_path):
    a=metrics()['common'].read_annotation(csv_file(tmp_path/'a.csv',[.5,.6,.6,0,.9]))
    assert [(e.start,e.end) for e in a.events]==[(1,3),(4,5)]


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,1.1,[.2,.2,0,0,0,0]])
def test_invalid_scores_rejected(tmp_path,value):
    with pytest.raises(ValueError):metrics()['common'].read_annotation(csv_file(tmp_path/'a.csv',[value]))


def test_f1_dataset_class_counts():
    one=annotation([('smiling',0,10)]);two=annotation([('smiling',0,10),('smiling',20,30)])
    empty=annotation()
    # TP=1 FN=2, giving smile F1=.5 then six-class mean. Per-clip mean would differ.
    pairs=[('a',one,one),('b',two,empty)]
    assert metrics()['r_f1'].compute_r_f1(pairs)==pytest.approx(.5/6)
    assert metrics()['r_f1'].compute_r_f1([('a',one,one)])==pytest.approx(1/6)


def test_no_match_conventions():
    pairs=[('a',annotation([('smiling',0,10)]),annotation([('nodding',0,10)]))]
    assert metrics()['r_f1'].compute_r_f1(pairs)==0
    assert metrics()['r_tiou'].compute_r_tiou(pairs)==0
    assert metrics()['r_atd'].compute_r_atd(pairs) is None


@pytest.mark.parametrize('pred,expected',[(('smiling',8,18),80),(('smiling',12,22),20),(('smiling',10,16),160)])
def test_atd_asymmetry_sum_scale(pred,expected):
    pairs=[('a',annotation([('smiling',10,20)]),annotation([pred]))]
    assert metrics()['r_atd'].compute_r_atd(pairs)==pytest.approx(expected)


def test_greedy_one_to_one_and_threshold():
    s=metrics()['common'];e=s.Event
    matches=s.match_events((e('smiling',0,10),e('smiling',10,20)),(e('smiling',0,20),e('nodding',0,10)))
    assert len(matches)==1 and matches[0].tiou==.5 and matches[0].ground_truth.start==0
    assert not s.match_events((e('smiling',0,10),),(e('smiling',0,21),))


def test_fid_moments_and_distance():
    f=metrics()['r_fid'];data=np.random.default_rng(9).normal(size=(30,3))
    a=f.FeatureMoments(3);b=f.FeatureMoments(3)
    a.update(data[:7]);a.update(data[7:]);b.update(data)
    mean,cov=a.statistics()
    np.testing.assert_allclose(mean,data.mean(0),atol=1e-14)
    np.testing.assert_allclose(cov,np.cov(data,rowvar=False),atol=1e-14)
    assert f.frechet_distance(a,b)==pytest.approx(0,abs=1e-12)
    with pytest.raises(ValueError):f.FeatureMoments(3).statistics()
    with pytest.raises(ValueError):a.update([[float('nan')]*3])


def test_seed_null_propagates():
    r=aggregate_seeds({'1':{'metrics':dict(zip(('R-F1','R-tIoU','R-ATD','R-FID'),(1,1,0,None)))},
                       '2':{'metrics':dict(zip(('R-F1','R-tIoU','R-ATD','R-FID'),(0,0,None,2)))}})
    assert r['mean']=={'R-F1':.5,'R-tIoU':.5,'R-ATD':None,'R-FID':None}
    assert r['valid_seed_counts']['R-FID']==1


def test_detector_missing_features_are_not_neutral():
    from p2_glare_eval.detector import valid_face,valid_emotion,EMOTIONS,BLENDSHAPES
    face=dict(detected=True,landmarks=[[0.,0.,0.]]*478,bbox_px=[0,0,10,10],pitch=0.,roll=0.,yaw=0.,
              blendshapes=dict.fromkeys(BLENDSHAPES,.1),au04_proxy=.1)
    assert valid_face(face)
    del face['pitch'];assert not valid_face(face)
    emotion=[dict(box=[0,0,10,10],emotions=dict.fromkeys(EMOTIONS,.1))]
    assert valid_emotion(emotion)
    assert not valid_emotion(emotion*2)
    del emotion[0]['emotions']['happy'];assert not valid_emotion(emotion)


def test_channel_missing_frame(tmp_path):
    from p2_glare_eval.detector import read_channel
    p=tmp_path/'scores.csv';p.write_text('frame_index,smiling_intensity\n0,0\n2,0\n')
    with pytest.raises(ValueError):read_channel(p,'smiling',3)


def test_artifact_identity_and_corruption(tmp_path):
    p=tmp_path/'out';p.write_text('original');complete(tmp_path,{'seed':1},[p])
    assert artifact_valid(tmp_path,{'seed':1})
    with pytest.raises(ValueError):artifact_valid(tmp_path,{'seed':2})
    p.write_text('changed')
    with pytest.raises(ValueError):artifact_valid(tmp_path,{'seed':1})


@pytest.fixture
def video(tmp_path):
    p=tmp_path/'v.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=black:s=64x64:r=25:d=2',
                    '-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(p)],check=True)
    return p


def pair_fixture(tmp_path,video):
    csv_path=csv_file(tmp_path/'zero.csv',[0]*50)
    row=dict(sample_id='fixture',sample_seed=137,gt_video=str(video),generated_video=str(video),
             gt_csv=str(csv_path),generated_csv=str(csv_path))
    manifest=tmp_path/'pairs.json';atomic_json(manifest,dict(schema='p2-glare-pairs-v1',
        annotation_provenance='synthetic engineering fixture',samples=[row]))
    return manifest,row


def test_video_probe_and_pts(video,tmp_path):
    assert probe(video)['frame_count']==50
    shifted=tmp_path/'shifted.mp4'
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-c','copy','-output_ts_offset','0.4',str(shifted)],check=True)
    with pytest.raises(ValueError,match='PTS'):probe(shifted)


def test_complete_neutral_pipeline(video,tmp_path):
    m,_=pair_fixture(tmp_path,video);work=tmp_path/'work';prepare_pairs(m,work)
    report=evaluate(work)
    assert report['status']=='complete'
    assert report['metrics']=={'R-F1':0,'R-tIoU':0,'R-ATD':None,'R-FID':None}
    assert evaluate(work)==report


def test_missing_generation_is_not_dropped(video,tmp_path):
    m,row=pair_fixture(tmp_path,video);row.pop('generated_video');row['generation_error']='CUDA failure'
    data=read(m);data['samples']=[row];atomic_json(m,data)
    work=tmp_path/'work';prepare_pairs(m,work);r=evaluate(work)
    assert r['status']=='incomplete' and r['failed_pairs']==1 and r['metrics'] is None


@pytest.mark.parametrize('mutation',['duplicates','seeds','provenance','probe'])
def test_bad_manifest(video,tmp_path,mutation):
    m,row=pair_fixture(tmp_path,video);data=read(m)
    if mutation=='duplicates':data['samples'].append(row)
    if mutation=='seeds':data['samples'].append(dict(row,sample_id='other',sample_seed=251))
    if mutation=='provenance':data['annotation_provenance']=''
    if mutation=='probe':data['annotation_provenance']='training_head probe.pt'
    atomic_json(m,data)
    with pytest.raises(ValueError):prepare_pairs(m,tmp_path/'work')


def test_source_change_rejected(video,tmp_path):
    m,row=pair_fixture(tmp_path,video);work=tmp_path/'work';prepare_pairs(m,work)
    Path(row['gt_csv']).write_text('changed')
    with pytest.raises(ValueError,match='changed'):load_protocol(work)


def test_csv_video_length_failure(video,tmp_path):
    m,row=pair_fixture(tmp_path,video);csv_file(Path(row['gt_csv']),[0]*49)
    work=tmp_path/'work';prepare_pairs(m,work);r=evaluate(work)
    assert r['status']=='incomplete' and r['metrics'] is None


def test_fid_pools_include_own_unmatched_frames(video):
    f=metrics()['r_fid'];gt=annotation([('smiling',0,10)]);gen=annotation([('nodding',20,40)])
    records=[('fixture',(gt,video),(gen,video))]
    assert f.collect_statistics(records,1,None,'cpu',8,25)[1]==10
    assert f.collect_statistics(records,2,None,'cpu',8,25)[1]==20


def test_generation_checkpoint_binding_reuse_and_no_gt(tmp_path,monkeypatch):
    from p2_glare_eval import generation as g
    checkpoint=tmp_path/'last.pt';checkpoint.write_bytes(b'checkpoint')
    atomic_json(checkpoint.with_suffix('.complete.json'),{})
    run=tmp_path/'run';atomic_json(run/'status.json',dict(step=5,stopped=False))
    ident={'id':'trained'};obs=dict(sample_id='test-fixture',experiment_split='test',speaker_audio='speaker.wav')
    p=dict(schema='p2-glare-generation-v1',code=code_identity(),source_files={},run_identity_sha256=digest(ident),
           budget=5,run=str(run),config={'model':'fixture'},observations=[obs],seeds=[137],source={},assets={})
    work=tmp_path/'work';atomic_json(work/'generation_protocol.json',p)
    # No evaluation_targets.json exists: worker must be source-only.
    calls=[]
    def native(config,observation,output,**kw):
        assert observation==obs and kw['seed']==137
        assert read(kw['release'])['checkpoint_sha256']==[sha(checkpoint)]
        output.mkdir(parents=True,exist_ok=False)
        for name in ('generated.mp4','listener.wav','completed.json'):(output/name).write_text('test fixture')
        calls.append(output)
    monkeypatch.setitem(sys.modules,'p2_listen.training',types.SimpleNamespace(checked_checkpoint=lambda path:dict(identity=ident,step=5)))
    monkeypatch.setitem(sys.modules,'p2_listen.generation',types.SimpleNamespace(generate=native))
    monkeypatch.setattr(g,'check_idle_device',lambda device:None)
    assert g.generate(work,checkpoint)['results'][0]['status']=='success'
    assert g.generate(work,checkpoint)['results'][0]['reused'] is True
    assert len(calls)==1
    checkpoint.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError,match='identity differs'):g.generate(work,checkpoint)


def test_export_preserves_generation_failure(tmp_path):
    from p2_glare_eval.generation import export_pairs
    gt=tmp_path/'gt.mp4';gt.write_bytes(b'fixture')
    targets=[dict(sample_id='x',gt_video=str(gt),sha256=sha(gt))]
    p=dict(code=code_identity(),evaluation_targets_digest=digest(targets),run_identity_sha256='run',
           seeds=[1,2],observations=[dict(sample_id='x')])
    atomic_json(tmp_path/'generation_protocol.json',p);atomic_json(tmp_path/'evaluation_targets.json',targets)
    atomic_json(tmp_path/'selected_checkpoint.json',dict(run_identity_sha256='run',sha256='checkpoint'))
    for seed in (1,2):atomic_json(tmp_path/'generation_failures'/(job_key('x',seed)+'.json'),dict(error='explicit failure'))
    assert len(export_pairs(tmp_path,tmp_path/'pairs.json')['samples'])==2
    gt.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):export_pairs(tmp_path,tmp_path/'pairs.json')


@pytest.mark.parametrize('violation',[None,'train_split','group_overlap','target_duration','target_hash'])
def test_generation_prepare_isolation_and_geometry(video,tmp_path,monkeypatch,violation):
    from p2_glare_eval.generation import prepare
    import p2_listen.config
    monkeypatch.setattr(p2_listen.config,'source_identity',lambda c:{})
    speaker=tmp_path/'speaker.wav';speaker.write_bytes(b'input audio')
    first=tmp_path/'first.png';first.write_bytes(b'input image')
    obs=dict(schema='p2-listen-observation-v1',sample_id='dev-sample',experiment_split='dev',official_split='dev',
             group_id='dev-group',grouping_ids=['participant:dev','session:dev'],speaker_audio=str(speaker),listener_image=str(first),
             requested_duration_s=1.99,geometry=dict(fps=25,width=64,height=64),
             media_hashes=dict(speaker_audio=sha(speaker),listener_image=sha(first)))
    target=dict(sample_id='dev-sample',gt_video=str(video))
    if violation=='train_split':obs['experiment_split']='train'
    if violation=='group_overlap':obs['grouping_ids']=['participant:train']
    if violation=='target_duration':obs['requested_duration_s']=2.05
    if violation=='target_hash':target['media_hashes']={'target_video':'changed'}
    dataset=tmp_path/'dataset';atomic_json(dataset/'manifests/observations.json',[
        dict(sample_id='train-sample',group_id='train-group',grouping_ids=['participant:train'])])
    run=tmp_path/'run';atomic_json(run/'run.json',dict(config={'output':str(dataset)},source={},s0=False,assets={},budget=5))
    observations=tmp_path/'obs.json';targets=tmp_path/'targets.json'
    atomic_json(observations,[obs]);atomic_json(targets,[target])
    if violation:
        with pytest.raises(ValueError):prepare(run,observations,targets,tmp_path/'work')
    else:
        assert prepare(run,observations,targets,tmp_path/'work')['jobs']==2
        protocol=read(tmp_path/'work/generation_protocol.json')
        assert str(video) not in str(protocol)
        assert protocol['observations'][0]['requested_duration_s']==1.99

@pytest.mark.parametrize('change',[None,'media','split'])
def test_explicit_train_diagnostic_preserves_exact_original_observation(video,tmp_path,monkeypatch,change):
    from p2_glare_eval.generation import prepare
    import p2_listen.config
    monkeypatch.setattr(p2_listen.config,'source_identity',lambda c:{})
    audio=tmp_path/'speaker.wav';audio.write_bytes(b'input');image=tmp_path/'first.png';image.write_bytes(b'first')
    obs=dict(schema='p2-listen-observation-v1',sample_id='train-sample',experiment_split='train',official_split='train',
             group_id='train-group',grouping_ids=['participant:train'],speaker_audio=str(audio),listener_image=str(image),
             requested_duration_s=1.99,geometry=dict(fps=25,width=64,height=64),media_hashes=dict(speaker_audio=sha(audio),listener_image=sha(image)))
    dataset=tmp_path/'dataset';atomic_json(dataset/'manifests/observations.json',[obs])
    run=tmp_path/'run';atomic_json(run/'run.json',dict(config={'output':str(dataset)},source={},s0=False,assets={},budget=5))
    if change=='media':obs=dict(obs,group_id='different')
    if change=='split':obs=dict(obs,experiment_split='dev')
    source=tmp_path/'obs.json';targets=tmp_path/'targets.json';atomic_json(source,[obs]);atomic_json(targets,[dict(sample_id='train-sample',gt_video=str(video))])
    if change:
        with pytest.raises(ValueError,match='unchanged'):prepare(run,source,targets,tmp_path/'work',train_diagnostic=True)
    else:
        prepare(run,source,targets,tmp_path/'work',train_diagnostic=True)
        assert read(tmp_path/'work/generation_protocol.json')['evaluation_split']=='train_diagnostic'
        with pytest.raises(ValueError):prepare(run,source,targets,tmp_path/'formal')
