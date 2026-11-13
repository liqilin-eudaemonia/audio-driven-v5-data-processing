import copy,json,sqlite3,sys,threading
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_richness.production import contract,messages,tokenized,statistic_record,validation_error
from listen_richness.prepare_cluster import database,pack_source,flush
from listen_richness.worker_cluster import save_chunk,load_records
from listen_richness.merge_cluster import merge
from listen_benchmark.j_annotation import JobQueue,read,write
from listen_richness.common import digest
from analyze_listen_richness_pilot import aggregate,report


class Tokenizer:
    def apply_chat_template(self,m,**kwargs):return [1,2,3]


def snapshot(sid='a',start=0):
    return dict(sample_id=sid,caption='She maintains a smile. She nods twice.',caption_file='original.json',caption_file_sha256='h',
      crop_revision='verified',crop_provenance=None,quality={},exclusion_reasons=[],
      source=dict(sample_id=sid,pair='pair',listener_id='person',split='train',group_id='group',duration_s=8.,source_start_s=start,source_end_s=start+8),
      face=dict(face_short_side_p10=300.,face_short_side_p50=320.),
      audio=dict(waveform_full_silence=True,official_core_s=0.,official_bout_count=0),
      nod=dict(candidate_interval_count=0,support_s=0.))


def payload():
    def b(name,sid,i,form='change',count=None):
        return dict(id=i,behavior=name,modality='visual',polarity='positive',form=form,source_ids=[sid],
                    repeat=dict(kind='explicit' if count else 'not_stated',count=count,source_ids=[sid] if count else []))
    return dict(behaviors=[b('smiling','s001','b1','state'),b('head nodding','s002','b2',count=2)],relations=[])


def attempts(p=None):return [dict(attempt=1,raw=json.dumps(p or payload()),finish_reason='stop',error=None)]


def test_contract_is_exactly_adopted_v3_not_current_default_v4():
    old=read(ROOT/'experiments/listen_richness_gpu7_pilot_20261031/pilot50_v3/protocol.json')['extraction_contract']
    assert contract()==old
    assert digest(messages('caption')[0]['content'])==old['system_sha256']


def test_unparseable_generation_is_retained_as_unknown_not_zero():
    req=tokenized(snapshot(),Tokenizer());r=statistic_record(req,[dict(attempt=1,raw='broken',error='bad')],'attempt.jsonl')
    assert r['retained'] and r['counted_type_count'] is None and r['record_status']=='unresolved_extraction'
    assert r['caption']==req['snapshot']['caption']


def test_local_reference_failure_keeps_record_and_original_payload():
    p=payload();p['behaviors'][1]['source_ids']=['s001'];old=copy.deepcopy(p)
    err=validation_error(json.dumps(p),snapshot()['caption'],'stop')
    assert err
    r=statistic_record(tokenized(snapshot(),Tokenizer()),[dict(attempt=1,raw=json.dumps(p),error=err)],'a')
    assert r['retained'] and r['record_status']=='recovered_with_review'
    assert p==old and r['known_occurrences_by_type']=={'visual:head nodding':2}


def test_chunk_resume_and_streaming_merge_match_pilot_semantics(tmp_path):
    w=tmp_path;db=database(w)
    ss=[snapshot('a',0),snapshot('b',5),snapshot('c',20)]
    for s in ss:
        s['source']['pair']=s['source']['pair']
        jid=s['sample_id'];db.execute("INSERT INTO sources(jid,status) VALUES (?,'pending')",(jid,));db.commit()
        pack_source(db,[dict(sample_id=s['sample_id'],job_id=jid,source=s['source'],status='caption_ready',snapshot=s)],Tokenizer())
    flush(db,w,2,True)
    assert db.execute('SELECT COUNT(*) FROM coverage').fetchone()[0]==3
    assert db.execute('SELECT COUNT(*) FROM pending').fetchone()[0]==0
    q=JobQueue(w,'extract');records=[]
    while claim:=q.claim('cpu-test'):
        jid,h=claim;j=read(w/'jobs'/f'{jid}.json');rs=load_records(j['path'],j['sha256']);records.extend(aggregate(r['snapshot'],payload()) for r in rs)
        rec=save_chunk(w,jid,rs,[attempts() for _ in rs],{},[],'cpu',0);q.finish(jid,receipt=rec);h.close()
    expected=report(records);db.close();out=merge(w)
    assert out['sample_count']==3 and out['source_overlap_pairs']==1
    cat={c['type']:c for c in out['catalog']}
    for key,c in expected['catalog'].items():
        assert cat[key]['sample_count']==c['sample_count']
        assert cat[key]['supported_occurrences_sum']==c['supported_occurrences_sum']
        assert cat[key]['unknown_occurrence_samples']==c['unknown_occurrence_samples']
    assert merge(w)['sample_count']==3  # idempotent chunk receipts, no doubled counts


def test_changed_packed_input_rejected(tmp_path):
    p=tmp_path/'data';p.write_text(json.dumps(tokenized(snapshot(),Tokenizer()))+'\n')
    with pytest.raises(ValueError,match='Packed input changed'):load_records(p,'wrong')


def test_source_batch_transaction_and_final_flush(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from listen_richness.prepare_cluster import prepare
    w=tmp_path/'work';u=tmp_path/'upstream';w.mkdir();u.mkdir()
    write(w/'config.json',dict(upstream=str(u),prepare_workers=2,chunk_size=512))
    (u/'ready.jsonl').write_text(''.join(json.dumps(dict(job_id=str(i)))+'\n' for i in range(8)))
    for i in range(8):write(u/'caption/completed'/f'{i}.json',{})
    write(u/'producer_complete.json',{})
    def source(_,jid):
        s=snapshot(jid,int(jid)*10)
        return [dict(sample_id=jid,job_id=jid,source=s['source'],status='caption_ready',snapshot=s)]
    monkeypatch.setattr('listen_richness.prepare_cluster.source_records',source)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    prepare(w)
    assert read(w/'producer_complete.json')['packed_chunks']==1
    with database(w) as db:assert db.execute('SELECT COUNT(*) FROM coverage').fetchone()[0]==8


def test_context_overflow_never_silently_truncated():
    class Long:
        def apply_chat_template(self,*a,**kw):return [1]*5000
    with pytest.raises(ValueError,match='Context overflow'):tokenized(snapshot(),Long())


def test_cached_schema_is_identical_and_keeps_open_behavior_names():
    from listen_richness.worker_cluster import schema_for_count
    from listen_richness.extraction import sentences,output_schema
    for text in ['She nods. Then smiles.','At 1.5 seconds she nods. She blinks.','He is quiet.']:
        assert schema_for_count(len(sentences(text)))==output_schema(text)


def test_sample_grid_tolerance_keeps_original_times_and_rejects_real_mismatch(tmp_path):
    from test_listen_richness import fixture
    from listen_richness.statistics import snapshot as make_snapshot
    from listen_richness.common import file_hash
    r,p,m=fixture(tmp_path);r['source']['source_end_s']+=.000001
    m['source']=r['source'];m['source_sha256']=digest(r['source'],ascii=True)
    mp=Path(r['visual_quality']['measurement_file']);write(mp,m)
    r['visual_quality'].update(source_sha256=m['source_sha256'],measurement_sha256=file_hash(mp))
    with pytest.raises(ValueError,match='duration'):make_snapshot(r,p,'hash')
    s=make_snapshot(r,p,'hash',sample_grid_tolerance=True)
    assert s['source']==r['source']
    r['source']['source_end_s']+=.1
    with pytest.raises(ValueError,match='duration'):make_snapshot(r,p,'hash',sample_grid_tolerance=True)


def test_real_pilot_50_all_retained_and_quantities_unchanged():
    p=ROOT/'experiments/listen_richness_gpu7_pilot_20261031'
    with sqlite3.connect(f'file:{p}/stats50_v4.sqlite?mode=ro',uri=True) as db:
        snapshots={sid:json.loads(s) for sid,s in db.execute('SELECT sid,snapshot FROM samples')}
    review=[json.loads(x) for x in (p/'review50.jsonl').read_text().splitlines()]
    got=[];expected=[]
    for r in review:
        s=snapshots[r['sample_id']];v=r['versions']['v3'];p0=v['payload']
        a=dict(attempt=v['attempt'],raw=json.dumps(p0),finish_reason='stop',error=v['validation_error'])
        out=statistic_record(tokenized(s,Tokenizer()),[a],'existing-v3')
        base=aggregate(s,p0)
        for key in ('counted_types','behavior_quantities','known_occurrences_by_type','evidence_ledger','caption'):
            assert out[key]==base[key]
        assert out['retained'];got.append(out);expected.append(base)
    assert len(got)==50 and report(got)['catalog']==report(expected)['catalog']


def test_cluster_process_identity_accepts_legacy_string_ticks():
    import ast, os
    from pathlib import Path
    import listen_richness_cluster_control as control
    tree=ast.parse(control.REMOTE_COMMON)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('ticks','alive')]
    namespace={'Path':Path}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<process identity>','exec'),namespace)
    ticks=namespace['ticks'](os.getpid())
    assert namespace['alive']({'pid':os.getpid(),'start_ticks':ticks})
    assert namespace['alive']({'pid':os.getpid(),'start_ticks':str(ticks)})
    assert not namespace['alive']({'pid':os.getpid(),'start_ticks':ticks+1})
    assert not namespace['alive'](None)


def test_handoff_waits_for_failed_deferred_and_inflight_sources(tmp_path):
    import listen_richness_cluster_control as c
    u=tmp_path/'upstream';u.mkdir();w=tmp_path/'work';w.mkdir()
    write(w/'config.json',dict(upstream=str(u)))
    (u/'ready.jsonl').write_text('source\n');write(u/'producer_complete.json',{})
    clean=dict(offset=7,inflight={},deferred={},failed={})
    for key in ('inflight','deferred','failed'):
        q=copy.deepcopy(clean);q[key]={'source':{}};write(u/'queues/caption/state.json',q)
        assert not c.upstream_done(w)
    write(u/'queues/caption/state.json',dict(clean,offset=0));assert not c.upstream_done(w)
    write(u/'queues/caption/state.json',clean);assert c.upstream_done(w)


def test_handoff_runs_merge_only_after_extraction_and_releases_on_error(tmp_path,monkeypatch):
    import listen_richness_cluster_control as c
    w=tmp_path;write(w/'config.json',dict(selected_profile='D512'))
    write(w/'producer_complete.json',{});(w/'ready.jsonl').write_text('')
    write(w/'queues/extract/state.json',dict(completed_clips=3,inflight={},offset=0,failed={}))
    calls=[]
    monkeypatch.setattr(c,'upstream_done',lambda *_:True)
    monkeypatch.setattr(c,'ensure_preparer',lambda *_:None)
    monkeypatch.setattr(c,'wait_free',lambda *a,**k:calls.append('free'))
    monkeypatch.setattr(c,'launch',lambda *a:calls.append('launch'))
    monkeypatch.setattr(c,'stop_own',lambda *a:calls.append('stop'))
    monkeypatch.setattr('listen_richness.merge_cluster.merge',lambda *a:calls.append('merge'))
    c.chain(w)
    assert calls==['free','launch','free','merge'] and read(w/'chain.json')['stage']=='complete'
    calls.clear();write(w/'preparation.json',dict(source_errors=[['source','bad provenance']]))
    with pytest.raises(RuntimeError,match='Input sources need recovery'):c.chain(w)
    assert calls==['free','launch','stop'] and read(w/'chain.json')['stage']=='error'


def test_preparation_progress_counts_use_covering_status_indexes(tmp_path):
    db=database(tmp_path)
    plan=' '.join(str(row) for row in db.execute('EXPLAIN QUERY PLAN SELECT status,COUNT(*) FROM coverage GROUP BY status'))
    assert 'COVERING INDEX coverage_status' in plan
    plan=' '.join(str(row) for row in db.execute("EXPLAIN QUERY PLAN SELECT COUNT(*) FROM sources WHERE status!='prepared'"))
    assert 'COVERING INDEX sources_status' in plan
    db.close()
