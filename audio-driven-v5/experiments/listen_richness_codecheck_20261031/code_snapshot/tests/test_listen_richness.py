"""Regression contracts for open-caption statistics; no model/GPU dependencies."""
import copy
import json
from pathlib import Path
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from listen_richness.common import digest, file_hash, read, rows, union, write, write_jsonl
from listen_richness.statistics import GATE, audio_stats, face_stats, nod_stats, snapshot
from listen_richness.extraction import alias_map, contract, request, summarize, validate
from listen_richness.selection import DEFAULT_POLICY, audit_catalog, compact, select
from listen_richness.cli import main, call_model, database


def fixture(tmp_path, i=0, caption="The listener smiles while gently nodding twice.", silent=True):
    sid = f"sample-{i}"; pair = f"naturalistic_V00_S{i:04}_I00000000"
    listener = f"V00_S{i:04}_I00000000_P{i*2:04}A"
    speaker = f"V00_S{i:04}_I00000000_P{i*2+1:04}A"
    source = dict(sample_id=sid, pair=pair, split="train", group_id=f"group-{i}",
                  listener_id=listener, speaker_id=speaker, source={"member": listener+".mp4"},
                  source_start_s=10., source_end_s=16., duration_s=6.,
                  normalized_audio=dict(path="/original.wav", sha256="original", sample_rate=100),
                  video_transform=dict(crop=dict(x=0, y=0, width=512, height=512)))
    measure = dict(status="ok", source=source, source_sha256=digest(source, ascii=True), contract_sha256="quality",
                   raw_measurements=[dict(detected=True, face_width_native=300., face_height_native=290.) for _ in range(30)],
                   focus=dict(low_fraction=0., measured_fraction=1., laplacian_reference_p80=45.))
    mp = tmp_path / f"measure-{i}.json"; write(mp, measure)
    decision = dict(sample_id=sid, decision="retain", source_sha256=digest(source, ascii=True),
                    measurement_file=str(mp), measurement_sha256=file_hash(mp), contract_sha256="quality")
    recipe = dict(schema="j-listener-segment-official-union-v1", sample_id=sid, listener_id=listener,
                  source=dict(path="/original.wav", sha256="original", sample_rate=100, channels=1,
                              source_samples=3000, first_sample=1000, last_sample=1600),
                  official_listener=dict(words=[], sentences=[], vad=[]), gate_policy=GATE,
                  extra_mute_intervals=[], source_time_origin_s=10.)
    if not silent: recipe["official_listener"]["vad"] = [dict(start=1., end=2.)]
    evidence = dict(audio_identity=dict(recipe_sha256=digest(recipe), full_silence=silent,
                    source_start_sample=1000, source_end_sample=1600, positive_samples=0 if silent else 120),
                    listener_audio_is_silent=silent,
                    nod_supplement=dict(sample_id=sid, fps=25, profile="recall", events=[]))
    # Fade rounds to one sample at sr=100, so there are no zero fade endpoints.
    record = dict(sample_id=sid, source=source, caption=caption, visual_quality=decision,
                  generation_checks_passed=True, finish_reason="stop", review_flags=[],
                  input_evidence=evidence, audio_recipe=recipe, training_admitted=False)
    cp = tmp_path / f"caption-{i}.json"; write(cp, record)
    return record, cp, measure


def snap(tmp_path, i=0, **kwargs):
    r, p, _ = fixture(tmp_path, i, **kwargs)
    return snapshot(r, p, file_hash(p))


def behavior(name="smiling", quote="The listener smiles", ident="b1", form="change", polarity="positive", modality="visual", count=None):
    return dict(id=ident, behavior=name, modality=modality, polarity=polarity, form=form, quote=quote,
                attributes=[], repeat=dict(kind="not_stated" if count is None else "explicit",
                                          count=count, quote="" if count is None else "twice"))


def payload(bs=None, relations=None):
    return dict(behaviors=bs or [], relations=relations or [])


def cat_for(rs):
    return dict(rows=[dict(pair=r["pair"], split=r["split"], group_id=r["group_id"],
                           a={"id":r["listener_id"]}, b={"id":r["speaker_id"]}) for r in rs])


def test_union_does_not_bridge_two_separate_okays():
    assert union([[1,1.2],[1.1,1.3],[1.5,1.7]]) == [[1,1.3],[1.5,1.7]]


def test_official_overlap_is_not_triple_counted(tmp_path):
    r, _, _ = fixture(tmp_path, silent=False)
    r["audio_recipe"]["official_listener"] = dict(words=[dict(start=1,end=1.2)],
        sentences=[dict(start=1,end=2)], vad=[dict(start=1.5,end=2)])
    r["input_evidence"]["audio_identity"]["recipe_sha256"] = digest(r["audio_recipe"])
    out = audio_stats(r["audio_recipe"], r["input_evidence"], r["source"])
    assert out["official_bout_count"] == 1 and out["official_core_s"] == 1
    assert out["nominal_keep_s"] == 1.2


def test_interval_metadata_outside_clip_cannot_become_audio(tmp_path):
    r, _, _ = fixture(tmp_path)
    r["audio_recipe"]["official_listener"]["words"] = [dict(start=6.05, end=6.2)]
    r["input_evidence"]["audio_identity"]["recipe_sha256"] = digest(r["audio_recipe"])
    out = audio_stats(r["audio_recipe"], r["input_evidence"], r["source"])
    assert out["official_bout_count"] == 0 and out["nominal_keep_s"] == 0


def test_optimized_support_matches_actual_production_mask(tmp_path):
    from listen_benchmark.official_audio_gate import make_gate
    rng = random.Random(123)
    for _ in range(80):
        r, _, _ = fixture(tmp_path)
        recipe = r["audio_recipe"]; recipe["source"].update(sample_rate=1000, first_sample=10000,last_sample=16000,source_samples=30000)
        r["source"]["normalized_audio"]["sample_rate"] = 1000
        off = {k: [dict(start=(a:=rng.uniform(-1,7)), end=a+rng.uniform(0,.5)) for _ in range(4)] for k in ("words","sentences","vad")}
        recipe["official_listener"] = off
        _, info = make_gate(off,6000,1000)
        r["input_evidence"].update(listener_audio_is_silent=info["full_silence"])
        r["input_evidence"]["audio_identity"].update(recipe_sha256=digest(recipe),
                source_start_sample=10000,source_end_sample=16000,positive_samples=info["positive_samples"],full_silence=info["full_silence"])
        result = audio_stats(recipe,r["input_evidence"],r["source"])
        assert result["mask_positive_s"] == info["positive_samples"]/1000
        assert result["official_core_s"] == info["core_samples"]/1000


@pytest.mark.parametrize("change", ["missing_vad", "nan_time", "wrong_origin", "wrong_recipe", "wrong_listener"])
def test_bad_audio_is_error_not_silence(tmp_path, change):
    r, _, _ = fixture(tmp_path); p = r["audio_recipe"]
    if change == "missing_vad": del p["official_listener"]["vad"]
    if change == "nan_time": p["official_listener"]["vad"] = [dict(start=float("nan"),end=1)]
    if change == "wrong_origin": p["source_time_origin_s"] = 11
    if change == "wrong_listener": p["listener_id"] = "other"
    if change != "wrong_recipe" and change != "nan_time": r["input_evidence"]["audio_identity"]["recipe_sha256"] = digest(p)
    if change == "wrong_recipe": r["input_evidence"]["audio_identity"]["recipe_sha256"] = "wrong"
    with pytest.raises((ValueError,KeyError)): audio_stats(p,r["input_evidence"],r["source"])


def test_nod_intervals_not_inferred_counts():
    n = nod_stats(dict(sample_id="s",fps=25,profile="recall",events=[
        dict(id="a",start=1,end=2),dict(id="b",start=1,end=2),dict(id="c",start=1.5,end=3)]),"s",5)
    assert n["candidate_interval_count"] == 2 and n["support_s"] == 2 and n["exact_nod_count"] is None
    with pytest.raises(ValueError): nod_stats(dict(sample_id="s",fps=25,profile="recall",events=[dict(id="a",start=4,end=6)]),"s",5)


def test_face_size_is_native_box_and_missing_is_unknown(tmp_path):
    _, _, m = fixture(tmp_path)
    f = face_stats(m)
    assert f["face_short_side_p10"] == 290 and f["face_short_side_p10"] != 512
    m["raw_measurements"] = [dict(detected=False)]
    assert face_stats(m)["face_short_side_p10"] is None


def test_snapshot_preserves_caption_and_checks_measurement(tmp_path):
    r,p,_ = fixture(tmp_path); before=copy.deepcopy(r)
    result=snapshot(r,p,file_hash(p)); assert result["caption"] == r["caption"] and r == before
    assert result["original_training_admitted"] is False and result["eligible_for_extraction"]
    r["visual_quality"]["measurement_sha256"] = "bad"
    with pytest.raises(ValueError,match="measurement hash"): snapshot(r,p,file_hash(p))


def test_failed_generation_excluded_and_missing_quality_not_assumed_pass(tmp_path):
    r,p,_=fixture(tmp_path); r["finish_reason"]="length"
    assert not snapshot(r,p,file_hash(p))["eligible_for_extraction"]
    del r["visual_quality"]
    with pytest.raises(ValueError,match="Quality receipt"): snapshot(r,p,file_hash(p))


def test_exact_quote_compound_actions_and_explicit_counts(tmp_path):
    s=snap(tmp_path); bs=[behavior(), behavior("head nodding",s["caption"],"b2",count=2)]
    p=payload(bs); validate(p,s["caption"]); out=summarize(s,p)
    assert out["behavior_type_count"]==2 and out["capped_change_score"]==3
    assert json.loads(request(s)[1]["content"]) == {"caption": s["caption"]}
    bs[1]["quote"]="The person nods twice."
    with pytest.raises(ValueError,match="substring"): validate(p,s["caption"])


def test_unknown_count_not_fabricated_and_no_invented_timestamps():
    b=behavior("head nodding","She nods repeatedly."); p=payload([b]); b["repeat"]=dict(kind="multiple_unspecified",count=3,quote="repeatedly")
    with pytest.raises(ValueError): validate(p,"She nods repeatedly.")
    b["repeat"]["count"]=None; validate(p,"She nods repeatedly.")
    b["start"]=1
    with pytest.raises(ValueError): validate(p,"She nods repeatedly.")


def test_wrong_explicit_count_is_rejected():
    b=behavior("head nodding","She nods twice.",count=3)
    with pytest.raises(ValueError,match="numerical evidence"): validate(payload([b]),b["quote"])


def test_sustained_repetition_and_negation_do_not_inflate_score(tmp_path):
    text="She maintains a smile. She keeps smiling. She does not nod."
    s=snap(tmp_path,caption=text)
    p=payload([behavior("smiling","She maintains a smile.",form="state"),
               behavior("smiling","She keeps smiling.","b2",form="state"),
               behavior("head nodding","She does not nod.","b3",polarity="negative")])
    v=summarize(s,p)
    assert v["behavior_type_count"]==1 and v["capped_change_score"]==0 and v["quiet_candidate"]


def test_nod_cross_source_not_double_counted_and_conflict_not_positive(tmp_path):
    s=snap(tmp_path); s["nod"]["candidate_interval_count"]=3
    p=payload([behavior("head nodding",s["caption"],count=2)])
    result=summarize(s,p)
    assert result["behavior_type_count"]==1 and result["capped_change_score"]==2
    assert result["groups"][0]["sources"]==["caption","nod_detector"]
    p["behaviors"][0]["polarity"]="negative"
    result=summarize(s,p)
    assert result["behavior_type_count"]==0 and "nod_caption_detector_conflict" in result["consistency_flags"]


def test_speech_on_zero_waveform_quarantined(tmp_path):
    s=snap(tmp_path,caption='He says "yeah".'); p=payload([behavior("speech",s["caption"],modality="audio")])
    result=summarize(s,p)
    assert result["behavior_type_count"]==0 and result["consistency_flags"]==["audible_caption_on_zero_waveform"]


def test_no_fuzzy_merge_of_head_dip_and_nod(tmp_path):
    s=snap(tmp_path,caption="She looks down and nods.")
    p=payload([behavior("looking down",s["caption"]),behavior("head nodding",s["caption"],"b2")])
    assert summarize(s,p)["behavior_type_count"]==2
    with pytest.raises(ValueError): alias_map(dict(schema="listen-richness-aliases-v1",reviewed=False,aliases=[]))


def test_temporal_cycles_rejected():
    text="She smiles, then nods."
    p=payload([behavior("smiling",text),behavior("head nodding",text,"b2")],
              [dict(kind="before",**{"from":"b1","to":"b2"},quote=text),dict(kind="before",**{"from":"b2","to":"b1"},quote=text)])
    with pytest.raises(ValueError,match="Cyclic"): validate(p,text)


def selection_pool(tmp_path,n=24):
    result=[]
    for i in range(n):
        s=snap(tmp_path,i); p=payload([behavior()])
        result.append(compact(s,summarize(s,p)))
    return result


def test_selection_reproducible_full_clips_rare_supplement(tmp_path):
    rs=selection_pool(tmp_path); rs[-1].update(face_short_side_p10=180,behaviors=["visual:hand raising"])
    p=dict(DEFAULT_POLICY,hours=60/3600,core_fraction=.8,quiet_fraction=0,target_pairs_per_behavior=1)
    chosen,report=select(rs,cat_for(rs),p)
    assert report["selection_complete"] and 60<=report["selected_seconds"]<66
    assert any(r["sample_id"]==rs[-1]["sample_id"] for r in chosen)
    assert select(list(reversed(rs)),cat_for(rs),p)[0]==chosen


def test_pair_cap_and_opposite_listener_overlap(tmp_path):
    rs=selection_pool(tmp_path,4); rs[1].update({k:rs[0][k] for k in ("pair","group_id","listener_id","speaker_id")})
    rs[1]["listener_id"],rs[1]["speaker_id"]=rs[1]["speaker_id"],rs[1]["listener_id"]
    catalog=cat_for([rs[0],*rs[2:]])
    p=dict(DEFAULT_POLICY,hours=30/3600,quiet_fraction=0)
    chosen,report=select(rs,catalog,p)
    assert len(chosen)==3 and not report["selection_complete"]
    assert report["constraint_skips"]["source_overlap"]>0


def test_test_split_cannot_leak_and_identity_catalog_conflict_fails(tmp_path):
    rs=selection_pool(tmp_path,3); rs[1]["split"]="test"
    chosen,report=select(rs,cat_for(rs),dict(DEFAULT_POLICY,hours=.01,quiet_fraction=0))
    assert all(r["split"]=="train" for r in chosen)
    c=cat_for(rs); c["rows"][1]["group_id"]=c["rows"][0]["group_id"]
    with pytest.raises(ValueError,match="Cross-split"): audit_catalog(c)


def test_silence_is_eligible_and_unknown_face_never_passes(tmp_path):
    s=snap(tmp_path,caption="The listener remains still.")
    row=compact(s,summarize(s,payload()))
    chosen,report=select([row],cat_for([row]),dict(DEFAULT_POLICY,hours=6/3600))
    assert chosen and chosen[0]["quiet_candidate"]
    row["face_short_side_p10"]=None
    assert not select([row],cat_for([row]),dict(DEFAULT_POLICY,hours=6/3600))[0]


def test_cli_prepare_resume_import_export_select(tmp_path):
    manifest=tmp_path/"inputs.jsonl"; db=tmp_path/"stats.sqlite"
    inputs=[]; snapshots=[]
    for i in range(5):
        r,p,_=fixture(tmp_path,i); inputs.append(dict(caption_file=str(p))); snapshots.append(snapshot(r,p,file_hash(p)))
    write_jsonl(manifest,inputs)
    assert main(["prepare","--manifest",str(manifest),"--db",str(db)])==0
    assert main(["prepare","--manifest",str(manifest),"--db",str(db)])==0
    assert read(db.with_suffix(".prepare.json"))["counts"]["reused"]==5
    requests=tmp_path/"requests.jsonl"
    main(["requests","--db",str(db),"--output",str(requests)])
    responses=[]
    for req in rows(requests):
        responses.append(dict(sample_id=req["sample_id"],snapshot_sha256=req["snapshot_sha256"],
                              extraction_contract=req["extraction_contract"],finish_reason="stop",extraction=payload([behavior()])))
    response_file=tmp_path/"responses.jsonl"; write_jsonl(response_file,responses)
    main(["import-responses","--db",str(db),"--responses",str(response_file)])
    out=tmp_path/"export"; main(["export","--db",str(db),"--output",str(out),"--parquet"])
    stats=list(rows(out/"sample_stats.jsonl")); assert len(stats)==5
    assert read(out/"EXPORT_COMPLETE.json")["all_prepared_resolved"]
    cat=tmp_path/"catalog.json"; write(cat,cat_for(stats))
    policy=tmp_path/"policy.json"; write(policy,dict(DEFAULT_POLICY,hours=18/3600,quiet_fraction=0))
    selected=tmp_path/"selected"
    assert main(["select","--db",str(db),"--catalog",str(cat),"--policy",str(policy),"--output",str(selected)])==0
    assert all(r["caption"]==snapshots[0]["caption"] for r in rows(selected/"selected_manifest.jsonl"))
    assert all(read(x["caption_file"])["caption"]==snapshots[0]["caption"] for x in inputs)
    responses[0]["snapshot_sha256"]="stale"; write_jsonl(response_file,responses)
    with pytest.raises(ValueError,match="contract mismatch"):
        main(["import-responses","--db",str(db),"--responses",str(response_file)])


def test_http_client_retries_invalid_json_then_validates(tmp_path):
    s=snap(tmp_path); calls=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            text="bad json" if len(calls)==1 else json.dumps(payload([behavior()]))
            body=json.dumps(dict(choices=[dict(finish_reason="stop",message=dict(content=text))])).encode()
            self.send_response(200); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self,*_): pass
    server=HTTPServer(("127.0.0.1",0),Handler); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        p,attempts=call_model(s,f"http://127.0.0.1:{server.server_port}/v1","test-model",3,2)
        assert p==payload([behavior()]) and len(attempts)==2 and "error" in attempts[0]
        assert "nod_supplement" not in json.dumps(calls)
    finally: server.shutdown(); server.server_close(); thread.join()


def test_existing_work_index_backfills_legacy_quality_without_scanning(tmp_path):
    import hashlib
    r,p,_=fixture(tmp_path); quality=r.pop("visual_quality")
    work=tmp_path/"production"; sid=r["sample_id"]; jid=r["source"]["pair"]
    path=work/"caption/results"/hashlib.sha256(sid.encode()).hexdigest()[:2]/(sid+".json")
    write(path,r)
    gate=work/"quality"
    write(work/"config.json",dict(visual_quality_gate=dict(work=str(gate),contract_sha256="quality")))
    write_jsonl(work/"ready.jsonl",[dict(job_id=jid)])
    write(work/"jobs"/(jid+".json"),dict(job_id=jid,source_result_sha256="source-hash",rows=[dict(sample_id=sid,source=r["source"])]))
    write(gate/"measure/completed"/(jid+".json"),dict(contract_sha256="quality",source_result_sha256="source-hash",decisions=[quality]))
    before=file_hash(path);db=tmp_path/"stats.sqlite"
    assert main(["prepare","--work",str(work),"--db",str(db)])==0
    assert file_hash(path)==before
    with database(db) as c:
        assert c.execute("select status from samples").fetchone()[0]=="pending"


def test_bounded_extraction_resume_and_failed_records(tmp_path,monkeypatch):
    import listen_richness.cli as cli
    inputs=[]
    for i in range(5):
        _,p,_=fixture(tmp_path,i);inputs.append(dict(caption_file=str(p)))
    manifest=tmp_path/"inputs.jsonl";write_jsonl(manifest,inputs);db=tmp_path/"stats.sqlite"
    main(["prepare","--manifest",str(manifest),"--db",str(db)])
    calls=[]
    def fake(s,*args):
        calls.append(s["sample_id"])
        if s["sample_id"]=="sample-1": return None,[dict(error="failed model reply")]
        return payload([behavior()]),[dict(finish_reason="stop")]
    monkeypatch.setattr(cli,"call_model",fake)
    command=["extract","--db",str(db),"--endpoint","http://unused/v1","--workers","2"]
    assert main(command+["--limit","2"])==1
    assert main(command+["--limit","0"])==0
    assert sorted(calls)==[f"sample-{i}" for i in range(5)]
    with database(db) as c:
        assert dict(c.execute("select status,count(*) from samples group by status"))==dict(ok=4,failed=1)
    monkeypatch.setattr(cli,"call_model",lambda s,*_: (payload([behavior()]),[dict(finish_reason="stop")]))
    assert main(command+["--retry-failed"])==0
    with database(db) as c:
        assert dict(c.execute("select status,count(*) from samples group by status"))==dict(ok=5)
