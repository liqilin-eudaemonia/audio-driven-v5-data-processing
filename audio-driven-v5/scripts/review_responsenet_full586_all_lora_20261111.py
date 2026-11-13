"""CPU source/cache preflight for the full586 GPU7 evaluation queue."""
import collections
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import p2_responsenet_full586_all_lora_20261111 as task
from p2_testsets_eval.protocol import read,save,sha


def main():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_responsenet_random200 import rn_speaker_wave
    from p2_ltx_all_lora.config import source_identity,validate
    from p2_testsets_eval.runtime import ready
    from PIL import Image
    p=task.verify(task.COMMON);items=read(task.COMMON/'shared_items.json')
    identities=read(p['source_media_identity']);assert sha(p['source_media_identity'])==p['source_media_identity_sha256']
    cached=0
    for x in items:
        folder=task.COMMON/'conditions'/x['key']
        for name,h in identities[x['key']].items():assert sha(folder/name)==h
        with Image.open(folder/'first_frame.png') as image:assert image.size==(512,512)
        # This official project adapter reads only speaker WAV and source-only audit.
        wave,audit=rn_speaker_wave(x,folder)
        assert audit['source_only'] and len(wave)==round(x['geometry']['duration_s']*16000)
        q=folder/'qwen';w=task.COMMON/'whisper'/x['key']
        if ready(q):
            plan=read(q/'plan.json');assert plan['adapter']==p['planner_adapter_identity']
            assert not plan['target_description_access'] and ready(folder/'text') and ready(w)
            meta=read(w/'metadata.json')
            assert meta['encoder']==p['whisper_encoder'] and meta['feature_code']==p['feature_code']
            for key in ('waveform_sha256','source_sha256','input_audit_sha256'):
                assert audit[key]==meta['source'][key]
            cached+=1
        else:
            assert not ready(folder/'text') and not ready(w)
        g=x['geometry'];assert g['fps']==float(x['reference']['media']['listener']['video']['fps'])
        assert g['frames']==x['reference']['media']['listener']['video']['frames']
    sources={}
    for step in (6000,6500):
        work=task.run_work(step);r=task.verify(work);rec=read(work/'export/RECEIPT.json')
        m=task.loading(step,work)
        expected=source_identity() if step==6000 else m.expected_source()
        assert rec['identity']['source']==expected
        validate(rec['identity']['experiment']) if step==6000 else m.model_config(rec['identity']['experiment'])
        assert r['arms'][0]['original_sha256']==rec['original_sha256']
        shards=[x for rank in range(8) for x in read(work/'shards'/f'{rank:02d}.json')]
        assert len(shards)==586 and {x['key'] for x in shards}=={x['key'] for x in items}
        sources[step]=dict(checkpoint=rec['original'],sha256=rec['original_sha256'],source_files=len(expected),
            model_tensors_bitwise_equal=rec['all_model_tensors_bitwise_equal'])
    assert cached==200
    save(task.WORK/'CPU_ACCEPTANCE.json',dict(success=True,at=time.time(),review_code_sha256=sha(__file__),
        pairs=586,outputs=1172,sources=135,original_source_count=140,zero_pair_sources=5,
        all_speaker_waveforms_checked=True,all_first_frames_checked=True,source_only=True,
        reused_SFT2752_and_Whisper=200,missing_conditions=386,checkpoints=sources,
        native_fps_counts=dict(collections.Counter(str(x['geometry']['fps']) for x in items)),
        group_counts=p['group_counts'],references_sha256=task.REF_SHA,
        physical_gpus=[7],metric_math='unchanged latest200 pipeline'))
    print(json.dumps(read(task.WORK/'CPU_ACCEPTANCE.json'),ensure_ascii=False))


if __name__=='__main__':main()
