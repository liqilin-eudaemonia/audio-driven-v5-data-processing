"""Independent completion checks without repeating GPU inference or scoring."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/p2_compare_responsenet_random200_20261106/dystream'
EXT=WORK/'node11_extension'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=read(WORK/'protocol.json');result=read(WORK/'RESULTS.json');expected={s['id']:s for s in p['samples']}
    previous=read(EXT/'PRE_EXISTING_OUTPUTS.json');assert previous['clips']==139
    assert sha(Path(p['references']))==p['references_sha256']
    for name,digest in p['code_sha256'].items():assert sha(Path(name))==digest,name
    protocol_hash=sha(WORK/'protocol.json');resource_hash=sha(EXT/'RESOURCE_EXTENSION.json')
    old_preserved=0;new=0;gpus=set();native_frames=0;score_frames=0;fallback=[]
    snapshots={f.stem:f for f in (EXT/'source_snapshots').glob('*.py')}
    adapter=ROOT/'compare/deployment/responsenet_dystream_node11.py';snapshots[sha(adapter)]=adapter
    for uid,s in expected.items():
        receipt=WORK/'generated'/uid/'complete.json';r=read(receipt)
        assert r['protocol_sha256']==protocol_hash and r['source_only'] and not r['listener_audio_generated']
        assert r['native_fps']==25 and r['native_size']==512
        assert r['score_frames']==s['frames'] and r['score_fps']==s['fps'] and r['score_size']==512
        assert len(r['mapping'])==s['frames']
        assert sha(WORK/'generated'/uid/'generated.mp4')==r['native_sha256']
        assert sha(WORK/'scoring/dystream'/(uid+'.listener.mp4'))==r['score_sha256']
        if uid in previous['records']:
            old=previous['records'][uid]
            assert sha(receipt)==old['receipt_sha256'] and r['native_sha256']==old['native_sha256'] and r['score_sha256']==old['score_sha256']
            old_preserved+=1
        else:
            execution=r['execution_resource'];gpu=execution['gpu'];assert gpu in range(7)
            assert execution['extension_sha256']==resource_hash
            assert execution['adapter_sha256'] in snapshots and sha(snapshots[execution['adapter_sha256']])==execution['adapter_sha256']
            new+=1;gpus.add(gpu)
            if 'diagnostic mask unavailable' in r.get('reference_preprocessing',''):fallback.append(uid)
        native_frames+=r['native_frames'];score_frames+=r['score_frames']
    assert old_preserved==139 and new==61 and gpus==set(range(7))
    pairs=[json.loads(line) for line in (WORK/'pairs.jsonl').read_text().splitlines() if line.strip()]
    assert len(pairs)==len({r['id'] for r in pairs})==200 and {r['id'] for r in pairs}==set(expected)
    errors={}
    for r in pairs:
        if 'media' not in r:raise AssertionError((r['id'],r.get('errors')))
        media=r['media'];s=expected[r['id']]
        assert media['gt_frames']==media['generated_frames']==media['aligned_frames']==s['frames']
        assert media['trimmed_gt']==media['trimmed_generated']==0
        assert abs(media['fps']-s['fps'])<.001
        if r['errors']:errors[r['id']]=r['errors']
    contracts=[read(WORK/'metric_shards'/f'{i:02d}'/'contract.json') for i in range(p['metric_shards'])]
    historical=read(ROOT/'experiments/p2_responsenet_random200_twelveway_20261105/metric_shards/av_only_6000/responsenet/00/contract.json')
    for c in contracts:
        assert c['code']==historical['code'] and c['assets']==historical['assets']
        for key in ['batch_size','face_padding','fid_frames','fvd_frames','fvd_stride','face_device','emotion_device','pixel_workers']:
            assert c['arguments'][key]==historical['arguments'][key],(key,c['arguments'][key],historical['arguments'][key])
    for path in (EXT/'preprocessing_parity').glob('*.json'):
        check=read(path);assert check['passed'] and check['resized_uint8_identical'] and check['latent_max_abs']<1e-5
    assert sorted(fallback)==sorted(f.stem for f in (EXT/'first_frame_fallback').glob('*.json'))
    assert read(WORK/'FINAL_AUDIT.json')['passed'] and read(WORK/'COMPLETE.json')['clips']==200
    report=dict(passed=True,clips=200,previous_clips_preserved=139,new_clips=61,new_gpus=sorted(gpus),excluded_gpu=7,
        native_frames=native_frames,score_frames=score_frames,first_frame_mask_fallback_ids=fallback,
        original_scorer_code_assets_and_parameters_match=True,no_scoring_frame_trimming=True,
        metric_pairs_with_errors=len(errors),metric_errors=errors,coverage=result['coverage'],
        status=result['status'],protocol_sha256=protocol_hash,results_sha256=sha(WORK/'RESULTS.json'),at=time.time())
    (EXT/'FINAL_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['metric_errors','coverage']},indent=2))


if __name__=='__main__':main()
