"""Small held-out planning experiment; reuse the native SFT reader and generation."""
import argparse
import collections
import contextlib
import difflib
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
CFG = ROOT / 'experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/config.json'
sys.path.insert(0, str(CFG.parent / 'vendor'))
os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT / 'scripts')
SEED = 20261113


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(exist_ok=True, parents=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, p)


def arms():
    parent = ROOT / 'experiments/p2_planning_sft_upright_3b_gpu7_20261105/training'
    latest = CFG.parent / 'training'
    out = [dict(name='base', step=0, adapter=None)]
    for step in (500, 1000, 1500, 2000, 2500, 2752):
        p = (parent if step <= 1500 else latest) / f'checkpoint-{step}'
        assert read(p / 'trainer_state.json')['global_step'] == step
        out.append(dict(name=f'sft{step}', step=step, adapter=str(p),
                        adapter_sha256=sha(p / 'adapter_model.safetensors'),
                        adapter_config_sha256=sha(p / 'adapter_config.json')))
    assert out[-1]['adapter_sha256'] == sha(latest / 'final_adapter/adapter_model.safetensors')
    return out


def prepare():
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.optimize import linear_sum_assignment
    from scipy.signal import resample_poly
    from p2_planning_sft.reference_data import ReferencePlanningDataset, connection
    from upright_lora_media import resolve_example
    from listen_benchmark.j_omni import check_file
    import math

    if (WORK / 'PREPARED.json').exists():
        return
    torch.set_num_threads(2)
    c = read(CFG)
    ds = ReferencePlanningDataset(c['data']['seal'], 'dev', input_pipeline=c['data']['input_pipeline'])
    rows = [ds.record(i) for i in range(len(ds))]
    by_id = {r['sample_id']: r for r in rows}
    indices = {sid: i for i, sid in enumerate(ds.ids)}
    rng = random.Random(SEED)
    buckets = collections.defaultdict(list)
    for row in rows:
        t = row['target']['text'].lower()
        if row['provenance']['processed_listener_waveform_nonzero']:
            tag = 'reference_audio_nonzero'
        elif 'smil' in t:
            tag = 'reference_audio_zero_smile'
        elif 'nod' in t:
            tag = 'reference_audio_zero_nod'
        else:
            tag = 'reference_audio_zero_other'
        buckets[tag].append(row)
    selected = []
    group_counts = collections.Counter()
    for tag in sorted(buckets):
        pool = list(buckets[tag]); rng.shuffle(pool)
        for _ in range(min(16, len(pool))):
            row = min(pool, key=lambda x: group_counts[x['group_id']])
            pool.remove(row); selected.append(dict(row, diagnostic_stratum=tag))
            group_counts[row['group_id']] += 1
    assert len(selected) == 64, {k: len(v) for k, v in buckets.items()}
    selected.sort(key=lambda x: (-x['duration_s'], x['sample_id']))
    groups = {x['group_id'] for x in selected}
    pairs = {x['pair'] for x in selected}
    with connection(ds.root / 'index.sqlite') as db:
        train_groups = {x[0] for x in db.execute("select distinct group_id from samples where split='train'")}
        params = ','.join('?' for _ in pairs)
        overlaps = db.execute(f"select count(*) from samples where split='train' and pair in ({params})", list(pairs)).fetchone()[0]
    assert not groups & train_groups and overlaps == 0
    # Select unique, different-source donors with very close durations. No time stretching.
    cost = np.array([[abs(a['duration_s'] - b['duration_s']) if a['group_id'] != b['group_id'] else 1e6
                      for b in rows] for a in selected])
    ri, ci = linear_sum_assignment(cost)
    assert list(ri) == list(range(64))
    assert float(cost[ri, ci].max()) <= .12, float(cost[ri, ci].max())
    mapping = dict(zip(ri.tolist(), ci.tolist()))
    candidates = arms()
    protocol = dict(schema='planning-checkpoint-audio64-v1', seed=SEED, config=str(CFG), config_sha256=sha(CFG),
        seal=str(ds.seal_path), seal_sha256=sha(ds.seal_path), data_identity=ds.identity,
        arms=candidates, cases=64, conditions=['original', 'mismatched_audio'], total_outputs=896,
        sampling='16 per reference nonzero/zero-smile/zero-nod/zero-other; deterministic source-group spreading',
        source_groups=len(groups), train_group_overlap=0, train_pair_overlap=0,
        reference_labels='Automatic annotation used for diagnostic sampling only; not unique human ground truth',
        generation=dict(batch_size=16, do_sample=False, max_new_tokens=512, retry_max_new_tokens=1024,
                        dtype='bfloat16', attn_implementation='sdpa', max_pixels=c['max_pixels']),
        caveats=['Text changes under audio replacement do not establish semantic understanding',
                 'Same response can be appropriate for different contexts; exact matches are not automatically errors',
                 'Balanced diagnostic dev subset does not estimate natural test distribution performance',
                 '500/1000/1500 preceded the32-GPU resume; effective training batch remained256'],
        at=time.time())
    save(WORK / 'protocol.json', protocol)
    manifest = []
    for i, row in enumerate(selected):
        sid = row['sample_id']; donor = rows[mapping[i]]; p = WORK / 'media' / sid; p.mkdir(parents=True, exist_ok=True)
        original = ds[indices[sid]]
        original['image'].save(p / 'first_frame.png')
        wave = original['audio']; assert len(wave) == round(row['duration_s'] * 16000)
        sf.write(p / 'original.wav', wave, 16000, subtype='FLOAT')
        resolved = resolve_example(donor, ds.crop_config)
        audio = resolved['speaker_audio']; source = resolved['source']; check_file(audio['path'], audio['sha256'])
        with sf.SoundFile(audio['path']) as f:
            rate = f.samplerate; assert f.channels == 1
            first, last = round(source['source_start_s'] * rate), round(source['source_end_s'] * rate)
            f.seek(first); other = f.read(last - first, dtype='float32')
        assert len(other) == last - first and np.isfinite(other).all()
        if rate != 16000:
            g = math.gcd(rate, 16000); other = resample_poly(other, 16000 // g, rate // g).astype(np.float32)
        other = other[:round(donor['duration_s'] * 16000)]
        adjustment = len(wave) - len(other)
        other = np.pad(other, (0, max(0, adjustment)))[:len(wave)]
        assert other.shape == wave.shape and np.isfinite(other).all() and np.any(other != wave)
        sf.write(p / 'mismatched_audio.wav', other, 16000, subtype='FLOAT')
        item = dict(id=sid, split='dev', group_id=row['group_id'], pair=row['pair'], duration=row['duration_s'],
            stratum=row['diagnostic_stratum'], reference_text=row['target']['text'],
            reference_human_ground_truth=False, donor_id=donor['sample_id'], donor_group_id=donor['group_id'],
            donor_duration=donor['duration_s'], donor_adjustment_samples=adjustment,
            input_audit=original['audit'], first_frame_rgb_sha256=hashlib.sha256(np.asarray(original['image']).tobytes()).hexdigest(),
            files={name: dict(path=str((p / name).relative_to(WORK)), sha256=sha(p / name))
                   for name in ('first_frame.png', 'original.wav', 'mismatched_audio.wav')})
        manifest.append(item); save(WORK / 'preparation_progress.json', dict(at=time.time(),done=i+1,total=64))
    save(WORK / 'samples.json', manifest)
    save(WORK / 'PREPARED.json', dict(success=True, at=time.time(), samples=64,
        source_groups=len(groups), strata=dict(collections.Counter(x['stratum'] for x in manifest)),
        max_donor_adjustment_s=max(abs(x['donor_adjustment_samples']) / 16000 for x in manifest),
        samples_sha256=sha(WORK / 'samples.json'), protocol_sha256=sha(WORK / 'protocol.json')))


def generate(slot, slots=2):
    import numpy as np
    from PIL import Image
    import soundfile as sf
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from p2_planning_sft.data import PlanningCollator, SYSTEM, USER
    from p2_planning_sft.model import build_model
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(64 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    protocol = read(WORK / 'protocol.json'); samples = read(WORK / 'samples.json'); c = read(CFG)
    selected = [a for i, a in enumerate(protocol['arms']) if i % slots == slot]
    unfinished = [a for a in selected if not (WORK / 'results' / a['name'] / 'COMPLETE.json').exists()]
    if not unfinished:
        return
    collator = PlanningCollator(c['model'], c['max_length'], c['max_pixels']); processor = collator.get_processor()
    processor.tokenizer.padding_side = 'left'; eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    rows = []
    for sample in samples:
        for condition in protocol['conditions']:
            image_path = WORK / sample['files']['first_frame.png']['path']
            audio_path = WORK / sample['files'][condition + '.wav']['path']
            assert sha(image_path) == sample['files']['first_frame.png']['sha256']
            assert sha(audio_path) == sample['files'][condition + '.wav']['sha256']
            wave, rate = sf.read(audio_path, dtype='float32'); assert rate == 16000 and wave.ndim == 1
            rows.append(dict(sample_id=sample['id'], duration=sample['duration'], condition=condition,
                             image=Image.open(image_path).convert('RGB'), audio=wave))
    accepted = []
    for arm in unfinished:
        if arm['adapter']:
            model = build_model(c, adapter=arm['adapter'], trainable=False)
            actual = get_peft_model_state_dict(model)
            expected = load_file(str(Path(arm['adapter']) / 'adapter_model.safetensors'), device='cpu')
            assert actual.keys() == expected.keys()
            assert all(torch.equal(actual[k].cpu(), expected[k]) for k in expected), arm['name']
            del actual, expected
            base = model.get_base_model()
        else:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                c['model'], local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
            base = model
        model.cuda().eval().requires_grad_(False)
        accepted.append(arm)
        save(WORK / f'MODEL_ACCEPTANCE_{slot}.json', dict(at=time.time(), pid=os.getpid(), host=socket.gethostname(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), arms=accepted,
            all_loaded_adapters_tensor_equal_to_disk=True, frozen=True, independent_native_load=True,
            system=SYSTEM,user=USER,target_in_prompt=False))
        started = time.monotonic(); folder = WORK / 'results' / arm['name']
        pending = [r for r in rows if not (folder / r['sample_id'] / (r['condition'] + '.json')).exists()]
        for offset in range(0, len(pending), 16):
            batch = pending[offset:offset+16]
            with torch.inference_mode():
                inputs = collator.source_inputs(batch).to('cuda', dtype=torch.bfloat16); base.rope_deltas = None
                generated = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True,
                    use_audio_in_video=False, eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id)
                outputs = generated[:, inputs.input_ids.shape[1]:].tolist(); del inputs, generated
            for row, ids in zip(batch, outputs):
                attempts = [dict(token_ids=ids, max_new_tokens=512)]
                if eos not in ids:
                    with torch.inference_mode():
                        inputs = collator.source_inputs([row]).to('cuda', dtype=torch.bfloat16); base.rope_deltas = None
                        generated = model.generate(**inputs,max_new_tokens=1024,do_sample=False,use_cache=True,
                            use_audio_in_video=False,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
                        ids = generated[0, inputs.input_ids.shape[1]:].tolist(); del inputs, generated
                    attempts.append(dict(token_ids=ids,max_new_tokens=1024))
                text = processor.tokenizer.decode(ids[:ids.index(eos)+1] if eos in ids else ids, skip_special_tokens=True).strip()
                result = dict(sample_id=row['sample_id'],condition=row['condition'],arm=arm['name'],text=text,
                    ended_with_eos=eos in ids,attempts=attempts,target_in_prompt=False,at=time.time())
                save(folder / row['sample_id'] / (row['condition'] + '.json'), result)
            save(WORK / f'progress_{slot}.json',dict(at=time.time(),pid=os.getpid(),arm=arm['name'],
                done=len(rows)-len(pending)+min(offset+16,len(pending)),total=len(rows),seconds=time.monotonic()-started))
        assert len(list(folder.glob('*/*.json'))) == len(rows)
        save(folder / 'COMPLETE.json',dict(success=True,at=time.time(),outputs=len(rows),seconds=time.monotonic()-started))
        del base, model
        import gc
        gc.collect();torch.cuda.empty_cache()
    save(WORK / f'WORKER_COMPLETE_{slot}.json',dict(at=time.time(),success=True,arms=[a['name'] for a in selected]))


def report():
    protocol = read(WORK / 'protocol.json'); samples = read(WORK / 'samples.json'); stats = []; views = []
    names = [a['name'] for a in protocol['arms']]; blind = list(names); random.Random(SEED).shuffle(blind)
    labels = {name: chr(65+i) for i,name in enumerate(blind)}
    for arm in names:
        original = []; swapped = []
        for row in samples:
            p = WORK / 'results' / arm / row['id']
            original.append(read(p / 'original.json'));swapped.append(read(p / 'mismatched_audio.json'))
        texts = [x['text'] for x in original]
        stats.append(dict(arm=arm,blind=labels[arm],n=64,valid_eos=sum(x['ended_with_eos'] for x in original+swapped),
            nonempty=sum(bool(x['text']) for x in original+swapped),mean_words=sum(len(t.split()) for t in texts)/64,
            unique_original_texts=len(set(texts)),shared_forward_gaze_opening=sum(t.startswith('The listener maintains a steady forward gaze') for t in texts),
            unchanged_when_audio_swapped=sum(x['text']==y['text'] for x,y in zip(original,swapped)),
            mean_token_edit_similarity=sum(difflib.SequenceMatcher(None,x['text'].split(),y['text'].split()).ratio() for x,y in zip(original,swapped))/64,
            retry_count=sum(len(x['attempts'])>1 for x in original+swapped)))
    for row in samples:
        plans = []
        for arm in blind:
            p = WORK / 'results' / arm / row['id']
            plans.append(dict(blind=labels[arm],arm=arm,original=read(p/'original.json')['text'],swapped=read(p/'mismatched_audio.json')['text']))
        views.append(dict(sample=row,plans=plans))
    save(WORK/'RESULTS.json',dict(at=time.time(),stats=stats,scope='Descriptive diagnostics; no semantic accuracy or blind human preference measured',blind_mapping=labels))
    payload=json.dumps(views,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Qwen规划检查点对照</title>
<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:24px auto;padding:0 18px;background:#f5f6fa;color:#182233}select,button{font:inherit;padding:8px}article{background:white;padding:18px;margin:16px 0;border-radius:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:180px;float:right}p{white-space:pre-wrap}audio{width:100%}.muted{color:#596579}@media(max-width:700px){.cols{grid-template-columns:1fr}}</style>
<h1>Qwen规划：原模型与中间检查点</h1><p>64条独立dev；同首帧，原音频／等长错配音频。先听音频，再判断反应是否合适。文字变化不等于理解，真实标签也不是唯一合理回应。</p>
<select id="choose"></select> <button id="prev">上一条</button> <button id="next">下一条</button> <button id="reveal">显示／隐藏权重</button><div id="view"></div>
<script>const data=PAYLOAD;let idx=0,show=false;const choose=document.getElementById('choose'),view=document.getElementById('view');for(let i=0;i<data.length;i++){let o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${data[i].sample.id}`;choose.append(o)}
function draw(){choose.value=idx;view.replaceChildren();let d=data[idx],s=d.sample,h=document.createElement('article');let image=document.createElement('img');image.src=s.files['first_frame.png'].path;h.append(image);let t=document.createElement('p');t.textContent=`${idx+1}/64 · ${s.duration.toFixed(3)}秒 · ${s.stratum}\n错配音频来自另一个dev来源；补齐／裁去 ${(Math.abs(s.donor_adjustment_samples)/16).toFixed(1)} 毫秒。`;h.append(t);let cols=document.createElement('div');cols.className='cols';for(let [label,key] of [['原speaker音频','original.wav'],['错配speaker音频','mismatched_audio.wav']]){let c=document.createElement('div');c.textContent=label;let a=document.createElement('audio');a.controls=true;a.preload='none';a.src=s.files[key].path;c.append(a);cols.append(c)}h.append(cols);view.append(h);for(let plan of d.plans){let art=document.createElement('article'),title=document.createElement('h2');title.textContent=`候选${plan.blind}${show?' · '+plan.arm:''}${plan.original===plan.swapped?' · 两种音频规划逐字相同':''}`;art.append(title);let cs=document.createElement('div');cs.className='cols';for(let [label,text] of [['原音频下规划',plan.original],['错配音频下规划',plan.swapped]]){let c=document.createElement('div'),b=document.createElement('b'),p=document.createElement('p');b.textContent=label;p.textContent=text;c.append(b,p);cs.append(c)}art.append(cs);view.append(art)}let details=document.createElement('details'),sum=document.createElement('summary'),p=document.createElement('p');sum.textContent='查看现有自动标注（仅供参考，不是唯一正确答案）';p.textContent=s.reference_text;details.append(sum,p);view.append(details)}
choose.onchange=()=>{idx=Number(choose.value);draw()};document.getElementById('prev').onclick=()=>{idx=(idx+63)%64;draw()};document.getElementById('next').onclick=()=>{idx=(idx+1)%64;draw()};document.getElementById('reveal').onclick=()=>{show=!show;draw()};draw();</script>'''.replace('PAYLOAD',payload)
    (WORK/'review.html').write_text(page)
    save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),samples=64,outputs=896,semantic_quality_review_pending=True,results_sha256=sha(WORK/'RESULTS.json')))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','generate','report']);parser.add_argument('--slot',type=int,choices=[0,1]);args=parser.parse_args()
    try:
        if args.action=='prepare':prepare()
        elif args.action=='generate':generate(args.slot)
        else:report()
    except BaseException as exc:
        import traceback
        save(WORK/f'FAILURE_{args.action}_{args.slot}.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()));raise
