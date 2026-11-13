import json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.native_caption_nod import PROMPT_STYLES
from listen_benchmark.j_omni import fingerprint,file_sha
work=Path(__file__).resolve().parent
scope=json.loads((ROOT/'experiments/listen_j_annotation_two_nodes_20261030/nodes.json').read_text())
nodes={}
for name,n in scope['nodes'].items():
 old=Path(n['work']);c=json.loads((old/'config.json').read_text());nodes[name]=dict(host=n['host'],hostname=c.get('expected_hostname','dedicated-developjob-lq-dev-machine-2-taaft'),old_work=n['work'],resource_work=n['work'],scope_sources=n['scope_sources'],scope_clips=n['scope_clips'],gpu_ids=list(range(8)),memory=.85)
paths=['src/listen_benchmark/j_omni.py','scripts/listen_j_omni.py','src/listen_benchmark/j_annotation.py','src/listen_benchmark/native_caption_nod.py','src/listen_benchmark/native_caption_quality.py','src/listen_benchmark/full_raw.py','src/listen_benchmark/media.py','src/listen_labeling/media.py','src/listen_benchmark/official_audio_gate.py','src/listen_audio_postprocess/recipes.py','src/listen_audio_postprocess/bulk.py','scripts/compare_listener_official_six.py','scripts/listen_visual_native8_nod.py']
hashes={p:file_sha(ROOT/p) for p in paths}
cfg=dict(schema='listen-j-omni-stream-v1',project_root=str(ROOT),scope_file=str(ROOT/'experiments/listen_j_annotation_two_nodes_20261030/nodes.json'),total_clips=scope['total_clips'],nodes=nodes,
 prompt_style='first_plus_evidence',prompt_version=PROMPT_STYLES['first_plus_evidence'][0],prompt=PROMPT_STYLES['first_plus_evidence'][1],model='/private/ckpts/Qwen3-Omni/Instruct',fps=8,output_size=[512,512],
 audio_policy='official_word_sentence_vad_union_only',chinese_filter=False,run_asr=False,new_wav_files=False,training_admitted=False,
 decoding_attempts=[dict(temperature=0,max_tokens=4096,repetition_penalty=1.05),dict(temperature=0,max_tokens=4096,repetition_penalty=1.12)],decode_slots=112,backlog_sources=1024,code_hashes=hashes)
cfg['pipeline_id']=fingerprint({k:v for k,v in cfg.items() if k not in ['nodes','code_hashes','project_root']})
if (work/'config.json').exists():assert json.loads((work/'config.json').read_text())==cfg,'Existing production config differs'
else:(work/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
for p in paths:
 dest=work/'code'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
(work/'prompt.txt').write_text(cfg['prompt']+'\n')
print('frozen',cfg['pipeline_id'],cfg['total_clips'])
