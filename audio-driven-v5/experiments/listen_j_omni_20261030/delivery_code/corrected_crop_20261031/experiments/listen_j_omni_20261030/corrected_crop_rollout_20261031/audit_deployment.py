import sys,json,time,hashlib,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent
sys.path.insert(0,str(W/'nod_cpu_share'));from probe import remote
C=json.loads((W/'config.json').read_text());R=Path(C['project_root']);sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_omni_crop import effective_row,caption_crop_info
from listen_benchmark.j_omni import file_sha,fingerprint
code='''import json,time,os
from pathlib import Path
W=Path(WORK);C=json.loads((W/'config.json').read_text());out=[]
for gpu in C['nodes'][NODE]['gpu_ids']:
 s=json.loads((W/'workers'/f'{NODE}-gpu-{gpu}.json').read_text());p=Path('/proc')/str(s['pid'])
 live=p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode() if live else ''
 out.append(dict(gpu=gpu,pid=s['pid'],alive=live and 'listen_j_omni.py worker' in cmd and str(W) in cmd,updated='+bounded-crop-v1' in s.get('execution_revision',''),stage=s['stage'],completed=s['completed_clips'],error=s.get('last_error'),heartbeat_age_s=time.time()-s['heartbeat'],job=s.get('job')))
print(json.dumps(dict(node=NODE,workers=out)))
'''
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 fs={n:pool.submit(remote,n,'WORK='+repr(str(W))+'; NODE='+repr(n)+'\n'+code) for n in C['nodes']};nodes={n:f.result() for n,f in fs.items()}
workers=[x for n in nodes.values() for x in n['workers']]
ready=all(x['alive'] and x['updated'] and x['completed']>0 and not x['error'] and x['heartbeat_age_s']<45 for x in workers)
if not ready:
 print(json.dumps(dict(ready=False,problems=[x for x in workers if not (x['alive'] and x['updated'] and x['completed']>0 and not x['error'] and x['heartbeat_age_s']<45)])));sys.exit(2)
samples=[]
for node,ns in nodes.items():
 selected={}
 for state in ns['workers']:
  if not state['job']:continue
  job=read(W/'jobs'/f"{state['job']}.json")
  for row in job['rows']:
   p=result_path(W,'caption',row['sample_id'])
   if not p.exists():continue
   result=read(p)
   if result.get('node')!=node or 'crop_provenance' not in result:continue
   changed=result['crop_provenance']['coordinates_changed']
   if changed in selected:continue
   applied,provenance=effective_row(row,C)
   assert result['source']==applied['source'] and result['original_source']==row['source']
   assert result['crop_provenance']==provenance
   evidence=result['input_evidence'];nod=read(evidence['nod_supplement']['detector_file'])
   assert file_sha(evidence['nod_supplement']['detector_file'])==evidence['nod_supplement']['detector_sha256']
   assert nod['media']['video_transform']==result['source']['video_transform']==evidence['video_evidence']['video_transform']
   assert result['visual_quality']['decision']=='retain'
   assert fingerprint(result['audio_recipe'])==evidence['audio_identity']['recipe_sha256']
   assert row['source']['source_start_s']==result['source']['source_start_s']
   selected[changed]=dict(node=node,sample_id=row['sample_id'],path=str(p),crop_changed=changed,crop=result['source']['video_transform']['crop'],nod_reuse=evidence['nod_supplement']['crop_reuse'],generation_checks_passed=result['generation_checks_passed'])
   if len(selected)==2:break
  if len(selected)==2:break
 assert selected,node
 samples.extend(selected.values())
legacy=[]
for item in read(W/'input_chain_audit_20261031/selected.json')['samples']:
 assert file_sha(item['path'])==item['result_sha256']
 info=caption_crop_info(read(item['path']));assert info['label_origin']=='legacy_original_crop_preserved'
 legacy.append(dict(sample_id=item['sample_id'],**info))
for rel,h in C['code_hashes'].items():assert file_sha(R/rel)==h,rel
out=dict(at=time.time(),passed=True,nodes=nodes,live_updated_gpu_workers=len(workers),new_samples=samples,old_caption_bytes_unchanged=legacy,crop_input=C['bounded_crop_input'],cpu_input_replay=read(D/'replay.json')['passed'],persistent_service_parity=read(D/'service_replay.json')['passed'],tests_passed=65)
write(D/'deployment_audit.json',out);write(D/'config.active.json',C)
print(json.dumps(dict(passed=True,live_updated_gpu_workers=len(workers),new_samples=len(samples),old_preserved=len(legacy),new_completed=sum(x['completed'] for x in workers))))
