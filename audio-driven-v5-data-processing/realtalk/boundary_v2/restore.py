"""Restore this approved pilot to a new directory; never overwrite an existing run."""
from pathlib import Path
import argparse,json,hashlib,shutil
ROOT=Path(__file__).resolve().parent

def restore(workdir,data_root,model_path=None):
 workdir=Path(workdir).resolve();data_root=Path(data_root).resolve()
 for entry in json.loads((ROOT/'BACKUP_MANIFEST.json').read_text())['files']:
  assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
 if workdir.exists():raise FileExistsError(f'Refusing existing workdir: {workdir}')
 workdir.mkdir(parents=True)
 shutil.copytree(ROOT/'snapshot/code',workdir/'code');shutil.copytree(ROOT/'dependencies/baseline',workdir/'inputs/baseline');shutil.copytree(ROOT/'snapshot/asr',workdir/'asr');shutil.copyfile(ROOT/'snapshot/words.json',workdir/'words.json');shutil.copyfile(ROOT/'LOCAL_OUTPUTS.json',workdir/'archived_source_evidence.json');shutil.copytree(ROOT/'dependencies/review',workdir/'review');(workdir/'logs').mkdir()
 patches=[]
 def replace(name,old,new):
  p=workdir/'code'/name;s=p.read_text();assert old in s,(name,old);p.write_text(s.replace(old,new));patches.append({'file':name,'old':old,'new':new})
 for name in ['prepare_source.py','run_pilot.py']:replace(name,"BASE=R.parent/'full_20261023'","BASE=R/'inputs/baseline'")
 replace('export_review.py',"O=R.parent/'full_20261023/reports/boundary_v2_review'","O=R/'review'")
 replace('export_review.py',"R.parent/'full_20261023/manifests'","R/'inputs/baseline/manifests'")
 env=json.loads((ROOT/'environment.json').read_text());old_model=env['whisper_model_path']
 if model_path:replace('transcribe_words.py',repr(old_model),repr(str(Path(model_path).resolve())))
 vid='-1wCGAxqT_4';p=workdir/'inputs/baseline/manifests'/f'{vid}.json';m=json.loads(p.read_text());src=m['source'];src['archive']=str(data_root/'videos'/Path(src['archive']).name);src['source']=f"subfile,,start,{src['offset']},end,{src['offset']+src['size']},,:{src['archive']}";src['annotation']=str(workdir/'inputs/baseline/annotations'/f'{vid}.json');src['asr']=str(data_root/'asr'/f'{vid}.json');p.write_text(json.dumps(m,ensure_ascii=False,indent=2))
 for name in ['verify_audio.py','publish_review.py']:shutil.copyfile(ROOT/'helpers'/name,workdir/'code'/name)
 replace('run_pilot.py',"if __name__=='__main__':main()","if __name__=='__main__':\n from verify_audio import verify\n verify();main()")
 html=workdir/'code/review.html';s=html.read_text().replace('<a href="../resolution_review/">旧分辨率审核</a>','旧分辨率审核仅在原环境提供');html.write_text(s)
 (workdir/'RESTORE.json').write_text(json.dumps({'backup':str(ROOT),'workdir':str(workdir),'data_root':str(data_root),'path_patches':patches,'frozen_asr_words_reused':True,'audio_payload_must_match_before_replay':True},ensure_ascii=False,indent=2)+'\n');print('RESTORED',workdir)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workdir',required=True);p.add_argument('--data-root',required=True);p.add_argument('--model-path');a=p.parse_args();restore(a.workdir,a.data_root,a.model_path)
