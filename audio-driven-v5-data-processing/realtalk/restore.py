"""Materialize an independent run, preserving the archived source snapshot."""
import argparse, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workdir',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;out=a.workdir.resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copytree(root/'snapshot/code',out/'code');shutil.copytree(root/'dependencies',out/'dependencies');shutil.copytree(root/'snapshot/reports/assets',out/'reports/assets')
for n in ['annotations','scans','manifests','logs']:(out/n).mkdir()
for f in (out/'code').glob('*.py'):
 s=f.read_text().replace(repr('/private/liqi_workspace/realtalk_inspection/full_video_v2_sentence/review/samples.json'),repr(str(out/'dependencies/approved_reference/review/samples.json'))).replace(repr('/public/0_DATA/2_Avatar/realtalk/preview/serve.py'),repr(str(out/'dependencies/preview/serve.py'))).replace(repr('/public/0_DATA/2_Avatar/realtalk'),repr(str(a.data_root.resolve())))
 if f.name=='index_dataset.py':s=s.replace("SRC/'preview/catalog.json'","ROOT/'dependencies/preview/catalog.json'")
 compile(s,str(f),'exec');f.write_text(s)
print(out)
