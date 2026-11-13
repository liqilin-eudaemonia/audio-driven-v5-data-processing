from huggingface_hub import snapshot_download
from pathlib import Path
import json,time
base=Path(__file__).resolve().parents[1]/'SpeakerVid-5M-Code'
snapshot_download('dorni/SpeakerVid-5M-data-curation-models',local_dir=base,allow_patterns=['assets/**','ckpt/**','weights/**'],max_workers=5)
# README requires expression metric assets in this location.
target=base/'metrics/express/assets'
target.mkdir(parents=True,exist_ok=True)
for item in (base/'assets').glob('*'):
 link=target/item.name
 if item.is_file() and not link.exists():link.symlink_to('../../../assets/'+item.name)
(base.parent/'deployment/logs/speakervid_download_complete.json').write_text(json.dumps({'finished':time.time(),'repo':'dorni/SpeakerVid-5M-data-curation-models','kind':'data curation and evaluation only; not video generation'}))
