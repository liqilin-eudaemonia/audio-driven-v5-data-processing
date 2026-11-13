from huggingface_hub import snapshot_download
from pathlib import Path
import json,time
base=Path(__file__).resolve().parents[1]
p=snapshot_download('robinwitch/DyStream',local_dir=base/'DyStream',allow_patterns=['tools/**','checkpoints/**'],max_workers=6)
(base/'deployment/logs/dystream_download_complete.json').write_text(json.dumps({'path':p,'finished':time.time()}))

# The official MediaPipe GPU delegate requires EGL, unavailable on headless node8.
face=base/'DyStream/tools/visualization_0416/utils/face_detector.py'
face.write_text(face.read_text().replace('mediapipe_model_asset_path, delegate=1,','mediapipe_model_asset_path, delegate=0,'))
