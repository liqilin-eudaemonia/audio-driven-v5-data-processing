"""Current-label references for the frozen planning SFT input pipeline."""
import json
from pathlib import Path
import sys
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from j_upright_response_sft_data import resolve,load_example as current_load

def resolve_example(row,crop_config):
    # resolve verifies latest label bytes, caption SHA/body, pairing, split,
    # original orientation lineage and final geometry before any cache lookup.
    item=resolve(row)
    cap=json.loads(Path(row['target']['source_file']).read_bytes())
    return dict(sample_id=row['sample_id'],split=row['split'],source=item['source'],
                speaker_audio=item['speaker'],target_text=row['target']['text'],
                crop_provenance=cap['crop_provenance'],annotation_crop_differs=False)

def load_example(row,crop_config):
    return current_load(row)
