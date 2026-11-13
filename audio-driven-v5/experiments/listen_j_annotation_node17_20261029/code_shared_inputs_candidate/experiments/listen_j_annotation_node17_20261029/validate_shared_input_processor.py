"""CPU-only equality check against the installed model's video processor."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import hashlib
import json
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parents[1] / 'src'))


def main():
    os.nice(10)
    import torch
    from transformers import AutoProcessor
    from listen_benchmark.visual_native8 import prepare_media, prepare_input, MODEL
    from listen_benchmark.frame_crop import crop_original_frame
    from listen_benchmark.shared_video_inputs import prepare_shared_inputs
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    row = json.loads((WORK / 'jobs/naturalistic_V02_S5078_I00000281.json').read_text())['rows'][0]
    whole = prepare_media(row['source'], frame_crop=crop_original_frame)
    shared = prepare_shared_inputs(processor, whole)
    records = []
    for view in [list(shared)[0], list(shared)[1], list(shared)[-1]]:
        start, end = shared[view]['interval']
        old, _ = prepare_input(processor, whole, start, end, view)
        new, _ = shared[view]['input']
        arr, _ = new['multi_modal_data']['video'][0]
        before = hashlib.sha256(arr.tobytes()).hexdigest()
        def convert(entry):
            a, m = entry['multi_modal_data']['video'][0]
            # Match the installed vLLM Qwen3-VL adapter: sampling is a processor
            # option, not a transformers VideoMetadata dataclass field.
            metadata = {k: v for k, v in m.items() if k != 'do_sample_frames'}
            return processor.video_processor(videos=[a], video_metadata=[metadata],
                do_resize=False, do_sample_frames=False, return_tensors='pt')
        a, b = convert(old), convert(new)
        assert set(a) == set(b)
        keys = []
        for key in a:
            if torch.is_tensor(a[key]):
                assert torch.equal(a[key], b[key]), key
                keys.append({'key': key, 'shape': list(a[key].shape)})
            else:
                assert a[key] == b[key], key
        assert hashlib.sha256(arr.tobytes()).hexdigest() == before
        records.append({'view': view, 'frames': len(arr), 'tensor_keys': keys,
                        'exact': True, 'source_array_unmodified': True})
    report = {'at': time.time(), 'sample_id': row['sample_id'],
        'scope': 'Installed Qwen3-VL video processor full/first/last windows; no GPU inference yet',
        'records': records}
    (WORK / 'shared_input_processor_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
