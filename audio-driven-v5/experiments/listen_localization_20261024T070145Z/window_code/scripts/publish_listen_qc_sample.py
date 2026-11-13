"""Publish only the bounded audio quality review; never launch bulk work."""
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
import preview_listen_v6 as preview
from listen_policy_v6 import POLICY as VIDEO_POLICY
from p3_listen.common import read, write

ROOT = preview.ROOT
OUT = ROOT / 'data_previews/seamless_listen_qc_review'


def main():
    selection = read(OUT / 'publication_selection.json')
    if not 1 <= len(selection['rows']) <= 20:
        raise ValueError('Bounded sample review only')
    preview.OUT = OUT
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(preview.publish, selection['rows']))
    screen = read(OUT / 'screened_candidates.json')
    write(OUT / 'samples.json', dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        status='human_review_pending_bulk_and_training_on_hold',
        seed=selection['seed'], samples=rows, counts=screen['counts'],
        inspected_pairs=48, audio_policy=screen['policy'], video_policy=VIDEO_POLICY,
        model_contract=read(ROOT / 'experiments/p3_listen_qc_review/model_contract.json'),
        note=selection['note'], denoising=False,
        pool_note='Previously normalized train/dev pairs with eligible crops; excludes previous review groups. Not a corpus yield estimate.'))
    shutil.copyfile(ROOT / 'scripts/listen_qc_review.html', OUT / 'index.html')
    link = ROOT / 'data_previews/seamless_face_listening_v1/listen_qc'
    if not link.exists():
        link.symlink_to(OUT, target_is_directory=True)
    print('Published http://127.0.0.1:8875/listen_qc/', flush=True)


if __name__ == '__main__':
    main()
