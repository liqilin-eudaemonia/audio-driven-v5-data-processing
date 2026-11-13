"""Independent media/role/timestamp checks on the complete original-view inputs."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import zlib

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from p2_avatarforcing_table4_generate import read, save, sha


def main():
    import cv2
    import numpy as np
    import soundfile as sf
    cv2.setNumThreads(1)
    metadata_path = Path('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv')
    with metadata_path.open() as stream:
        canonical = {r['uuid']: r for r in csv.DictReader(stream) if r['data_split']=='test'}
    assert len(canonical) == 43
    manifest = read(WORK / 'ORIGINAL_TEST43_MANIFEST.json')
    sources = {(r['id'], r['role']): r for r in manifest['records']}
    generation = read(WORK / 'NATIVE_GENERATION_PROTOCOL.json')
    assert set(canonical) == {s['id'] for s in generation['samples']}
    source_checks = {}
    with zipfile.ZipFile(manifest['archive']) as archive:
        for record in manifest['records']:
            path = Path(record['path'])
            assert path.stem == canonical[record['id']][record['role']]
            info = archive.getinfo(record['source_member'])
            if str(path) not in source_checks:
                checksum, digest, count = 0, hashlib.sha256(), 0
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(4*1024*1024), b''):
                        checksum=zlib.crc32(block, checksum)
                        digest.update(block)
                        count+=len(block)
                assert count == info.file_size == record['bytes']
                assert checksum == info.CRC == record['crc32']
                assert digest.hexdigest() == record['sha256']
                source_checks[str(path)] = dict(bytes=count, crc32=checksum, sha256=digest.hexdigest())
    records, frame_totals = [], dict(speaker=0, listener=0)
    for sample in generation['samples']:
        uid = sample['id']
        expected = round(sample['duration']*25)
        wave, sr = sf.read(sample['speaker'], dtype='float32')
        silent, zr = sf.read(sample['silent'], dtype='float32')
        assert sr == zr == 16000 and wave.shape == silent.shape == (round(sample['duration']*16000),)
        assert np.count_nonzero(silent) == 0
        assert hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest() == sample['speaker_waveform_sha256']
        for role in ['speaker', 'listener']:
            folder = WORK / 'inputs' / uid / role
            receipt = read(folder / 'complete.json')
            assert receipt['frames'] == expected and receipt['fps'] == 25
            for path, digest in receipt['files'].items():
                assert sha(path) == digest, path
            paths = sorted((folder / 'cropped25').glob('*.jpg'))
            raw = sorted((folder / 'raw25').glob('*.jpg'))
            assert len(paths) == len(raw) == expected
            trace = read(folder / 'detection_trace.json')
            assert len(trace['boxes']) == expected and trace['source_rectangle'] == receipt['source_rectangle']
            x1,y1,x2,y2 = receipt['source_rectangle']
            assert 0 <= x1 < x2 <= trace['source_size'][0]
            assert 0 <= y1 < y2 <= trace['source_size'][1]
            assert x2-x1 == y2-y1
            for index in sorted({0, expected//2, expected-1}):
                image = cv2.imread(str(raw[index]))
                resized = cv2.resize(image[y1:y2,x1:x2], (512,512), interpolation=cv2.INTER_AREA)
                okay, jpeg = cv2.imencode('.jpg', resized)
                assert okay and jpeg.tobytes() == paths[index].read_bytes(), (uid,role,index)
            cap = cv2.VideoCapture(str(folder / 'cropped25.mkv'))
            fps, count = cap.get(cv2.CAP_PROP_FPS), round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            okay, first = cap.read()
            cap.release()
            assert okay and first.shape == (512,512,3) and fps == 25 and count == expected
            # The container was encoded from JPEGs by ffmpeg. JPEG chroma
            # upsampling can differ from OpenCV/libjpeg, so test losslessness
            # against the actual ffmpeg input decoder and report that delta.
            raw_bytes = subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','1',
                '-i',str(paths[0]),'-frames:v','1','-f','rawvideo','-pix_fmt','bgr0','-threads','1','pipe:1'])
            decoded = np.frombuffer(raw_bytes,dtype=np.uint8).reshape(512,512,4)[...,:3]
            assert np.array_equal(first,decoded), (uid,role,'FFV1 differs from actual ffmpeg input pixels')
            cv_jpeg = cv2.imread(str(paths[0]))
            decoder_delta = np.abs(first.astype(np.int16)-cv_jpeg.astype(np.int16))
            if role == 'listener':
                cap = cv2.VideoCapture(sources[uid,role]['path'])
                okay, first_original = cap.read()
                cap.release()
                assert okay and np.array_equal(first_original,cv2.imread(str(folder/'first_original.png')))
            records.append(dict(id=uid, role=role, frames=expected, fps=fps,
                source_stem=Path(sources[uid,role]['path']).stem,
                canonical_person_id=canonical[uid][role+'_id'], rectangle=receipt['source_rectangle'],
                hashed_files=len(receipt['files']), three_crop_reencodes_byte_equal=True,
                lossless_video_matches_ffmpeg_input_decoder=True,
                ffmpeg_vs_opencv_jpeg_max_abs=int(decoder_delta.max()),
                ffmpeg_vs_opencv_jpeg_mean_abs=float(decoder_delta.mean()),
                first_original_frame_verified=role=='listener'))
            frame_totals[role]+=expected
        print('Audited pair',len(records)//2,'/43',uid,flush=True)
    save(WORK / 'ORIGINAL_INPUTS_INDEPENDENT_AUDIT.json', dict(passed=True, pairs=43, roles=86,
        originals=len(source_checks), frame_totals=frame_totals, records=records,
        canonical_metadata_sha256=sha(metadata_path), script_sha256=sha(__file__),
        generation_protocol_sha256=sha(WORK/'NATIVE_GENERATION_PROTOCOL.json'),
        originals_verified_against_zip_crc_and_size=True, condition_audio_full_length_and_silent_own=True,
        longest_seconds=max(s['duration'] for s in generation['samples']),
        clips_exceeding_cli30_seconds=[dict(id=s['id'],seconds=s['duration']) for s in generation['samples'] if s['duration']>30],
        decoder_note='FFV1 preserves ffmpeg JPEG-decoded bgr0 bytes; OpenCV JPEG chroma upsampling differs and is explicitly quantified, not claimed byte-identical',
        scope='Role mapping, source bytes, all hashed cropped files, exact representative crop JPEGs, video headers and first pixels, first-original references, condition waveforms',
        full_video_decode_covered_by='Subsequent129stream extractor and final feature coverage audit; header checks alone do not prove full decode'))


if __name__ == '__main__':
    main()
