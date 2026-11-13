import copy
import tempfile
import unittest
from pathlib import Path
from reference_media_cache import bind,remember
from p2_listen.contracts import read,sha,atomic_json
from p2_event_presence.contracts import digest,save_rows,rows


def fixture(root,name,reference='reference',generated='generated',sid='clip'):
    work=root/name;work.mkdir()
    contract=dict(policy_identity='same-policy',source={'detector.py':'version1'})
    atomic_json(work/'contract.json',contract);media={};detections={}
    for side,content in [('reference',reference),('generated',generated)]:
        identity=dict(rgb_sha256=digest(content+'pixels'),audio_sha256=digest(content+'audio'),duration_s=1.,errors={})
        mid=digest(identity);folder=work/'media'/mid;folder.mkdir(parents=True,exist_ok=True)
        for f in ('video.mkv','audio.wav','frames.npy'):(folder/f).write_bytes((content+f).encode())
        m=dict(media_id=mid,identity=identity,folder=str(folder),duration_s=1.,errors={},
            visual25=str(folder/'video.mkv'),visual25_sha256=sha(folder/'video.mkv'),
            audio=str(folder/'audio.wav'),audio_file_sha256=sha(folder/'audio.wav'),
            windows=[dict(frames=str(folder/'frames.npy'),frames_sha256=sha(folder/'frames.npy'))])
        atomic_json(folder/'candidates.json',dict(cues=['fixed-cue']))
        atomic_json(folder/'detection.json',dict(detector_identity='same-detector',media_identity=identity,
            cache_identity=digest([m,'same-detector',sha(folder/'candidates.json')]),labels={'example':'unknown'}))
        media[side]=m;detections[mid]=sha(folder/'detection.json')
    save_rows(work/'prepared_pairs.jsonl',[dict(dataset='seamless',id=sid,prepared_media=media)])
    atomic_json(work/'PREPARED.json',dict(prepared_pairs_sha256=sha(work/'prepared_pairs.jsonl'),**contract))
    atomic_json(work/'detector_contract.json',dict(identity_hash='same-detector'))
    atomic_json(work/'DETECTION_COMPLETE.json',dict(detector_identity='same-detector',detections=detections))
    return work


class ReferenceMediaCacheChecks(unittest.TestCase):
    def test_reuses_verified_reference_and_keeps_own_generated_media(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);cache=root/'cache';base=fixture(root,'base');remember(base,cache)
            other=fixture(root,'other');before=rows(other/'prepared_pairs.jsonl')[0]
            report=bind(other,cache);after=rows(other/'prepared_pairs.jsonl')[0]
            self.assertEqual(report['reference_cache_hits'],1)
            self.assertEqual(after['prepared_media']['generated'],before['prepared_media']['generated'])
            self.assertEqual(after['prepared_media']['reference'],rows(base/'prepared_pairs.jsonl')[0]['prepared_media']['reference'])
            self.assertEqual(read(other/'PREPARED.json')['prepared_pairs_sha256'],sha(other/'prepared_pairs.jsonl'))
            m=after['prepared_media']['reference'];d=read(Path(m['folder'])/'detection.json')
            self.assertEqual(d['cache_identity'],digest([m,'same-detector',sha(Path(m['folder'])/'candidates.json')]))
            self.assertEqual(d['labels'],{'example':'unknown'})

    def test_no_cache_leaves_prepared_manifest_unchanged(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);work=fixture(root,'new');before=sha(work/'prepared_pairs.jsonl')
            self.assertEqual(bind(work,root/'missing-cache')['reference_cache_hits'],0)
            self.assertEqual(sha(work/'prepared_pairs.jsonl'),before)

    def test_changed_pixels_or_policy_rejected_without_rewriting(self):
        for changed in ('pixels','policy'):
            with tempfile.TemporaryDirectory()as folder:
                root=Path(folder);base=fixture(root,'base');remember(base,root/'cache')
                other=fixture(root,'other',reference='changed'if changed=='pixels'else'reference')
                if changed=='policy':
                    c=read(other/'contract.json');c['policy_identity']='different';atomic_json(other/'contract.json',c)
                before=sha(other/'prepared_pairs.jsonl')
                with self.assertRaises(ValueError):bind(other,root/'cache')
                self.assertEqual(sha(other/'prepared_pairs.jsonl'),before)

    def test_changed_cached_media_or_detection_rejected(self):
        for file in ('audio.wav','detection.json','candidates.json'):
            with tempfile.TemporaryDirectory()as folder:
                root=Path(folder);base=fixture(root,'base');remember(base,root/'cache');other=fixture(root,'other')
                m=rows(base/'prepared_pairs.jsonl')[0]['prepared_media']['reference']
                (Path(m['folder'])/file).write_text('corrupt')
                with self.assertRaises(ValueError):bind(other,root/'cache')

    def test_identical_generated_content_uses_one_native_media_id(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);base=fixture(root,'base');remember(base,root/'cache')
            other=fixture(root,'other',generated='reference');report=bind(other,root/'cache')
            m=rows(other/'prepared_pairs.jsonl')[0]['prepared_media']
            self.assertEqual(m['reference'],m['generated']);self.assertEqual(report['generated_identical_content_hits'],1)

    def test_expansion_preserves_existing_entries(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);cache=root/'cache';base=fixture(root,'base');remember(base,cache)
            old={p.name:sha(p)for p in cache.glob('*.json')}
            expanded=fixture(root,'expanded',reference='newref',sid='newclip');remember(expanded,cache)
            self.assertEqual(len(list(cache.glob('*.json'))),2)
            for name,h in old.items():self.assertEqual(sha(cache/name),h)


if __name__=='__main__':unittest.main(verbosity=2)
