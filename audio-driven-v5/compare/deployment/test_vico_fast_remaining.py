"""Resume must not write through archived links or request GPUs for completed work."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from vico_more_common import save, sha
from vico_fast_registry import l2l_group


class RemainingTests(unittest.TestCase):
    def test_l2l_corruption_invalidates_entire_sampling_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'generated/conan';folder.mkdir(parents=True)
            p=dict(samples=[dict(id='one'),dict(id='two')])
            save(folder/'COMPLETE.json',dict(model='conan',samples=2))
            for item in p['samples']:
                path=folder/(item['id']+'.npz');path.write_bytes(b'valid')
                save(path.with_suffix('.json'),dict(id=item['id'],model='conan',feature_sha256=sha(path)))
            self.assertEqual(len(l2l_group(root,'conan',p)),2)
            (folder/'two.npz').write_bytes(b'corrupt')
            self.assertIsNone(l2l_group(root,'conan',p))

    def test_l2l_restore_manifest_without_gpu(self):
        import vico_l2l_fast as module
        old=module.WORK
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(module,'node8_guard',side_effect=AssertionError('unexpected GPU')):
                    module.prepare(Path(tmp));module.generate()
                self.assertTrue((Path(tmp)/'GENERATION_COMPLETE.json').exists())
        finally:module.WORK=old

    def test_l2l_invalid_archive_link_detached(self):
        import vico_l2l_fast as module
        old=module.WORK
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);dest=root/'generated/conan';dest.parent.mkdir()
                target=root/'readonly';target.mkdir();(target/'sentinel').write_text('keep')
                dest.symlink_to(target,target_is_directory=True)
                with patch.object(module,'assets_match',return_value=True),patch.object(module,'l2l_group',return_value=None):
                    module.prepare(root)
                self.assertFalse(dest.is_symlink())
                self.assertEqual((target/'sentinel').read_text(),'keep')
        finally:module.WORK=old

    def test_dystream_restore_without_gpu(self):
        import vico_dystream_fast as module
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(module,'worker',side_effect=AssertionError('unexpected inference')):
                module.generate(Path(tmp))
            self.assertTrue((Path(tmp)/'GENERATION_COMPLETE.json').exists())


if __name__=='__main__':unittest.main()
