"""Data integrity checks for the full caption filtering pipeline (no GPU)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full_filter',ROOT/'scripts/filter_full_response_sft.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FilterTests(unittest.TestCase):
    def fixture(self,directory,split_counts=(('dev',1),('train',2))):
        work=Path(directory)/'job'
        source=Path(directory)/'source'
        source.mkdir()
        work.mkdir()
        (source/'crop_config.json').write_bytes((module.SOURCE/'crop_config.json').read_bytes())
        originals={}
        cfg=dict(schema='test',source=str(source),total=sum(n for _,n in split_counts),sources={},splits=[s for s,_ in split_counts])
        for split,count in split_counts:
            with (module.SOURCE/f'{split}.jsonl').open('rb') as f:
                lines=[f.readline() for _ in range(count)]
            (source/f'{split}.jsonl').write_bytes(b''.join(lines))
            originals[split]=lines
            cfg['sources'][split]=dict(path=str(source/f'{split}.jsonl'),count=count,
                sha256=module.sha(source/f'{split}.jsonl'))
            records=[]
            offset=0
            counts={}
            for i,raw in enumerate(lines):
                row=json.loads(raw)
                decision='keep' if split=='dev' else ('reject' if i==0 else 'needs_review')
                if split=='test':
                    decision=('keep','reject','needs_review')[i]
                counts[decision]=counts.get(decision,0)+1
                records.append(dict(sample_id=row['sample_id'],split=split,duration_s=row['duration_s'],
                    source_offset=offset,source_bytes=len(raw),source_line_sha256=module.hashlib.sha256(raw).hexdigest(),
                    caption_sha256=module.hashlib.sha256(row['target']['text'].encode()).hexdigest(),
                    decision=decision,in_current_500h_trial=row['provenance']['in_current_500h_trial'],
                    listener_waveform_nonzero=row['provenance']['processed_listener_waveform_nonzero']))
                offset+=len(raw)
            out=work/'batches'/split/'000000.jsonl'
            out.parent.mkdir(parents=True)
            out.write_text(''.join(json.dumps(r)+'\n' for r in records))
            module.save(out.with_suffix('.meta.json'),dict(split=split,block=0,start_offset=0,end_offset=offset,
                count=count,counts=counts,result_file=str(out.relative_to(work)),result_sha256=module.sha(out)))
        module.save(work/'config.json',cfg)
        return work,source,originals

    def test_exact_evidence_and_consistent_decision_required(self):
        caption='She nods once and remains silent.'
        self.assertTrue(module.parse_answer('{"keep":true,"defects":[]}', 'stop',caption)['keep'])
        for response,finish in [('{"keep":true,"defects":[]}','length'),
                ('{"keep":false,"defects":[]}','stop'),
                ('{"keep":false,"defects":[{"category":"A","quote":"supplementary data","reason":"external"}]}','stop')]:
            with self.assertRaises(ValueError):
                module.parse_answer(response,finish,caption)

    def test_quote_alignment_never_changes_evidence_words(self):
        answer=json.dumps(dict(keep=False,defects=[dict(category='B',quote='wearing a red shirt',reason='clothing')]))
        result=module.parse_answer(answer,'stop','Wearing  a red shirt, she nods.')
        self.assertEqual(result['defects'][0]['quote'],'Wearing  a red shirt')
        self.assertEqual(result['defects'][0]['original_model_quote'],'wearing a red shirt')
        with self.assertRaises(ValueError):
            module.parse_answer(answer,'stop','Wearing a blue shirt, she nods.')

    def test_missing_vocal_information_is_not_inferred_as_silence(self):
        self.assertTrue(module.rule_evidence('She nods once and looks down.'))
        self.assertTrue(module.rule_evidence('Her mouth moves, though no speech is audible.'))
        self.assertEqual(module.rule_evidence('She nods once and remains silent.'),[])
        self.assertEqual(module.rule_evidence('She says "Yeah" and smiles.'),[])
        self.assertEqual(module.rule_evidence('No audible speech or vocalizations occur.'),[])

    def test_exports_preserve_source_lines_splits_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            work,source,originals=self.fixture(directory)
            module.finalize(work)
            self.assertEqual((work/'filtered/dev.jsonl').read_bytes(),originals['dev'][0])
            self.assertEqual((work/'filtered/train.jsonl').read_bytes(),b'')
            self.assertEqual((work/'filtered/rejected/train.jsonl').read_bytes(),originals['train'][0])
            self.assertEqual((work/'filtered/needs_review/train.jsonl').read_bytes(),originals['train'][1])
            self.assertFalse((work/'filtered/test.jsonl').exists())
            with module.sqlite3.connect(work/'filtered/index.sqlite') as db:
                row=db.execute('SELECT split,file,offset,bytes FROM samples').fetchone()
                self.assertEqual(row,('dev','dev.jsonl',0,len(originals['dev'][0])))
            self.assertEqual(module.read(work/'COMPLETE.json')['processed'],3)
            module.finalize(work)  # Publishing again is a no-op after verified completion.

    def test_changed_source_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            work,source,originals=self.fixture(directory)
            (source/'dev.jsonl').write_bytes(originals['dev'][0].replace(b'The listener',b'The speaker',1))
            with self.assertRaises(AssertionError):
                module.finalize(work)
            self.assertFalse((work/'COMPLETE.json').exists())
            self.assertFalse((work/'filtered').exists())

    def test_test_only_partition_exports_without_train_or_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            work,source,originals=self.fixture(directory,(('test',3),))
            cfg=module.read(work/'config.json')
            metas,starts=module.completed(work,cfg)
            self.assertEqual(set(starts),{'test'})
            self.assertEqual(module.aggregate(metas,cfg)['processed'],3)
            module.finalize(work)
            self.assertEqual((work/'filtered/test.jsonl').read_bytes(),originals['test'][0])
            self.assertEqual((work/'filtered/rejected/test.jsonl').read_bytes(),originals['test'][1])
            self.assertEqual((work/'filtered/needs_review/test.jsonl').read_bytes(),originals['test'][2])
            self.assertFalse((work/'filtered/train.jsonl').exists())
            self.assertFalse((work/'filtered/dev.jsonl').exists())
            self.assertTrue(module.read(work/'filtered/summary.json')['test_processed'])
            with module.sqlite3.connect(work/'filtered/index.sqlite') as db:
                self.assertEqual(db.execute('SELECT split,file,offset,bytes FROM samples').fetchall(),[('test','test.jsonl',0,len(originals['test'][0]))])

    def test_resume_requires_contiguous_committed_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            work,_,_=self.fixture(directory)
            meta=work/'batches/dev/000000.meta.json'
            data=module.read(meta)
            data['block']=1
            module.save(meta,data)
            with self.assertRaises(AssertionError):
                module.completed(work,module.read(work/'config.json'))


if __name__=='__main__':
    unittest.main()
