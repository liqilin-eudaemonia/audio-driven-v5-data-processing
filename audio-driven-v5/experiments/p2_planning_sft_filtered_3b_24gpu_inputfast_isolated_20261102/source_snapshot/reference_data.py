"""Indexed latest reference manifests; original cached-data mode remains separate."""
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3

import numpy as np
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from p2_listen.contracts import read,sha,digest,atomic_json
from listen_response_sft import load_example

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='planning-reference-data-seal-v1'


def fingerprint(path):
    s=Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def reader_identity():
    names=['listen_response_sft/data.py','listen_benchmark/j_omni_crop.py','listen_benchmark/j_omni.py',
           'listen_benchmark/frame_crop.py','listen_visual_quality/measurement.py','listen_visual_quality/io.py']
    return {n:sha(ROOT/'src'/n) for n in names}


def connection(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def prepare(directory,output,expected_train=690337,expected_dev=5238,seed=20261102,diagnostic_count=256):
    directory=Path(directory).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing data seals')
    if type(diagnostic_count) is not int or diagnostic_count<1:raise ValueError('Invalid diagnostic count')
    summary=read(directory/'summary.json');files={};splits={};groups={};pairs={}
    # All checks precede exporting a usable seal. Only sequential JSONL reads.
    index=directory/'index.sqlite';before_index=fingerprint(index);conn=connection(index)
    extra=conn.execute("select count(*) from samples where split not in ('train','dev') or split is null").fetchone()[0]
    if extra:raise ValueError('Unexpected split in index')
    try:
        for split,expected in [('train',expected_train),('dev',expected_dev)]:
            path=directory/(split+'.jsonl');before=fingerprint(path);h=hashlib.sha256();count=0;hours=0.;seen=set();grp=set();pair=set()
            cursor=conn.execute('select sid,split,file,offset,bytes,duration,group_id,pair from samples where split=? order by offset',(split,))
            offset=0
            with path.open('rb') as stream:
                for raw in stream:
                    h.update(raw);record=json.loads(raw);loc=cursor.fetchone()
                    if loc is None:raise ValueError('JSONL has more rows than index')
                    sid,sp,filename,pos,size,duration,g,p=loc
                    if (sid,sp,filename,pos,size)!=(record['sample_id'],record['split'],split+'.jsonl',offset,len(raw)):
                        raise ValueError('JSONL/index identity or byte range mismatch')
                    if sid in seen or sp!=split:raise ValueError('Duplicate/cross-split sample')
                    if not math.isfinite(duration) or duration<=0 or abs(duration-record['duration_s'])>1e-7:raise ValueError('Invalid/index-mismatched duration')
                    if record['group_id']!=g or record['pair']!=p:raise ValueError('Index grouping mismatch')
                    text=record['target']['text']
                    if not isinstance(text,str) or not text.strip() or '<|' in text or '|>' in text:raise ValueError('Invalid assistant target: '+sid)
                    if set(record['inputs'])!={'speaker_audio','listener_first_frame'}:raise ValueError('Unexpected model condition')
                    seen.add(sid);grp.add(g);pair.add(p);hours+=duration/3600;count+=1;offset+=len(raw)
            if cursor.fetchone() is not None or count!=expected:raise ValueError(f'{split}: wrong indexed sample count {count}/{expected}')
            if count!=summary['splits'][split]['counts']['keep'] or h.hexdigest()!=summary['manifests'][split+'.jsonl']['sha256']:
                raise ValueError('Source differs from approved summary')
            if abs(hours-summary['splits'][split]['clip_hours']['keep'])>1e-6:raise ValueError('Hours differ from approved summary')
            if fingerprint(path)!=before:raise ValueError('Manifest changed during preflight')
            files[path.name]=dict(**before,sha256=h.hexdigest());groups[split]=grp;pairs[split]=pair
            ordered=sorted(seen)
            splits[split]=dict(count=count,hours=hours,ids_sha256=digest(ordered))
            if split=='dev':
                sampled=random.Random(seed).sample(ordered,min(diagnostic_count,len(ordered)))
                splits[split]['diagnostic_ids']=sorted(sampled)
            print(f'{split}: verified {count} rows, {hours:.5f}h',flush=True)
        if groups['train']&groups['dev'] or pairs['train']&pairs['dev']:raise ValueError('train/dev source group overlap')
        if fingerprint(index)!=before_index:raise ValueError('Index changed during preflight')
        for name in ('index.sqlite','crop_config.json','summary.json'):
            path=directory/name;before=fingerprint(path);files[name]=dict(**before,sha256=sha(path))
            if fingerprint(path)!=before:raise ValueError('File changed while hashing')
        result=dict(schema=SCHEMA,directory=str(directory),files=files,splits=splits,seed=seed,
            reader_code=reader_identity(),source_groups_disjoint=True,source_pairs_disjoint=True,
            scope='all manifest rows/index offsets/text checked; media and provenance decoded only in sampled verification')
        atomic_json(output,result,immutable=True);return result
    finally:conn.close()


class ReferencePlanningDataset(Dataset):
    def __init__(self,seal,split,selection=None,limit=None,seed=20261102,input_pipeline=None):
        if split not in ('train','dev'):raise ValueError('Only train/dev allowed')
        self.seal_path=Path(seal).resolve();self.seal=read(self.seal_path)
        if self.seal['schema']!=SCHEMA:raise ValueError('Unknown data seal')
        self.root=Path(self.seal['directory']);self.split=split
        for name,expected in self.seal['files'].items():
            if fingerprint(self.root/name)!={k:expected[k] for k in ('bytes','mtime_ns')}:raise ValueError('Prepared input changed: '+name)
        if self.seal['reader_code']!=reader_identity():raise ValueError('Source media reader changed; revalidate data seal')
        self.crop_config=read(self.root/'crop_config.json')
        with closing(connection(self.root/'index.sqlite')) as c:
            values=c.execute('select sid,duration from samples where split=? order by sid',(split,)).fetchall()
        ids=[x[0] for x in values]
        if len(ids)!=self.seal['splits'][split]['count'] or digest(ids)!=self.seal['splits'][split]['ids_sha256']:raise ValueError('Indexed split membership changed')
        durations=dict(values)
        if selection=='diagnostic':
            if split!='dev':raise ValueError('Diagnostic selection must be dev')
            ids=self.seal['splits']['dev']['diagnostic_ids']
        elif selection is not None:raise ValueError('Unknown selection')
        if limit is not None:
            if type(limit) is not int or limit<1:raise ValueError('Invalid sample limit')
            ids=sorted(random.Random(seed).sample(ids,min(limit,len(ids))))
        if not ids:raise ValueError('Empty selection')
        self.ids=ids;self.lengths=[round(durations[sid]*25)+256 for sid in ids]
        self.manifest_sha256=self.seal['files'][split+'.jsonl']['sha256']
        self.identity=dict(schema=SCHEMA,seal_sha256=sha(self.seal_path),split=split,selection=selection,
            selected_ids_sha256=digest(ids),count=len(ids),index_sha256=self.seal['files']['index.sqlite']['sha256'],
            crop_config_sha256=self.seal['files']['crop_config.json']['sha256'],reader_code=self.seal['reader_code'])
        self._pid=None;self._conn=None;self._stream=None
        self.input_pipeline=input_pipeline;self._input_reader=None
    def __len__(self):return len(self.ids)
    def close(self):
        if self._conn is not None:self._conn.close()
        if self._stream is not None:self._stream.close()
        self._conn=self._stream=None;self._pid=None
    def __getstate__(self):
        state=dict(self.__dict__);state.update(_conn=None,_stream=None,_pid=None);return state
    def _open(self):
        if self._pid!=os.getpid():
            self.close();self._conn=connection(self.root/'index.sqlite')
            self._stream=(self.root/(self.split+'.jsonl')).open('rb');self._pid=os.getpid()
    def record(self,index):
        self._open();sid=self.ids[index]
        loc=self._conn.execute('select split,file,offset,bytes,duration from samples where sid=?',(sid,)).fetchone()
        if loc is None or loc[0]!=self.split or loc[1]!=self.split+'.jsonl':raise ValueError('Wrong split/file in index')
        self._stream.seek(loc[2]);raw=self._stream.read(loc[3]);row=json.loads(raw)
        if row['sample_id']!=sid or row['split']!=self.split or abs(row['duration_s']-loc[4])>1e-7:raise ValueError('Wrong indexed row')
        return row
    def __getitem__(self,index):
        row=self.record(index)
        if self.input_pipeline is not None:
            if self._input_reader is None:
                from .fast_media import FastInputs
                self._input_reader=FastInputs(self.input_pipeline)
            example=self._input_reader(row,self.crop_config)
        else:example=load_example(row,self.crop_config)
        inputs=example['inputs']
        rate=inputs['speaker_sample_rate'];wave=np.asarray(inputs['speaker_audio'],dtype=np.float32)
        if wave.ndim!=1:raise ValueError('Expected mono speaker audio')
        if rate!=16000:
            g=math.gcd(rate,16000);wave=resample_poly(wave,16000//g,rate//g).astype(np.float32)
        count=round(row['duration_s']*16000)
        if not count<=len(wave)<=count+1 or not np.isfinite(wave).all():raise ValueError('Speaker resampling duration mismatch')
        return dict(sample_id=row['sample_id'],split=self.split,duration=row['duration_s'],image=inputs['listener_first_frame'],
            audio=wave[:count],target=example['target_text'],audit=example['audit'])


def datasets(config,smoke=False):
    from .data import PlanningDataset
    spec=config.get('data',{})
    if not spec:
        train=PlanningDataset(config['review_cache'] if smoke else config['train_cache'],'train')
        dev=PlanningDataset(config['review_cache'],'dev');return train,dev,dev
    if spec.get('mode')!='reference_jsonl':raise ValueError('Unknown planning dataset mode')
    if config.get('extend_from'):raise ValueError('New reference data is not an extension of the old cached-data run')
    train=ReferencePlanningDataset(spec['seal'],'train',limit=spec.get('smoke_train',8) if smoke else None,seed=config['seed'],input_pipeline=spec.get('input_pipeline'))
    full=ReferencePlanningDataset(spec['seal'],'dev',limit=spec.get('smoke_dev',4) if smoke else None,seed=config['seed'],input_pipeline=spec.get('input_pipeline'))
    diagnostic=full if smoke else ReferencePlanningDataset(spec['seal'],'dev',selection='diagnostic',input_pipeline=spec.get('input_pipeline'))
    return train,diagnostic,full
