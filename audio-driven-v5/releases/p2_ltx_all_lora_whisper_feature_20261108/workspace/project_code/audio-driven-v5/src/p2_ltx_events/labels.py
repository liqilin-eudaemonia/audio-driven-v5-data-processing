"""Historical intervals joined by original media identity and physical time.

Original model candidates remain explicitly weak. Successful historical clip
coverage supplies downweighted background hypotheses, never confirmed absence.
Outside that coverage, targets are unknown. No text/count-to-time inference.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,math,multiprocessing,sqlite3
from pathlib import Path
from itertools import islice
from functools import lru_cache
import os

from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.annotations import union,intersection_length
from .config import TEMPORAL_CLASSES,CURRENT_TEMPORAL_CLASSES
from . import VERSION

ALL=tuple(n for m in ('video','audio') for n in TEMPORAL_CLASSES[m])
CURRENT_ALL=tuple(n for m in ('video','audio') for n in CURRENT_TEMPORAL_CLASSES[m])


@lru_cache(maxsize=4)
def _connection(path,pid):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


@lru_cache(maxsize=32)
def _caption(path,expected):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Frozen caption evidence changed')
    return json.loads(raw)


def media_key(source):
    media=source['source']
    if Path(media['member']).stem!=source['listener_id']:
        raise ValueError('Listener/source identity mismatch')
    return digest({k:media[k] for k in ('archive','member','offset','byte_size')})


def history_source(item):
    source=item['source'];old=item.get('historical_source',source)
    if (old['listener_id']!=source['listener_id'] or
            any(abs(old[k]-source[k])>1e-6 for k in ('source_start_s','source_end_s','duration_s'))):
        raise ValueError('Restored source changed listener/time before historical join')
    return old


def event_spans(event):
    values=event.get('intervals',[{'start':event['start'],'end':event['end']}])
    spans=[]
    for value in values:
        a,b=(value['start'],value['end']) if isinstance(value,dict) else value
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in (a,b)) or b<=a:
            raise ValueError('Invalid historical interval')
        spans.append([float(a),float(b)])
    return union(spans)


def normalize_old(label):
    source=label['source'];start=float(source['source_start_s']);end=float(source['source_end_s'])
    if abs(end-start-label['duration_s'])>1e-5:raise ValueError('Historical duration mismatch')
    complete=label.get('raw_completion_status')=='complete_candidates'
    classes={name:dict(spans=[],coverage=[[start,end]] if complete else [],unknown=[]) for name in ALL}
    for event in label['events']:
        categories=set(event.get('categories',[]));mod=event['modality']
        selected=categories & set(TEMPORAL_CLASSES['video']) if mod=='visual' else (
            {'listener_vocalization'} if mod=='audio' and categories & {'listener_vocalization','laugh'} else set())
        if not selected:continue
        spans=event_spans(event)
        if any(a<0 or b>label['duration_s']+1e-5 for a,b in spans):raise ValueError('Historical event outside clip')
        uncertain=event.get('mapping_status') in {'pending','uncertain','unusable'} or event.get('status',{}).get('content') in {'unusable','uncertain','rejected'}
        for name in selected:
            classes[name]['unknown' if uncertain else 'spans'].extend([[start+a,start+b] for a,b in spans])
    return dict(sample_id=label['sample_id'],media_key=media_key(source),start=start,end=end,classes=classes,
        weak=True,human_ground_truth=False,source_completion=label.get('raw_completion_status'))


def _index_file(path):
    raw=Path(path).read_bytes();rows=[]
    for line in raw.splitlines():
        if line.strip():
            row=normalize_old(json.loads(line));rows.append((row['media_key'],row['start'],row['end'],json.dumps(row,separators=(',',':'))))
    return str(path),hashlib.sha256(raw).hexdigest(),rows


def index_history(source,output,processes=8):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    db=output/'history.sqlite'
    if db.exists():raise FileExistsError('Use a new historical index output')
    paths=sorted((source/'integrated').glob('*.jsonl'))
    if not paths:raise ValueError('No historical integrated labels')
    hashes={};count=0
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE history(media TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL, value TEXT NOT NULL)')
        with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork')) as pool:
            iterator=iter(paths)
            while batch:=list(islice(iterator,256)):
                for path,h,rows in pool.map(_index_file,batch,chunksize=1):
                    hashes[path]=h;count+=len(rows);conn.executemany('INSERT INTO history VALUES (?,?,?,?)',rows)
                conn.commit();print(json.dumps(dict(stage='history_index',files=len(hashes),total_files=len(paths),samples=count)),flush=True)
        conn.execute('CREATE INDEX history_media ON history(media,start,end)');conn.commit()
    result=dict(schema=VERSION,kind='history-index',source=str(source),files=hashes,samples=count,
        database_sha256=sha(db),weak=True,human_ground_truth=False,classes=TEMPORAL_CLASSES)
    atomic_json(output/'manifest.json',result,immutable=True)
    return result


def current_audio_support(item):
    """Exact nonzero gate support without allocating or decoding a waveform."""
    from listen_benchmark.official_audio_gate import POLICY,merge
    recipe=item['audio_recipe'];src=recipe['source']
    if recipe['schema']!='j-listener-segment-official-union-v1' or recipe['gate_policy']!=POLICY or recipe['extra_mute_intervals']:
        raise ValueError('Unsupported current listener gate')
    if recipe['listener_id']!=item['source']['listener_id']:
        raise ValueError('Current audio/listener mismatch')
    rate=src['sample_rate'];n=src['last_sample']-src['first_sample'];duration=n/rate
    if abs(recipe['source_time_origin_s']-item['source']['source_start_s'])>1e-6:
        raise ValueError('Current audio time origin mismatch')
    core=[]
    for kind in POLICY['union_sources']:
        for interval in recipe['official_listener'][kind]:
            a,b=interval['start'],interval['end']
            if not math.isfinite(a+b) or b<a:raise ValueError('Invalid current official interval')
            a,b=max(0.,a),min(duration,b)
            if b>a:core.append([max(0,math.floor(a*rate+1e-7)),min(n,math.ceil(b*rate-1e-7))])
    pad=round(POLICY['padding_seconds']*rate);fade=round(POLICY['fade_seconds']*rate)
    keep=merge([[max(0,a-pad),min(n,b+pad)] for a,b in merge(core)])
    positive=[]
    for a,b in keep:
        k=min(fade,b-a)
        a+=int(a>0 and k>=2);b-=int(b<n and k>=2)
        if b>a:positive.append([a,b])
    if sum(b-a for a,b in positive)!=item['listener_audio_identity']['positive_samples']:
        raise ValueError('Current gate differs from cached waveform support')
    return [[a/rate,b/rate] for a,b in positive]


def intersect(spans,allowed):
    return union([[max(a,x),min(b,y)] for a,b in spans for x,y in allowed if min(b,y)>max(a,x)])


def current_nod(item):
    """Frozen production GLARE recall evidence already used by this caption.

    The caption embeds the checked detector intervals and its SHA. Reuse those
    authenticated intervals without reopening a large raw frame-score file.
    """
    path=Path(item['statistics']['caption_file'])
    expected=item['statistics']['caption_file_sha256']
    caption=_caption(str(path),expected)
    if caption['sample_id']!=item['sample_id'] or media_key(caption['source'])!=media_key(item['source']):
        raise ValueError('Production nod/caption listener mismatch')
    if abs(caption['source']['source_start_s']-item['source']['source_start_s'])>1e-6:
        raise ValueError('Production nod/caption time mismatch')
    nod=caption.get('input_evidence',{}).get('nod_supplement')
    if nod is None:return None
    if nod['sample_id']!=item['sample_id'] or nod['fps']!=25 or nod['profile']!='recall':
        raise ValueError('Wrong production nod sample/FPS/profile')
    duration=item['source']['duration_s']
    for event in nod['events']:
        if not 0<=event['start']<event['end']<=duration+1e-6:raise ValueError('Production nod outside current clip')
    return dict(nod,caption_file=str(path),caption_file_sha256=expected)


@lru_cache(maxsize=8)
def _production_gate(parameters,model_sha):
    from listen_benchmark.caption_guard import reaction_cache_problems
    return reaction_cache_problems(dict(parameters=json.loads(parameters),model_sha256=model_sha))


def current_supplements(item):
    """Use the CURRENT qualified detector, not rejected old six-class pilots.

    Keep physical brow observations separate from frown/surprise labels. An
    expression proxy must not silently become an emotional/expression class.
    """
    s=item['statistics'];cap=_caption(s['caption_file'],s['caption_file_sha256']);evidence=cap.get('input_evidence',{})
    reaction=evidence.get('reaction_supplement')
    if reaction is None:return None
    if reaction['sample_id']!=item['sample_id'] or reaction['fps']!=25:raise ValueError('Wrong production reaction identity')
    raw=Path(reaction['detector_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reaction['detector_sha256']:raise ValueError('Production reaction detector changed')
    record=json.loads(raw);duration=item['source']['duration_s']
    if (record['sample_id']!=item['sample_id'] or record['source']!=cap['source']
            or record['status']!='ok' or record['fps']!=25 or record['profile']!='recall'
            or abs(record['duration_s']-duration)>1e-6 or record['reaction_cues']!=reaction['cues']):
        raise ValueError('Production reaction source/time/cues mismatch')
    problems=_production_gate(json.dumps(record['parameters'],sort_keys=True),record['model_sha256'])
    if problems:raise ValueError('Unqualified production reaction gates: '+','.join(problems))
    detected=record['pose']['detected'];fps=record['fps']
    coverage=union([[i/fps,min((i+1)/fps,duration)] for i,ok in enumerate(detected) if ok and i/fps<duration])
    observation={'head shaking':'head_shake','brow lowering':'brow_lowering','brows raised':'brows_raised'}
    classes={name:dict(spans=[],coverage=coverage if name=='head_shake' or reaction['fer_status']=='ok' else [],unknown=[])
             for name in observation.values()}
    for cue in reaction['cues']:
        if cue['observation'] not in observation:raise ValueError('Unrecognized production physical observation')
        a,b=clip_cue_interval(cue,duration,item['sample_id'])
        classes[observation[cue['observation']]]['spans'].append([a,b])
    vocal=[]
    for cue in evidence.get('vocal_supplement',[]):
        a,b=clip_cue_interval(cue,duration,item['sample_id'])
        vocal.append([a,b])
    return dict(classes=classes,nod_coverage=coverage,vocal_spans=union(vocal),
        detector_file=reaction['detector_file'],detector_sha256=reaction['detector_sha256'],
        physical_observations_preserved=True,fer_status=reaction['fer_status'])


def clip_cue_interval(cue,duration,sid):
    # Production prompt cues round endpoints to milliseconds. Clip only this
    # documented quantization overhang to the physical target support.
    a,b=cue['start'],cue['end']
    if not all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in (a,b)):
        raise ValueError('Nonfinite production interval: '+sid)
    if not (-.001001<=a<b<=duration+.001001):
        raise ValueError(f'Production interval outside clip: {sid}: {a},{b} vs {duration}')
    a,b=max(0.,a),min(duration,b)
    if b<=a:raise ValueError('Empty production interval after support clipping: '+sid)
    return a,b


def join_target(item,histories,nod=None,production=None):
    source=history_source(item);key=media_key(source);start=source['source_start_s']
    # Source endpoints are displayed at rounded precision; duration_s is the
    # sample-accurate duration used by all CURRENT native cache geometry.
    duration=item['source']['duration_s'];rate=item['audio_recipe']['source']['sample_rate']
    if abs(source['source_end_s']-start-duration)>1/rate+1e-6:
        raise ValueError('Source endpoints / native media duration mismatch')
    end=start+duration
    classes={name:dict(spans=[],coverage=[],unknown=[]) for name in CURRENT_ALL};origins=[]
    for row in histories:
        if row['media_key']!=key:raise ValueError('Historical/target media identity mismatch')
        if row['end']<=start or row['start']>=end:continue
        origins.append(row['sample_id'])
        for name in ALL:
            for field in ('spans','coverage','unknown'):
                classes[name][field].extend([[max(a,start)-start,min(b,end)-start]
                    for a,b in row['classes'][name][field] if min(b,end)>max(a,start)])
    for fields in classes.values():
        for field in fields:fields[field]=union(fields[field])
    if nod is not None:
        if nod['sample_id']!=item['sample_id']:raise ValueError('Production nod ID mismatch')
        data=classes['head_nod']
        data['spans']=union(data['spans']+[[e['start'],e['end']] for e in nod['events']])
        # Detector success supplies weak observational coverage. An empty list
        # is only a 0.1-weight background hypothesis, not confirmed absence.
        data['coverage']=union(data['coverage']+(production['nod_coverage'] if production is not None else [[0.,duration]]))
    if production is not None:
        for name,fields in production['classes'].items():
            for field in ('spans','coverage','unknown'):classes[name][field]=union(classes[name][field]+fields[field])
    # Historical retained audio already passed the old gate. Intersect again
    # with the CURRENT target gate: old and new cuts/normalization can differ.
    allowed=current_audio_support(item)
    if production is not None:
        data=classes['listener_vocalization'];vocal=intersect(production['vocal_spans'],allowed)
        data['spans']=union(data['spans']+vocal)
        zero=[];cursor=0.
        for a,b in allowed:
            if a>cursor:zero.append([cursor,a])
            cursor=b
        if cursor<duration:zero.append([cursor,duration])
        data['coverage']=union(data['coverage']+zero+vocal)
    for name in TEMPORAL_CLASSES['audio']:
        classes[name]['spans']=intersect(classes[name]['spans'],allowed)
    for fields in classes.values():
        for field in fields:
            fields[field]=union([[max(0.,a),min(duration,b)] for a,b in fields[field]
                                 if min(duration,b)>max(0.,a)])
    return dict(sample_id=item['sample_id'],source_digest=digest(item),media_key=key,duration_s=duration,
        classes=classes,origins=sorted(set(origins)),nod_source=nod,production_source=production,
        policy='historical-weak-intervals-v1',human_ground_truth=False)


def _prepare_target(args):
    work,sid,expected,index,*reuse=args
    if (Path(work)/'sources.sqlite').exists():
        source_db=_connection(str(Path(work)/'sources.sqlite'),os.getpid())
        item=json.loads(source_db.execute('SELECT value FROM samples WHERE sid=?',(sid,)).fetchone()[0])
    else:item=read(Path(work)/'samples'/(sid+'.json'))
    if digest(item)!=expected:raise ValueError('Current sample changed: '+sid)
    conn=_connection(str(index),os.getpid());source=history_source(item)
    rows=conn.execute('SELECT value FROM history WHERE media=? AND start<? AND end>?',
        (media_key(source),source['source_end_s'],source['source_start_s']))
    if reuse:
        old=json.loads(_connection(reuse[0],os.getpid()).execute('SELECT value FROM labels WHERE sid=?',(sid,)).fetchone()[0])
        if old['source_digest']!=expected:raise ValueError('Evidence reuse source changed: '+sid)
        nod=old.get('nod_source');production=old.get('production_source')
        if nod is not None and nod['caption_file_sha256']!=item['statistics']['caption_file_sha256']:
            raise ValueError('Evidence reuse caption changed: '+sid)
        return join_target(item,(json.loads(r[0]) for r in rows),nod,production)
    return join_target(item,(json.loads(r[0]) for r in rows),current_nod(item),current_supplements(item))


def prepare(work,history,output,processes=8,sample_ids=None,evidence_from=None):
    work=Path(work).resolve();history=Path(history).resolve();output=Path(output).resolve()
    m=read(work/'manifest.json');hm=read(history/'manifest.json')
    if hm['schema']!=VERSION or sha(history/'history.sqlite')!=hm['database_sha256']:
        raise ValueError('Historical index identity mismatch')
    training=m['training_ids'] if 'training_ids' in m else m['splits']['train']
    ids=list(training if sample_ids is None else sample_ids)
    if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(training):
        raise ValueError('Use frozen training members only')
    previous=None
    if evidence_from is not None:
        previous=Path(evidence_from).resolve();pm=read(previous/'manifest.json')
        if (pm['schema']!=VERSION or pm['policy']!='historical-weak-intervals-v1'
                or pm['cache_manifest_sha256']!=sha(work/'manifest.json')
                or pm['history_manifest_sha256']!=sha(history/'manifest.json')
                or pm['sample_ids_sha256']!=digest(ids)
                or pm['database_sha256']!=sha(previous/'labels.sqlite')):
            raise ValueError('Clock reprojection must reuse the exact verified source/evidence set')
    output.mkdir(parents=True,exist_ok=True);db=output/'labels.sqlite'
    if db.exists():raise FileExistsError('Use a new temporal-label output')
    classes=CURRENT_TEMPORAL_CLASSES if m.get('kind')=='current-label-sources' else TEMPORAL_CLASSES
    stats={k:Counter() for values in classes.values() for k in values};coverage=Counter()
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE labels(ordinal INTEGER PRIMARY KEY,sid TEXT UNIQUE NOT NULL,value TEXT NOT NULL)')
        args=((str(work),sid,m['entries'][sid],str(history/'history.sqlite'),*([str(previous/'labels.sqlite')] if previous else [])) for sid in ids)
        count=0
        with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork')) as pool:
            while batch:=list(islice(args,2048)):
                for row in pool.map(_prepare_target,batch,chunksize=8):
                    conn.execute('INSERT INTO labels VALUES (?,?,?)',(count,row['sample_id'],json.dumps(row,separators=(',',':'))));count+=1
                    coverage['samples']+=1;coverage['with_history']+=bool(row['origins'])
                    coverage['with_production_nod']+=row['nod_source'] is not None
                    coverage['with_any_time_source']+=bool(row['origins']) or row['nod_source'] is not None
                    coverage['with_current_production_reactions']+=row['production_source'] is not None
                    for k,fields in row['classes'].items():
                        if k not in stats:continue
                        known=union(fields['coverage']);unknown=fields['unknown']
                        known_s=sum(b-a for a,b in known)-sum(intersection_length(known,a,b) for a,b in union(unknown))
                        pos=intersect(fields['spans'],known)
                        pos_s=sum(b-a for a,b in pos)-sum(intersection_length(pos,a,b) for a,b in union(unknown))
                        stats[k].update(positive_seconds=max(0.,pos_s),background_seconds=max(0.,known_s-pos_s),
                                        positive_samples=int(pos_s>0),covered_samples=int(known_s>0))
                conn.commit();print(json.dumps(dict(stage='temporal_labels',samples=count,total=len(ids))),flush=True)
    result=dict(schema=VERSION,kind='target-intervals',cache_work=str(work),cache_manifest_sha256=sha(work/'manifest.json'),
        history_manifest_sha256=sha(history/'manifest.json'),database_sha256=sha(db),samples=count,
        sample_ids_sha256=digest(ids),full_train=ids==training,classes=classes,
        source_split_counts=m.get('source_split_counts'),source_split_preserved=True,
        policy='historical-weak-intervals-v1',statistics=stats,coverage=coverage,human_ground_truth=False,
        background_semantics='downweighted hypothesis within completed historical clips; unknown outside coverage',
        audio_policy='historical retained events intersect CURRENT nonzero official gate')
    result['duration_policy']='native source.duration_s; rounded source endpoints tolerated within one sample; clip to real support'
    if previous:result['evidence_parent_manifest_sha256']=sha(previous/'manifest.json')
    atomic_json(output/'manifest.json',result,immutable=True)
    return result


class Reader:
    def __init__(self,folder,verify=True):
        self.folder=Path(folder)
        if (self.folder/'INVALIDATED.json').exists():raise ValueError('Temporal targets were superseded; see INVALIDATED.json')
        self.manifest=read(self.folder/'manifest.json')
        if self.manifest['schema']!=VERSION or self.manifest['kind']!='target-intervals':raise ValueError('Wrong temporal label schema')
        if verify and sha(self.folder/'labels.sqlite')!=self.manifest['database_sha256']:raise ValueError('Temporal label database changed')
        self.conn=sqlite3.connect((self.folder/'labels.sqlite').resolve().as_uri()+'?mode=ro',uri=True)

    def row(self,sid):
        value=self.conn.execute('SELECT value FROM labels WHERE sid=?',(sid,)).fetchone()
        if value is None:raise KeyError(sid)
        return json.loads(value[0])

    def ids(self):return [r[0] for r in self.conn.execute('SELECT sid FROM labels ORDER BY ordinal')]
