"""Export plot-ready statistics from final sealed Seamless metadata only.

No media reads, inference, relabeling, synonym merging, or training-file writes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time

ROOT=Path(__file__).resolve().parents[1]
FULL=ROOT/'experiments/listen_j_upright_av500h_v3_20261105/tables'
LTX=ROOT/'experiments/listen_j_upright_av500h_train_dev_v3_20261105/selection'
QWEN=ROOT/'experiments/listen_j_upright_response_sft_20261105/final'
OUT=ROOT/'experiments/seamless_final_data_overview_20261112'
ZH={
 'visual:head nodding':'点头','visual:smiling':'微笑','visual:head shaking':'摇头',
 'visual:gaze shifting':'视线转移','visual:blinking':'眨眼','visual:eye closing':'闭眼',
 'visual:eye opening':'睁眼','visual:mouth opening':'张嘴','visual:mouth closing':'闭嘴',
 'visual:mouth movement':'口部运动','visual:head tilting':'头部倾斜',
 'visual:eyebrow raising':'抬眉','visual:eyebrow lowering':'眉毛下压',
 'visual:brow lowering':'眉部下压','visual:frowning':'皱眉表情',
 'visual:brow furrowing':'眉部皱起','visual:lip pursing':'抿嘴／噘嘴',
 'visual:hand raising':'抬手','visual:hand lowering':'放下手',
 'visual:head turning':'转头','visual:head movement':'头部运动',
 'visual:eye widening':'睁大眼睛','visual:laughing':'可见笑／大笑动作',
 'audio:speech':'口头回应／说话','audio:vocalization':'其他发声（未细分）',
 'audio:audible laughter':'可听笑声','audio:chuckling':'轻笑',
 'audio:humming':'轻哼','audio:exhalation':'呼气','audio:breathing':'呼吸声',
 'audio:coughing':'咳嗽','audio:throat clearing':'清嗓',
}


def read(path):return json.loads(path.read_text())
def db(path):return sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True)
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvwrite(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def bucket(value,edges):
    for a,b in zip(edges,edges[1:]):
        if a<=value<b:return f'[{a},{b})'
    return f'>={edges[-1]}'


def export(output):
    start=time.time();output.mkdir(parents=True,exist_ok=True)
    full_summary=read(FULL/'summary.json');ltx_summary=read(LTX/'SUMMARY.json');q_summary=read(QWEN/'summary.json')
    assert ltx_summary['manifest_sha256']==q_summary['labels_sha256']=='0851714c52e17dec6c7b5145b82dc53998ab09af0cc7ceda0b9c807192c2b389'
    ltx={sid:split for sid,split in db(LTX/'index.sqlite').execute('select sid,split from samples')}
    qwen={sid:split for sid,split in db(QWEN/'index.sqlite').execute('select sid,split from samples')}
    assert len(ltx)==284218 and len(qwen)==704368+5296+7087
    assert Counter(ltx.values())=={'train':281060,'dev':3158}
    assert Counter(qwen.values())=={'train':704368,'dev':5296,'test':7087}
    conn=db(FULL/'sample_stats.sqlite');conn.execute('pragma query_only=on')
    totals=defaultdict(lambda:dict(samples=0,seconds=0.,zero_adopted_labels=0,assignments=0))
    distributions=defaultdict(lambda:[0,0.])
    infos={};found_ltx=set();found_q=set()
    overlap=Counter();overlap_seconds=Counter()
    for sid,split,seconds,p10,p50,types in conn.execute('select sid,split,duration,face_p10,face_p50,types from samples'):
        scopes=['full_all','full_'+split]
        in_ltx=sid in ltx;in_q=sid in qwen;in_qtrain=in_q and qwen[sid]=='train'
        if in_ltx:
            assert ltx[sid]==split;found_ltx.add(sid);scopes+=['ltx500h_all','ltx500h_'+split]
        if in_q:
            assert qwen[sid]==split;found_q.add(sid);scopes+=['qwen_sft_all','qwen_sft_'+split]
        if in_ltx or in_qtrain:scopes+=['actual_training_union']
        if in_ltx and in_qtrain:key='both_ltx_and_qwen_train'
        elif in_ltx:key='ltx_only'
        elif in_qtrain:key='qwen_train_only'
        else:key='neither_training_set'
        overlap[key]+=1;overlap_seconds[key]+=seconds
        infos[sid]=(tuple(scopes),seconds,types)
        for scope in scopes:
            t=totals[scope];t['samples']+=1;t['seconds']+=seconds;t['zero_adopted_labels']+=types==0
            for variable,value,edges in [('face_p10_px',p10,[0,128,160,192,224,256,320,384,512]),
                                         ('face_p50_px',p50,[0,128,160,192,224,256,320,384,512]),
                                         ('clip_duration_s',seconds,[0,3,5,7,10,12,15]),
                                         ('adopted_labels_per_clip',types,[0,1,2,3,4,5,6,10])]:
                b='missing' if value is None else bucket(value,edges)
                x=distributions[(scope,variable,b)];x[0]+=1;x[1]+=seconds
    assert len(infos)==full_summary['sample_count']==1107313
    assert found_ltx==set(ltx) and found_q==set(qwen)
    print(f'Indexed {len(infos)} final samples; aggregating existing behavior records',flush=True)
    # Each behavior row is unique on (sid,type); coverage counts a clip once.
    # Fields: clips, seconds, literal_count, derived_count, known_clips,
    # unknown_count_clips, state_groups, state_clips, pending_adopted_clips.
    categories=defaultdict(lambda:[0,0.,0,0,0,0,0,0,0])
    audio_sids=set();visual_sids=set();pending_only=Counter();behavior_rows=0
    for sid,label,counted,pending,quantity,basis,unit,state in conn.execute('select sid,type,counted,pending,quantity,basis,unit,state from behaviors'):
        scopes,seconds,types=infos[sid]
        if not counted:
            for scope in scopes:pending_only[(scope,label)]+=1
            continue
        behavior_rows+=1
        if label.startswith('audio:'):audio_sids.add(sid)
        elif label.startswith('visual:'):visual_sids.add(sid)
        for scope in scopes:
            t=totals[scope];t['assignments']+=1
            a=categories[(scope,label)];a[0]+=1;a[1]+=seconds;a[8]+=bool(pending)
            if unit=='state_group':a[6]+=quantity or 0;a[7]+=1
            elif quantity is None:a[5]+=1
            else:
                a[4]+=1
                if basis=='literal':a[2]+=quantity
                else:a[3]+=quantity
    modality=defaultdict(lambda:[0,0.])
    for sid,(scopes,seconds,types)in infos.items():
        group='visual_and_audio'if sid in visual_sids and sid in audio_sids else 'visual_only'if sid in visual_sids else 'audio_only'if sid in audio_sids else 'no_adopted_label'
        for scope in scopes:
            a=modality[(scope,group)];a[0]+=1;a[1]+=seconds
    rows=[];label_sets=defaultdict(set)
    for (scope,label),a in sorted(categories.items()):
        label_sets[scope].add(label)
        rows.append(dict(scope=scope,label=label,modality=label.split(':',1)[0],name_en=label.split(':',1)[1],name_zh=ZH.get(label,''),
            clip_count=a[0],clip_percent=100*a[0]/totals[scope]['samples'],covered_clip_seconds=a[1],covered_clip_hours=a[1]/3600,
            covered_clip_duration_percent=100*a[1]/totals[scope]['seconds'],literal_occurrences_sum=a[2] if a[4]else None,
            derived_occurrences_sum=a[3] if a[4]else None,known_occurrence_clip_count=a[4],unknown_occurrence_clip_count=a[5],
            state_group_sum=a[6],state_clip_count=a[7],adopted_but_pending_clip_count=a[8],candidate_status='adopted_open_name_not_human_ground_truth'))
    summary_rows=[]
    for scope,t in sorted(totals.items()):
        labels=label_sets[scope]
        summary_rows.append(dict(scope=scope,clip_count=t['samples'],clip_hours=t['seconds']/3600,mean_clip_seconds=t['seconds']/t['samples'],
            adopted_label_names=len(labels),visual_label_names=sum(x.startswith('visual:')for x in labels),audio_label_names=sum(x.startswith('audio:')for x in labels),
            sample_label_assignments=t['assignments'],mean_labels_per_clip=t['assignments']/t['samples'],zero_adopted_label_clips=t['zero_adopted_labels'],
            clips_with_audio_label=sum(v[0]for (s,g),v in modality.items()if s==scope and g in ('audio_only','visual_and_audio'))))
    by_scope={x['scope']:x for x in summary_rows}
    assert by_scope['full_all']['adopted_label_names']==full_summary['distinct_candidate_types']
    assert by_scope['ltx500h_all']['adopted_label_names']==2605
    assert abs(by_scope['ltx500h_all']['clip_hours']-ltx_summary['hours'])<1e-7
    # Independently compare the exported LTX category coverage with its frozen CSV.
    expected=list(csv.DictReader((LTX/'category_distribution.csv').open(encoding='utf-8-sig')))
    actual={x['label']:x for x in rows if x['scope']=='ltx500h_all'}
    assert len(expected)==len(actual)
    for x in expected:
        a=actual[x['type']];assert a['clip_count']==int(x['samples'])
        assert abs(a['covered_clip_hours']-float(x['covered_clip_hours']))<1e-7
    for split in ('train','dev','test'):
        s=by_scope['qwen_sft_'+split];e=q_summary['splits'][split]
        assert s['clip_count']==e['counts']['keep'] and abs(s['clip_hours']-e['hours']['keep'])<1e-7
    assert by_scope['actual_training_union']['clip_count']==len(ltx)+sum(s=='train'for s in qwen.values())-overlap['both_ltx_and_qwen_train']
    for scope,t in totals.items():
        assert sum(v[0]for (s,g),v in modality.items()if s==scope)==t['samples']
        assert sum(a[0]for (s,l),a in categories.items()if s==scope)==t['assignments']
        for variable in ('face_p10_px','face_p50_px','clip_duration_s','adopted_labels_per_clip'):
            assert sum(v[0]for (s,k,b),v in distributions.items()if s==scope and k==variable)==t['samples']
    csvwrite(output/'dataset_summary.csv',summary_rows,list(summary_rows[0]))
    csvwrite(output/'category_counts_long.csv',rows,list(rows[0]))
    primary=['full_all','ltx500h_all','qwen_sft_train','actual_training_union'];lookup={(x['scope'],x['label']):x for x in rows}
    wide=[]
    for label in sorted(label_sets['full_all'],key=lambda x:(-lookup['full_all',x]['clip_count'],x)):
        d=dict(label=label,name_zh=ZH.get(label,''),modality=label.split(':',1)[0])
        for scope in primary:
            a=lookup.get((scope,label),{})
            for field in ('clip_count','clip_percent','covered_clip_hours'):d[scope+'_'+field]=a.get(field,0)
        wide.append(d)
    csvwrite(output/'category_comparison_wide.csv',wide,list(wide[0]))
    for scope in primary:
        subset=sorted((x for x in rows if x['scope']==scope),key=lambda x:(-x['clip_count'],x['label']))
        csvwrite(output/(scope+'_categories.csv'),subset,list(rows[0]))
    dist=[dict(scope=s,variable=k,bin=b,clip_count=v[0],clip_hours=v[1]/3600,clip_percent=100*v[0]/totals[s]['samples'])for(s,k,b),v in sorted(distributions.items())]
    csvwrite(output/'distributions.csv',dist,list(dist[0]))
    mod=[dict(scope=s,coverage=g,clip_count=v[0],clip_hours=v[1]/3600,clip_percent=100*v[0]/totals[s]['samples'])for(s,g),v in sorted(modality.items())]
    csvwrite(output/'modality_coverage.csv',mod,list(mod[0]))
    overlaps=[dict(membership=k,clip_count=v,clip_hours=overlap_seconds[k]/3600)for k,v in sorted(overlap.items())]
    csvwrite(output/'training_set_overlap.csv',overlaps,list(overlaps[0]))
    pending=[dict(scope=s,label=l,pending_only_clip_count=n)for(s,l),n in sorted(pending_only.items())]
    csvwrite(output/'pending_only_labels.csv',pending,list(pending[0]))
    report=dict(success=True,created_at=time.time(),elapsed_seconds=time.time()-start,datasets=summary_rows,overlap=overlaps,
        counted_behavior_rows=behavior_rows,source_labels_sha256=ltx_summary['manifest_sha256'],
        inputs=[str(FULL/'sample_stats.sqlite'),str(LTX/'index.sqlite'),str(QWEN/'index.sqlite')],
        definitions={'category':'Original adopted open label name; synonyms are not merged; not a validated functional taxonomy.',
            'clip_count':'Clips containing this label, unique per (clip,label). Categories overlap.',
            'covered_clip_hours':'Whole lengths of clips containing this label; NOT action duration.',
            'occurrences':'Only explicit or supported derived quantities; unknowns stay unknown. State groups are separate.',
            'actual_training_union':'Deduplicated union of LTX500h (source train+dev) and Qwen SFT train. Does not include Qwen dev/test unless already used in LTX.',
            'full_all':'Latest caption/statistics-ready 1,107,313 clips; supersedes historical 1,107,323 and 4,764-name overview.',
            'human_ground_truth':False},
        checks={'membership_complete':True,'source_splits_match':True,'ltx_all_category_counts_and_hours_match_frozen_export':True,
            'qwen_counts_and_hours_match_final_summary':True,'distribution_counts_sum_back':True,'training_overlap_not_double_counted':True})
    dump(output/'summary.json',report)
    print(json.dumps({'output':str(output),'elapsed_seconds':report['elapsed_seconds'],'datasets':summary_rows,'overlap':overlaps},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();export(a.output.resolve())
