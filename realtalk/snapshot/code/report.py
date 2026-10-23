"""Aggregate native-pixel/retained-duration statistics without threshold filtering."""
import json,csv,hashlib,collections,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports';OUT.mkdir(exist_ok=True);FPS=25
BINS=[0,64,96,128,160,192,224,256,320,384,448,512,640,768,1024,1536,100000]
THRESHOLDS=[64,96,128,160,192,224,256,288,320,384,448,512,640,768,1024]
DUR_BINS=[0,1,2,4,6,8,10,11.00001]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def binlabel(a,b):return f'{a}–{b}' if b<100000 else f'≥{a}'
def histogram(counter):
 out=[]
 for a,b in zip(BINS,BINS[1:]):
  frames=sum(v for k,v in counter.items() if a<=k<b);out.append(dict(min_px=a,max_px=b if b<100000 else None,label=binlabel(a,b),frames=frames,seconds=frames/FPS,hours=frames/FPS/3600))
 return out
def weighted_quantile(counter):
 if not counter:return None
 items=sorted(counter.items());xs=np.array([k for k,v in items]);cum=np.cumsum([v for k,v in items]);return {name:int(xs[min(np.searchsorted(cum,cum[-1]*q),len(xs)-1)]) for name,q in [('p01',.01),('p10',.1),('p25',.25),('p50',.5),('p75',.75),('p90',.9),('p99',.99)]}
rows=[];byvideo=[];byrole=[];role_distribution=[];facehist=collections.Counter();crophist=collections.Counter();totals=collections.Counter();boundary=collections.Counter();checks=[];manifest_hashes={}
files=sorted((ROOT/'manifests').glob('*.json'))
for f in files:
 m=json.loads(f.read_text());manifest_hashes[m['video_id']]=hashlib.sha256(f.read_bytes()).hexdigest();totals.update(m['frame_totals']);totals['source_frames']+=m['frames'];totals['valid_face_frames']+=m['face_valid_frames'];totals['crop_unresolved_frames']+=m['crop_unresolved_pair_frames'];facehist.update({int(k):v for k,v in m['listener_face_short_side_frame_histogram'].items()});boundary.update(m['boundary_methods'])
 assert sum(m['frame_totals'].values())==m['frames'];assert sum(m['listener_face_short_side_frame_histogram'].values())==m['face_valid_frames'];assert m['checks']['known_role_frames_retained']
 v=dict(video_id=m['video_id'],source_seconds=m['duration'],pair_count=len(m['pairs']),pair_seconds=m['frame_totals']['pair']/FPS,role_unknown_seconds=m['frame_totals']['role_unknown']/FPS,black_seconds=m['frame_totals']['black']/FPS,face_known_seconds=m['face_valid_frames']/FPS,crop_unknown_seconds=m['crop_unresolved_pair_frames']/FPS,source_width=m['width'],source_height=m['height']);byvideo.append(v)
 for role in ['p0','p1']:
  kids=[r for r in m['pairs'] if r['listener']==role]
  if not kids:continue
  fh={int(k):v for k,v in m['listener_face_short_side_frame_histogram_by_role'][role].items()};ch=collections.Counter()
  for r in kids:
   if r['crop']:ch[r['crop']['native_edge']]+=r['end_frame']-r['start_frame']
  fq=weighted_quantile(fh);cq=weighted_quantile(ch);total=sum(r['end_frame']-r['start_frame'] for r in kids)
  byrole.append(dict(video_id=m['video_id'],listener=role,pair_count=len(kids),total_seconds=total/FPS,valid_face_seconds=sum(fh.values())/FPS,missing_face_seconds=(total-sum(fh.values()))/FPS,pre_crop_known_seconds=sum(ch.values())/FPS,pre_crop_unknown_seconds=(total-sum(ch.values()))/FPS,face_short_duration_weighted_p10_px=fq['p10'] if fq else None,face_short_duration_weighted_median_px=fq['p50'] if fq else None,pre_crop_duration_weighted_median_px=cq['p50'] if cq else None))
  for kind,hist in [('face_short_side',fh),('pre_crop_edge',ch)]:
   for bucket in histogram(hist):role_distribution.append(dict(video_id=m['video_id'],listener=role,measurement=kind,**bucket))
 for r in m['pairs']:
  c=r['crop'];face=r['face'];short=face['short_side'];frames=r['end_frame']-r['start_frame'];
  if c:crophist[c['native_edge']]+=frames
  consistency=r.get('crop_box_consistency',{});row=dict(face_box_inside_crop_fraction=consistency.get('full_face_box_inside_fraction'),face_center_inside_crop_fraction=consistency.get('face_center_inside_fraction'),pair_id=r['pair_id'],video_id=m['video_id'],speaker=r['speaker'],listener=r['listener'],shot=r['shot'],start_frame=r['start_frame'],end_frame=r['end_frame'],start_seconds=r['start'],end_seconds=r['end'],duration_seconds=r['duration'],source_width=m['width'],source_height=m['height'],fps=FPS,crop_x=c['x'] if c else None,crop_y=c['y'] if c else None,crop_width=c['width'] if c else None,crop_height=c['height'] if c else None,crop_native_edge=c['native_edge'] if c else None,crop_status=c['status'] if c else 'unresolved',face_valid_frames=face['valid_frames'],face_missing_frames=face['missing_or_invalid_frames'],face_out_of_bounds_frames=face['out_of_bounds_frames'],face_valid_fraction=face['valid_frames']/frames,face_width_median=face['width']['median'] if face['width'] else None,face_height_median=face['height']['median'] if face['height'] else None,face_short_min=short['min'] if short else None,face_short_p10=short['p10'] if short else None,face_short_median=short['median'] if short else None,face_short_p90=short['p90'] if short else None,start_boundary=r['start_boundary']['reason'],end_boundary=r['end_boundary']['reason'],flags=';'.join(r['flags']),manifest_file=str(f),source_archive=m['source']['archive'],source_member=m['source']['member'])
  rows.append(row)
df=pd.DataFrame(rows);assert len(df)>0;df.to_csv(OUT/'pairs.csv',index=False,float_format='%.6f');df.to_parquet(OUT/'pairs.parquet',index=False);pd.DataFrame(byvideo).to_csv(OUT/'videos.csv',index=False,float_format='%.6f')
facebins=histogram(facehist);cropbins=histogram(crophist);pd.DataFrame(facebins).to_csv(OUT/'face_resolution_duration.csv',index=False);pd.DataFrame(cropbins).to_csv(OUT/'pre_crop_resolution_duration.csv',index=False)
# Exact native crop sizes are separately available; no resizing to 512 is involved.
exact=[dict(crop_width=k,crop_height=k,frames=v,seconds=v/FPS,hours=v/FPS/3600,pair_count=int((df.crop_native_edge==k).sum())) for k,v in sorted(crophist.items())];pd.DataFrame(exact).to_csv(OUT/'pre_crop_exact_sizes.csv',index=False)
thresholds=[];duration=df.duration_seconds.to_numpy();allseconds=float(duration.sum());coverage=df.face_valid_fraction.to_numpy()
for t in THRESHOLDS:
 row={'threshold_px':t,'total_candidate_seconds':allseconds}
 for criterion,column in [('face_min','face_short_min'),('face_p10','face_short_p10'),('face_median','face_short_median'),('crop_edge','crop_native_edge')]:
  keep=df[column].fillna(-1).to_numpy()>=t;row[criterion+'_pair_count']=int(keep.sum());row[criterion+'_seconds']=float(duration[keep].sum());row[criterion+'_hours']=row[criterion+'_seconds']/3600;row[criterion+'_fraction']=row[criterion+'_seconds']/allseconds
  if criterion!='crop_edge':
   cov=keep&(coverage>=.95);row[criterion+'_coverage95_pair_count']=int(cov.sum());row[criterion+'_coverage95_seconds']=float(duration[cov].sum());row[criterion+'_coverage95_hours']=row[criterion+'_coverage95_seconds']/3600
 row['face_frame_seconds_ge_threshold']=sum(v for k,v in facehist.items() if k>=t)/FPS;thresholds.append(row)
pd.DataFrame(thresholds).to_csv(OUT/'threshold_retention.csv',index=False)
# 2D: clip-duration buckets versus crop native resolution, both counts and hours.
joint=[]
for a,b in zip(BINS,BINS[1:]):
 for lo,hi in zip(DUR_BINS,DUR_BINS[1:]):
  mask=(df.crop_native_edge>=a)&(df.crop_native_edge<b)&(df.duration_seconds>=lo)&(df.duration_seconds<hi)
  joint.append(dict(crop_bin=binlabel(a,b),duration_bin=f'{lo:g}–{min(hi,11):g}s',pair_count=int(mask.sum()),seconds=float(df.loc[mask,'duration_seconds'].sum()),hours=float(df.loc[mask,'duration_seconds'].sum()/3600)))
pd.DataFrame(joint).to_csv(OUT/'pre_crop_size_by_pair_duration.csv',index=False)
# p0/p1 denote screen position within a video, NOT a persistent person identity.
pd.DataFrame(byrole).to_csv(OUT/'per_video_listener.csv',index=False);pd.DataFrame(role_distribution).to_csv(OUT/'per_video_listener_resolution_duration.csv',index=False)
df[df.crop_status=='unresolved'].to_csv(OUT/'crop_review.csv',index=False);df[df.start_boundary.str.contains('fallback')|df.end_boundary.str.contains('fallback')].to_csv(OUT/'boundary_alignment_review.csv',index=False)
summary=dict(version='realtalk-native-size-statistics-v1',expected_videos=len(json.loads((ROOT/'index.json').read_text())),completed_videos=len(files),complete=len(files)==692,pair_count=len(rows),source_hours=totals['source_frames']/FPS/3600,candidate_hours=totals['pair']/FPS/3600,role_unknown_hours=totals['role_unknown']/FPS/3600,black_hours=totals['black']/FPS/3600,valid_face_hours=totals['valid_face_frames']/FPS/3600,missing_face_hours=(totals['pair']-totals['valid_face_frames'])/FPS/3600,pre_crop_known_hours=sum(crophist.values())/FPS/3600,pre_crop_unknown_hours=totals['crop_unresolved_frames']/FPS/3600,stacked_layout_pending_hours=float(df.loc[df['flags'].str.contains('stacked_layout_crop_needs_review'),'duration_seconds'].sum()/3600),crop_box_mismatch_pair_hours=float(df.loc[df.face_box_inside_crop_fraction.notna()&(df.face_box_inside_crop_fraction<.95),'duration_seconds'].sum()/3600),geometry_audited_videos=sum(json.loads(f.read_text()).get('geometry_audit_version')=='v3' for f in files),maximum_pair_seconds=float(df.duration_seconds.max()),short_pair_count=int((df.duration_seconds<1).sum()),frame_totals=dict(totals),face_short_side_duration_weighted_quantiles_px=weighted_quantile(facehist),pre_crop_edge_duration_weighted_quantiles_px=weighted_quantile(crophist),face_resolution_bins=facebins,pre_crop_resolution_bins=cropbins,thresholds=thresholds,pre_crop_size_by_pair_duration=joint,boundary_methods=dict(boundary),definitions={'face_histogram':'每个候选帧中，listener人脸框裁到原图范围后的短边像素；按帧时长加权。缺失/无效框单列。','pre_crop_histogram':'镜头内固定预裁剪正方形的原生边长；按整个pair时长加权。不是输出512尺寸。未确定裁剪框单列。','threshold_table':'假设整条pair按尺寸条件保留的时长估计；未真正应用阈值。min/p10/median基于有框帧，coverage95列额外要求至少95%的帧有有效框。','identity':'p0/p1是视频内局部人物标签，常见布局中对应左右，但上下布局不能照搬左右解释；不是跨视频人物身份。','scope':'所有输出为切分候选，未完成音质筛选或人工验收；待定区间不是废片。','sentence_boundaries':'全量优先原ASR句末候选和声学停顿，原ASR不可靠的边界标为复核；无可行切点时保留帧级兜底并标记。审核过的示例沿用其清单。'})
assert totals['source_frames']==totals['pair']+totals['role_unknown']+totals['black'];assert sum(facehist.values())==totals['valid_face_frames'];assert sum(crophist.values())+totals['crop_unresolved_frames']==totals['pair'];assert summary['maximum_pair_seconds']<=11
save(OUT/'summary.json',summary);save(OUT/'manifest_hashes.json',manifest_hashes)
plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(1,2,figsize=(16,5))
for ax,bins,title in [(axes[0],facebins,'Listener face short side (frame-weighted)'),(axes[1],cropbins,'Native pre-crop edge (pair-duration-weighted)')]:
 ax.bar(range(len(bins)),[r['hours'] for r in bins],color='#299b77');ax.set_xticks(range(len(bins)));ax.set_xticklabels([r['label'] for r in bins],rotation=55,ha='right');ax.set_ylabel('Hours');ax.set_xlabel('Native pixels');ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.suptitle(f'RealTalk: {len(files)}/692 videos; {len(rows):,} candidate pairs');fig.tight_layout();fig.savefig(OUT/'resolution_duration.png',dpi=180);fig.savefig(OUT/'resolution_duration.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for key,label in [('crop_edge','Pre-crop edge'),('face_p10_coverage95','Face p10, >=95% frames valid'),('face_median_coverage95','Face median, >=95% frames valid')]:ax.plot(THRESHOLDS,[r[key+'_hours'] for r in thresholds],marker='o',label=label)
ax.set(xlabel='Minimum native pixels',ylabel='Candidate hours retained',title='Threshold simulation (no filtering applied)');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'threshold_retention.png',dpi=180);plt.close(fig)
save(OUT/'validation.json',dict(frame_partition=True,face_histogram_conservation=True,crop_histogram_conservation=True,max11_seconds=True,completed_videos=len(files),pair_count=len(rows),parquet_rows=len(pd.read_parquet(OUT/'pairs.parquet')),thresholds_monotonic=all(all(thresholds[i][key+'_seconds']>=thresholds[i+1][key+'_seconds']-1e-8 for i in range(len(thresholds)-1)) for key in ['crop_edge','face_min','face_p10','face_median'])))
print(json.dumps({k:summary[k] for k in ['complete','completed_videos','pair_count','candidate_hours','pre_crop_unknown_hours','missing_face_hours','maximum_pair_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':pass
