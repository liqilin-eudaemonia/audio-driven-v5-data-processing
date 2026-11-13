"""Compare with the exact previous page result, and separately with production."""
import sys,collections,json
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'experiments/pair_boundary_holdout3_20261029'),str(ROOT/'src')]
import compare
from p3_listen.common import sha
read=compare.read
HEAD=ROOT/'experiments/pair_head_attach5_20261029'
PAIRS=['naturalistic_V02_S3610_I00000088','improvised_V01_S0122_I00000125','naturalistic_V02_S5228_I00000302','naturalistic_V03_S2098_I00000195','improvised_V00_S0862_I00000794','naturalistic_V02_S3579_I00000175','naturalistic_V03_S1909_I00000205']
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def kept(d):return [r for r in d['clips'] if r['audio_final']['pool']=='keep_candidate']
def main():
 reports=[]
 for i,pair in enumerate(PAIRS):
  source=read(EXP/(pair+'.json'));new=read(EXP/'results'/pair/'result.json');view=read(EXP/'baseline_views'/(pair+'.json'))
  if i<2:
   path=ROOT/'experiments/p3_listen_full_source_recovery_20261023/results'/(pair+'.json');pilot=read(path);attempts=pilot['attempts'];parent_ids={r.get('recovery_parent_id') for r in attempts}
   leaves=[r for r in attempts if r['sample_id'] not in parent_ids];parents=[r for r in attempts if r['sample_id'] in parent_ids]
   old=dict(pair=pair,revision='published-full-source-recovery-20261023',source_seconds=source['limit'],clips=leaves,superseded_attempts=parents,initial_ids=[r['sample_id'] for r in attempts if not r.get('recovery_parent_id')],pool_counts=dict(collections.Counter(r['audio_final']['pool'] for r in leaves)),counts={},kept_seconds=pilot['summary']['new_kept_seconds'],complete=True)
   assert set(view['kept_ids'])=={r['sample_id'] for r in kept(old)}
   write(EXP/'baseline_results'/pair/'result.json',old)
   baseline_kind='previously_published_20261023_pilot'
  else:
   path=HEAD/'results'/pair/'result.json';old=read(path);baseline_kind='previous_H_five_source_trial'
   assert {(r['speaker_index'],r['start'],r['end']) for r in kept(old)}=={(r['speaker_index'],r['start'],r['end']) for r in view['fixed_pairs']['rows'] if r['id'] in view['fixed_pairs']['terminal_ids'] and r['pool']=='keep_candidate'}
  prod=read(ROOT/'data_corpora/seamless_listen_local_full/results'/(pair+'.json'))
  v=compare.evaluate(source,kept(old),kept(new));initial=compare.evaluate(source,compare.roots(old),compare.roots(new))
  quality_changed=[]
  prior={compare.sig(r):r for r in old['clips']+old['superseded_attempts']}
  for r in new['clips']+new['superseded_attempts']:
   prev=prior.get(compare.sig(r))
   if prev and (prev['audio_final']['pool'],prev['audio_final']['reasons'])!=(r['audio_final']['pool'],r['audio_final']['reasons']):quality_changed.append(dict(start=r['start'],end=r['end'],before=prev['audio_final']['pool'],after=r['audio_final']['pool']))
  reports.append(dict(pair=pair,source_seconds=source['limit'],baseline_kind=baseline_kind,baseline_file=str(path.relative_to(ROOT)),baseline_sha256=sha(path),initial=initial,kept=v,old_kept_seconds=old['kept_seconds'],new_kept_seconds=new['kept_seconds'],same_interval_quality_changes=quality_changed,vs_production=compare.evaluate(source,kept(prod),kept(new))))
 write(EXP/'baseline_results/contract.json',dict(pause_threshold_seconds=3,revision='published-full-source-recovery-20261023'))
 blockers=[dict(pair=r['pair'],reason='previously_retained_speaker_words_now_uncovered',words=r['kept']['previous_speaker_words_lost']) for r in reports if r['kept']['previous_speaker_words_lost']]
 result=dict(pairs=reports,full_run_gate=dict(passed=not blockers,blocking_findings=blockers,full_run_started=False,rule_unchanged_during_validation=True))
 write(EXP/'comparison.json',result)
 for r in reports:print(r['pair'],'kept',r['kept']['old_count'],'->',r['kept']['new_count'],'unchanged',r['kept']['unchanged'],'seconds',round(r['old_kept_seconds'],3),'->',round(r['new_kept_seconds'],3),'lost_words',len(r['kept']['previous_speaker_words_lost']),'added_words',len(r['kept']['added_speaker_words']))
 print('GATE',result['full_run_gate'])
if __name__=='__main__':main()
