import copy
from p2_stage2_joint.policy import decide,DEFAULT
from p2_stage2_joint.validation import OverlapIndex


def metric():
 return dict(samples=64,complete=True,planning_ce=.5,lpips=.2,false_sound_rate=.05,format_error_rate=0.,reaction_f1=.4)


def test_100_safety_gate_and_300_positive_benefit_gate():
 base=metric();current=metric()
 assert decide(base,current,DEFAULT,100)['decision']=='continue_to_300'
 assert decide(base,current,DEFAULT,300)['decision']=='hold'
 current['reaction_f1']=.43
 result=decide(base,current,DEFAULT,300)
 assert result['decision']=='eligible_for_1000_review'and result['auto_launch']is False


def test_regression_or_unknown_never_passes():
 base=metric()
 for key,value in [('planning_ce',.6),('lpips',.23),('false_sound_rate',.09),('format_error_rate',.04)]:
  row=metric();row[key]=value
  assert decide(base,row,DEFAULT,100)['decision']=='stop'
 for key,value in [('planning_ce',None),('samples',63),('complete',False)]:
  row=metric();row[key]=value
  assert decide(base,row,DEFAULT,100)['decision']=='hold'
 row=metric();row['reaction_f1']=float('nan')
 assert decide(base,row,DEFAULT,300)['decision']=='hold'
 assert decide(base,row,DEFAULT,100)['decision']=='continue_to_300'


def row(a,b,listener='listener',speaker='speaker'):
 return dict(source=dict(source_start_s=a,source_end_s=b,listener_id=listener,speaker_id=speaker))


def test_overlap_excludes_both_roles_and_near_boundaries():
 index=OverlapIndex();index.add(row(1,3));index.add(row(2,5));index.freeze()
 assert index.overlaps(row(4,6,'other','listener'))
 assert index.overlaps(row(5.02,6))
 assert not index.overlaps(row(5.1,6))
 assert not index.overlaps(row(1,2,'unseen1','unseen2'))


def test_plan_execution_omission_unknown_never_counts_as_explicit_absence():
 from p2_stage2_joint.plan_evidence import execution_metrics,prompt,label_schema
 from p2_event_presence.contracts import labels
 plan=labels('unknown');plan['visual']['smiling']=dict(state='present',evidence='smile')
 plan['audio']['audible_laughter']=dict(state='absent',evidence='silent')
 generated=labels('absent');generated['visual']['smiling']['state']='present'
 generated['visual']['nodding']['state']='present' # Plan omission is unknown, not a violation.
 result=execution_metrics([dict(sample_id='a',labels=plan)],[dict(id='a',generated=generated)])
 assert result['planned_positive_known']==1 and result['executed']==1
 assert result['explicit_negative_known']==1 and result['violated']==0
 assert result['execution_recall']==1 and result['explicit_contradiction_rate']==0
 assert 'unknown for omitted' in prompt('smile')
 assert set(label_schema()['properties'])=={'visual','audio'}
