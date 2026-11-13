"""Prepared real trial contract, CLI rank routing, report tamper and source-only boundaries."""
from pathlib import Path
from unittest.mock import patch
import copy,json
import pytest
from p2_listen.contracts import ROOT,read,sha,atomic_json,digest
from p2_stage2_joint_v2.config import validate

CONFIG=ROOT/'experiments/p2_stage2_joint_balance_v2_20261112/candidate_full6000/config.json'


@pytest.mark.parametrize('mutate',[
 lambda c:c.update(global_batch=32),lambda c:c.update(steps=1000),lambda c:c.update(qwen_lr=float('nan')),
 lambda c:c['generator']['components'].update(feature=False),lambda c:c['generator']['loss_weights'].update(event=0.),
 lambda c:c['generator'].update(j_cache='/old/H'),lambda c:c['evaluation']['sampling'].update(steps=50),
 lambda c:c.update(output=read(c['parent_config'])['output'])])
def test_real_candidate_rejects_objective_architecture_data_and_budget_mutations(mutate):
 c=read(CONFIG);validate(c);mutate(c)
 with pytest.raises(ValueError):validate(c)


def test_real_31rank_commands_use_joint_entry_and_reserve_gpu7():
 import p2_stage2_joint_v2_train as cli
 c=validate(read(CONFIG));cluster=read(Path(c['output'])/'cluster.json');mapping=cli.layout(cluster)
 assert [len(x['ranks'])for x in mapping]==[8,8,8,7]
 commands=cli.commands(c)
 assert len(commands)==4
 assert all(str(cli.SELF)in cmd and 'P2_COMPONENT_STOP_AFTER=6000'not in cmd for cmd in commands)
 assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6'in commands[-1]
 with pytest.raises(ValueError):cli.commands(c,Path('/wrong/step_000100.pt'))


def test_evaluation_report_detects_changed_evidence_and_source(tmp_path):
 from p2_stage2_joint_v2.evaluation import verify_report
 out=tmp_path/'evaluation/step_000000';out.mkdir(parents=True)
 evidence=out/'metric.json';atomic_json(evidence,{'score':.4})
 contract={'step':0,'checkpoint_sha256':'x','validation_sha256':'v','policy_sha256':'p','source':{'code':'hash'}}
 atomic_json(out/'CONTRACT.json',contract)
 atomic_json(out/'REPORT.json',dict(success=True,**contract,summary={'score':.4},metrics={'metric.json':sha(evidence)}))
 with patch('p2_stage2_joint_v2.evaluation.stage_contract',return_value=contract),patch('p2_stage2_joint_v2.evaluation.derive_summary',return_value={'score':.4}):
  assert verify_report({'output':str(tmp_path)},0)['success']
  atomic_json(evidence,{'score':.9})
  with pytest.raises(ValueError,match='evidence changed'):verify_report({'output':str(tmp_path)},0)


def test_resume_inside_window_still_checks_prior_evaluation_barriers():
 from p2_stage2_joint_v2.training import evaluation_barrier
 with patch('p2_stage2_joint_v2.policy.require_evaluation')as req:
  evaluation_barrier({},57);assert [c.args[1]for c in req.call_args_list]==[0]
  req.reset_mock();evaluation_barrier({},137);assert [c.args[1]for c in req.call_args_list]==[0,100]


def test_frozen_gt_labels_are_shared_but_generated_labels_remain_independent():
 from p2_stage2_joint_v2.evaluation import fixed_reference_records
 frozen=[dict(id=str(i),reference_identity='ref'+str(i),protocol_identity='same',reference={'label':'present'})for i in range(64)]
 generated=[dict(r,reference={'label':'absent'},generated={'label':'prediction'})for r in frozen]
 result=fixed_reference_records(generated,frozen)
 assert all(r['reference']=={'label':'present'}and r['generated']=={'label':'prediction'}for r in result)
 generated[0]['reference_identity']='different-media'
 with pytest.raises(ValueError):fixed_reference_records(generated,frozen)
