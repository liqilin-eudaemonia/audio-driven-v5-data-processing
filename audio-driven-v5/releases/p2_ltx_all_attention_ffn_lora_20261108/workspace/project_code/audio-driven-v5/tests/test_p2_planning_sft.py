from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from p2_planning_sft.data import append_targets, messages
from p2_planning_sft.model import response_loss
from p2_planning_sft.continuation import align_loaded_learning_rate, validate_extension


def test_assistant_only_labels_include_eos_and_exclude_all_padding():
    ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    batch = append_targets(ids, mask, [[6, 7], [8]], 0, 9, 12)
    assert batch['input_ids'].tolist() == [[1, 2, 3, 6, 7, 9], [4, 5, 8, 9, 0, 0]]
    assert batch['labels'].tolist() == [[-100, -100, -100, 6, 7, 9], [-100, -100, 8, 9, -100, -100]]
    assert batch['attention_mask'].tolist() == [[1]*6, [1, 1, 1, 1, 0, 0]]


@pytest.mark.parametrize('target,maximum,match', [([], 12, 'Empty'), ([6,9], 12, 'EOS'), ([6,7], 4, 'truncation')])
def test_bad_targets_and_overflow_fail_without_silent_truncation(target, maximum, match):
    with pytest.raises(ValueError, match=match):
        append_targets(torch.tensor([[1,2,3]]), torch.ones(1,3), [target], 0, 9, maximum)


def test_causal_response_loss_matches_native_ce_and_ignored_positions_have_zero_gradient():
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 10, requires_grad=True)
    labels = torch.tensor([[-100,-100,-100,6,7,9],[-100,-100,8,9,-100,-100]])
    actual = response_loss(SimpleNamespace(logits=logits), labels)
    expected = F.cross_entropy(logits[:,:-1].reshape(-1,10),labels[:,1:].reshape(-1))
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert not logits.grad[:,:-1][labels[:,1:]==-100].any()
    assert logits.grad[0,2].abs().sum() > 0  # Last prompt position predicts first response token.


def test_global_token_normalization_with_unequal_rank_and_microbatch_token_counts():
    torch.manual_seed(4)
    logits = torch.randn(2, 6, 10, requires_grad=True)
    labels = torch.tensor([[-100,-100,-100,6,7,9],[-100,-100,8,9,-100,-100]])
    total_tokens = labels.ne(-100).sum()
    global_loss = response_loss(SimpleNamespace(logits=logits), labels)
    # Trainer multiplies each local sum/global_token_count by world_size;
    # DDP averages gradients over ranks, recovering the global token mean.
    local_losses = [2 * response_loss(SimpleNamespace(logits=logits[i:i+1]),
                    labels[i:i+1], total_tokens) for i in range(2)]
    distributed = sum(local_losses)/2
    torch.testing.assert_close(distributed, global_loss)
    torch.testing.assert_close(torch.autograd.grad(distributed,logits,retain_graph=True)[0],
                               torch.autograd.grad(global_loss,logits)[0])


def test_source_messages_never_include_a_target_or_listener_audio():
    rows = messages(4.2)
    assert [r['role'] for r in rows] == ['system','user']
    assert [p['type'] for p in rows[1]['content']] == ['image','audio','text']
    assert '4.200' in rows[1]['content'][2]['text']


def test_extension_refreshes_terminal_lr_without_resetting_adam_or_taking_a_step():
    from transformers import get_cosine_schedule_with_warmup
    p = torch.nn.Parameter(torch.tensor([1.0]))
    old = torch.optim.AdamW([p], lr=1e-4)
    schedule = get_cosine_schedule_with_warmup(old, 1, 4)
    for _ in range(4):
        p.grad = torch.ones_like(p)
        old.step(); schedule.step()
    assert old.param_groups[0]['lr'] == 0
    new = torch.optim.AdamW([p], lr=1e-4)
    extended = get_cosine_schedule_with_warmup(new, 1, 8)
    new.load_state_dict(old.state_dict()); extended.load_state_dict(schedule.state_dict())
    before = new.state[p]['exp_avg'].clone()
    rates = align_loaded_learning_rate(new, extended)
    assert 0 < rates[0] < 1e-4
    assert extended.last_epoch == 4 and new.state[p]['step'].item() == 4
    assert torch.equal(before, new.state[p]['exp_avg'])


def test_extension_only_allows_one_more_epoch_with_same_inputs(tmp_path):
    import copy
    import json
    parent = tmp_path/'parent'; checkpoint = parent/'checkpoint-4'; checkpoint.mkdir(parents=True)
    old_config = dict(output=str(parent), training=dict(epochs=1,warmup_ratio=.03,learning_rate=1e-4))
    identity = dict(train_manifest='a',dev_manifest='b',train_ids='c',train_count=20,dev_count=4,
                    world_size=1,system='s',user='u',smoke=True,code={'data.py':'d','model.py':'m'})
    (parent/'run.json').write_text(json.dumps(dict(identity,config=old_config)))
    (parent/'COMPLETE.json').write_text(json.dumps(dict(success=True,smoke=True,global_step=4)))
    (checkpoint/'trainer_state.json').write_text(json.dumps(dict(global_step=4,max_steps=4)))
    for name in ('adapter_model.safetensors','optimizer.pt','scheduler.pt'):
        (checkpoint/name).touch()
    config = copy.deepcopy(old_config);config['output']=str(tmp_path/'new');config['extend_from']=str(checkpoint)
    config['training']['epochs']=2
    result = validate_extension(config,identity,tmp_path/'new')
    assert result['parent_step']==4 and result['warmup_steps']==1
    config['training']['learning_rate']=2e-4
    with pytest.raises(ValueError,match='Only output'):
        validate_extension(config,identity,tmp_path/'new')
