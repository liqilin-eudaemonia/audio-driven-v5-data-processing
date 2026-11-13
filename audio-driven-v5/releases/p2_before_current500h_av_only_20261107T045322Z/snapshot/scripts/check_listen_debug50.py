"""Verify frozen real samples and exercise seven-class positive-only BCE on CPU."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from p2_listen.contracts import EVENTS,observation,target
from p2_listen.annotations import validate,rasterize,class_statistics
from p2_listen.dataset import verify_media
from p2_listen.events import EventHeads,event_sums,normalized_event_loss
from p2_listen.layout import make_layout

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();torch.set_num_threads(2);torch.manual_seed(20261025)
 observations=json.loads((run/'observations.json').read_text());targets={r['sample_id']:r for r in json.loads((run/'targets.debug.json').read_text())};labels={r['sample_id']:r for r in map(json.loads,(run/'labels.debug-positive.jsonl').read_text().splitlines())};assert len(observations)==len(targets)==len(labels)==50
 receipts=json.loads((run/'raw_receipts.json').read_text());bybranch=collections.defaultdict(list)
 for r in receipts:bybranch[r['branch']].append(r)
 for b,records in bybranch.items():
  lines=(run/'raw'/f'{b}.jsonl').read_bytes().splitlines(keepends=True);assert len(lines)==len(records)
  assert all(hashlib.sha256(line).hexdigest()==r['sha256'] for line,r in zip(lines,records))
 data=json.loads((run/'integrated/review.json').read_text());preserved=0
 original_open={r['sample_id']:r for r in map(json.loads,(run/'open_input/labels.jsonl').read_text().splitlines())}
 for s in data['samples']:
  orig={e['event_id']:e for e in original_open[s['id']]['events']}
  for e in s['original_events']:
   assert all(e[k]==orig[e['event_id']][k] for k in ['start','end','modality','behavior']);preserved+=1
  allowed={q['description'] for o in s['original_events'] for q in o['source_descriptions']}
  for e in s['events']:
   assert 0<=e['start']<e['end']<=s['duration_s']
   if 'timing_evidence' in e:
    assert e['description']==e['text'];assert any((x['start'],x['end'])==(e['start'],e['end']) for x in e['timing_evidence']['members'])
   else:assert e['description'] in allowed
  assert not s['automatic_fusion']['content_substitutions'] and not s['automatic_fusion']['human_feedback_read']
 all_grid=[];known=collections.Counter();losses=[];head=EventHeads(16,12,width=16);gradient_samples=0
 for o in observations:
  o=observation(o);t=target(targets[o['sample_id']],o);verify_media(o,t);label=validate(labels[o['sample_id']],o)
  layout=make_layout(o['requested_duration_s'],(1,1));grid=rasterize(label,layout);all_grid.append(grid)
  for k in EVENTS:
   known[k]+=int(grid[k]['known'].sum());assert bool((grid[k]['y'][grid[k]['known']]==1).all())
  u=SimpleNamespace(video=torch.randn(1,layout['video_groups'].shape[1],16,requires_grad=True),audio=torch.randn(1,layout['audio_groups'].shape[1],12,requires_grad=True),video_valid=layout['video_valid'],audio_valid=layout['audio_valid'])
  logits=head(u,layout,'joint_event');sums,counts=event_sums(logits,grid,{k:1. for k in EVENTS});loss=normalized_event_loss(sums,counts);assert torch.isfinite(loss);loss.backward();losses.append(float(loss))
  if any(int(v)>0 for v in counts.values()):
   gradient_samples+=1;assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
  head.zero_grad(set_to_none=True)
  if int(counts['listener_vocalization']):assert u.audio.grad.abs().sum()>0
  if any(int(counts[k]) for k in EVENTS[:-1]):assert u.video.grad.abs().sum()>0
  u.video.grad=None;u.audio.grad=None;logits=head(u,layout,'probe_only');sums,counts=event_sums(logits,grid,{k:1. for k in EVENTS});normalized_event_loss(sums,counts).backward();assert u.video.grad is None and u.audio.grad is None;head.zero_grad(set_to_none=True)
 stats=class_statistics(all_grid);assert all(not v['enabled'] for v in stats.values())
 report=dict(samples=50,media_verified=50,raw_lines_verified=len(receipts),raw_descriptions_and_intervals_preserved=preserved,seven_head_keys=list(EVENTS),known_positive_bins={k:known[k] for k in EVENTS},known_negative_bins=0,nonzero_gradient_samples=gradient_samples,finite_loss_all_samples=True,E0_isolated_E1_gradient_checked=True,production_class_statistics=stats,smoke_uses_explicit_unit_pos_weights=True,full_LTX_training_run=False,cpu_only_smoke=True)
 (run/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
