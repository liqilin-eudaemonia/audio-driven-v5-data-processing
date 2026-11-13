#!/usr/bin/env python3
"""Two-rank CPU/Gloo integration check of global metrics and rank-zero event files."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch.distributed as dist
from p2_listen.contracts import EVENTS,atomic_json
from p2_listen.monitoring import global_scalars,open_monitor,rank_zero_action

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size()
    if world!=2:raise ValueError('Run with two ranks')
    logger=None
    try:
        record=dict(base=2. if rank==0 else 10.,av=2. if rank==0 else 10.,
            av_loss_by_modality=dict(video=1. if rank==0 else 6.,audio=1. if rank==0 else 4.),
            feature_losses=None,event=.8,event_loss_per_class={k:.8 if k=='head_nod' else None for k in EVENTS},
            event_loss_by_modality=dict(video=.8,audio=0.),event_known_bins={k:10 if k=='head_nod' else 0 for k in EVENTS},
            grad_norm=dict(main=2.,events=.1))
        settings=dict(training=dict(tensorboard=dict(enabled=True)))
        logger=open_monitor(a.output,settings,start=0,resume=False,rank=rank,world=world)
        assert (logger is not None)==(rank==0)
        metrics=global_scalars(record,local_samples=1 if rank==0 else 3,media_seconds=2. if rank==0 else 6.,
            timings=dict(step_seconds=2. if rank==0 else 4.,data_seconds=1.,compute_seconds=1.,checkpoint_seconds=0.,
                allocated_gib=rank+1.,reserved_gib=rank+2.,peak_allocated_gib=rank+3.,peak_reserved_gib=rank+4.),
            optimizer=SimpleNamespace(param_groups=[dict(name='connector',lr=.001)]),event_weight=.05,arm='E1',
            step=1,dataset_size=50,device='cpu',world=world)
        assert metrics['loss/av']==8.,metrics
        assert metrics['loss/event']==.8  # Already globally normalized; never double-reduce.
        assert metrics['loss/av_video']==4.75 and metrics['loss/av_audio']==3.25
        assert metrics['throughput/samples_per_second']==1.
        assert metrics['memory/peak_allocated_gib_max_rank']==4.
        assert 'event_loss/head_shake' not in metrics
        rank_zero_action(lambda:logger.write(1,metrics,flush=True),rank=rank,world=world)
        # Failure on the writer rank is surfaced on both ranks, not hidden.
        try:rank_zero_action(lambda:1/0,rank=rank,world=world)
        except RuntimeError:pass
        else:raise AssertionError('Writer failure did not propagate')
        dist.barrier()
        if rank==0:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea=EventAccumulator(str(a.output/'tensorboard'));ea.Reload()
            assert ea.Scalars('loss/av')[0].value==8.
            assert len(list((a.output/'tensorboard').glob('events.out.tfevents.*')))==1
            atomic_json(a.output/'summary.json',dict(passed=True,world=world,weighted_av=8.,event=.8,
                rank_zero_only=True,writer_failure_propagated=True,scalar_tags=ea.Tags()['scalars']))
            print('Global weighting, event normalization, rank-zero writer and failure propagation passed.',flush=True)
    finally:
        if logger is not None:logger.close()
        dist.destroy_process_group()

if __name__=='__main__':main()
