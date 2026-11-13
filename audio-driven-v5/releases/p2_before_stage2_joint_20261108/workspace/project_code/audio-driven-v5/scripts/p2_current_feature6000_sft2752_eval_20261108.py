"""Current500h AV+Feature6000, same SFT2752 dual200 inputs and native AV sampler."""
import argparse
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
ORIGINAL = ROOT/'scripts/p2_current_av6000_sft2752_eval_20261107.py'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read, save, sha
import p2_current_av6000_sft2752_eval_20261107 as base
WORK = ROOT/'experiments/p2_current500h_feature6000_sft2752_dual200_20261108'
TRAIN = ROOT/'experiments/p2_current500h_av_feature31_20261107'
AV_RESULT = ROOT/'experiments/p2_current500h_av6000_sft2752_dual200_20261107'
ARM = 'current500h_av_feature_6000_sft2752'

native_identity = base.code_identity
def code_identity():
    from p2_current_av_feature.config import source_identity
    return dict(native_identity(), **source_identity(), **{str(SELF):sha(SELF), str(ORIGINAL):sha(ORIGINAL)})


def adapt(function, replacements):
    original = inspect.getsource(function)
    source = original
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Native adapter source changed: '+old)
        source = source.replace(old, new)
    exec(compile(source, str(SELF)+':'+function.__name__, 'exec'), base.__dict__)
    return dict(function=function.__name__, original_sha256=__import__('hashlib').sha256(original.encode()).hexdigest(),
                replacements=[dict(old=a,new=b) for a,b in replacements])


ADAPTATIONS = [adapt(base.prepare, [
    ('from p2_current_av_only.checkpoint import checked', 'from p2_current_av_feature.checkpoint import checked'),
    ('from p2_current_av_only.config import source_identity, validate', 'from p2_current_av_feature.config import source_identity, validate'),
    ('from p2_current_av_only.data import data_config', 'from p2_current_av_feature.data import data_config'),
    ("assert c['loss_weights'] == dict(av=1., event=0., video_feature=0., audio_feature=0.)",
     "assert c['loss_weights'] == dict(av=1., event=0., video_feature=.025, audio_feature=.025)"),
    ("schema='current500h-pure-av6000-sft2752-dual200-v1'", "schema='current500h-av-feature6000-sft2752-dual200-v1'"),
    ("kind='current500h_av_only'", "kind='current500h_av_feature'"),
    ("comparison='Current500h pureAV6000 with SFT2752 inference; joint6000 changes Event+Feature+Whisper together'",
     "comparison='Same current500h train selection/schedule and same SFT2752 dual200 conditions; feature-only increment versus current pureAV6000; joint differs by Event and Whisper'"),
    ('pure_av=True, source_identity_exact=True', 'pure_av=False, feature_training=True, source_identity_exact=True'),
]), adapt(base.load_model, [
    ('from p2_current_av_only.checkpoint import checked', 'from p2_current_av_feature.checkpoint import checked'),
    ('from p2_current_av_only.config import source_identity, validate', 'from p2_current_av_feature.config import source_identity, validate'),
    ('from p2_current_av_only.model import Model', 'from p2_current_av_feature.model import Model'),
    ('assert model.qwen_targets == [] and len(model.decoders) == 0',
     "assert model.qwen_targets == [] and set(model.decoders) == {'video', 'audio'}"),
]), adapt(base.render, [
    ('decoders=0, text_targets=len(model.text_targets), qwen_lora=0',
     'decoders=len(model.decoders), text_targets=len(model.text_targets), qwen_lora=0'),
]), adapt(base.final, [
    ('planner_step=2752, pure_av=True, no_whisper=True, same_joint400_conditions=True',
     'planner_step=2752, pure_av=False, feature_training=True, no_whisper=True, same_joint400_conditions=True'),
    ('# 当前500h AV-only6000＋最新SFT2752', '# 当前500h AV＋Feature6000＋最新SFT2752'),
])]

native_prepare = base.prepare
native_final = base.final
base.SELF = SELF
base.TRAIN = TRAIN
base.WORK = WORK
base.ARM = ARM
base.code_identity = code_identity


def prepare(work):
    p = native_prepare(work)
    c = p['checkpoint_identity']['experiment']
    av = read(AV_RESULT/'protocol.json')
    original = av['checkpoint_identity']['experiment']
    for key in ['training','connector','qwen_lora','text_lora','current_cache_index','sample_list','qwen_hidden_source']:
        assert c[key] == original[key], 'Feature baseline training mismatch: '+key
    assert c['duration_sampling']['sha256'] == original['duration_sampling']['sha256']
    for key in ['datasets','seed','generation','planner_adapter_identity','shared_items_sha256']:
        assert p[key] == av[key], 'Inference control differs: '+key
    save(work/'RUNTIME_ADAPTER.json', dict(success=True, at=time.time(), original=str(ORIGINAL),
         original_sha256=sha(ORIGINAL), adaptations=ADAPTATIONS,
         av_sampler_math_unchanged=True, no_teacher_or_event_labels_at_inference=True,
         feature_weights=dict(video=.025,audio=.025), same_training_schedule_as_current_AV=True,
         same400_SFT2752_conditions_as_current_AV=True))
    return p


def final(work):
    native_final(work)
    save(work/'COMPARISON.json', dict(current=read(work/'RESULTS.json'),
         av_only6000=read(AV_RESULT/'RESULTS.json'),
         joint6000=read(base.CONDITIONS/'RESULTS.json'),
         note='Feature-only training increment vs current pureAV6000 with same500h schedule and same400 SFT2752 inference conditions; joint adds Event and Whisper.'))
    (work/'RESULTS.md').write_text('# 当前500h AV＋Feature6000＋最新SFT2752\n\n固定双200。与当前纯AV6000为Feature单项对照；全部原值及覆盖见RESULTS.json，AV与联合模型对照见COMPARISON.json。\n')


def wait_memory(work):
    good = 0
    while not (work/'STOP').exists():
        try:
            raw = subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=20)
            used,total,util = [float(x.strip()) for x in raw.split(',')]
            free = total-used
            good = good+1 if free >= 80*1024 else 0
            save(work/'GPU_WAIT.json',dict(at=time.time(),used_mib=used,free_mib=free,util_percent=util,
                 consecutive_ready=good,required_free_mib=80*1024,physical_gpu=7,
                 admission='authorized same-card use; no other-task termination'))
            if good >= 2:
                return True
        except Exception as exc:
            good=0
            save(work/'GPU_WAIT.json',dict(at=time.time(),error=repr(exc)))
        save(work/'status.json',dict(stage='waiting_gpu7_memory',at=time.time(),physical_gpu=7))
        time.sleep(10)
    return False


base.prepare = prepare
base.final = final
base.wait_idle = wait_memory


def queue(work):
    from p2_testsets_eval import metrics
    from p2_current_av6000_score_overlay_20261108 import free_gpu7, launch_budget
    latest = {}
    def available(report):
        latest.clear()
        latest.update(free_gpu7(report))
        save(work/'METRIC_RESOURCE.json',dict(at=time.time(),**latest))
        return [7] if launch_budget(latest['free_mib'],0) else []
    def slots(reports,active):
        count=sum(v[2]==7 for v in active.values())
        budget=launch_budget(latest.get('free_mib',0),count)
        choices=[(name,gpu) for name,cap,gpus in reports for gpu in gpus if gpu==7]
        return [choices[0]]*budget if choices else []
    with patch.object(metrics,'idle_gpus',available),patch.object(metrics,'balanced_metric_slots',slots):
        base.queue(work)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','queue','worker'])
    parser.add_argument('--work',type=Path,default=WORK)
    parser.add_argument('--stage',choices=['render','decode'])
    parser.add_argument('--slot',type=int,choices=range(4),default=0)
    args=parser.parse_args()
    try:
        if args.action=='prepare':prepare(args.work.resolve())
        elif args.action=='queue':queue(args.work.resolve())
        else:base.worker(args.work.resolve(),args.stage,args.slot)
    except Exception as exc:
        save(args.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
