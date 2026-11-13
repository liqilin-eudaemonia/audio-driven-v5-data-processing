"""Wait for node11 GPU7, then evaluate current500h pure AV6000 with SFT2752."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read, save, sha, PYTHON
WORK = ROOT/'experiments/p2_current500h_av6000_sft2752_dual200_20261107'
TRAIN = ROOT/'experiments/p2_current500h_av_only31_20261107'
CONDITIONS = ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
ARM = 'current500h_av_only_6000_sft2752'


def install():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()


def code_identity():
    from p2_current_av_only.config import source_identity
    paths = [SELF, ROOT/'src/p2_testsets_eval/runtime.py',
             ROOT/'src/p2_testsets_eval/shared_items.py', ROOT/'src/p2_testsets_eval/metrics.py',
             ROOT/'src/p2_testsets_eval/cluster.py', ROOT/'scripts/p2_pipelined_testsets_eval.py',
             ROOT/'scripts/p2_pipelined_video_metrics.py', ROOT/'src/p2_testsets_eval/metric_pipeline.py']
    return dict(source_identity(), **{str(p): sha(p) for p in paths})


def prepare(work):
    install()
    from p2_testsets_eval.runtime import ready
    from p2_current_av_only.checkpoint import checked
    from p2_current_av_only.config import source_identity, validate
    from p2_current_av_only.data import data_config
    from p2_combined_whisper_eval import link
    import torch
    torch.set_num_threads(2)
    work.mkdir(parents=True, exist_ok=True)
    if (work/'protocol.json').exists():
        p = read(work/'protocol.json')
        assert p['execution_code'] == code_identity(), 'Prepared source changed'
        return p
    saved = checked(TRAIN/'step_006000.pt')
    identity = read(TRAIN/'run.json')
    assert saved['identity'] == identity and saved['step'] == 6000
    assert identity['source'] == source_identity()
    assert len(saved['rank_rng']) == 31
    c = validate(identity['experiment'])
    assert c['loss_weights'] == dict(av=1., event=0., video_feature=0., audio_feature=0.)
    assert not any(k.startswith('whisper') for k in c)
    parent = read(CONDITIONS/'protocol.json')
    assert read(CONDITIONS/'FINAL_AUDIT.json')['success']
    assert read(CONDITIONS/'HANDOFF_COMPLETE.json')['planner_step'] == 2752
    items = read(CONDITIONS/'shared_items.json')
    assert len(items) == len({x['key'] for x in items}) == 400
    assert sha(CONDITIONS/'shared_items.json') == parent['shared_items_sha256']
    plans = {}
    for ds in parent['datasets']:
        link(CONDITIONS/'conditions'/ds, work/'conditions'/ds)
        spec = parent['datasets'][ds]
        assert sha(Path(spec['prepared'])/'references.jsonl') == spec['references_sha256']
    for x in items:
        f = work/'conditions'/x['key']
        assert ready(f/'qwen') and ready(f/'text')
        plan = read(f/'qwen/plan.json')
        assert plan['adapter'] == parent['planner_adapter_identity']
        assert plan['target_description_access'] is False
        plans[x['key']] = dict(plan_sha256=sha(f/'qwen/plan.json'),
                              qwen_complete_sha256=sha(f/'qwen/complete.json'),
                              text_complete_sha256=sha(f/'text/complete.json'))
    for rank in range(8):
        link(CONDITIONS/'negative'/f'{rank:02d}', work/'negative'/f'{rank:02d}')
        save(work/'shards'/f'{rank:02d}.json', read(CONDITIONS/'shards'/f'{rank:02d}.json'))
    save(work/'shared_items.json', items)
    cluster = read(CONDITIONS/'cluster.json')
    assert len(cluster['nodes']) == 1 and cluster['nodes'][0]['host'] == '172.29.189.119'
    cluster['nodes'][0]['gpus'] = [7]
    cluster['training_script'] = str(SELF)
    save(work/'cluster.json', cluster)
    p = copy.deepcopy(parent)
    for k in list(p):
        if k.startswith(('whisper', 'feature_code', 'sft_hidden_cache', 'upstream_')):
            del p[k]
    checkpoint = TRAIN/'step_006000.pt'
    p.update(schema='current500h-pure-av6000-sft2752-dual200-v1', created_at=time.time(),
        arms=[dict(name=ARM, kind='current500h_av_only', step=6000, checkpoint=str(checkpoint),
              checkpoint_sha256=sha(checkpoint), original_checkpoint=str(checkpoint),
              original_sha256=read(checkpoint.with_suffix('.complete.json'))['sha256'],
              planner=parent['planner'])], checkpoint_identity=identity,
        shared_items_manifest=str(work/'shared_items.json'), shared_items_sha256=sha(work/'shared_items.json'),
        cluster=str(work/'cluster.json'), metrics_cluster=str(work/'cluster.json'),
        metrics_workers_per_gpu=8, metrics_inference_workers_per_gpu=8,
        worker_cli=str(SELF), metrics_cpu_pipeline=True,
        execution_order='wait GPU7 empty twice -> pureAV render -> four decode -> eight pipelined metric shards -> aggregate',
        authorized_scope='node11 physical GPU7 only after idle; do not stop any other job',
        execution_code=code_identity(), sampling='native guided_sample; Euler30, no speaker Whisper branch',
        conditioning='Exact same400 SFT2752 planned text/H/Gemma and source first frame as joint6000/6500; no Whisper',
        comparison='Current500h pureAV6000 with SFT2752 inference; joint6000 changes Event+Feature+Whisper together')
    assert p['generation'] == parent['generation'] and p['seed'] == parent['seed']
    save(work/'protocol.json', p)
    save(work/'DATA_CONFIG.json', data_config(c))
    save(work/'REUSED_CONDITIONS.json', dict(source=str(CONDITIONS), same400=True, plans=plans))
    save(work/'CPU_ACCEPTANCE.json', dict(success=True, at=time.time(), step=6000,
        original_sha256=p['arms'][0]['original_sha256'], pure_av=True, source_identity_exact=True,
        conditions_reused=400, planner_step=2752, no_whisper=True,
        no_listener_future_or_target_caption=True, gpu_model_validation='pending idle GPU7'))
    save(work/'status.json', dict(stage='prepared_waiting_gpu7', at=time.time(), outputs=400))
    return p


@contextmanager
def runtime(work):
    install()
    from p2_listen.config import extractor_config
    d = read(work/'DATA_CONFIG.json')
    cfg = extractor_config(d)
    with patch('p2_testsets_eval.runtime.data_config', lambda: (d, cfg)):
        yield d, cfg


def load_model(p, cfg):
    import torch
    from p2_current_av_only.checkpoint import checked
    from p2_current_av_only.config import source_identity, validate
    from p2_current_av_only.model import Model
    from p2_listen.features import weights_identity
    from omni_av_p2.ltx import load_renderer
    saved = checked(p['arms'][0]['checkpoint'])
    assert saved['step'] == 6000 and saved['identity'] == p['checkpoint_identity']
    assert saved['identity']['source'] == source_identity()
    assert saved['identity']['assets'] == {k: weights_identity(read(Path(p['cluster']).parent/'DATA_CONFIG.json'), k)
                                           for k in ('omni', 'ltx', 'gemma')}
    c = validate(saved['identity']['experiment'])
    renderer, strategy, sampler = load_renderer(cfg, 'cuda:0')
    renderer.set_gradient_checkpointing(False)
    torch.manual_seed(c['training']['seed'])
    model = Model(c, cfg, renderer).to('cuda:0')
    assert model.text_targets == saved['identity']['text_targets']
    assert model.qwen_targets == [] and len(model.decoders) == 0
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    renderer.eval().requires_grad_(False)
    return model, strategy, sampler


def render(work, p, cfg):
    import torch
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.shared_items import claim_items, all_items
    from p2_listen.cache import atomic_torch, cpu, tree_to
    from p2_listen.features import unload, make_observation
    from p2_listen.runtime import guided_sample, append_negative
    from omni_av_p2.connector import ConditionInputs
    from omni_av_p2.inference import reference_latents
    from omni_av_p2.ltx import append_conditions, unpack_latents
    from omni_av.ltx_runtime import with_text
    arm = p['arms'][0]
    model, strategy, sampler = load_model(p, cfg)
    save(work/'GPU_MODEL_ACCEPTANCE.json', dict(success=True, at=time.time(), physical_gpu=7,
         native_model_strict_load=True, step=6000, no_whisper=True,
         decoders=0, text_targets=len(model.text_targets), qwen_lora=0))
    negative = tree_to(torch.load(work/'negative/07/payload.pt', map_location='cpu', weights_only=True), 'cuda:0')
    with rt.cached_loaders(['load_video_vae_encoder']), torch.inference_mode():
        for i, item in enumerate(claim_items(work, all_items(work, [], p), p, arm, 'render')):
            start = time.monotonic()
            folder = work/'conditions'/item['key']
            out = work/'latents'/ARM/item['key']
            g = item['geometry']
            plan = read(folder/'qwen/plan.json')
            assert plan['adapter'] == p['planner_adapter_identity'] and plan['target_description_access'] is False
            payload = torch.load(folder/'qwen/hidden.pt', map_location='cpu', weights_only=True)
            text = torch.load(folder/'text/payload.pt', map_location='cpu', weights_only=True)
            assert text['prompt'] == payload['prompt'] == plan['prompt']
            inputs = ConditionInputs(**tree_to(payload['condition'], 'cuda:0'))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                u = model.connector(inputs)
            native = dict(fps=g['fps'], height=g['height'], width=g['width'],
                          frames=g['native_frames'], duration_s=g['native_duration_s'])
            av = reference_latents(cfg, make_observation(rt.observation(item, folder)), native, 'cuda:0')
            torch.manual_seed(p['seed'])
            prepared = strategy.prepare_training_inputs(with_text(av, tree_to(text, 'cuda:0')), sampler)
            prepared = append_conditions(prepared, u)
            neg = append_negative(negative, u)
            torch.manual_seed(p['seed'])
            sampled = guided_sample(model.renderer, prepared, neg, **p['generation'])
            video, audio = unpack_latents(sampled, av)
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            atomic_torch(out/'payload.pt', cpu(dict(video=video, audio=audio)))
            save(out/'metadata.json', dict(key=item['key'], arm=ARM, step=6000,
                 checkpoint=arm['original_checkpoint'], checkpoint_sha256=arm['original_sha256'],
                 geometry=g, seed=p['seed'], generation=p['generation'],
                 planning_adapter=p['planner_adapter_identity'], planning_sha256=sha(folder/'qwen/plan.json'),
                 target_description_access=False, listener_future_input=False, whisper_condition=False,
                 seconds=time.monotonic()-start))
            rt.complete(out, [out/'payload.pt', out/'metadata.json'])
            rt.progress(work, 'render', 7, done=i+1, total=400, last_seconds=time.monotonic()-start)
            del payload, text, inputs, u, av, prepared, neg, sampled, video, audio
    del model, strategy, sampler, negative
    unload()


def worker(work, stage, slot):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7' or 'js-infer-machine-ws3qe' not in os.uname().nodename:
        raise ValueError('Only authorized node11 physical GPU7')
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    p = read(work/'protocol.json')
    assert p['execution_code'] == code_identity()
    rank = 7 if slot == 0 else 70+slot
    (work/'locks').mkdir(exist_ok=True)
    with (work/'locks'/f'{stage}_slot{slot}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with runtime(work) as (_, cfg):
            from p2_testsets_eval import runtime as rt
            from p2_testsets_eval.shared_items import all_items
            rt.progress(work, stage, rank, status='starting', slot=slot, physical_gpu=7)
            if stage == 'render':
                assert slot == 0
                render(work, p, cfg)
            else:
                rt.decode(work, all_items(work, [], p), rank, p, p['arms'])
            rt.progress(work, stage, rank, status='complete', slot=slot, physical_gpu=7)


def gpu_is_idle(report, processes):
    used, util = [float(v.strip()) for v in report.strip().split(',')]
    return used < 2048 and util < 10 and not processes.strip()


def wait_idle(work):
    good = 0
    while True:
        if (work/'STOP').exists():
            return False
        try:
            report = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], text=True, timeout=20)
            processes = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits'], text=True, timeout=20)
            idle = gpu_is_idle(report, processes)
            good = good+1 if idle else 0
            save(work/'GPU_WAIT.json', dict(at=time.time(), report=report, processes=processes,
                  idle=idle, consecutive_idle=good, required_idle_checks=2, physical_gpu=7))
            if good >= 2:
                return True
        except Exception as exc:
            good = 0
            save(work/'GPU_WAIT.json', dict(at=time.time(), error=repr(exc), action='wait and retry'))
        save(work/'status.json', dict(stage='waiting_node11_gpu7_empty', at=time.time(), physical_gpu=7))
        time.sleep(30)


def phase(work, stage, count):
    from p2_testsets_eval.runtime import ready
    p = read(work/'protocol.json')
    items = read(work/'shared_items.json')
    base = work/('latents' if stage == 'render' else 'generated')/ARM
    if all(ready(base/x['key']) for x in items):
        return True
    for attempt in range(1, 4):
        if (work/'STOP').exists():
            return False
        save(work/'status.json', dict(stage=stage, at=time.time(), physical_gpu=7,
             concurrency=count, completed=sum(ready(base/x['key']) for x in items), total=400))
        processes = []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2',
                   PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false')
        for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'P2_TRAIN_MEMORY_GIB']:
            env.pop(key, None)
        for slot in range(count):
            log = work/'logs'/f'{stage}_{slot}_attempt{attempt}.log'
            log.parent.mkdir(exist_ok=True)
            with log.open('ab', buffering=0) as f:
                proc = subprocess.Popen([str(PYTHON), '-u', str(SELF), 'worker', '--work', str(work),
                       '--stage', stage, '--slot', str(slot)], cwd=ROOT, env=env,
                       stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
            processes.append(proc)
            save(work/'processes'/f'{stage}_{slot}.json', dict(pid=proc.pid, at=time.time(), physical_gpu=7, log=str(log)))
        codes = [proc.wait() for proc in processes]
        if (work/'STOP').exists():
            return False
        if all(code == 0 for code in codes) and all(ready(base/x['key']) for x in items):
            save(work/'stages'/(stage+'.json'), dict(complete=True, at=time.time(), physical_gpu=7, workers=count))
            return True
        time.sleep(5)
    raise RuntimeError('Phase failed after bounded retries: '+stage)


def final(work):
    from p2_testsets_eval.runtime import ready
    p = read(work/'protocol.json')
    results = read(work/'RESULTS.json')
    for x in read(work/'shared_items.json'):
        folder = work/'generated'/ARM/x['key']
        assert ready(folder)
        m = read(folder/'metadata.json')
        assert m['step'] == 6000 and m['checkpoint_sha256'] == p['arms'][0]['original_sha256']
        assert m['planning_adapter'] == p['planner_adapter_identity']
        assert m['whisper_condition'] is False and m['listener_future_input'] is False
        assert m['target_description_access'] is False
    for ds in p['datasets']:
        score = results[ARM+'/'+ds]
        assert score['requested_pairs'] == 200 and score['audio']['requested_pairs'] == 200
        assert score['audio']['pairs_with_errors'] == 0
        assert all(score['coverage'][k]['pairs'] == 200 for k in ['psnr', 'ssim', 'lpips', 'fid', 'fvd'])
    save(work/'FINAL_AUDIT.json', dict(success=True, at=time.time(), outputs=400, step=6000,
         planner_step=2752, pure_av=True, no_whisper=True, same_joint400_conditions=True))
    save(work/'COMPARISON.json', dict(current=results, joint6000=read(CONDITIONS/'RESULTS.json'),
         note='Same SFT2752 conditions; pure AV versus the joint Event+Feature+Whisper pipeline, a bundled comparison'))
    (work/'RESULTS.md').write_text('# 当前500h AV-only6000＋最新SFT2752\n\n两套固定200条，全部结果见RESULTS.json，与联合6000对照见COMPARISON.json。\n')


def queue(work):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:
        raise ValueError('Controller must run on node11')
    work.mkdir(parents=True, exist_ok=True)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = prepare(work)
        save(work/'CONTROLLER.json', dict(pid=os.getpid(), at=time.time(), physical_gpu=7))
        if (work/'HANDOFF_COMPLETE.json').exists():
            return
        if not wait_idle(work):
            return
        save(work/'GPU_IDLE_ACCEPTED.json', read(work/'GPU_WAIT.json'))
        for stage, count in [('render', 1), ('decode', 4)]:
            if not phase(work, stage, count):
                save(work/'status.json', dict(stage='stopped', at=time.time()))
                return
        save(work/'GENERATION_COMPLETE.json', dict(total_outputs=400, at=time.time(), physical_gpu=7))
        from p2_testsets_eval.cluster import bind
        bind(work)
        from p2_testsets_eval import metrics
        import listen_cluster
        save(work/'SMOKE_COMPLETE.json', dict(status='generation_passed', outputs=400, at=time.time()))
        metrics.prepare_smoke(work)
        original_transport = listen_cluster.transport
        def transport(cfg, node, cmd):
            return original_transport(cfg, node, ['env', 'CUDA_VISIBLE_DEVICES=7', *cmd])
        save(work/'status.json', dict(stage='metrics', at=time.time(), physical_gpu=7))
        # Preserve idle-only GPU acquisition; the user explicitly requested waiting.
        with patch.object(metrics, 'CLI', ROOT/'scripts/p2_pipelined_testsets_eval.py'), \
             patch.object(listen_cluster, 'transport', transport):
            metrics.queue(work)
        if (work/'STOP').exists():
            return
        final(work)
        save(work/'HANDOFF_COMPLETE.json', dict(outputs=400, step=6000, planner_step=2752, at=time.time()))
        save(work/'status.json', dict(stage='complete', outputs=400, at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'queue', 'worker'])
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--stage', choices=['render', 'decode'])
    parser.add_argument('--slot', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare(args.work.resolve())
        elif args.action == 'queue':
            queue(args.work.resolve())
        else:
            worker(args.work.resolve(), args.stage, args.slot)
    except Exception as exc:
        save(args.work/'FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
        raise
