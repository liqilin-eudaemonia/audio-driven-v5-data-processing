"""Native Trainer/Accelerate DDP: PEFT checkpoints, optimizer/RNG resume and TB."""
import json
import os
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler

from p2_listen.contracts import atomic_json, read, sha, digest
from .data import PlanningCollator, SYSTEM, USER
from .reference_data import datasets
from .model import build_model, response_loss


class PlanningTrainer(Trainer):
    def get_total_train_batch_size(self, args):
        return getattr(self, 'fixed_global_batch', None) or super().get_total_train_batch_size(args)

    def get_train_dataloader(self):
        if getattr(self, 'fixed_global_batch', None):
            from .global_batch import loader
            return loader(self)
        return super().get_train_dataloader()

    def _load_rng_state(self, checkpoint):
        super()._load_rng_state(checkpoint)
        if getattr(self, 'starting_performance_migration', False):
            from .performance_resume import audit_loaded_rng
            audit_loaded_rng(self, checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if getattr(self, 'starting_performance_migration', False):
            from .performance_resume import audit_loaded_optimizer
            audit_loaded_optimizer(self, checkpoint)
        if getattr(self, 'starting_extension', False):
            from .continuation import align_loaded_learning_rate
            rates = align_loaded_learning_rate(self.optimizer, self.lr_scheduler)
            steps = {int(v['step'].item()) for v in self.optimizer.state.values() if 'step' in v}
            if steps != {self.extension_parent_step}:
                raise ValueError('Optimizer step history was not restored from the parent run')
            if self.is_world_process_zero():
                atomic_json(Path(self.args.output_dir)/'extension_load.json', dict(
                    checkpoint=str(checkpoint), optimizer_steps=sorted(steps),
                    optimizer_state_entries=len(self.optimizer.state), learning_rates=rates,
                    scheduler_last_epoch=self.lr_scheduler.last_epoch, optimizer_preserved=True))

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=dataset.lengths)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = self.accelerator.unwrap_model(model).get_base_model()
        base.rope_deltas = None
        # Keep frozen perception modules deterministic while LoRA is in train mode.
        base.visual.eval()
        base.audio_tower.eval()
        inputs['use_cache'] = False
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class GradientAudit(TrainerCallback):
    """Record a real finite LoRA gradient before the first optimizer update."""
    def __init__(self, output):
        self.output = Path(output)
        self.written = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.written:
            return
        rows = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for _, p in rows)
        norm = sum(p.grad.float().square().sum().item() for _, p in rows if p.grad is not None) ** .5
        if not finite or not norm > 0:
            raise ValueError('Missing, nonfinite or zero LoRA gradient')
        frozen_gradient = any(p.grad is not None for p in model.parameters() if not p.requires_grad)
        if frozen_gradient:
            raise ValueError('Frozen base model received a gradient')
        if state.is_world_process_zero:
            atomic_json(self.output / 'gradient_audit.json', dict(finite=finite, norm=norm,
                trainable_tensors=len(rows), trainable_parameters=sum(p.numel() for _, p in rows),
                frozen_gradient=frozen_gradient, world_size=args.world_size))
        self.written = True


class StopAfter(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step is not None and state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


class StepMetrics(TrainerCallback):
    def __init__(self, output):
        self.output = Path(output)

    def on_train_begin(self, args, state, control, **kwargs):
        self.previous = time.monotonic()

    def on_step_begin(self, args, state, control, **kwargs):
        self.begin = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        if state.is_world_process_zero:
            row = dict(step=state.global_step, time=time.time(),
                step_seconds=now-self.begin, wall_since_previous_step_s=now-self.previous,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            with (self.output/'step_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        self.previous = now


def train(config_path, resume=None, smoke=False, output=None, stop_after=None):
    config = read(config_path)
    torch.set_num_threads(config.get('cpu_threads', 2))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        if os.environ.get('PLANNING_MEMORY_GIB'):
            limit = float(os.environ['PLANNING_MEMORY_GIB']) * 2**30
            torch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(0).total_memory)
    set_seed(config['seed'])
    t = config['training']
    evaluate = t.get('evaluate', True)
    if type(evaluate) is not bool:
        raise ValueError('training.evaluate must be boolean')
    root = Path(output or config['output'])
    if smoke and output is None:
        raise ValueError('Smoke training must use a separate --output')
    training, validation, full_validation = datasets(config, smoke)
    if set(training.ids) & set(full_validation.ids):
        raise ValueError('Training and validation IDs overlap')
    identity = dict(config=config, smoke=smoke, train_manifest=training.manifest_sha256,
        dev_manifest=validation.manifest_sha256, train_ids=digest(training.ids),
        train_count=len(training), dev_count=len(validation), system=SYSTEM, user=USER,
        code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        world_size=int(os.environ.get('WORLD_SIZE', 1)))
    if config.get('data'):
        identity['reference_data'] = dict(train=training.identity, diagnostic_dev=validation.identity, full_dev=full_validation.identity)
        identity['full_dev_count'] = len(full_validation)
    migration = None
    if config.get('performance_resume'):
        from .performance_resume import validate_performance_resume
        migration = validate_performance_resume(config, identity, root)
        identity['performance_migration'] = migration
    extension = None
    if config.get('extend_from'):
        from .continuation import validate_extension
        extension = validate_extension(config, identity, root)
        identity['extension'] = extension
    if resume:
        if read(root / 'run.json') != identity:
            raise ValueError('Resume data/config/code/world size differs from original run')
        if Path(resume).resolve().parent != root.resolve():
            raise ValueError('Resume checkpoint must belong to the same output directory')
    elif (root / 'run.json').exists():
        raise FileExistsError('Existing run: use --resume or a new output')
    # All ranks check first, rank0 writes after Trainer initializes the process group.
    args = TrainingArguments(output_dir=str(root), num_train_epochs=t['epochs'],
        per_device_train_batch_size=t['micro_batch'], per_device_eval_batch_size=t['eval_batch'],
        gradient_accumulation_steps=t['gradient_accumulation'], learning_rate=t['learning_rate'],
        warmup_ratio=t['warmup_ratio'], warmup_steps=extension['warmup_steps'] if extension else 0,
        lr_scheduler_type='cosine', weight_decay=t['weight_decay'],
        max_grad_norm=1., bf16=True, tf32=True, optim='adamw_torch_fused',
        gradient_checkpointing=t.get('gradient_checkpointing', True), gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=1, logging_first_step=True, logging_nan_inf_filter=False,
        save_strategy='steps', save_steps=1 if smoke else t.get('save_steps', 100), save_total_limit=t['save_total_limit'],
        eval_strategy='steps' if evaluate else 'no',
        eval_steps=(1 if smoke else t.get('eval_steps', 100)) if evaluate else None,
        do_eval=evaluate, prediction_loss_only=True,
        report_to=['tensorboard'], logging_dir=str(root/'tensorboard'),
        remove_unused_columns=False, label_names=['labels'],
        dataloader_num_workers=t['workers'], dataloader_pin_memory=True,
        dataloader_persistent_workers=t['workers'] > 0, dataloader_drop_last=False,
        dataloader_prefetch_factor=t.get('prefetch_factor',2) if t['workers']>0 else None,
        ddp_find_unused_parameters=False, ddp_timeout=600,
        average_tokens_across_devices=True, seed=config['seed'], data_seed=config['seed'],
        disable_tqdm=True, save_safetensors=True)
    model = build_model(config)
    trainer = PlanningTrainer(model=model, args=args, train_dataset=training,
        eval_dataset=validation if evaluate else None, data_collator=PlanningCollator(config['model'], config['max_length'], config.get('max_pixels', 50176)),
        compute_loss_func=response_loss, callbacks=[GradientAudit(root), StopAfter(stop_after), StepMetrics(root)])
    # The custom loss receives a global token denominator through compute_loss_func.
    # Do not pass that kwarg to native Qwen, which does not use it in its own CE.
    trainer.model_accepts_loss_kwargs = False
    trainer.fixed_global_batch = t.get('global_batch_size')
    trainer.starting_performance_migration = migration is not None and resume is None
    trainer.performance_migration = migration
    trainer.starting_extension = extension is not None and resume is None
    trainer.extension_parent_step = None if extension is None else extension['parent_step']
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        atomic_json(root/'run.json', identity, immutable=True)
        atomic_json(root/'trainable.json', dict(names=[n for n,p in model.named_parameters() if p.requires_grad],
            parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print(json.dumps(dict(stage='train', samples=len(training), dev_samples=len(validation) if evaluate else 0,
            world=args.world_size, global_batch=trainer.get_total_train_batch_size(args),
            smoke=smoke)), flush=True)
    trainer.accelerator.wait_for_everyone()
    result = trainer.train(resume_from_checkpoint=resume or (migration['checkpoint'] if migration else None) or (extension['checkpoint'] if extension else None))
    complete = trainer.state.global_step >= trainer.state.max_steps
    # A disabled evaluation policy also skips final/stopped-run evaluation.
    metrics = None
    if evaluate:
        metrics = trainer.evaluate(eval_dataset=full_validation if complete else validation,
                                   metric_key_prefix='eval_full' if complete and config.get('data') else 'eval')
    if complete:
        trainer.save_model(str(root/'final_adapter'))
    trainer.save_state()
    if trainer.is_world_process_zero():
        atomic_json(root/('COMPLETE.json' if complete else 'STOPPED.json'), dict(success=True, smoke=smoke, global_step=trainer.state.global_step,
            train=result.metrics, validation_enabled=evaluate, validation=metrics, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    trainer.accelerator.wait_for_everyone()
