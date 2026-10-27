"""Read all 50 real cached samples with no teacher directory and an I/O guard."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from p2_listen.config import load,activate
from p2_listen.contracts import atomic_json
from p2_listen.dataset import load_observations,load_targets
from p2_listen.training_data import load_training_sample
from p2_listen.features import load_text
from p2_listen.annotations import class_statistics
from p2_listen.model import build_model

OUT=Path(__file__).resolve().parent
c=load(OUT/"runtime/config.json");activate(c)
assert not (OUT/"runtime/cache/teacher").exists()
blocked=[]
def audit(event,args):
    if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.path.abspath(os.fsdecode(args[0]))
        if "/cache/teacher/" in path:
            blocked.append(path)
            raise AssertionError("Teacher accessed during AV-only validation: "+path)
sys.addaudithook(audit)
torch.set_num_threads(2)
observations=load_observations(c,"train");targets=load_targets(c,observations)
labels=[];sample_reports=[]
def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite(v)
    elif hasattr(value,"__dataclass_fields__"):
        for name in value.__dataclass_fields__:finite(getattr(value,name))
for obs in observations:
    sample=load_training_sample(c,obs,targets[obs["sample_id"]])
    assert sample[2] is None
    finite(sample);labels.append(sample[4])
    sample_reports.append(dict(sample_id=obs["sample_id"],input_shape=list(sample[0].hidden.shape),
        video_shape=list(sample[3]["av"]["latents"]["latents"].shape),
        audio_shape=list(sample[3]["av"]["audio_latents"]["latents"].shape)))
finite(load_text(c))
stats=class_statistics(labels,c["events"]["pos_weight_cap"],c["events"]["K_train"])
# Metadata-only construction of the actual pretrained LTX architecture.
from safetensors import safe_open
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from p2_listen.config import extractor_config
with safe_open(extractor_config(c).ltx_path,framework="pt",device="cpu") as checkpoint:
    native_config=json.loads(checkpoint.metadata()["config"])
with torch.device("meta"):
    renderer=LTXModelConfigurator.from_config(native_config)
    model=build_model(c,renderer=renderer)
model.assert_parameter_ownership()
assert len(renderer.transformer_blocks)==48
assert len(model.text_lora_targets)==48*2*4
assert all(p.is_meta for p in model.parameters())
assert not any("decoder" in name for name in model.state_dict())
groups=model.optimizer_groups(c["training"])
assert [g["name"] for g in groups]==["connector","events","text_lora"]
full=dict(c,training=dict(c["training"],feature_loss_enabled=True,feature_lr=3e-5))
try:
    load_training_sample(full,observations[0],targets[observations[0]["sample_id"]])
except FileNotFoundError as error:
    assert "teacher" in str(error)
else:raise AssertionError("Full mode incorrectly accepted missing teacher")
assert not blocked
result=dict(passed=True,samples=len(observations),teacher_directory_absent=True,teacher_open_attempts=blocked,
    all_cache_identities_and_hashes_verified=True,all_tensors_finite=True,
    optimizer_groups=[g["name"] for g in groups],trainable_parameters=sum(p.numel() for p in model.parameters()),
    full_mode_rejects_missing_teacher=True,class_statistics=stats,sample_shapes=sample_reports,
    pretrained_models_loaded=False,real_gpu_s0=False,
    native_architecture_constructed_on_meta=True,transformer_blocks=len(renderer.transformer_blocks),
    text_lora_target_count=len(model.text_lora_targets),text_lora_targets=model.text_lora_targets,
    text_lora_parameters=sum(p.numel() for p in model.text_lora.parameters()))
atomic_json(OUT/"real_cache_validation.json",result)
print(json.dumps({k:v for k,v in result.items() if k not in ("sample_shapes","class_statistics","text_lora_targets")},indent=2))
