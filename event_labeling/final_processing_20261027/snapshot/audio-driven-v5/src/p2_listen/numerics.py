"""Explicit, checkpoint-bound numerical policy for matched GPU training."""
import os

import torch


POLICY = "p2-deterministic-training-v2"
WORKSPACES = (":4096:8", ":16:8")


def configure_training_numerics():
    """Configure before CUDA initialization; fail rather than silently degrade.

    Frozen AV attention still backpropagates into U. Its default backward can
    vary even for identical input/state, which invalidates cross-world checks.
    Keep BF16 and the objectives, but require deterministic kernel selection.
    """
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        if torch.cuda.is_initialized():
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before initializing CUDA for P2 training")
        workspace = WORKSPACES[0]
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    elif workspace not in WORKSPACES:
        raise ValueError("P2 deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dict(policy=POLICY, deterministic_algorithms=True, warn_only=False,
                gradient_accumulation="fp64", gradient_reduction="sum", loss_normalization="global_batch",
                cublas_workspace_config=workspace, cudnn_benchmark=False, cudnn_deterministic=True,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda)


def validate_numerics(saved, current):
    if saved != current:
        raise ValueError("Checkpoint training numerics differ; use the recorded policy or start a new run")
