"""
Training utilities for the VLA fine-tuning + closed-loop notebooks.

Helpers shared across the course notebooks (02 fine-tuning, 03 serving +
sim eval). They live here so the notebooks stay focused on the Ray-specific
orchestration, while this module owns the model plumbing.

Sections:
  * PI0.5 attention-mask patch
  * load_pi05_policy
  * NumpyToTorchCollate
  * train_step / optimizer_step
  * truncate_batch
  * build_lr_scheduler
  * make_checkpoint / load_checkpoint  (preserves dataset stats for serving)
  * stage_model_to_local / stage_on_all_nodes (model only -- datasets stream)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from ray.data.iterator import NumpyBatchCollateFn


# ============================================================================
# PI0.5 attention-mask patch
# ============================================================================
def apply_pi05_attention_mask_patch():
    """Tolerate pad/attention mask length mismatches in PI0.5's preprocessor.

    lerobot's preprocessor can produce pad_masks and att_masks of slightly
    different sequence lengths (typically off-by-one after image tokenization);
    upstream make_att_2d_masks doesn't handle that and crashes. We truncate
    both masks to the shorter length on mismatch. Idempotent.
    """
    import lerobot.policies.pi05.modeling_pi05 as mp
    if getattr(mp, "_PI05_MASK_PATCH_APPLIED", False):
        return
    _orig = mp.make_att_2d_masks

    def _patched(pad_masks, att_masks):
        pl, al = pad_masks.shape[-1], att_masks.shape[-1]
        if pl != al:
            L = min(pl, al)
            return _orig(pad_masks[..., :L], att_masks[..., :L])
        return _orig(pad_masks, att_masks)

    mp.make_att_2d_masks = _patched
    mp._PI05_MASK_PATCH_APPLIED = True


# ============================================================================
# Model loading
# ============================================================================
def load_pi05_policy(pretrained_path):
    """Load PI0.5 in fp16, freeze backbone, train only 4 projection heads.

    train_expert_only=True configures the model's expert branch as trainable
    but doesn't actually call requires_grad_(True) on those params -- we do
    that manually here for the 4 action-head modules.
    """
    apply_pi05_attention_mask_patch()
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(
        str(pretrained_path), device="cuda", dtype=torch.float16, train_expert_only=True,
    )
    for p in policy.parameters():
        p.requires_grad = False
    for name, module in policy.model.named_children():
        if name in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}:
            for p in module.parameters():
                p.requires_grad = True
    return policy


# ============================================================================
# Collation: numpy dicts -> GPU tensors
# ============================================================================
class NumpyToTorchCollate(NumpyBatchCollateFn):
    """Convert a numpy batch dict into tensors on the target device.

    Ray Data delivers batches as numpy arrays. This moves them to GPU as
    torch tensors, preserving dtype semantics: integer -> torch.long, bool
    -> torch.bool, everything else -> torch.float32. The ``task`` column
    stays as a Python list of strings (language conditioning).
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        task = list(batch.pop("task"))
        result = {}
        for k, v in batch.items():
            arr = np.asarray(v)
            if arr.dtype == object:
                arr = np.stack([np.asarray(x) for x in v])
            if np.issubdtype(arr.dtype, np.integer):
                result[k] = torch.tensor(arr, dtype=torch.long, device=self.device)
            elif np.issubdtype(arr.dtype, np.bool_):
                result[k] = torch.tensor(arr, dtype=torch.bool, device=self.device)
            else:
                result[k] = torch.tensor(arr, dtype=torch.float32, device=self.device)
        result["task"] = task
        return result


# ============================================================================
# Sequence truncation
# ============================================================================
def truncate_batch(batch, max_len):
    """Clip 2D+ sequence/mask tensors to max_len tokens. max_len=0 disables."""
    if not max_len:
        return batch
    for k in ("tokens", "input_ids", "masks", "attention_mask",
              "pad_masks", "att_masks", "img_masks", "image_masks"):
        if k in batch and hasattr(batch[k], "ndim") and batch[k].ndim >= 2:
            batch[k] = batch[k][..., :max_len]
    return batch


# ============================================================================
# Training step helpers — vanilla PyTorch wrapped in autocast
# ============================================================================
def train_step(policy, batch, preprocessor, max_len, grad_accum, scaler):
    """One forward + scaled backward. Returns scalar loss value."""
    batch = preprocessor(batch)
    batch = truncate_batch(batch, max_len)
    batch.pop("task", None)
    batch.pop("task_index", None)
    with torch.autocast("cuda", torch.float16):
        out = policy(batch)
        loss = out.loss if hasattr(out, "loss") else out[0]
    scaler.scale(loss / grad_accum).backward()
    return float(loss.detach())


def optimizer_step(policy, optimizer, scaler, scheduler):
    """Unscale, clip grads, step optimizer + LR schedule."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], max_norm=1.0,
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


# ============================================================================
# LR schedule
# ============================================================================
def build_lr_scheduler(optimizer, config, num_workers, last_step):
    """Linear warmup -> cosine decay LR schedule."""
    import math
    bs = int(config.get("batch_size", 1))
    ga = int(config.get("grad_accum", 1))
    nepochs = int(config.get("num_epochs", 1))
    rows = int(config.get("total_rows", 10000))
    warm_fr = float(config.get("warmup_frac", 0.1))

    rows_per_worker = rows // num_workers
    total_steps = max(rows_per_worker // (bs * ga), 1) * nepochs
    warmup_steps = int(total_steps * warm_fr)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(warmup_steps, 1)
        progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_step - 1 if last_step > 0 else -1,
    )


# ============================================================================
# Checkpoint I/O
# ============================================================================
def make_checkpoint(policy, optimizer, scaler, epoch, step, stats,
                    base_model_repo, camera_rename):
    """Pickle trainable-only state + dataset stats into a Ray Train Checkpoint.

    Stats are included so policy_server.py can rebuild the same preprocessor
    at inference time without re-reading the dataset. base_model_repo and
    camera_rename are saved as breadcrumbs for downstream consumers.
    """
    import ray.cloudpickle as pickle
    import ray.train

    trainable_keys = {k for k, p in policy.module.named_parameters() if p.requires_grad}
    full_sd = policy.module.state_dict()
    trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys}

    ckpt_dir = tempfile.mkdtemp(prefix="pi05_ckpt_")
    with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
        pickle.dump(
            {"model":            trainable_sd,
             "optim":            optimizer.state_dict(),
             "scaler":           scaler.state_dict(),
             "epoch":            epoch,
             "step":             step,
             "stats":            stats,
             "base_model_repo":  base_model_repo,
             "camera_rename":    camera_rename},
            f,
        )
    return ray.train.Checkpoint.from_directory(ckpt_dir)


def load_checkpoint(checkpoint, policy, optimizer, scaler):
    """Restore from a Ray Train checkpoint. Returns (start_epoch, start_step)."""
    import ray.cloudpickle as pickle
    with checkpoint.as_directory() as d:
        with open(os.path.join(d, "state.pkl"), "rb") as f:
            state = pickle.load(f)
    policy.module.load_state_dict(state["model"], strict=False)
    optimizer.load_state_dict(state["optim"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state["epoch"] + 1, state.get("step", 0)


# ============================================================================
# Per-node HF snapshot staging (model only -- datasets are streamed via hf://)
# ============================================================================
def stage_model_to_local(repo_id, local_dir):
    """Download a HF model to `local_dir` if not already present."""
    local_dir = Path(local_dir)
    if (local_dir / "config.json").exists():
        return f"cached: {local_dir}"
    local_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id, repo_type="model",
        local_dir=str(local_dir),
        token=os.environ.get("HF_TOKEN"),
        max_workers=8,
    )
    return f"downloaded: {local_dir}"


def stage_on_all_nodes(ray_module, stage_fn, label, dest, log_fn=print):
    """Run stage_fn on the head node and every live GPU worker node.

    Each GPU node has its own /mnt/local_storage (per-node disk), so we
    pin a tiny num_cpus=0 task to each node and have it run the same
    snapshot_download call. log_fn defaults to print but can be log.info.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    log_fn(f"Staging {label} -> {dest} ...")
    log_fn(f"  on head: {stage_fn()}")

    nodes = [n for n in ray_module.nodes()
             if n.get("Alive") and n.get("Resources", {}).get("GPU")]

    @ray_module.remote(num_cpus=0)
    def _stage():
        return stage_fn()

    futures = [
        _stage.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=n["NodeID"], soft=False,
            )
        ).remote()
        for n in nodes
    ]
    for n, status in zip(nodes, ray_module.get(futures)):
        log_fn(f"  {n['NodeManagerHostname']}: {status}")
