# Scaling Physical AI & Robotics Systems with Ray

An end-to-end physical-AI workflow on **Ray + Anyscale**, taught as a series of
self-contained, runnable notebooks. You stream robotics data, fine-tune a
Vision-Language-Action (VLA) policy, serve it and evaluate it in simulation,
close the loop by folding sim trajectories back into training, pre-train a
**world model** at scale, and finally **distill** a large model into a small
backbone you could deploy on a robot.

> **Scale up to learn; scale down to deploy.**

The point of the course is the **infrastructure**. The models (PI0.5, V-JEPA,
ResNet/MobileNet) are the workload; the lesson is how a handful of Ray
primitives — **Ray Data**, **Ray Train**, **Ray Serve**, and **Ray remote
tasks** — handle streaming data, distributed training, live serving, and
parallel simulation on one cluster. Swap in your own models and the
orchestration code barely changes.

---

## The lifecycle

```
   ┌─────────┐   ┌──────────────┐   ┌────────────┐   ┌─────────────┐
   │  DATA   │──▶│  FINE-TUNE   │──▶│   SERVE    │──▶│   SIMULATE  │
   │ Ray Data│   │  a VLA       │   │ the policy │   │  & evaluate │
   │ stream  │   │  Ray Train   │   │ Ray Serve  │   │ Ray tasks   │
   └─────────┘   └──────────────┘   └────────────┘   └──────┬──────┘
        ▲                                                    │
        │                  CLOSE THE LOOP                    │
        └───────────  filter by reward, union  ◀────────────┘
                              │
            ┌─────────────────┴───────────────────┐
            ▼                                      ▼
   ┌──────────────────┐                  ┌────────────────────┐
   │  WORLD MODEL     │                  │   DISTILL FOR EDGE │
   │  pre-train at    │  ───────────▶    │  big teacher →     │
   │  scale (V-JEPA)  │   scale down     │  small student     │
   │  Ray Train       │                  │  Ray Train         │
   └──────────────────┘                  └────────────────────┘
```

---

## Course map

| # | Notebook | You learn | Ray primitive |
|---|----------|-----------|---------------|
| 00 | `00_overview.ipynb` | The lifecycle, the through-line, how to navigate | — |
| 01 | `01_robotics_data_pipelines.ipynb` | Stream LeRobot v3 video, partition, preprocess | **Ray Data** |
| 02 | `02_vla_finetuning.ipynb` | Distributed DDP fine-tune of PI0.5 (3.4B) | **Ray Train** |
| 03 | `03_serving_and_sim_eval.ipynb` | Serve the policy + fan out Isaac Lab rollouts + close the loop | **Ray Serve**, **Ray tasks** |
| 04 | `04_world_model_pretraining.ipynb` | Pre-train a V-JEPA world model + online adaptation | **Ray Data**, **Ray Train** |
| 05 | `05_distillation_for_edge.ipynb` | Distill a large teacher into a deployable student | **Ray Data**, **Ray Train** |

**Outline coverage:** robotics data prep (01) · VLA pre-training & fine-tuning
(02 fine-tune, 04 pre-train) · distributed simulation & evaluation (03) ·
scalable inference (03 serving, 05 edge) · world-foundation model pre-training
at scale (04).

The notebooks are designed to be read in order, and they cross-reference each
other. They ship **pre-run** — every notebook has committed outputs, so you can
read the whole story without a cluster, then re-run any of them on your own.

---

## Shared modules

| File | Used by | Role |
|------|---------|------|
| `lerobot_datasource.py` | 01/02/03/04 | Ray Data `Datasource` for LeRobot v3 (streaming parquet + mp4) |
| `util.py` | 02/03 | Model load/freeze, checkpoint I/O, LR schedule, node staging |
| `policy_server.py` | 03 | `@serve.deployment` PI0.5 HTTP policy server |
| `franka_env.py` | 03 | Isaac Lab `Isaac-Lift-Cube-Franka-v0` wrapper |
| `sim_worker.py` | 03 | Standalone subprocess: boots Isaac Lab, queries Serve, saves GIF + trajectory |

---

## Prerequisites

### Cluster

A **4× L4 Anyscale cluster** running the image defined in [`Dockerfile`](./Dockerfile). The head node is CPU-only;
the 4 L4s are worker nodes accessed via Ray. Resource use is phased so the
cluster is never over-subscribed:

| Phase | GPUs |
|-------|------|
| Training (Ray Train DDP) | 4 × L4 |
| Sim eval (1 Serve replica + 2 sim workers) | 3 × L4 |

Ray releases GPUs between phases.

### HuggingFace token

**No HF token is needed anywhere — not at runtime, and not even to build the image.**

All datasets, the PI0.5 model, and the PaliGemma tokenizer are mirrored to a public S3
bucket (`s3://anyscale-public-materials/ray_summit_robotics_2026/`). The notebooks run with
`HF_HUB_OFFLINE=1`, and the Dockerfile bakes the tokenizer from that public bucket
**anonymously** (`s3fs(anon=True)`). Nothing reaches Hugging Face at build time or run
time, so hundreds of attendees can each build the image and run the tutorial with zero
secrets — and HF cannot throttle the event.

(The gated `google/paligemma-3b-pt-224` tokenizer was fetched once by the presenter to seed
the mirror; it is redistributed under the Gemma Terms of Use — see `GEMMA_NOTICE.txt` at the
S3 prefix.)

### Tested configuration

| Component | Version |
|-----------|---------|
| Ray | 2.53.0. & 2.55.0 |
| Python | 3.11 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.4 (`main@b0542fe`, pinned commit, built from source) |
| lerobot | 0.4.3 (`--no-deps`) |
| transformers | `huggingface/transformers@dcddb97` (patched fork, pinned commit) |

**Why the patched transformers fork?** PI0.5's checkpoint stores Gemma
layernorm parameters under a different key layout than mainline `transformers
>= 4.57`, and `PI05Pytorch.__init__` aborts unless `transformers.models.siglip.check`
exists — a symbol only in the fork.

**Why `lerobot --no-deps`?** lerobot's `rerun-sdk` dependency requires
`numpy >= 2`, which breaks Isaac Sim's compiled ABI. Install `--no-deps` and
pin `numpy>=1.26,<2`.

**Why `TORCHDYNAMO_DISABLE=1`?** PI0.5 calls `torch.compile` internally; the
worker nodes have no C compiler, so dynamo falls back to eager cleanly.

### Cluster image (Dockerfile)

The image is defined by [`Dockerfile`](./Dockerfile) in this directory, which is the single
source of truth — build that file as-is.

Two things worth knowing before you build:

- **The NVIDIA graphics-userspace block is not optional.** The container runtime injects only
  *compute* driver libs, so without it Isaac Sim's Vulkan/RTX renderer has no graphics libs to
  load and dies with `ERROR_INCOMPATIBLE_DRIVER` — all-black frames plus a PhysX-GPU init hang.
  The Dockerfile bakes them from the version-matched driver `.run`, so `NV_DRIVER_VERSION` must
  match the host kernel driver reported by `nvidia-smi`.
- **Isaac Lab is pinned to a tag and the transformers fork to a commit**, so independent builds
  resolve to the same code instead of tracking a moving branch.

(No Weights & Biases in the tutorial — metrics are reported through Ray Train. `wandb` is
installed only because lerobot expects it to be importable.)

---

## A note on scope

Both the LIBERO training data and Isaac Lab's `Isaac-Lift-Cube-Franka-v0` use a
**Franka Panda**, so the action and state dimensions line up cleanly. What PI0.5
has *not* seen is this exact setup — Isaac Lab's action/control convention, scene
and coordinate frame, and camera views (we feed one render into both of PI0.5's
camera inputs). So in 02/03 expect **exploratory motion, not task success**: we're
validating the **orchestration loop**, not manipulation skill. Every run here is
at smoke scale (small step counts); the lesson is that the *same code* scales to
production by changing config, not logic.
