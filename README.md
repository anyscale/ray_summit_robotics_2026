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

A **4× L4 Anyscale cluster** running the image below. The head node is CPU-only;
the 4 L4s are worker nodes accessed via Ray. Resource use is phased so the
cluster is never over-subscribed:

| Phase | GPUs |
|-------|------|
| Training (Ray Train DDP) | 4 × L4 |
| Sim eval (1 Serve replica + 2 sim workers) | 3 × L4 |

Ray releases GPUs between phases.

### HuggingFace token

PI0.5 (notebooks 02/03) depends on the gated `google/paligemma-3b-pt-224`.
Accept the license at <https://huggingface.co/google/paligemma-3b-pt-224> and:

```bash
export HF_TOKEN=hf_...
```

### Tested configuration

| Component | Version |
|-----------|---------|
| Ray | 2.53.0. & 2.55.0 |
| Python | 3.11 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | latest from source |
| lerobot | 0.4.3 (`--no-deps`) |
| transformers | `huggingface/transformers@fix/lerobot_openpi` (patched fork) |

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

<details>
<summary>Full Anyscale cluster image used by this course</summary>

```dockerfile
FROM anyscale/ray:2.55.0-slim-py311-cu128
# =============================================================================
# Isaac Sim 5.1.0 + Isaac Lab on Anyscale Ray
#   - Isaac Sim 5.X requires Python 3.11 (NOT 3.10, NOT 3.12)
#   - Isaac Lab recommends torch 2.7.0 + CUDA 12.8
#   - Headless GPU rendering needs Vulkan + EGL userspace libs
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=""
ENV OMNI_KIT_RENDERING_MODE=headless
ENV __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    libgl1 libegl1 libegl-mesa0 libgles2 libglvnd0 libglvnd-dev \
    libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers \
    libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libx11-6 \
    libxext6 libxt6 libglu1-mesa libsm6 libice6 libfontconfig1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}' \
    > /etc/vulkan/icd.d/nvidia_icd.json
USER ray
WORKDIR /home/ray
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
    "torch==2.7.0" "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install --no-cache-dir "isaacsim[all]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
RUN git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git /home/ray/IsaacLab
RUN cd /home/ray/IsaacLab && python -m pip install --no-cache-dir \
    -e source/isaaclab -e source/isaaclab_tasks -e source/isaaclab_rl
RUN python -m pip install --no-cache-dir pillow gymnasium
ENV PYTHONPATH=/home/ray/IsaacLab/source/isaaclab:\
/home/ray/IsaacLab/source/isaaclab_tasks:\
/home/ray/IsaacLab/source/isaaclab_rl:${PYTHONPATH}
WORKDIR /home/ray/default

# -------- VLA / lerobot overlay (verified compatible with Isaac Sim) --------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER ray
RUN python -m pip install --no-cache-dir --no-deps lerobot==0.4.3
RUN python -m pip install --no-cache-dir \
    "numpy>=1.26,<2" "datasets>=4.0,<4.2" "accelerate>=1.10,<2" \
    "deepdiff>=7,<9" "diffusers>=0.27.2,<0.36" "draccus==0.10.0" \
    "einops>=0.8,<0.9" "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36" \
    "imageio[ffmpeg]==2.37.0" "jsonlines>=4,<5" "packaging>=24.2,<26" \
    "pyserial>=3.5,<4" "sentencepiece" "termcolor>=2.4,<4" \
    "torchcodec>=0.2.1,<0.6" "av>=15,<16" "s3fs>=2024.1" "fsspec[s3]>=2024.1"
# Patched transformers fork (see "Why the patched fork?" above).
RUN python -m pip uninstall -y transformers tokenizers || true \
 && python -m pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
RUN python -c "import lerobot, transformers, isaaclab; \
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Config; \
PI05Config(); print('env OK')"
```

(No Weights & Biases anywhere — metrics are reported through Ray Train.)
</details>

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
