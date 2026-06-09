# BUILD PLAN — Scaling Physical AI & Robotics Systems with Ray

> **Read this whole document before touching anything.** It is the build brief
> for assembling a standalone, self-contained Ray Summit 2026 training course
> from material that already exists on this workspace. Most of the work is
> *curation, re-sequencing, restyling, and re-tying narrative* — not writing new
> ML logic. The hard ML/infra code is already written and (for the VLA half)
> demo-tested.

---

## 1. Mission

Assemble an instructor-led, take-home notebook course** titled
**"Scaling Physical AI & Robotics Systems with Ray"** into this directory
(`/home/ray/default/ray_summit_robotics_2026/`).

The course teaches the full physical-AI lifecycle on Ray + Anyscale through one
through-line:

> **Scale up to learn; scale down to deploy.**
> Stream robotics data → fine-tune a VLA → serve & evaluate it in simulation →
> close the loop (sim data → training) → pre-train a world model *at scale* →
> distill it down for the edge.

The pedagogical spine is that **one Ray surface** (`TorchTrainer` +
`prepare_model` + `train.report` + `FailureConfig` + `ScalingConfig`, plus the
shared `lerobot_datasource`) carries unchanged across VLA fine-tuning, world-model
pre-training, and distillation — "change the config, not the code."

It maps 1:1 to the official outline bullets:
- Robotics data preparation pipelines
- VLA model pre-training and fine-tuning
- Distributed simulation and evaluation
- Scalable inference for physical AI systems
- World-foundation model pre-training at scale

---

## 2. Hard constraints (do not violate)

1. **Environment: this is an Anyscale workspace with 4× L4 GPUs (24 GB each).**
   All notebooks must run, and produce committed outputs, on 4×L4. Pick configs
   accordingly (see §7 per-notebook configs). The L4 budget is the binding
   constraint — the source JEPA/distillation notebooks were authored for
   2×A10G; re-tune their `num_workers`/batch sizes for 4×L4.
2. **Notebooks are pre-run.** The deliverable is the notebook *with outputs
   committed*. You must actually execute each notebook on this cluster and save
   the executed version. Use smoke-scale configs (small step counts) so runs
   finish in minutes — we are validating plumbing, not optimizing loss.
3. **Standalone & self-contained.** Notebooks may cross-reference *each other
   only*. Strip/rewrite every reference to any external course, prior series,
   or notebooks not in this directory (the source `06_`/`08_` notebooks contain
   such references — see §8).
4. **Notebook-only deliverable.** Markdown cells are the slides. No separate
   slide deck, no instructor run-of-show file.
5. **Unified house style** across all six notebooks (see §6).
6. **No Weights & Biases anywhere.** Strip it if you find it.
7. **HuggingFace token required.** PI0.5 depends on the gated
   `google/paligemma-3b-pt-224`. `export HF_TOKEN=hf_...` before running 01–03.
   Accept the license at https://huggingface.co/google/paligemma-3b-pt-224.
8. **Cluster image.** Use the existing Anyscale cluster image documented in
   `/home/ray/default/vla_sim_closed_loop/README.md` (§"Cluster Image") and
   `/home/ray/default/vla_sim_closed_loop/docker_image_copy.txt`:
   `anyscale/ray:2.53.0-slim-py311-cu128` + Isaac Sim 5.1 + Isaac Lab + lerobot
   0.4.3 (`--no-deps`) + patched transformers fork. Do not change it.

---

## 3. Source material inventory (copy/lift from here)

All paths are absolute on this workspace.

| Source | Use for |
|---|---|
| `/home/ray/default/vla_sim_closed_loop/closed_loop_vla_sim.ipynb` | **Split into 01, 02, 03.** Most complete VLA asset. |
| `/home/ray/default/vla_sim_closed_loop/lerobot_datasource.py` | Shared data engine → copy into course dir (used by 01/02/03/04). |
| `/home/ray/default/vla_sim_closed_loop/util.py` | Training helpers → copy (02/03). |
| `/home/ray/default/vla_sim_closed_loop/policy_server.py` | Ray Serve PI0.5 server → copy (03). **Has stale docstrings — see §9.** |
| `/home/ray/default/vla_sim_closed_loop/franka_env.py` | Isaac Lab wrapper → copy (03). **Has stale docstrings — see §9.** |
| `/home/ray/default/vla_sim_closed_loop/sim_worker.py` | Sim subprocess → copy (03). |
| `/home/ray/default/vla_sim_closed_loop/README.md` + `context.md` | Source prose for 00 framing, cluster image, gotchas, design rationale. |
| `/home/ray/default/08_jepa_world_model_pretraining.ipynb` | → **04** (renumber + re-tie). Already in house style. |
| `/home/ray/default/06_teacher_student_ray_distillation.ipynb` | → **05** (renumber + re-tie + bridge). Already in house style. |

> Note: `/home/ray/default/vla_sim_closed_loop/vla_finetuning_w_sim_eval.ipynb`
> ("Tutorial 1") becomes redundant — it is a subset of what 02+03 cover. Do not
> port it. The job scripts (`vla_finetune.py`, `run_demo.py`, `closed_loop_demo.py`)
> are also out of scope for the course; ignore them.

---

## 4. Target repo layout

```
ray_summit_robotics_2026/
├── BUILD_PLAN.md                      # this file
├── README.md                          # course index + arc + prereqs (NEW)
├── 00_overview.ipynb                  # framing, lifecycle, Ray-primitive map (NEW)
├── 01_robotics_data_pipelines.ipynb   # Ray Data + LeRobot v3 streaming
├── 02_vla_finetuning.ipynb            # Ray Train DDP fine-tune of PI0.5
├── 03_serving_and_sim_eval.ipynb      # Ray Serve policy + Isaac Lab fan-out + close the loop
├── 04_world_model_pretraining.ipynb   # V-JEPA pretrain at scale + online adaptation
├── 05_distillation_for_edge.ipynb     # teacher→student for on-robot deploy
├── lerobot_datasource.py              # shared (01/02/03/04)
├── util.py                            # shared (02/03)
├── policy_server.py                   # 03
├── franka_env.py                      # 03
└── sim_worker.py                      # 03
```

---

## 5. Datasets & models referenced

| Asset | Where | Used by |
|---|---|---|
| `lerobot/libero` (dataset, ~273k frames, streamed via `hf://`) | HuggingFace | 01/02/03 |
| `lerobot/pi05_libero_finetuned` (PI0.5, 3.4B, staged to `/mnt/local_storage`) | HuggingFace | 02/03 |
| `google/paligemma-3b-pt-224` (gated backbone) | HuggingFace | 02/03 (transitively) |
| FMB dataset (~814 MB) | `s3://anyscale-public-robotics-datasets/lerobot/lerobot/fmb` | 04 |
| CIFAR-10 | HuggingFace (`load_dataset` / `from_huggingface`) | 05 |

Shared cluster FS for checkpoints/data: `/mnt/cluster_storage/...` (visible to
all nodes). Per-node model staging: `/mnt/local_storage/...`.

---

## 6. House style (apply to ALL six notebooks)

Match the structure of `08_jepa_world_model_pretraining.ipynb` and
`06_teacher_student_ray_distillation.ipynb`:

- **Top intro block** (markdown): `# Title`, then `## TLDR`, `## Introduction`,
  `## Key concepts used in this notebook`, `## What you will learn`,
  `## Why Ray on Anyscale for <X>?` (comparison table), architecture ASCII
  diagram, and a `## How this scales on Anyscale` table (tutorial vs production).
- **Each code cell preceded by a `## Cell N:` markdown header** with the
  pattern: a short intro, then **What you do**, **What to check**, **Why it
  matters** bullet sections. (06 uses these three sub-headers consistently; 08
  uses lighter prose — converge on the 06 pattern where practical.)
- **`## Conclusion`** cell at the end recapping the Ray primitives used and the
  scaling levers.
- Keep the existing rich ASCII diagrams and LaTeX objective blocks from 04/05;
  add equivalent diagrams to 01/02/03 where they clarify (e.g., the closed-loop
  architecture diagram already in `closed_loop_vla_sim.ipynb` cell 2).
- Honest framing about scope: these are smoke-scale runs validating the
  *infrastructure*, not optimizing task success (the VLA half deliberately has
  a train/eval mismatch — see §9).

---

## 7. Per-notebook build instructions

### 00 — `00_overview.ipynb` (NEW)
- **Purpose:** establish the lifecycle arc, the Ray-primitive map, and how to
  navigate 01→05. This is where all cross-references get their home.
- **Build:** assemble from the narrative prose in
  `vla_sim_closed_loop/README.md` (the "Narrative" + "Why infrastructure is the
  hard part" sections) and the intros of 04/05. Draw the "scale up / scale down"
  arc explicitly. Include a table mapping each module → Ray primitive → outline
  bullet.
- **Mostly markdown.** A tiny `ray.init(address="auto")` + `ray.cluster_resources()`
  cell to confirm the 4×L4 environment is fine.

### 01 — `01_robotics_data_pipelines.ipynb`
- **Source:** `closed_loop_vla_sim.ipynb` cells 9–12 (stage model, peek dataset,
  `LeRobotDatasource`, `rename_columns`/`transpose_images`, `build_libero_dataset`)
  + the temporal-windowing idea from 04's `build_temporal_windows`.
  + `lerobot_datasource.py`.
- **Teach:** LeRobot v3 layout (parquet + mp4), streaming via `read_lerobot`,
  the 5 **partitioning strategies** and their tradeoffs (this is the richest
  scaling content in `lerobot_datasource.py` — surface it), `map`/`map_batches`
  preprocessing on the CPU pool, normalization stats, frame preview.
- **Bullet:** Robotics data preparation pipelines.
- **Config (4×L4):** data prep is CPU-bound; just stream and preview. No GPU
  needed. Keep `.take()` previews small.

### 02 — `02_vla_finetuning.ipynb`
- **Source:** `closed_loop_vla_sim.ipynb` cells 14 (`train_loop_per_worker`),
  16 (`run_training`), config cell, + `util.py`.
- **Teach:** Ray Train `TorchTrainer` + DDP via `prepare_model`,
  `get_dataset_shard`, grad-accum, LR schedule, `FailureConfig`,
  checkpoint I/O (`make_checkpoint`/`load_checkpoint`), the frozen-backbone /
  train-only-action-heads design. Emphasize the "4 → 400 GPUs = one number in
  `ScalingConfig`" point.
- **Bullet:** VLA fine-tuning.
- **Config (4×L4):** `num_workers=4, use_gpu=True`, `batch_size=1`,
  `grad_accum=8`, `MAX_TRAIN_STEPS≈50–200` (smoke). Uses all 4 GPUs.

### 03 — `03_serving_and_sim_eval.ipynb`
- **Source:** `closed_loop_vla_sim.ipynb` cells 18 (`run_sim_eval`), 21
  (`build_mixed_dataset`), the GIF-display cells, + `policy_server.py`,
  `franka_env.py`, `sim_worker.py`.
- **Teach:** Ray Serve deployment (`@serve.deployment(num_gpus=1)`, FastAPI
  ingress, HTTP boundary, why-not-DeploymentHandle), `@ray.remote(num_gpus=1)`
  sim fan-out, why-subprocess-not-actor (Isaac Sim asyncio), then **close the
  loop**: filter trajectories by reward → `ray.data.from_items` → `.union()` →
  retrain (Round 2). The close-the-loop section is the "depth flex" — keep it
  self-contained so it can be skipped live.
- **Bullets:** Distributed simulation & evaluation + Scalable inference (server).
- **Config (4×L4):** **resource phasing matters.** Training phase uses 4 GPUs;
  sim-eval phase uses 1 (Serve) + 2 (sim workers) = 3 GPUs. Ray releases GPUs
  between phases — run phases sequentially within the notebook. Set
  `SIM_WORKERS=2`, `SIM_EPISODES=1–2`, `MAX_SIM_STEPS≈100`. Isaac Sim cold start
  is ~60–90s per worker; budget for it.

### 04 — `04_world_model_pretraining.ipynb`
- **Source:** `08_jepa_world_model_pretraining.ipynb` (port largely as-is).
- **Teach:** V-JEPA latent-space world model — Ray Data temporal windowing,
  ViT-Small dual-encoder (context + EMA target), spacetime tube masking,
  multi-camera + action/state conditioning, Phase 1 distributed pretrain +
  Phase 2 online adaptation (same `TorchTrainer`).
- **Bullets:** World-foundation model pre-training at scale (+ the *pre-train*
  side of the VLA bullet).
- **Re-tie (REQUIRED, §8):** rewrite any framing implying a broader series;
  point infra references back at 01 (shared `lerobot_datasource`) and 02 ("the
  same `TorchTrainer` you used to fine-tune the VLA").
- **Config (4×L4):** source defaults are 2×A10G / 50 episodes / 2 epochs. On
  4×L4 you may raise `num_workers` to 4, but keep episode counts and ViT depth
  small for fast pre-run. Verify A10G→L4 memory (both 24 GB, so batch sizes
  should transfer). Confirm FMB S3 staging works from this workspace.

### 05 — `05_distillation_for_edge.ipynb`
- **Source:** `06_teacher_student_ray_distillation.ipynb` (port as-is — option
  (a): keep the ResNet-50→MobileNetV3 / CIFAR-10 example).
- **Teach:** teacher–student feature distillation, frozen teacher forward at
  scale, Ray Data image preprocessing, Ray Train DDP with cross-worker
  all-reduce metric aggregation + best-checkpoint selection, exporting a small
  deployable backbone.
- **Bullet:** Scalable inference for physical AI (edge).
- **Re-tie (REQUIRED, §8):** rewrite the "Where this notebook fits in the
  series" cell — it currently references BEV perception + old-series notebooks.
  Replace with: *"In 02 you fine-tuned a 3.4B VLA and in 04 you pre-trained a
  world model — both far too large for an on-robot compute budget. This module
  distills a large vision encoder into a small, deployable student backbone."*
  This narration is what makes the edge-deployment framing land despite the
  CIFAR/ResNet example being generic.
- **Config (4×L4):** small — CIFAR-10, 2–4 workers, few epochs. Fast pre-run.

---

## 8. Cross-reference / self-containment rules

- The course is standalone. **Search every ported notebook for references to:**
  BEV perception, "earlier in the series", "the previous notebook" (when it
  means an old-series one), specific old filenames/numbers, or any course other
  than this one. Rewrite each to reference only `00`–`05` in this directory, or
  delete.
- Add *helpful* forward/back references between our notebooks (e.g., 04 → "same
  `TorchTrainer` as 02"; 05 → "the models from 02 and 04 are too big for edge").
- `00` must establish the arc so these references resolve.

---

## 9. Known gotchas & required fixes

1. **Stale SO-101 docstrings (MUST FIX).** The active code in
   `franka_env.py` and `policy_server.py` was migrated from an earlier SO-101
   design to LIBERO, but several docstrings/comments still describe the old
   SO-101 world and now contradict the code. Fix before shipping (take-home
   readers will be misled):
   - `franka_env.py`: header + `_format_obs`/`_flatten_action` docstrings say
     "SO-101 6-DOF", `base_0_rgb`/`left_wrist_0_rgb`, `(n_steps, 6)`, state
     `(6,)`. Actual constants: `PI05_STATE_DIM=8`, `FRANKA_ARM_FROM_PI05=7`,
     camera keys `observation.images.image`/`image2`, `PI05_IMAGE_HW=(256,256)`.
     The `_resize_chw_uint8` comment says `(224,224,3)` but the constant is 256.
   - `policy_server.py`: module docstring describes the SO-101 request format
     (`base_0_rgb`, state `(6,)`, `(50,6)` output) though the live defaults are
     the LIBERO finetuned model with `image`/`image2` and state `(8,)`.
   - Update docstrings to match the LIBERO reality. Do **not** change the live
     constants/keys — they are correct.
2. **The train/eval mismatch is deliberate.** PI0.5 (fine-tuned on LIBERO's
   Panda) drives an Isaac Lab Franka; expect "exploratory motion," not task
   success. The point is orchestration, not lifting the cube. Keep this framing
   honest in 03's markdown.
3. **Cross-file invariants (do not break when splitting):**
   - Checkpoint pickle layout `{model, optim, scaler, epoch, step, stats,
     base_model_repo, camera_rename}` is written by `util.make_checkpoint` and
     read by `policy_server` + `util.load_checkpoint`. Keep both in sync.
   - Image keys `observation.images.image`/`image2` + 8-D state flow
     `franka_env._format_obs` → HTTP → `policy_server._build_batch` →
     preprocessor. Renaming in one place breaks the chain.
   - The `make_att_2d_masks` patch must run in BOTH the training workers and the
     Serve replica (it's idempotent).
4. **Ray init env vars (02/03):** `TORCHDYNAMO_DISABLE=1` (no C compiler on
   workers), `NCCL_P2P_DISABLE/SHM_DISABLE/IB_DISABLE=1` (containerized GPU
   safety), Vulkan/EULA vars for Isaac Sim headless, `HF_TOKEN`,
   `HF_HUB_ENABLE_HF_TRANSFER=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   `working_dir="."` so `@ray.remote` sim workers can import `franka_env`/`sim_worker`.
5. **Isaac Sim teardown can hang.** `sim_worker.py` writes all outputs before
   `env.close()` and force-exits via SIGALRM after 10s. Preserve this.

---

## 10. Build order (checklist)

1. [ ] `cd` into `ray_summit_robotics_2026/`. Copy the 5 shared `.py` modules
       from `vla_sim_closed_loop/`. Fix the stale docstrings (§9.1). Verify
       imports resolve within the dir (`python -c "import lerobot_datasource,
       util, policy_server, franka_env, sim_worker"` — note some need the Isaac
       env).
2. [ ] Write `README.md` + `00_overview.ipynb` — lock the arc, numbering, and
       cross-reference targets.
3. [ ] Build `01` (data) — lift, restyle, run, commit outputs.
4. [ ] Build `02` (VLA fine-tune) — lift, restyle, run on 4×L4 smoke config,
       commit outputs + verify checkpoint lands in `/mnt/cluster_storage`.
5. [ ] Build `03` (serve + sim + loop) — lift, restyle, run (sequential phases),
       commit outputs incl. GIFs.
6. [ ] Port `04` (world model) — renumber, re-tie (§8), re-tune for 4×L4, run,
       commit outputs.
7. [ ] Port `05` (distillation) — renumber, re-tie + bridge (§8), run, commit
       outputs.
8. [ ] Final pass: grep for external references (§8); confirm house-style
       consistency (§6); confirm all six notebooks have committed outputs;
       confirm no W&B.

---

## 11. Time budget (context — informs core-vs-depth, not a hard script)

Instructor walks pre-run notebooks for 2.5h (150 min), deep-diving ~3 modules
and touring the rest. Design each section so skipping reads gracefully.

| Beat | Min | Spine |
|---|---|---|
| Framing / lifecycle (00) | 12 | narrate |
| Data pipelines (01) | 15 | core |
| VLA fine-tuning (02) | 20 | core |
| Serve + sim eval + close loop (03) | 28 | core (close-loop = depth flex) |
| World model at scale (04) | 30 | core |
| Distill for edge (05) | 25 | core |
| Wrap / scaling knobs / Q&A | 20 | — |

---

## 12. Decisions locked (do not relitigate)

- Distillation notebook kept as-is (ResNet→MobileNet / CIFAR); tied in by
  narration only — **option (a)**.
- Course is **standalone & self-contained**; notebooks reference each other only.
- **House style** applied to all six notebooks.
- **Notebook-only** deliverable (no slides, no run-of-show).
- Standalone numbering `00`–`05`.
- Environment: **4×L4 Anyscale workspace**; all notebooks pre-run there.

---

## 13. Out of scope

- `vla_finetuning_w_sim_eval.ipynb` (Tutorial 1) and the job scripts
  (`vla_finetune.py`, `run_demo.py`, `closed_loop_demo.py`) — not ported.
- Re-targeting distillation to the V-JEPA/PI0.5 encoder (option (b)) — a future
  tightening, not part of this build.
- Real edge-runtime export (ONNX/TensorRT/on-device) — out of scope for 05.
- Changing the cluster image.
