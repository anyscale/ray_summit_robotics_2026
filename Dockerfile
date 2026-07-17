FROM anyscale/ray:2.53.0-slim-py311-cu128
# =============================================================================
# Isaac Sim 5.1.0 + Isaac Lab on Anyscale Ray
# =============================================================================
# Key constraints:
#   - Isaac Sim 5.X requires Python 3.11 (NOT 3.10, NOT 3.12)
#   - Isaac Lab recommends torch 2.7.0 + CUDA 12.8
#   - Headless GPU rendering needs Vulkan + EGL userspace libs
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PYTHONUNBUFFERED=1
# ---------- headless rendering ----------
ENV DISPLAY=""
ENV OMNI_KIT_RENDERING_MODE=headless
# Force EGL over GLX (no X server on Anyscale nodes)
ENV __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
# Vulkan ICD — NVIDIA driver mounts this at runtime
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
USER root
# ---------- system deps: EGL + Vulkan + misc ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    # EGL / OpenGL
    libgl1 \
    libegl1 \
    libegl-mesa0 \
    libgles2 \
    libglvnd0 \
    libglvnd-dev \
    # Vulkan (required for Isaac Sim GPU rendering)
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    # X11 libs (Isaac Sim links against these even headless)
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxkbcommon0 \
    libx11-6 \
    libxext6 \
    libxt6 \
    libglu1-mesa \
    # Misc
    libsm6 \
    libice6 \
    libfontconfig1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Create NVIDIA Vulkan ICD file (in case NVIDIA container runtime doesn't mount it)
RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}' \
    > /etc/vulkan/icd.d/nvidia_icd.json
USER ray
WORKDIR /home/ray
# ---------- PyTorch (pinned to Isaac Lab's recommended version) ----------
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
# ---------- Isaac Sim 5.1.0 (skip extscache to save ~10GB, lazy-loads at runtime) ----------
RUN python -m pip install --no-cache-dir \
    "isaacsim[all]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
# ---------- Isaac Lab from source ----------
RUN git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git /home/ray/IsaacLab
# Install Isaac Lab extensions as editable packages
RUN cd /home/ray/IsaacLab && \
    python -m pip install --no-cache-dir \
    -e source/isaaclab \
    -e source/isaaclab_tasks \
    -e source/isaaclab_rl
# ---------- Training deps ----------
RUN python -m pip install --no-cache-dir \
    pillow \
    wandb \
    gymnasium
# ---------- PYTHONPATH fallback (belt + suspenders with editable installs above) ----------
ENV PYTHONPATH=/home/ray/IsaacLab/source/isaaclab:\
/home/ray/IsaacLab/source/isaaclab_tasks:\
/home/ray/IsaacLab/source/isaaclab_rl:\
${PYTHONPATH}
WORKDIR /home/ray/default

# =============================================================================
# VLA / lerobot overlay (added 2026-05-05 -- verified compatible with Isaac Sim)
# =============================================================================

USER root
# torchcodec needs libav* shared libs at runtime; lerobot decodes mp4 frames
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER ray

# lerobot transitively requires rerun-sdk which requires numpy>=2 -- incompatible
# with Isaac Sim's compiled ABI. Install lerobot --no-deps and provide its
# runtime deps explicitly with numpy<2 preserved.
RUN python -m pip install --no-cache-dir --no-deps lerobot==0.4.3

RUN python -m pip install --no-cache-dir \
    "numpy>=1.26,<2" \
    "datasets>=4.0,<4.2" \
    "accelerate>=1.10,<2" \
    "deepdiff>=7,<9" \
    "diffusers>=0.27.2,<0.36" \
    "draccus==0.10.0" \
    "einops>=0.8,<0.9" \
    "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36" \
    "imageio[ffmpeg]==2.37.0" \
    "jsonlines>=4,<5" \
    "packaging>=24.2,<26" \
    "pyserial>=3.5,<4" \
    "sentencepiece" \
    "termcolor>=2.4,<4" \
    "torchcodec>=0.2.1,<0.6" \
    "av>=15,<16" \
    "wandb>=0.24,<0.25" \
    "s3fs>=2024.1" \
    "fsspec[s3]>=2024.1"

# Patched transformers fork (huggingface/transformers@fix/lerobot_openpi).
# PI0.5's `lerobot/pi05_base` checkpoint stores Gemma layernorm parameters
# under a different key layout than mainline 4.57+, AND PI05Pytorch.__init__
# aborts unless `transformers.models.siglip.check` exists (only in this fork).
# Install with deps so tokenizers gets pulled to the fork's required <0.22.
RUN python -m pip uninstall -y transformers tokenizers || true \
 && python -m pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"

# Sanity check (fails the build if the env is broken)
RUN python -c "import lerobot, transformers, isaaclab; \
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Config; \
PI05Config(); \
print(f'lerobot={lerobot.__version__} transformers={transformers.__version__} OK')"