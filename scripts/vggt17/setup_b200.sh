#!/usr/bin/env bash
set -euo pipefail
[ $# -eq 1 ] || { echo "Usage: bash scripts/vggt17/setup_b200.sh /absolute/path/to/venv" >&2; exit 2; }
cd "$(dirname "$0")/../.."
prefix=$1
[[ "$prefix" = /* ]] || { echo "Use an absolute environment path." >&2; exit 2; }

bootstrap=${BOOTSTRAP_PYTHON:-/usr/bin/python3}
if ! "$bootstrap" -m uv --version >/dev/null 2>&1; then
    "$bootstrap" -m pip install --user --upgrade uv
fi
# Native uv can select a managed Python without inheriting the bootstrap interpreter.
uv=$("$bootstrap" -c 'from uv import find_uv_bin; print(find_uv_bin())')
if [ ! -x "$prefix/bin/python" ]; then
    "$uv" venv --python 3.10 --managed-python --seed "$prefix"
fi
py="$prefix/bin/python"
"$py" -B - <<'PY'
from pathlib import Path
import shlex
import sys
import sysconfig

include = Path(sysconfig.get_path("include"))
missing = [name for name in ("Python.h", "pyconfig.h") if not (include / name).is_file()]
if missing:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    command = f'uv venv --allow-existing --python "$(uv python find --system --managed-python {version})" {shlex.quote(sys.prefix)}'
    raise SystemExit(
        f"Python development headers missing under {include}: {', '.join(missing)}.\n"
        f"Triton needs these headers even without a CUDA toolkit. Install a managed Python and rebase this environment without deleting packages:\n"
        f"uv python install --no-bin {version}\n{command}"
    )
print("Python development headers:", include)
PY
# Accelerate imports installed DeepSpeed even in plain DDP.
"$uv" pip uninstall --python "$py" deepspeed
"$uv" pip install --python "$py" --upgrade pip 'setuptools<81' wheel packaging ninja
"$uv" pip install --python "$py" --torch-backend cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
"$uv" pip install --python "$py" --no-build-isolation --torch-backend cu128 -r scripts/vggt17/requirements_b200.txt
"$uv" pip install --python "$py" --no-deps --no-build-isolation -e . -e ./vggt
wheel=$("$py" - <<'PY'
import sys
import torch

tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abi{abi}-{tag}-{tag}-linux_x86_64.whl")
PY
)
"$uv" pip install --python "$py" --no-deps "$wheel"
PYTHONPATH="$PWD/lmms-eval:$PWD${PYTHONPATH:+:$PYTHONPATH}" LMMS_EVAL_LAUNCHER=python PYTHONDONTWRITEBYTECODE=1 "$py" -B - <<'PY'
from importlib.util import find_spec

assert find_spec("deepspeed") is None, "This recipe uses CUDA DDP; DeepSpeed must not be installed in this environment"
import torch
from flash_attn import flash_attn_func
from cambrianp.train.train import TrainingArguments
import datasets
import pyarrow
import lmms_eval.__main__
from lmms_eval.models.simple.cambrian_p import CambrianP

assert torch.cuda.device_count() == 2, "This recipe expects both allocated B200 GPUs to be visible"
assert torch.version.cuda == "12.8", f"Expected the CUDA 12.8 build, got {torch.version.cuda}"
assert "sm_100" in torch.cuda.get_arch_list(), "Install the CUDA 12.8 PyTorch wheel, not the old CUDA 12.1 environment"
for device in range(2):
    with torch.cuda.device(device):
        q = torch.randn(1, 128, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        flash_attn_func(q, q, q, causal=True).float().square().mean().backward()
        torch.cuda.synchronize()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        print(device, torch.cuda.get_device_name(device), "BF16 FlashAttention forward/backward passed")
print("Training/evaluation imports passed (CUDA DDP, no DeepSpeed):", torch.__version__, datasets.__version__, pyarrow.__version__)
PY
printf '\nEnvironment ready: %s\n' "$prefix"
