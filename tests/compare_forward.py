"""
端到端数值对比脚本。

用法：
    python tests/compare_forward.py

逻辑：
  1. 以子进程方式分别运行 _run_orig.py 和 _run_ebgs.py
  2. 两个脚本使用完全相同的随机种子 + 相同的合成输入
  3. 各自保存中间张量（xt, label, pred, loss）到 npz 文件
  4. 主进程加载两份结果并逐一比较
"""

import subprocess
import sys
import os
import tempfile
import numpy as np
from pathlib import Path

PYTHON  = sys.executable
HERE    = Path(__file__).parent
REPO    = HERE.parent.parent                         # DESI2Euclid/
LOG_DIR = REPO / "logs" / "2026-06-01T20-48-48_generation-astroIR_SBn"
CKPT    = LOG_DIR / "checkpoints" / "last.ckpt"
CFG     = LOG_DIR / "configs" / "2026-06-01T20-48-48-project.yaml"
EBGS    = HERE.parent                                # EBGS/

SEED = 2026

# ------------------------------------------------------------------ #
# Worker script: original log snapshot
# ------------------------------------------------------------------ #

ORIG_SCRIPT = r"""
import sys, os, numpy as np, torch
sys.path.insert(0, "{log_dir}")   # makes `sgm` resolve to log snapshot
sys.path.insert(1, "{log_dir}/sgm/modules/autoencoding/lpips")  # NLayerDiscriminator

import sgm.models.astreIR_SBn as M
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

torch.manual_seed({seed})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = OmegaConf.load("{cfg}")
mc  = cfg.model
model = instantiate_from_config(
    {{"target": mc.target,
      "params": OmegaConf.to_container(mc.params, resolve=True)}}
)
ckpt  = torch.load("{ckpt}", map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
model.load_state_dict(state, strict=False)
model.to(device).eval()

B, H, W = 2, 64, 64
torch.manual_seed({seed})
x0   = torch.randn(B, 1, H, W, device=device)
x1   = torch.randn(B, 1, H, W, device=device)
cond = torch.randn(B, 1, H, W, device=device)
step = torch.tensor([100, 250], device=device)

torch.manual_seed({seed} + 1)
xt    = model.diffusion.q_sample(step, x0, x1, ot_ode=False, sqrt_w=None)
label = model.compute_label(step, x0, xt, sqrt_w=None)
with torch.no_grad():
    pred = model.run_network(xt, step, cond, log_w=None)
# masked_mse(pred, label, pixel_mask=None, x0=x0) with sb_bright_weight=0 => plain MSE
loss  = model.masked_mse(pred, label, pixel_mask=None, x0=x0)

np.savez("{out}",
    xt    = xt.cpu().numpy(),
    label = label.cpu().numpy(),
    pred  = pred.cpu().numpy(),
    loss  = np.array([loss.item()]),
)
print(f"ORIG loss = {{loss.item():.8f}}")
"""

# ------------------------------------------------------------------ #
# Worker script: EBGS simplified code
# ------------------------------------------------------------------ #

EBGS_SCRIPT = r"""
import sys, os, numpy as np, torch
sys.path.insert(0, "{ebgs}")

import sgm.models.astreIR_SBn as M
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

torch.manual_seed({seed})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = OmegaConf.load("{cfg}")
mc  = cfg.model
model = instantiate_from_config(
    {{"target": mc.target,
      "params": OmegaConf.to_container(mc.params, resolve=True)}}
)
ckpt  = torch.load("{ckpt}", map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
model.load_state_dict(state, strict=False)
model.to(device).eval()

B, H, W = 2, 64, 64
torch.manual_seed({seed})
x0   = torch.randn(B, 1, H, W, device=device)
x1   = torch.randn(B, 1, H, W, device=device)
cond = torch.randn(B, 1, H, W, device=device)
step = torch.tensor([100, 250], device=device)

torch.manual_seed({seed} + 1)
xt    = model.diffusion.q_sample(step, x0, x1, ot_ode=False)
label = model.compute_label(step, x0, xt)
with torch.no_grad():
    pred = model.run_network(xt, step, cond)
loss  = model.sb_loss(pred, label)

np.savez("{out}",
    xt    = xt.cpu().numpy(),
    label = label.cpu().numpy(),
    pred  = pred.cpu().numpy(),
    loss  = np.array([loss.item()]),
)
print(f"EBGS loss = {{loss.item():.8f}}")
"""


def run_worker(script_tpl, out_path, label):
    script = script_tpl.format(
        log_dir = str(LOG_DIR),
        ebgs    = str(EBGS),
        cfg     = str(CFG),
        ckpt    = str(CKPT),
        out     = str(out_path),
        seed    = SEED,
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(script)
    tmp.close()
    try:
        result = subprocess.run(
            [PYTHON, tmp.name],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"\n=== {label} STDERR ===\n{result.stderr[-3000:]}")
            raise RuntimeError(f"{label} worker failed (rc={result.returncode})")
        print(result.stdout.strip())
    finally:
        os.unlink(tmp.name)


def compare(orig_npz, ebgs_npz):
    o = np.load(orig_npz)
    e = np.load(ebgs_npz)

    keys = ["xt", "label", "pred", "loss"]
    all_ok = True
    print("\n{:=<60}".format(""))
    print(f"{'Tensor':<10} {'max|diff|':>15} {'rel_err':>15} {'OK?':>6}")
    print("{:-<60}".format(""))
    for k in keys:
        diff    = np.abs(o[k] - e[k])
        max_abs = diff.max()
        rel_err = max_abs / (np.abs(o[k]).max() + 1e-12)
        ok      = max_abs < 1e-4
        all_ok  = all_ok and ok
        flag    = "✓" if ok else "✗ FAIL"
        print(f"{k:<10} {max_abs:>15.2e} {rel_err:>15.2e} {flag:>6}")
    print("{:=<60}".format(""))

    if all_ok:
        print("\n✓ 所有张量数值一致（atol < 1e-4），EBGS 与原始训练完全等价。")
    else:
        print("\n✗ 存在数值差异，请检查上方标记为 FAIL 的张量。")
    return all_ok


def main():
    if not CKPT.exists():
        print(f"Checkpoint not found: {CKPT}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_out = Path(tmpdir) / "orig.npz"
        ebgs_out = Path(tmpdir) / "ebgs.npz"

        print("Running original log snapshot...")
        run_worker(ORIG_SCRIPT, orig_out, "ORIG")

        print("Running EBGS simplified code...")
        run_worker(EBGS_SCRIPT, ebgs_out, "EBGS")

        ok = compare(orig_out, ebgs_out)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
