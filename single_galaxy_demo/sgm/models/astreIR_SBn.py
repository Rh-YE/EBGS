# ---------------------------------------------------------------
# astreIR_SBn.py — 多模态 DESI → Euclid 生成模型
#
# 支持两种工作模式：
#   [MODE-SB]  use_i2sb=True  ← 异方差薛定谔桥 (I2SB)，基线参考 NVlabs/I2SB
#   [MODE-DET] use_i2sb=False ← 传统 UNet 确定性回归（无扩散过程）
#
# 可消融的扩展功能（两种模式均支持）：
#   [EXT-1] 像素拉伸 (pixel_transform)        — 数据端动态范围压缩
#   [EXT-2] 异方差桥动力学 (heteroscedastic)   — 逐像素噪声缩放 w(i)（仅 SB 模式）
#   [EXT-3] PatchGAN 对抗损失 (adversarial)    — 逐 patch 感受野对抗训练
#
# ============================================================
# 数据流（训练，SB 模式）：
# ============================================================
#
#   batch → get_input() → euclid, desi, err (变换域)
#     → build_x1/build_cond
#     → [EXT-2] _build_hetero() → sqrt_w, log_w
#     → diffusion.q_sample() → xt
#     → compute_label() → label
#     → run_network(xt, step, cond, [log_w]) → pred
#     → sb_loss(pred, label, [mask])  +  [EXT-3] adv_loss
#
# 数据流（训练，传统 UNet 模式）：
# ============================================================
#
#   batch → get_input() → euclid (x0), desi (x1), err
#     → run_network_det(desi) → pred_x0
#     → det_loss(pred_x0, euclid, [euclid_err], loss_type)  +  [EXT-3] adv_loss
#
#   loss_type 选项：
#     "l2"   — MSE（等价于原始 masked_mse）
#     "l1"   — MAE（对离群像素更鲁棒）
#     "chi2" — 约化 χ²（需要 euclid_err，天文意义最强）
#
# ---------------------------------------------------------------

import logging
import numpy as np
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from tqdm import tqdm
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import ListConfig, OmegaConf

from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from ..modules.ema import LitEma
from ..util import default, get_obj_from_str, instantiate_from_config
from ..transforms.pixel_stretch import BasePixelTransform, IdentityTransform
from ..modules.autoencoding.lpips.model.model import NLayerDiscriminator, weights_init

logpy = logging.getLogger(__name__)


# ============================================================
# §0  常量与工具函数（I2SB 扩散相关）
# ============================================================

def _compute_gaussian_product_coef(
    sigma1: np.ndarray,
    sigma2: np.ndarray,
):
    """
    两个高斯分布的乘积系数（I2SB 公式 10）。

    给定 p1 = N(x_t | x_0, σ₁²) 和 p2 = N(x_t | x_1, σ₂²)，
    计算 p1·p2 = N(x_t | coef1·x₀ + coef2·x₁, var)。
    """
    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom
    return coef1, coef2, var


def _unsqueeze_xdim(z: torch.Tensor, xdim) -> torch.Tensor:
    """将 (B,) 张量广播为 (B, 1, 1, ...) 以便逐像素运算。"""
    bc = (...,) + (None,) * len(xdim)
    return z[bc]


def make_sb_betas(n_timestep: int = 1000, linear_end: float = 2e-2) -> np.ndarray:
    """
    构建 I2SB 使用的对称 beta 调度。

    ← 等价于 NVlabs/I2SB 的 make_beta_schedule
    """
    linear_start = 1e-10
    assert linear_end >= linear_start, (
        f"linear_end={linear_end:.2e} < linear_start={linear_start:.2e}！"
        f"beta_max 必须 >= {linear_start * n_timestep:.4f}（= linear_start × interval）。"
    )
    betas = (
        torch.linspace(
            linear_start ** 0.5, linear_end ** 0.5,
            n_timestep, dtype=torch.float64,
        ) ** 2
    ).numpy()
    half = n_timestep // 2
    betas = np.concatenate([betas[:half], np.flip(betas[:half])])
    return betas


def space_indices(num_steps: int, count: int) -> List[int]:
    """在 [0, num_steps-1] 中均匀选取 count 个索引。"""
    assert count <= num_steps
    frac = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur, taken = 0.0, []
    for _ in range(count):
        taken.append(round(cur))
        cur += frac
    return taken


# ============================================================
# §1  薛定谔桥扩散过程（仅 SB 模式使用）
# ============================================================

class SBDiffusion:
    """
    I2SB 薛定谔桥扩散过程，支持可选的逐像素异方差缩放。
    仅在 use_i2sb=True 时被实例化。
    """

    def __init__(self, betas: np.ndarray, device: torch.device):
        self.device = device
        self.num_timesteps = len(betas)

        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = _compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        to_t = partial(torch.tensor, dtype=torch.float32)
        self.betas   = to_t(betas).to(device)
        self.std_fwd = to_t(std_fwd).to(device)
        self.std_bwd = to_t(std_bwd).to(device)
        self.std_sb  = to_t(std_sb).to(device)
        self.mu_x0   = to_t(mu_x0).to(device)
        self.mu_x1   = to_t(mu_x1).to(device)

    def get_std_fwd(self, step: torch.Tensor, xdim=None) -> torch.Tensor:
        s = self.std_fwd[step]
        return s if xdim is None else _unsqueeze_xdim(s, xdim)

    def q_sample(
        self,
        step: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        ot_ode: bool = False,
        sqrt_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x0.shape == x1.shape
        _, *xdim = x0.shape
        mu0  = _unsqueeze_xdim(self.mu_x0[step], xdim)
        mu1  = _unsqueeze_xdim(self.mu_x1[step], xdim)
        s_sb = _unsqueeze_xdim(self.std_sb[step], xdim)
        xt = mu0 * x0 + mu1 * x1
        if not ot_ode:
            noise = torch.randn_like(xt)
            if sqrt_w is not None:
                noise = noise * sqrt_w
            xt = xt + s_sb * noise
        return xt.detach()

    def p_posterior(
        self,
        nprev: int,
        n: int,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode: bool = False,
        sqrt_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n ** 2 - std_nprev ** 2).sqrt()
        mu_x0, mu_xn, var = _compute_gaussian_product_coef(std_nprev, std_delta)
        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            noise = torch.randn_like(xt_prev)
            if sqrt_w is not None:
                noise = noise * sqrt_w
            xt_prev = xt_prev + var.sqrt() * noise
        return xt_prev

    def ddpm_sampling(
        self,
        steps: List[int],
        pred_x0_fn,
        x1: torch.Tensor,
        ot_ode: bool = False,
        log_steps: Optional[List[int]] = None,
        verbose: bool = True,
        sqrt_w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xt = x1.detach().to(self.device)
        xs, pred_x0s = [], []
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0
        rev = steps[::-1]
        pairs = list(zip(rev[1:], rev[:-1]))
        if verbose:
            pairs = tqdm(pairs, desc="SB DDPM 采样", total=len(pairs))
        for prev_step, step in pairs:
            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, sqrt_w=sqrt_w)
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
        stack = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack(xs), stack(pred_x0s)

    # ============================================================
    # [SPEEDUP] 共享前缀分叉采样
    # ------------------------------------------------------------
    # 设计：3 条采样轨迹共享前 K 步(确定性主导阶段),
    #       从第 K 步开始沿 batch 维度复制为 num_samples 份,
    #       各自独立加噪走完后 N-K 步。
    #
    # 用途：在保留逐像素 std 估计的前提下,把 num_samples 次完整
    #       反向轨迹的 forward 总次数从 N·num_samples·B 降到
    #       K·B + (N-K)·num_samples·B。例如 N=50, K=35, ns=3:
    #         独立: 3 × 50 × B = 150·B
    #         分叉: 35·B + 15·3B = 80·B  (省 47%)
    #
    # 输入约定：
    #   x1, cond, sqrt_w, log_w 形状均为 (B, ...) 单份;
    #   pred_x0_fn 是个闭包,内部会调用 run_network 时使用
    #   *已经在外层 broadcast 过的* cond/log_w(见 sample_forked)。
    #
    # 返回：
    #   final_x0  : (num_samples, B, 1, H, W) 最终 x0 估计(用于 mean/std)
    #   bridge_xs : (B, log_count, 1, H, W) 仅来自分叉前 + 第一条轨迹的
    #               bridge 可视化(image_logger 兼容);num_samples>1 时
    #               其它分支不参与 bridge_xs(节省 CPU 拷贝)。
    # ============================================================
    def ddpm_sampling_forked(
        self,
        steps: List[int],
        pred_x0_fn_shared,
        pred_x0_fn_forked,
        x1: torch.Tensor,
        num_samples: int,
        split_step: int,
        ot_ode: bool = False,
        log_steps: Optional[List[int]] = None,
        verbose: bool = True,
        sqrt_w_shared: Optional[torch.Tensor] = None,
        sqrt_w_forked: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数
        ----
        pred_x0_fn_shared : (xt[B,...], step_int) -> pred_x0[B,...]
            分叉前的预测函数,batch 维度=B
        pred_x0_fn_forked : (xt[ns*B,...], step_int) -> pred_x0[ns*B,...]
            分叉后的预测函数,batch 维度=ns*B
        split_step : int
            已走过的反向步数到达此值时分叉。0=全程独立,len(steps)-1=全程共享。
            注意:必须 1 <= split_step <= len(pairs)-1,否则等价于普通 sampling。
        """
        B = x1.shape[0]
        device = self.device
        xt = x1.detach().to(device)
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        rev = steps[::-1]
        pairs = list(zip(rev[1:], rev[:-1]))  # 反向相邻 step 对
        n_pairs = len(pairs)

        # 边界裁剪:确保 split_step 合法
        split_step = max(0, min(n_pairs, int(split_step)))

        # bridge 可视化只在前缀阶段记录;后缀只取第 0 条 batch 的轨迹
        xs_bridge = []
        # 用 set 加速 in 判断
        log_steps_set = set(log_steps)

        # ---------- 阶段 1:共享前缀(batch=B) ----------
        iter1 = pairs[:split_step]
        if verbose and len(iter1) > 0:
            iter1 = tqdm(iter1, desc=f"SB 共享前缀 (B={B})", total=len(iter1), leave=False)
        for prev_step, step in iter1:
            pred_x0 = pred_x0_fn_shared(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0,
                                  ot_ode=ot_ode, sqrt_w=sqrt_w_shared)
            if prev_step in log_steps_set:
                xs_bridge.append(xt.detach().cpu())

        # ---------- 分叉点:沿 batch 维度复制 num_samples 份 ----------
        # 复制不加额外噪声,因为下一步 p_posterior 内部已加噪;
        # 这里只是把 batch 扩成 ns*B,使后续 forward 一次跑完所有分支
        if num_samples > 1 and split_step < n_pairs:
            # (B, ...) -> (num_samples, B, ...) -> (num_samples*B, ...)
            xt = xt.unsqueeze(0).expand(num_samples, *xt.shape).contiguous()
            xt = xt.reshape(num_samples * B, *xt.shape[2:])
            sqrt_w_curr = sqrt_w_forked
        else:
            sqrt_w_curr = sqrt_w_shared

        # ---------- 阶段 2:分叉后缀(batch=ns*B) ----------
        iter2 = pairs[split_step:]
        if verbose and len(iter2) > 0:
            iter2 = tqdm(iter2, desc=f"SB 分叉后缀 (B={xt.shape[0]})",
                         total=len(iter2), leave=False)
        for prev_step, step in iter2:
            pred_x0 = pred_x0_fn_forked(xt, step) if (num_samples > 1 and split_step < n_pairs) \
                      else pred_x0_fn_shared(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0,
                                  ot_ode=ot_ode, sqrt_w=sqrt_w_curr)
            # bridge 可视化:后缀阶段只取 num_samples 中的第 0 条
            if prev_step in log_steps_set and num_samples > 1 and split_step < n_pairs:
                xs_bridge.append(xt[:B].detach().cpu())
            elif prev_step in log_steps_set:
                xs_bridge.append(xt.detach().cpu())

        # ---------- 整理输出 ----------
        # 最终 x0:reshape 回 (num_samples, B, C, H, W)
        if num_samples > 1 and split_step < n_pairs:
            final_x0 = xt.reshape(num_samples, B, *xt.shape[1:])
        else:
            # 全程共享(split_step >= n_pairs)或 num_samples=1
            final_x0 = xt.unsqueeze(0).expand(num_samples, *xt.shape).contiguous() \
                       if num_samples > 1 else xt.unsqueeze(0)

        # bridge_xs 形状对齐原版:(B, log_count, C, H, W)
        if len(xs_bridge) > 0:
            bridge_xs = torch.flip(torch.stack(xs_bridge, dim=1), dims=(1,))
        else:
            bridge_xs = xt[:B].detach().cpu().unsqueeze(1)

        return final_x0, bridge_xs


# ============================================================
# §2  主 Lightning 模块
# ============================================================

class MultiModalSBDiffusion(pl.LightningModule):
    """
    多模态 DESI → Euclid 生成模型。

    通过 use_i2sb 开关在两种模式间切换：
      - use_i2sb=True  : 薛定谔桥扩散模型（原始功能，含 EXT-1/2/3）
      - use_i2sb=False : 传统 UNet 确定性回归，损失函数由 det_loss_type 指定

    所有扩展设为 False + IdentityTransform = 原始 I2SB 基线。
    use_i2sb=False 时 EXT-2（异方差桥）自动失效（不涉及扩散过程）。
    """

    def __init__(
        self,
        # ===== 网络 =====
        network_config: Dict,

        # ===== [NEW] 模式开关 =====
        use_i2sb: bool = True,                   # True=I2SB桥，False=传统UNet直接回归
        det_loss_type: str = "l2",               # 传统模式损失: "l1" | "l2" | "chi2"
        det_loss_reduction: str = "mean",        # "mean" | "sum"

        # ===== 原始 I2SB 参数 (cf. NVlabs/I2SB) =====
        interval: int = 1000,
        beta_max: float = 0.3,
        ot_ode: bool = False,
        clip_denoise: bool = False,
        nfe: int = 100,
        log_count: int = 10,
        x1_mode: str = "desi_mean",
        desi_bands: int = 3,

        # ===== [EXT-1] 像素拉伸 =====
        pixel_transform_config: Union[None, Dict, ListConfig, OmegaConf] = None,

        # ===== [EXT-2] 异方差桥动力学（仅 use_i2sb=True 时生效）=====
        heteroscedastic: bool = False,
        hetero_cond_channel: bool = False,
        snr_eps: float = 1e-3,
        w_clamp_min: float = 0.1,
        w_clamp_max: float = 10.0,

        # ===== [EXT-3] PatchGAN 对抗损失 =====
        adversarial: bool = False,
        disc_in_channels: int = 1,
        disc_ndf: int = 64,
        disc_n_layers: int = 3,
        disc_use_actnorm: bool = False,
        adv_weight: float = 0.1,
        disc_loss_type: str = "hinge",
        adv_start_step: int = 0,

        # ===== 优化器 / 调度器 =====
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,

        # ===== EMA =====
        use_ema: bool = True,
        ema_decay: float = 0.9999,

        # ===== 检查点 =====
        ckpt_path: Union[None, str] = None,

        # ===== 输入键名 =====
        input_key_euclid: str = "euclid_img",
        input_key_desi: str = "desi_img",
        input_key_desi_error: str = "desi_error",
        input_key_euclid_error: str = "euclid_error",
        input_key_pixel_mask: str = "pixel_mask",

        # ===== [LOSS-ROBUST] 鲁棒 SB 扩散损失 =====
        # 针对亮源 ε 残差重尾的修正：
        #   sb_loss_type:
        #     "mse"    — 原始 I2SB MSE（默认）
        #     "huber"  — Huber 损失，δ 由 sb_huber_delta 控制
        #   sb_bright_weight: 若 >0，对像素加权 1/(|x0|+sb_bright_weight)
        #     这把亮核处的梯度按 1/√|x0| 等价地压低，避免 outlier 主导。
        sb_loss_type: str = "mse",
        sb_huber_delta: float = 1.0,
        sb_bright_weight: float = 0.0,

        # ===== [SB-X0] x₀ 预测模式 + χ² 加权损失 =====
        # sb_pred_type:
        #   "epsilon" — 原始模式，网络预测 score/ε（默认，与 NVlabs/I2SB 一致）
        #   "x0"      — 直接预测 x₀，label=x0，compute_pred_x0 直接返回 net_out
        # sb_chi2_loss:
        #   若 True，SB 扩散损失用 euclid_err 做逐像素 σ² 加权（χ² 形式）；
        #   仅在 euclid_err 不为 None 时生效，否则自动退化为普通 MSE/Huber。
        sb_pred_type: str = "epsilon",
        sb_chi2_loss: bool = False,

        **kwargs,
    ):
        super().__init__()

        # ---- 模式 ----
        self.use_i2sb        = use_i2sb
        self.det_loss_type   = det_loss_type.lower()
        self.det_loss_reduction = det_loss_reduction

        # ---- [LOSS-ROBUST] SB 扩散损失参数 ----
        self.sb_loss_type     = sb_loss_type.lower()
        self.sb_huber_delta   = float(sb_huber_delta)
        self.sb_bright_weight = float(sb_bright_weight)
        assert self.sb_loss_type in ("mse", "huber"), (
            f"sb_loss_type='{sb_loss_type}' 未知，请选择 'mse' | 'huber'"
        )

        # ---- [SB-X0] x₀ 预测模式 + χ² 加权损失 ----
        self.sb_pred_type = sb_pred_type.lower()
        self.sb_chi2_loss = bool(sb_chi2_loss)
        assert self.sb_pred_type in ("epsilon", "x0"), (
            f"sb_pred_type='{sb_pred_type}' 未知，请选择 'epsilon' | 'x0'"
        )

        assert self.det_loss_type in ("l1", "l2", "chi2"), (
            f"det_loss_type='{det_loss_type}' 未知，请选择 'l1' | 'l2' | 'chi2'"
        )

        logpy.info(
            f"[模式] use_i2sb={use_i2sb}, "
            + (f"det_loss_type={det_loss_type}" if not use_i2sb
               else f"I2SB 桥模式, sb_pred_type={sb_pred_type}, sb_chi2_loss={sb_chi2_loss}")
        )

        # ---- 输入键名 ----
        self.input_key_euclid       = input_key_euclid
        self.input_key_desi         = input_key_desi
        self.input_key_desi_error   = input_key_desi_error
        self.input_key_euclid_error = input_key_euclid_error
        self.input_key_pixel_mask   = input_key_pixel_mask

        # ---- 原始 I2SB 参数 ----
        self.interval     = interval
        self._beta_max    = beta_max
        self.ot_ode       = ot_ode
        self.clip_denoise = clip_denoise
        self.nfe          = nfe
        self.log_count    = log_count
        self.x1_mode      = x1_mode
        self.desi_bands   = desi_bands

        # ---- [EXT-1] 像素拉伸 ----
        if pixel_transform_config is not None:
            self.pixel_transform: BasePixelTransform = instantiate_from_config(
                pixel_transform_config
            )
        else:
            self.pixel_transform = IdentityTransform()
        logpy.info(f"[EXT-1] 像素拉伸: {self.pixel_transform.__class__.__name__}")

        # ---- [EXT-2] 异方差桥（仅 SB 模式）----
        self.heteroscedastic     = heteroscedastic and use_i2sb
        self.hetero_cond_channel = hetero_cond_channel and use_i2sb
        self.snr_eps             = snr_eps
        self.w_clamp_min         = w_clamp_min
        self.w_clamp_max         = w_clamp_max
        if heteroscedastic and not use_i2sb:
            logpy.warning("[EXT-2] heteroscedastic=True 在 use_i2sb=False 时自动忽略")

        # ---- [EXT-3] PatchGAN 对抗损失 ----
        self.adversarial     = adversarial
        self.adv_weight      = adv_weight
        self.disc_loss_type  = disc_loss_type
        self.adv_start_step  = adv_start_step

        if adversarial:
            self.discriminator = NLayerDiscriminator(
                input_nc    = disc_in_channels,
                ndf         = disc_ndf,
                n_layers    = disc_n_layers,
                use_actnorm = disc_use_actnorm,
            ).apply(weights_init)
            logpy.info(
                f"[EXT-3] PatchGAN 判别器已启用: "
                f"in_ch={disc_in_channels}, ndf={disc_ndf}, "
                f"n_layers={disc_n_layers}, loss={disc_loss_type}, "
                f"adv_weight={adv_weight}, adv_start_step={adv_start_step}"
            )
            self.automatic_optimization = False
        else:
            self.discriminator = None

        logpy.info(
            f"[消融开关] "
            f"MODE={'I2SB' if use_i2sb else 'DET'}, "
            f"EXT-1={self.pixel_transform.__class__.__name__}, "
            f"EXT-2={self.heteroscedastic}, "
            f"EXT-3={adversarial}"
        )

        # ---- 优化器 ----
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        self.scheduler_config = scheduler_config

        # ---- 动态 in_channels ----
        # SB 模式：xt(1) + cond(desi_bands-1) + [log_w(1)]
        # DET 模式：desi(desi_bands)  — 直接输入全波段 DESI 图
        if use_i2sb:
            if x1_mode == "rz":
                cond_ch = 1  # build_cond 实际只返回 desi[:, 2].unsqueeze(1)
            else:  # desi_mean
                cond_ch = desi_bands - 1
            _in_ch = 1 + cond_ch + int(self.hetero_cond_channel)
            # rz: 1(xt) + 1(cond) + 0/1(log_w) = 2 或 3
            # desi_mean: 1(xt) + (desi_bands-1)(cond) + 0/1(log_w)

        if isinstance(network_config, dict):
            network_config = dict(network_config)
            if "params" in network_config:
                network_config["params"] = dict(network_config["params"])
                network_config["params"]["in_channels"] = _in_ch
        else:
            from omegaconf import OmegaConf as _OC
            network_config = _OC.to_container(network_config, resolve=True)
            network_config["params"]["in_channels"] = _in_ch

        # ---- UNet ----
        self.model: nn.Module = instantiate_from_config(network_config)

        # SB 模式：噪声水平嵌入
        if use_i2sb:
            noise_levels = torch.linspace(1e-4, 1.0, interval) * interval
            self.register_buffer("noise_levels", noise_levels)

        # ---- EMA ----
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay)
            logpy.info(f"[EMA] 跟踪 {len(list(self.model_ema.buffers()))} 个缓冲区")

        # ---- SB 扩散对象（延迟初始化）----
        self._sb: Optional[SBDiffusion] = None

        # ---- 加载检查点 ----
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        # print(f"[DEBUG] _in_ch = {_in_ch}, x1_mode = {self.x1_mode}, hetero = {self.hetero_cond_channel}")
    # ----------------------------------------------------------
    # SB 扩散对象属性（仅 SB 模式）
    # ----------------------------------------------------------

    @property
    def diffusion(self) -> SBDiffusion:
        assert self.use_i2sb, "diffusion 属性仅在 use_i2sb=True 时可用"
        if self._sb is None:
            betas = make_sb_betas(
                n_timestep=self.interval,
                linear_end=self._beta_max / self.interval,
            )
            self._sb = SBDiffusion(betas, self.device)
        return self._sb

    # ----------------------------------------------------------
    # 检查点
    # ----------------------------------------------------------

    def init_from_ckpt(self, path: str) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError(f"不支持的检查点格式: {path}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        logpy.info(f"从 {path} 恢复: {len(missing)} 个缺失, {len(unexpected)} 个多余键")

    # ----------------------------------------------------------
    # 数据输入与拉伸
    # ----------------------------------------------------------

    def get_input(self, batch: Dict):
        """
        从 batch 中提取数据，统一应用 [EXT-1] 像素拉伸与标准化。

        返回
        ------
        euclid     : (B, 1, H, W)         变换域（SB目标 x0）
        desi       : (B, C_desi, H, W)     变换域（SB起点/DET输入）
        euclid_err : (B, 1, H, W) or None
        desi_err   : (B, C_desi, H, W) or None
        pixel_mask : (B, 1, H, W) bool or None
        """
        euclid_raw     = batch.get(self.input_key_euclid)
        desi_raw       = batch.get(self.input_key_desi)
        if desi_raw is None:
            desi_raw = batch.get("images")
        euclid_err_raw = batch.get(self.input_key_euclid_error, None)
        desi_err_raw   = batch.get(self.input_key_desi_error, None)
        pixel_mask     = batch.get(self.input_key_pixel_mask, None)

        pt = self.pixel_transform
        euclid = pt.forward(euclid_raw, source="euclid") if euclid_raw is not None else None
        desi   = pt.forward(desi_raw, source="desi")

        euclid_err = None
        if euclid_err_raw is not None and euclid_raw is not None:
            euclid_err = pt.transform_error(euclid_err_raw, euclid_raw, source="euclid")

        desi_err = None
        if desi_err_raw is not None:
            desi_err = pt.transform_error(desi_err_raw, desi_raw, source="desi")

        norm_eu = pt._get_norm("euclid")
        norm_de = pt._get_norm("desi")

        if euclid is not None:
            euclid = pt.normalize(euclid, source="euclid")
            if euclid_err is not None and norm_eu is not None:
                euclid_err = euclid_err / norm_eu[1]

        desi = pt.normalize(desi, source="desi")
        if desi_err is not None and norm_de is not None:
            desi_err = desi_err / norm_de[1]

        return euclid, desi, euclid_err, desi_err, pixel_mask

    # ----------------------------------------------------------
    # 构建桥端点与条件（SB 模式）
    # ----------------------------------------------------------

    def build_x1(self, x0: torch.Tensor, desi: torch.Tensor) -> torch.Tensor:
        if self.x1_mode == "desi_mean":
            return desi.mean(dim=1, keepdim=True)
        elif self.x1_mode == "gaussian":
            return torch.randn_like(x0)
        elif self.x1_mode == "rz":
            return desi[:, 1, :, :].unsqueeze(1)
        else:
            raise ValueError(f"未知 x1_mode='{self.x1_mode}'")

    def build_cond(self, desi: torch.Tensor) -> torch.Tensor:
        if self.x1_mode == "desi_mean":
            return desi[:, :-1, :, :] - desi[:, 1:, :, :]
        elif self.x1_mode == "rz":
            return desi[:, 2, :, :].unsqueeze(1)
        else:
            raise ValueError(f"未知 x1_mode='{self.x1_mode}'")

    def build_desi_mean_err(self, desi_err: torch.Tensor) -> torch.Tensor:
        C_desi = desi_err.shape[1]
        return (desi_err.pow(2).sum(dim=1, keepdim=True)).sqrt() / C_desi

    # ----------------------------------------------------------
    # [EXT-2] 异方差权重图（仅 SB 模式）
    # ----------------------------------------------------------

    def compute_weight_map(self, x1: torch.Tensor, desi_mean_err: torch.Tensor) -> torch.Tensor:
        var = desi_mean_err.pow(2)
        B = var.shape[0]
        flat = var.view(B, -1)
        median_val = flat.median(dim=1).values.clamp(min=1e-10)
        w = var / median_val.view(B, 1, 1, 1)
        return w.clamp(min=self.w_clamp_min, max=self.w_clamp_max)

    def _build_hetero(self, x1: torch.Tensor, desi_err: Optional[torch.Tensor]) -> Dict:
        result: Dict[str, Optional[torch.Tensor]] = {"sqrt_w": None, "log_w": None}
        if self.heteroscedastic and desi_err is not None:
            desi_mean_err = self.build_desi_mean_err(desi_err)
            w = self.compute_weight_map(x1, desi_mean_err)
            result["sqrt_w"] = w.sqrt()
            if self.hetero_cond_channel:
                result["log_w"] = w.clamp(min=1e-10).log()
        return result

    # ----------------------------------------------------------
    # Label / pred_x0 计算（SB 模式）
    # ----------------------------------------------------------

    def compute_label(self, step, x0, xt, sqrt_w=None):
        if self.sb_pred_type == "x0":
            return x0.detach()
        std = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        if sqrt_w is not None:
            std = std * sqrt_w
        return ((xt - x0) / std).detach()

    def compute_pred_x0(self, step, xt, net_out, sqrt_w=None):
        if self.sb_pred_type == "x0":
            if self.clip_denoise:
                return net_out.clamp(-1.0, 1.0)
            return net_out
        std = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        if sqrt_w is not None:
            std = std * sqrt_w
        pred = xt - std * net_out
        if self.clip_denoise:
            pred = pred.clamp(-1.0, 1.0)
        return pred

    # ----------------------------------------------------------
    # 网络调用
    # ----------------------------------------------------------

    def run_network(self, xt, step, cond, log_w=None):
        """SB 模式：UNet(xt ⊕ cond [⊕ log_w], t) → score"""
        parts = [xt, cond]
        if log_w is not None:
            parts.append(log_w)
        x_in = torch.cat(parts, dim=1)
        t = self.noise_levels[step]
        return self.model(x_in, t)

    def run_network_det(self, desi: torch.Tensor) -> torch.Tensor:
        """
        传统 UNet 模式：UNet(desi[:, 1:]) → pred_x0

        调用方负责传入已切片的 desi（即 desi[:, 1:, :, :]），
        与 x1_mode='rz' 保持一致（z 波段及之后作为输入）。
        时间嵌入传入全零以兼容带时间嵌入的 UNetModel 接口。
        """
        B = desi.shape[0]
        t = torch.zeros(B, device=desi.device, dtype=torch.long)
        return self.model(desi, t)

    # ----------------------------------------------------------
    # EMA 上下文
    # ----------------------------------------------------------

    @contextmanager
    def ema_scope(self, context: Optional[str] = None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
        try:
            yield
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())

    # ----------------------------------------------------------
    # 指标与损失函数
    # ----------------------------------------------------------

    def get_reduced_chi2(self, gt, recon, euclid_err):
        """逐像素约化 χ²（监控指标，两种模式均用）。"""
        if euclid_err is None:
            return torch.tensor(0.0, device=gt.device)
        chi2 = ((gt - recon) ** 2 / euclid_err.clamp(min=1e-10) ** 2).sum()
        dof = gt.numel()
        return chi2 / dof if dof > 0 else chi2
    def get_reduced_chi2_seg(self, gt, recon, euclid_err):
        """逐像素约化 χ²（监控指标，两种模式均用）。"""
        from astropy.stats import sigma_clipped_stats
        if euclid_err is None:
            return torch.tensor(0.0, device=gt.device)
        chi2 = ((gt - recon) ** 2 / euclid_err.clamp(min=1e-10) ** 2).sum()
        # dof = gt.numel()
        with torch.no_grad():
            gt_np = gt.float().cpu().numpy()
            flat = gt_np.reshape(-1)
            _, bg_median, bg_std = sigma_clipped_stats(flat, sigma=3.0, maxiters=5)
            threshold = bg_median + 3.0 * bg_std
            mask = torch.tensor(gt_np > threshold, device=gt.device)

        dof = mask.sum().item()
        if dof == 0:
            return torch.tensor(0.0, device=gt.device)

        chi2 = ((gt - recon) ** 2 / euclid_err.clamp(min=1e-10) ** 2)
        return chi2[mask].sum() / dof
    def masked_mse(self, pred, label, pixel_mask=None, x0=None, euclid_err=None):
        """
        SB 模式扩散损失（对 score/ε 或 x₀ 的回归损失）。

        参数
        ------
        pred, label : (B, 1, H, W) 网络输出与目标
        pixel_mask  : (B, 1, H, W) bool/float，可选
        x0          : (B, 1, H, W) 变换+标准化域的 Euclid GT，可选；
                      若 self.sb_bright_weight>0，则按
                      w = 1 / (|x0| + sb_bright_weight) 对每个像素加权。
        euclid_err  : (B, 1, H, W) or None；
                      若 self.sb_chi2_loss=True 且此值不为 None，
                      损失改为 χ² 形式：(pred-label)²/σ²，
                      其中 σ=euclid_err（x₀ 预测模式下直接是像素误差；
                      ε 预测模式下近似用同一图作逐像素缩放权重）。

        损失类型由 self.sb_loss_type 控制（"mse" | "huber"），
        χ² 加权由 self.sb_chi2_loss 控制，两者可叠加。
        """
        diff = pred - label
        if self.sb_loss_type == "huber":
            d = self.sb_huber_delta
            abs_d = diff.abs()
            err = torch.where(
                abs_d <= d, 0.5 * diff.pow(2), d * (abs_d - 0.5 * d)
            )
        else:  # mse
            err = diff.pow(2)

        # χ² 加权：除以逐像素 σ²（优先级高于 sb_bright_weight）
        if self.sb_chi2_loss and euclid_err is not None:
            sigma2 = euclid_err.clamp(min=1e-10).pow(2)
            err = err / sigma2
            w = None  # χ² 已含权重，不再叠加亮端权重
        elif self.sb_bright_weight > 0.0 and x0 is not None:
            # 亮端权重：对应"信号相关方差"近似补偿
            w = 1.0 / (x0.abs() + self.sb_bright_weight)
        else:
            w = None

        if pixel_mask is None and w is None:
            return err.mean()

        if pixel_mask is not None:
            mask_f = pixel_mask.float()
        else:
            mask_f = torch.ones_like(err)

        if w is not None:
            mask_f = mask_f * w

        denom = mask_f.sum().clamp(min=1.0)
        return (err * mask_f).sum() / denom

    def det_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        euclid_err: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        传统 UNet 模式的像素空间损失函数。

        参数
        ------
        pred        : (B, 1, H, W)  网络预测
        target      : (B, 1, H, W)  GT（x0，已变换域）
        euclid_err  : (B, 1, H, W) or None  （仅 chi2 需要）
        pixel_mask  : (B, 1, H, W) bool or None

        loss_type
        ----------
        "l2"   : MSE(pred, target)
        "l1"   : MAE(pred, target)
        "chi2" : Σ (pred - target)² / σ² / N_valid
                 若 euclid_err 为 None，自动退化为 l2 并警告。
        """
        # 构建掩码权重
        if pixel_mask is not None:
            mask_f = pixel_mask.float()
        else:
            mask_f = torch.ones_like(pred)

        n_valid = mask_f.sum().clamp(min=1.0)

        loss_type = self.det_loss_type

        if loss_type == "l2":
            err = (pred - target).pow(2)

        elif loss_type == "l1":
            err = (pred - target).abs()

        elif loss_type == "chi2":
            if euclid_err is None:
                logpy.warning(
                    "[det_loss/chi2] euclid_err 为 None，自动退化为 l2！"
                    "请检查数据管道或改用 det_loss_type=l2。"
                )
                err = (pred - target).pow(2)
            else:
                sigma2 = euclid_err.clamp(min=1e-10).pow(2)
                err = (pred - target).pow(2) / sigma2

        else:
            raise ValueError(f"未知 det_loss_type='{loss_type}'")

        if self.det_loss_reduction == "mean":
            return (err * mask_f).sum() / n_valid
        else:  # "sum"
            return (err * mask_f).sum()

    # ----------------------------------------------------------
    # GAN 损失（两种模式均支持）
    # ----------------------------------------------------------

    def _disc_loss(self, real, fake):
        real_pred = self.discriminator(real)
        fake_pred = self.discriminator(fake)
        if self.disc_loss_type == "hinge":
            loss_real = torch.nn.functional.relu(1.0 - real_pred).mean()
            loss_fake = torch.nn.functional.relu(1.0 + fake_pred).mean()
        elif self.disc_loss_type == "vanilla":
            loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
        else:
            raise ValueError(f"未知 disc_loss_type='{self.disc_loss_type}'")
        return (loss_real + loss_fake) * 0.5

    def _adv_loss(self, fake):
        fake_pred = self.discriminator(fake)
        if self.disc_loss_type == "hinge":
            return -fake_pred.mean()
        elif self.disc_loss_type == "vanilla":
            return torch.nn.functional.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )
        else:
            raise ValueError(f"未知 disc_loss_type='{self.disc_loss_type}'")

    # ----------------------------------------------------------
    # §3  训练步骤（按模式分发）
    # ----------------------------------------------------------

    def on_train_start(self, *args, **kwargs):
        if self.use_i2sb:
            _ = self.diffusion

    def training_step(self, batch: Dict, batch_idx: int):
        if self.use_i2sb:
            return self._training_step_sb(batch, batch_idx)
        else:
            return self._training_step_det(batch, batch_idx)

    # ---- SB 训练步骤（原始逻辑）----

    def _training_step_sb(self, batch: Dict, batch_idx: int):
        euclid, desi, euclid_err, desi_err, pixel_mask = self.get_input(batch)
        x0, desi = euclid.to(self.device), desi.to(self.device)
        x1, cond = self.build_x1(x0, desi), self.build_cond(desi)

        if euclid_err is not None: euclid_err = euclid_err.to(self.device)
        if desi_err   is not None: desi_err   = desi_err.to(self.device)
        if pixel_mask is not None: pixel_mask = pixel_mask.to(self.device)

        hetero = self._build_hetero(x1, desi_err)
        sqrt_w, log_w = hetero["sqrt_w"], hetero["log_w"]

        B    = x0.shape[0]
        step = torch.randint(0, self.interval, (B,), device=self.device)
        xt    = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode, sqrt_w=sqrt_w)
        label = self.compute_label(step, x0, xt, sqrt_w=sqrt_w)

        if self.adversarial:
            opt_g, opt_d = self.optimizers()
        else:
            opt_g = self.optimizers()

        global_step = self.global_step
        use_adv = self.adversarial and (global_step >= self.adv_start_step)

        pred    = self.run_network(xt, step, cond, log_w=log_w)
        pred_x0 = self.compute_pred_x0(step, xt, pred, sqrt_w=sqrt_w)

        # -- 生成器 --
        diff_loss = self.masked_mse(
            pred, label, pixel_mask=pixel_mask, x0=x0, euclid_err=euclid_err
        )

        if use_adv:
            adv_loss = self._adv_loss(pred_x0)
            gen_loss = diff_loss + self.adv_weight * adv_loss
        else:
            adv_loss = torch.tensor(0.0, device=self.device)
            gen_loss = diff_loss

        if self.adversarial:
            opt_g.zero_grad()
            self.manual_backward(gen_loss)
            opt_g.step()

        # -- 判别器 --
        disc_loss = torch.tensor(0.0, device=self.device)
        if use_adv:
            opt_d.zero_grad()
            disc_loss = self._disc_loss(x0, pred_x0.detach())
            self.manual_backward(disc_loss)
            opt_d.step()

        if self.adversarial:
            self._step_schedulers()

        with torch.no_grad():
            reduced_chi2 = self.get_reduced_chi2(x0, pred_x0.detach(), euclid_err)

        self.log("train/diff_loss",      diff_loss,    prog_bar=True)
        self.log("train/gen_total_loss", gen_loss,     prog_bar=True)
        self.log("train/reduced_chi2",   reduced_chi2, prog_bar=True)
        if use_adv:
            self.log("train/adv_loss_g", adv_loss)
            self.log("train/disc_loss",  disc_loss)

        # 非对抗模式下返回标量损失（Lightning 自动优化）
        return gen_loss if not self.adversarial else None

    # ---- 传统 UNet 确定性训练步骤 ----

    def _training_step_det(self, batch: Dict, batch_idx: int):
        """
        传统 UNet 直接回归训练步骤。

        前向：pred_x0 = UNet(desi)
        损失：det_loss(pred_x0, euclid, euclid_err, pixel_mask)
              + [EXT-3] adv_weight * adv_loss（可选）
        """
        euclid, desi, euclid_err, desi_err, pixel_mask = self.get_input(batch)
        x0   = euclid.to(self.device)
        desi = desi.to(self.device)

        if euclid_err is not None: euclid_err = euclid_err.to(self.device)
        if pixel_mask is not None: pixel_mask = pixel_mask.to(self.device)

        if self.adversarial:
            opt_g, opt_d = self.optimizers()
        else:
            opt_g = self.optimizers()

        global_step = self.global_step
        use_adv = self.adversarial and (global_step >= self.adv_start_step)

        # 前向（仅取 desi[:, 1:] 作为输入，与 x1_mode='rz' 对应）
        pred_x0 = self.run_network_det(desi[:, 1:, :, :])

        # -- 生成器 --
        opt_g.zero_grad() if self.adversarial else None

        pixel_loss = self.det_loss(pred_x0, x0, euclid_err=euclid_err, pixel_mask=pixel_mask)

        if use_adv:
            adv_loss = self._adv_loss(pred_x0)
            gen_loss = pixel_loss + self.adv_weight * adv_loss
        else:
            adv_loss = torch.tensor(0.0, device=self.device)
            gen_loss = pixel_loss

        if self.adversarial:
            self.manual_backward(gen_loss)
            opt_g.step()

        # -- 判别器 --
        disc_loss = torch.tensor(0.0, device=self.device)
        if use_adv:
            opt_d.zero_grad()
            disc_loss = self._disc_loss(x0, pred_x0.detach())
            self.manual_backward(disc_loss)
            opt_d.step()

        if self.adversarial:
            self._step_schedulers()

        with torch.no_grad():
            reduced_chi2 = self.get_reduced_chi2(x0, pred_x0.detach(), euclid_err)

        loss_name = f"train/{self.det_loss_type}_loss"
        self.log(loss_name,              pixel_loss,   prog_bar=True)
        self.log("train/gen_total_loss", gen_loss,     prog_bar=True)
        self.log("train/reduced_chi2",   reduced_chi2, prog_bar=True)
        if use_adv:
            self.log("train/adv_loss_g", adv_loss)
            self.log("train/disc_loss",  disc_loss)

        # 非对抗模式下返回标量损失（Lightning 自动优化）
        return gen_loss if not self.adversarial else None

    # ----------------------------------------------------------
    # 验证步骤（按模式分发）
    # ----------------------------------------------------------

    def validation_step(self, batch: Dict, batch_idx: int):
        if self.use_i2sb:
            return self._validation_step_sb(batch, batch_idx)
        else:
            return self._validation_step_det(batch, batch_idx)

    def _validation_step_sb(self, batch: Dict, batch_idx: int):
        euclid, desi, euclid_err, desi_err, pixel_mask = self.get_input(batch)
        x0   = euclid.to(self.device)
        desi = desi.to(self.device)
        x1   = self.build_x1(x0, desi)
        cond = self.build_cond(desi)

        if euclid_err is not None: euclid_err = euclid_err.to(self.device)
        if desi_err   is not None: desi_err   = desi_err.to(self.device)
        if pixel_mask is not None: pixel_mask = pixel_mask.to(self.device)

        hetero = self._build_hetero(x1, desi_err)
        sqrt_w, log_w = hetero["sqrt_w"], hetero["log_w"]

        B    = x0.shape[0]
        step = torch.randint(0, self.interval, (B,), device=self.device)
        xt    = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode, sqrt_w=sqrt_w)
        label = self.compute_label(step, x0, xt, sqrt_w=sqrt_w)

        with self.ema_scope():
            pred = self.run_network(xt, step, cond, log_w=log_w)

        val_loss = self.masked_mse(
            pred, label, pixel_mask=pixel_mask, x0=x0, euclid_err=euclid_err
        )
        pred_x0      = self.compute_pred_x0(step, xt, pred, sqrt_w=sqrt_w)
        reduced_chi2 = self.get_reduced_chi2_seg(x0, pred_x0, euclid_err)

        self.log("val/loss",         val_loss,     on_epoch=True, sync_dist=True)
        self.log("val/reduced_chi2", reduced_chi2, on_epoch=True, sync_dist=True)
        return val_loss

    def _validation_step_det(self, batch: Dict, batch_idx: int):
        euclid, desi, euclid_err, desi_err, pixel_mask = self.get_input(batch)
        x0   = euclid.to(self.device)
        desi = desi.to(self.device)

        if euclid_err is not None: euclid_err = euclid_err.to(self.device)
        if pixel_mask is not None: pixel_mask = pixel_mask.to(self.device)

        with self.ema_scope():
            pred_x0 = self.run_network_det(desi[:, 1:, :, :])

        val_loss     = self.det_loss(pred_x0, x0, euclid_err=euclid_err, pixel_mask=pixel_mask)
        reduced_chi2 = self.get_reduced_chi2(x0, pred_x0, euclid_err)

        self.log("val/loss",         val_loss,     on_epoch=True, sync_dist=True)
        self.log("val/reduced_chi2", reduced_chi2, on_epoch=True, sync_dist=True)
        return val_loss

    # ----------------------------------------------------------
    # 辅助：scheduler 步进
    # ----------------------------------------------------------

    def _step_schedulers(self):
        sch = self.lr_schedulers()
        if sch is None:
            return
        if isinstance(sch, list):
            for s in sch:
                s.step()
        else:
            sch.step()

    # ----------------------------------------------------------
    # EMA 更新
    # ----------------------------------------------------------

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    # ----------------------------------------------------------
    # 推理采样（SB 模式）
    # ----------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
        desi_err: Optional[torch.Tensor] = None,
        nfe: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SB 模式：完整反向轨迹 x₁ → x₀。
        返回 (xs, pred_x0s)，形状 (B, log_count, 1, H, W)。
        """
        assert self.use_i2sb, "sample() 仅在 use_i2sb=True 时可用，传统模式请用 predict()"
        nfe   = nfe or self.nfe
        steps = space_indices(self.interval, nfe + 1)
        log_count = min(len(steps) - 1, self.log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        hetero = self._build_hetero(
            x1=x1,
            desi_err=desi_err.to(self.device) if desi_err is not None else None,
        )
        sqrt_w, log_w = hetero["sqrt_w"], hetero["log_w"]
    
        hetero = self._build_hetero(x1, desi_err)
        # print("[DEBUG sample] log_w is None:", hetero["log_w"] is None)  # 加这行
        sqrt_w, log_w = hetero["sqrt_w"], hetero["log_w"]

        with self.ema_scope():
            self.model.eval()

            def pred_x0_fn(xt, step_int):
                step_t = torch.full(
                    (xt.shape[0],), step_int, device=self.device, dtype=torch.long
                )
                net_out = self.run_network(xt, step_t, cond, log_w=log_w)
                return self.compute_pred_x0(step_t, xt, net_out, sqrt_w=sqrt_w)

            xs, pred_x0s = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1,
                ot_ode=self.ot_ode,
                log_steps=log_steps,
                verbose=verbose,
                sqrt_w=sqrt_w,
            )

        return xs, pred_x0s

    # ============================================================
    # [SPEEDUP] 高效推理采样:共享前缀分叉 + num_samples batch 并行
    # ------------------------------------------------------------
    # 与 sample() 的区别:
    #   1. 不进 ema_scope():推理脚本应在循环外一次性 swap EMA 权重,
    #      避免每 batch 重复 store/copy_to。
    #   2. num_samples > 1 时使用共享前缀分叉(见 SBDiffusion.
    #      ddpm_sampling_forked),返回 (num_samples, B, 1, H, W)。
    #   3. 适合 torch.compile(model.model) 后调用:UNet forward 的
    #      输入 shape 在前缀阶段=B,后缀阶段=num_samples*B,只有两种
    #      shape,compile 缓存友好。
    # ============================================================
    @torch.no_grad()
    def sample_forked(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
        desi_err: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        split_ratio: float = 0.7,
        nfe: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        参数
        ----
        num_samples : 重复采样数(并行,通过共享前缀实现)
        split_ratio : 共享前缀比例 ∈ [0, 1]
            0.0 = 全程独立(等价于 num_samples 次独立 sample)
            0.7 = 推荐起点,前 70% 步共享、后 30% 分叉
            1.0 = 全程共享(等价于单 sample,std=0,不要这么用)

        返回
        ----
        final_x0 : (num_samples, B, 1, H, W) 最终 x0 估计(变换域)
        """
        assert self.use_i2sb, "sample_forked() 仅在 use_i2sb=True 时可用"
        assert num_samples >= 1
        assert 0.0 <= split_ratio <= 1.0

        nfe   = nfe or self.nfe
        steps = space_indices(self.interval, nfe + 1)
        log_count = min(len(steps) - 1, self.log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        # 分叉点:前 split_ratio 比例的 pair 共享,其余分叉
        n_pairs = len(steps) - 1  # = nfe
        split_step = int(round(split_ratio * n_pairs))

        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        # ---- 异方差权重(单份,后缀阶段需要 broadcast)----
        hetero = self._build_hetero(
            x1=x1,
            desi_err=desi_err.to(self.device) if desi_err is not None else None,
        )
        sqrt_w, log_w = hetero["sqrt_w"], hetero["log_w"]

        # ---- 为分叉后缀准备 broadcast 版本的 cond / sqrt_w / log_w ----
        # 形状从 (B, ...) 扩到 (num_samples*B, ...),仅在 num_samples>1 时需要
        def _expand_ns(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None or num_samples == 1:
                return t
            return t.unsqueeze(0).expand(num_samples, *t.shape).reshape(
                num_samples * t.shape[0], *t.shape[1:]
            ).contiguous()

        cond_forked   = _expand_ns(cond)
        sqrt_w_forked = _expand_ns(sqrt_w)
        log_w_forked  = _expand_ns(log_w)

        self.model.eval()

        # ---- 两个 pred_x0 闭包:shape 不同,确保 compile 缓存命中 ----
        def pred_x0_fn_shared(xt, step_int):
            step_t = torch.full(
                (xt.shape[0],), step_int, device=self.device, dtype=torch.long
            )
            net_out = self.run_network(xt, step_t, cond, log_w=log_w)
            return self.compute_pred_x0(step_t, xt, net_out, sqrt_w=sqrt_w)

        def pred_x0_fn_forked(xt, step_int):
            step_t = torch.full(
                (xt.shape[0],), step_int, device=self.device, dtype=torch.long
            )
            net_out = self.run_network(xt, step_t, cond_forked, log_w=log_w_forked)
            return self.compute_pred_x0(step_t, xt, net_out, sqrt_w=sqrt_w_forked)

        final_x0, _ = self.diffusion.ddpm_sampling_forked(
            steps=steps,
            pred_x0_fn_shared=pred_x0_fn_shared,
            pred_x0_fn_forked=pred_x0_fn_forked,
            x1=x1,
            num_samples=num_samples,
            split_step=split_step,
            ot_ode=self.ot_ode,
            log_steps=log_steps,
            verbose=verbose,
            sqrt_w_shared=sqrt_w,
            sqrt_w_forked=sqrt_w_forked,
        )

        return final_x0  # (num_samples, B, 1, H, W)

    @torch.no_grad()
    def predict(self, desi: torch.Tensor) -> torch.Tensor:
        """
        传统 UNet 模式推理：desi → pred_x0（变换域）。
        反变换回物理单位需再调用 pixel_transform.inverse()。
        """
        assert not self.use_i2sb, "predict() 仅在 use_i2sb=False 时可用，SB 模式请用 sample()"
        with self.ema_scope():
            self.model.eval()
            return self.run_network_det(desi.to(self.device)[:, 1:, :, :])

    # ----------------------------------------------------------
    # 图像日志
    # ----------------------------------------------------------

    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict:
        log = {}
        euclid, desi, euclid_err, desi_err, pixel_mask = self.get_input(batch)
        x0   = euclid.to(self.device)
        desi = desi.to(self.device)
        pt   = self.pixel_transform

        def to3(t): return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t[:, :3]

        if self.use_i2sb:
            x1   = self.build_x1(x0, desi)
            cond = self.build_cond(desi)
            xs, _ = self.sample(
                x1, cond,
                desi_err=desi_err.to(self.device) if desi_err is not None else None,
                verbose=False,
            )
            generated = xs[:, 0].to(self.device)

            x0_vis        = pt.inverse(pt.denormalize(x0,        source="euclid"), source="euclid")
            generated_vis = pt.inverse(pt.denormalize(generated, source="euclid"), source="euclid")
            desi_vis = pt.inverse(pt.denormalize(desi[:, :3], source="desi"), source="desi")
            # x1 是单波段（build_x1 从 desi 提取），直接从已反变换的 desi_vis 取对应通道
            _x1_band = 1 if self.x1_mode == "rz" else 0  # rz→r(idx1), desi_mean→第0通道近似
            x1_vis   = to3(desi_vis[:, _x1_band:_x1_band+1, :, :])
            eu_vis   = to3(x0_vis)
            gen_vis  = to3(generated_vis)
            residual = eu_vis - gen_vis

            log["comparison"] = torch.cat([desi_vis, x1_vis, eu_vis, gen_vis, residual], dim=-1)

            
            B, L, C, H, W = xs.shape
            xs_vis = pt.inverse(
                pt.denormalize(xs.to(self.device).reshape(B * L, C, H, W), source="euclid"),
                source="euclid"
            )
            log["bridge_trajectory"] = make_grid(to3(xs_vis), nrow=L)

        else:
            pred_x0 = self.predict(desi)

            x0_vis   = pt.inverse(pt.denormalize(x0,      source="euclid"), source="euclid")
            pred_vis = pt.inverse(pt.denormalize(pred_x0, source="euclid"), source="euclid")
            desi_vis = pt.inverse(pt.denormalize(desi[:, :3], source="desi"), source="desi")
            eu_vis   = to3(x0_vis)
            gen_vis  = to3(pred_vis)
            residual = eu_vis - gen_vis

            log["comparison"] = torch.cat([desi_vis, eu_vis, gen_vis, residual], dim=-1)

        return log

    # ----------------------------------------------------------
    # 优化器 / 调度器
    # ----------------------------------------------------------

    def _build_optimizer(self, params, cfg: Dict):
        return get_obj_from_str(cfg["target"])(params, **cfg.get("params", {}))

    def configure_optimizers(self):
        params_g = list(self.model.parameters())
        opt_g    = self._build_optimizer(params_g, self.optimizer_config)

        if self.adversarial:
            params_d = list(self.discriminator.parameters())
            opt_d    = self._build_optimizer(params_d, self.optimizer_config)

            if self.scheduler_config is not None:
                sched_fn = instantiate_from_config(self.scheduler_config)
                sched_g  = {
                    "scheduler": LambdaLR(opt_g, lr_lambda=sched_fn.schedule),
                    "interval":  "step",
                    "frequency": 1,
                }
                return [opt_g, opt_d], [sched_g]

            return [opt_g, opt_d]

        if self.scheduler_config is not None:
            sched_fn = instantiate_from_config(self.scheduler_config)
            sched = {
                "scheduler": LambdaLR(opt_g, lr_lambda=sched_fn.schedule),
                "interval":  "step",
                "frequency": 1,
            }
            return [opt_g], [sched]

        return opt_g