# ---------------------------------------------------------------
# astreIR_SBn.py — Multimodal DESI → Euclid generative model
#
# Two operating modes:
#   [MODE-SB]  use_i2sb=True  ← Heteroscedastic Schrodinger bridge (I2SB), based on NVlabs/I2SB
#   [MODE-DET] use_i2sb=False ← Deterministic UNet regression without diffusion
#
# Optional extensions for ablation studies (except where noted):
#   [EXT-1] Pixel stretching (pixel_transform) — Compress the input dynamic range
#   [EXT-2] Heteroscedastic bridge dynamics — Per-pixel noise scaling w(i), SB mode only
#   [EXT-3] PatchGAN adversarial loss — Adversarial training over patch receptive fields
#
# ============================================================
# Training data flow in SB mode:
# ============================================================
#
#   batch → get_input() → euclid, desi, err (transformed space)
#     → build_x1/build_cond
#     → [EXT-2] _build_hetero() → sqrt_w, log_w
#     → diffusion.q_sample() → xt
#     → compute_label() → label
#     → run_network(xt, step, cond, [log_w]) → pred
#     → sb_loss(pred, label, [mask])  +  [EXT-3] adv_loss
#
# Training data flow in deterministic UNet mode:
# ============================================================
#
#   batch → get_input() → euclid (x0), desi (x1), err
#     → run_network_det(desi) → pred_x0
#     → det_loss(pred_x0, euclid, [euclid_err], loss_type)  +  [EXT-3] adv_loss
#
#   loss_type options:
#     "l2"   — MSE, equivalent to the original masked_mse
#     "l1"   — MAE, more robust to outlier pixels
#     "chi2" — Reduced χ², requiring euclid_err for uncertainty weighting
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
# Section 0: Constants and utilities for I2SB diffusion
# ============================================================

def _compute_gaussian_product_coef(
    sigma1: np.ndarray,
    sigma2: np.ndarray,
):
    """
    Coefficients for the product of two Gaussians (I2SB Equation 10).

    Given p1 = N(x_t | x_0, σ₁²) and p2 = N(x_t | x_1, σ₂²),
    compute p1*p2 = N(x_t | coef1*x₀ + coef2*x₁, var).
    """
    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom
    return coef1, coef2, var


def _unsqueeze_xdim(z: torch.Tensor, xdim) -> torch.Tensor:
    "Broadcast a (B,) tensor to (B, 1, 1, ...) for pixelwise operations."
    bc = (...,) + (None,) * len(xdim)
    return z[bc]


def make_sb_betas(n_timestep: int = 1000, linear_end: float = 2e-2) -> np.ndarray:
    """
    Construct the symmetric beta schedule used by I2SB.

    Equivalent to make_beta_schedule in NVlabs/I2SB.
    """
    linear_start = 1e-10
    assert linear_end >= linear_start, (
        f"linear_end={linear_end:.2e} < linear_start={linear_start:.2e}！"
        f"beta_max must be >= {linear_start * n_timestep:.4f} (= linear_start * interval)."
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
    "Choose count uniformly spaced indices from [0, num_steps-1]."
    assert count <= num_steps
    frac = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur, taken = 0.0, []
    for _ in range(count):
        taken.append(round(cur))
        cur += frac
    return taken


# ============================================================
# Section 1: Schrodinger bridge diffusion, used only in SB mode
# ============================================================

class SBDiffusion:
    """
    I2SB Schrodinger bridge diffusion with optional per-pixel
    heteroscedastic scaling. Instantiated only when use_i2sb=True.
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
            pairs = tqdm(pairs, desc="SB DDPM sampling", total=len(pairs))
        for prev_step, step in pairs:
            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, sqrt_w=sqrt_w)
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
        stack = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack(xs), stack(pred_x0s)

    # ============================================================
    # [SPEEDUP] Forked sampling with a shared prefix
    # ------------------------------------------------------------
    # Design: three trajectories share the first K steps, where deterministic evolution dominates.
    # At step K, replicate the batch num_samples times,
    # then apply independent noise during the remaining N-K steps.
    #
    # This retains per-pixel standard-deviation estimates while reducing the
    # total forward workload from N*num_samples*B to
    # K*B + (N-K)*num_samples*B. For example, N=50, K=35, ns=3:
    #   Independent: 3 * 50 * B = 150*B
    #   Forked: 35*B + 15*3B = 80*B, saving 47%
    #
    # Input conventions:
    #   x1, cond, sqrt_w, and log_w each contain one batch with shape (B, ...).
    #   pred_x0_fn is a closure whose run_network call uses cond/log_w
    #   already broadcast by the caller; see sample_forked.
    #
    # Returns:
    #   final_x0: (num_samples, B, 1, H, W), final x0 estimates for mean/std
    #   bridge_xs: (B, log_count, 1, H, W), visualization from the shared prefix
    #   and first trajectory only, compatible with image_logger. For num_samples>1,
    #   other branches are omitted from bridge_xs to reduce CPU copies.
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
        Parameters
        ----------
        pred_x0_fn_shared : (xt[B,...], step_int) -> pred_x0[B,...]
            Prediction before the fork, with batch size B.
        pred_x0_fn_forked : (xt[ns*B,...], step_int) -> pred_x0[ns*B,...]
            Prediction after the fork, with batch size ns*B.
        split_step : int
            Number of completed reverse steps before forking.
            0 means independent trajectories; len(steps)-1 means full sharing.
            Actual forking requires 1 <= split_step <= len(pairs)-1;
            otherwise the procedure reduces to ordinary sampling.
        """
        B = x1.shape[0]
        device = self.device
        xt = x1.detach().to(device)
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        rev = steps[::-1]
        pairs = list(zip(rev[1:], rev[:-1]))  # Adjacent reverse-step pairs.
        n_pairs = len(pairs)

        # Clamp split_step to a valid range.
        split_step = max(0, min(n_pairs, int(split_step)))

        # Record the shared prefix and only the first trajectory of the suffix for visualization.
        xs_bridge = []
        # Use a set for faster membership checks.
        log_steps_set = set(log_steps)

        # ---------- Stage 1: shared prefix (batch=B) ----------
        iter1 = pairs[:split_step]
        if verbose and len(iter1) > 0:
            iter1 = tqdm(iter1, desc=f"SB shared prefix (B={B})", total=len(iter1), leave=False)
        for prev_step, step in iter1:
            pred_x0 = pred_x0_fn_shared(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0,
                                  ot_ode=ot_ode, sqrt_w=sqrt_w_shared)
            if prev_step in log_steps_set:
                xs_bridge.append(xt.detach().cpu())

        # ---------- Fork: replicate num_samples times along the batch dimension ----------
        # Do not add noise during replication; the next p_posterior call adds it.
        # Expand the batch to ns*B so one forward pass handles every branch.
        if num_samples > 1 and split_step < n_pairs:
            # (B, ...) -> (num_samples, B, ...) -> (num_samples*B, ...)
            xt = xt.unsqueeze(0).expand(num_samples, *xt.shape).contiguous()
            xt = xt.reshape(num_samples * B, *xt.shape[2:])
            sqrt_w_curr = sqrt_w_forked
        else:
            sqrt_w_curr = sqrt_w_shared

        # ---------- Stage 2: forked suffix (batch=ns*B) ----------
        iter2 = pairs[split_step:]
        if verbose and len(iter2) > 0:
            iter2 = tqdm(iter2, desc=f"SB forked suffix (B={xt.shape[0]})",
                         total=len(iter2), leave=False)
        for prev_step, step in iter2:
            pred_x0 = pred_x0_fn_forked(xt, step) if (num_samples > 1 and split_step < n_pairs) \
                      else pred_x0_fn_shared(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0,
                                  ot_ode=ot_ode, sqrt_w=sqrt_w_curr)
            # Bridge visualization: keep only the first of num_samples suffix trajectories.
            if prev_step in log_steps_set and num_samples > 1 and split_step < n_pairs:
                xs_bridge.append(xt[:B].detach().cpu())
            elif prev_step in log_steps_set:
                xs_bridge.append(xt.detach().cpu())

        # ---------- Assemble outputs ----------
        # Reshape final x0 to (num_samples, B, C, H, W).
        if num_samples > 1 and split_step < n_pairs:
            final_x0 = xt.reshape(num_samples, B, *xt.shape[1:])
        else:
            # Full sharing (split_step >= n_pairs) or num_samples=1.
            final_x0 = xt.unsqueeze(0).expand(num_samples, *xt.shape).contiguous() \
                       if num_samples > 1 else xt.unsqueeze(0)

        # Match the original bridge_xs shape: (B, log_count, C, H, W).
        if len(xs_bridge) > 0:
            bridge_xs = torch.flip(torch.stack(xs_bridge, dim=1), dims=(1,))
        else:
            bridge_xs = xt[:B].detach().cpu().unsqueeze(1)

        return final_x0, bridge_xs


# ============================================================
# Section 2: Main Lightning module
# ============================================================

class MultiModalSBDiffusion(pl.LightningModule):
    """
    Multimodal DESI → Euclid generative model.

    Select the mode with use_i2sb:
      - True: Schrodinger bridge diffusion, including EXT-1/2/3.
      - False: deterministic UNet regression with det_loss_type.

    Disabling all extensions and using IdentityTransform recovers the
    original I2SB baseline. With use_i2sb=False, EXT-2 (heteroscedastic
    bridge dynamics) is automatically inactive because there is no diffusion.
    """

    def __init__(
        self,
        # ===== Network =====
        network_config: Dict,

        # ===== [NEW] Mode selection =====
        use_i2sb: bool = True,                   # True: I2SB bridge; False: direct deterministic UNet regression
        det_loss_type: str = "l2",               # Deterministic-mode loss: "l1" | "l2" | "chi2"
        det_loss_reduction: str = "mean",        # "mean" | "sum"

        # ===== Original I2SB parameters (cf. NVlabs/I2SB) =====
        interval: int = 1000,
        beta_max: float = 0.3,
        ot_ode: bool = False,
        clip_denoise: bool = False,
        nfe: int = 100,
        log_count: int = 10,
        x1_mode: str = "desi_mean",
        desi_bands: int = 3,

        # ===== [EXT-1] Pixel stretching =====
        pixel_transform_config: Union[None, Dict, ListConfig, OmegaConf] = None,

        # ===== [EXT-2] Heteroscedastic bridge dynamics, use_i2sb=True only =====
        heteroscedastic: bool = False,
        hetero_cond_channel: bool = False,
        snr_eps: float = 1e-3,
        w_clamp_min: float = 0.1,
        w_clamp_max: float = 10.0,

        # ===== [EXT-3] PatchGAN adversarial loss =====
        adversarial: bool = False,
        disc_in_channels: int = 1,
        disc_ndf: int = 64,
        disc_n_layers: int = 3,
        disc_use_actnorm: bool = False,
        adv_weight: float = 0.1,
        disc_loss_type: str = "hinge",
        adv_start_step: int = 0,

        # ===== Optimizer / scheduler =====
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,

        # ===== EMA =====
        use_ema: bool = True,
        ema_decay: float = 0.9999,

        # ===== Checkpoint =====
        ckpt_path: Union[None, str] = None,

        # ===== Input keys =====
        input_key_euclid: str = "euclid_img",
        input_key_desi: str = "desi_img",
        input_key_desi_error: str = "desi_error",
        input_key_euclid_error: str = "euclid_error",
        input_key_pixel_mask: str = "pixel_mask",

        # ===== [LOSS-ROBUST] Robust SB diffusion loss =====
        # Options for heavy-tailed epsilon residuals around bright sources:
        #   sb_loss_type:
        #   "mse" — Original I2SB MSE, the default
        #   "huber" — Huber loss, with delta controlled by sb_huber_delta
        #   sb_bright_weight > 0 applies pixel weights 1/(|x0|+sb_bright_weight).
        #   This reduces the contribution of bright cores so outliers do not dominate.
        sb_loss_type: str = "mse",
        sb_huber_delta: float = 1.0,
        sb_bright_weight: float = 0.0,

        # ===== [SB-X0] x₀ prediction and χ²-weighted loss =====
        # sb_pred_type:
        #   "epsilon" — Predict score/epsilon; default, matching NVlabs/I2SB
        #   "x0" — Predict x₀ directly; label=x0 and compute_pred_x0 returns net_out
        # sb_chi2_loss:
        #   When True, weight the SB diffusion loss by per-pixel variance from euclid_err.
        #   If euclid_err is None, fall back to ordinary MSE/Huber.
        sb_pred_type: str = "epsilon",
        sb_chi2_loss: bool = False,

        **kwargs,
    ):
        super().__init__()

        # ---- Mode ----
        self.use_i2sb        = use_i2sb
        self.det_loss_type   = det_loss_type.lower()
        self.det_loss_reduction = det_loss_reduction

        # ---- [LOSS-ROBUST] SB diffusion loss parameters ----
        self.sb_loss_type     = sb_loss_type.lower()
        self.sb_huber_delta   = float(sb_huber_delta)
        self.sb_bright_weight = float(sb_bright_weight)
        assert self.sb_loss_type in ("mse", "huber"), (
            f"Unknown sb_loss_type='{sb_loss_type}'; choose 'mse' or 'huber'."
        )

        # ---- [SB-X0] x₀ prediction and χ²-weighted loss ----
        self.sb_pred_type = sb_pred_type.lower()
        self.sb_chi2_loss = bool(sb_chi2_loss)
        assert self.sb_pred_type in ("epsilon", "x0"), (
            f"Unknown sb_pred_type='{sb_pred_type}'; choose 'epsilon' or 'x0'."
        )

        assert self.det_loss_type in ("l1", "l2", "chi2"), (
            f"Unknown det_loss_type='{det_loss_type}'; choose 'l1', 'l2', or 'chi2'."
        )

        logpy.info(
            f"[Mode] use_i2sb={use_i2sb}, "
            + (f"det_loss_type={det_loss_type}" if not use_i2sb
               else f"I2SB bridge mode, sb_pred_type={sb_pred_type}, sb_chi2_loss={sb_chi2_loss}")
        )

        # ---- Input keys ----
        self.input_key_euclid       = input_key_euclid
        self.input_key_desi         = input_key_desi
        self.input_key_desi_error   = input_key_desi_error
        self.input_key_euclid_error = input_key_euclid_error
        self.input_key_pixel_mask   = input_key_pixel_mask

        # ---- Original I2SB parameters ----
        self.interval     = interval
        self._beta_max    = beta_max
        self.ot_ode       = ot_ode
        self.clip_denoise = clip_denoise
        self.nfe          = nfe
        self.log_count    = log_count
        self.x1_mode      = x1_mode
        self.desi_bands   = desi_bands

        # ---- [EXT-1] Pixel stretching ----
        if pixel_transform_config is not None:
            self.pixel_transform: BasePixelTransform = instantiate_from_config(
                pixel_transform_config
            )
        else:
            self.pixel_transform = IdentityTransform()
        logpy.info(f"[EXT-1] Pixel transform: {self.pixel_transform.__class__.__name__}")

        # ---- [EXT-2] Heteroscedastic bridge, SB mode only ----
        self.heteroscedastic     = heteroscedastic and use_i2sb
        self.hetero_cond_channel = hetero_cond_channel and use_i2sb
        self.snr_eps             = snr_eps
        self.w_clamp_min         = w_clamp_min
        self.w_clamp_max         = w_clamp_max
        if heteroscedastic and not use_i2sb:
            logpy.warning("[EXT-2] heteroscedastic=True is ignored when use_i2sb=False")

        # ---- [EXT-3] PatchGAN adversarial loss ----
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
                f"[EXT-3] PatchGAN discriminator enabled: "
                f"in_ch={disc_in_channels}, ndf={disc_ndf}, "
                f"n_layers={disc_n_layers}, loss={disc_loss_type}, "
                f"adv_weight={adv_weight}, adv_start_step={adv_start_step}"
            )
            self.automatic_optimization = False
        else:
            self.discriminator = None

        logpy.info(
            f"[Ablation switches] "
            f"MODE={'I2SB' if use_i2sb else 'DET'}, "
            f"EXT-1={self.pixel_transform.__class__.__name__}, "
            f"EXT-2={self.heteroscedastic}, "
            f"EXT-3={adversarial}"
        )

        # ---- Optimizer ----
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        self.scheduler_config = scheduler_config

        # ---- Dynamic in_channels ----
        # SB mode: xt(1) + cond(desi_bands-1) + [log_w(1)]
        # DET mode: desi(desi_bands), using DESI bands directly
        if use_i2sb:
            if x1_mode == "rz":
                cond_ch = 1  # build_cond actually returns only desi[:, 2].unsqueeze(1).
            else:  # desi_mean
                cond_ch = desi_bands - 1
            _in_ch = 1 + cond_ch + int(self.hetero_cond_channel)
            # rz: 1(xt) + 1(cond) + 0/1(log_w) = 2 or 3
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

        # SB mode: noise-level embedding
        if use_i2sb:
            noise_levels = torch.linspace(1e-4, 1.0, interval) * interval
            self.register_buffer("noise_levels", noise_levels)

        # ---- EMA ----
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay)
            logpy.info(f"[EMA] Tracking {len(list(self.model_ema.buffers()))} buffers")

        # ---- SB diffusion object, initialized lazily ----
        self._sb: Optional[SBDiffusion] = None

        # ---- Load checkpoint ----
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        # print(f"[DEBUG] _in_ch = {_in_ch}, x1_mode = {self.x1_mode}, hetero = {self.hetero_cond_channel}")
    # ----------------------------------------------------------
    # SB diffusion object property, available only in SB mode
    # ----------------------------------------------------------

    @property
    def diffusion(self) -> SBDiffusion:
        assert self.use_i2sb, "The diffusion property is available only when use_i2sb=True."
        if self._sb is None:
            betas = make_sb_betas(
                n_timestep=self.interval,
                linear_end=self._beta_max / self.interval,
            )
            self._sb = SBDiffusion(betas, self.device)
        return self._sb

    # ----------------------------------------------------------
    # Checkpoints
    # ----------------------------------------------------------

    def init_from_ckpt(self, path: str) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError(f"Unsupported checkpoint format: {path}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        logpy.info(f"Restored from {path}: {len(missing)} missing and {len(unexpected)} unexpected keys")

    # ----------------------------------------------------------
    # Data input and pixel stretching
    # ----------------------------------------------------------

    def get_input(self, batch: Dict):
        """
        Extract batch data and apply [EXT-1] pixel transforms and normalization.

        Returns
        -------
        euclid     : (B, 1, H, W), transformed SB target x0
        desi       : (B, C_desi, H, W), transformed SB source / DET input
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
    # Build the bridge endpoint and conditioning in SB mode.
    # ----------------------------------------------------------

    def build_x1(self, x0: torch.Tensor, desi: torch.Tensor) -> torch.Tensor:
        if self.x1_mode == "desi_mean":
            return desi.mean(dim=1, keepdim=True)
        elif self.x1_mode == "gaussian":
            return torch.randn_like(x0)
        elif self.x1_mode == "rz":
            return desi[:, 1, :, :].unsqueeze(1)
        else:
            raise ValueError(f"Unknown x1_mode='{self.x1_mode}'")

    def build_cond(self, desi: torch.Tensor) -> torch.Tensor:
        if self.x1_mode == "desi_mean":
            return desi[:, :-1, :, :] - desi[:, 1:, :, :]
        elif self.x1_mode == "rz":
            return desi[:, 2, :, :].unsqueeze(1)
        else:
            raise ValueError(f"Unknown x1_mode='{self.x1_mode}'")

    def build_desi_mean_err(self, desi_err: torch.Tensor) -> torch.Tensor:
        C_desi = desi_err.shape[1]
        return (desi_err.pow(2).sum(dim=1, keepdim=True)).sqrt() / C_desi

    # ----------------------------------------------------------
    # [EXT-2] Heteroscedastic weight map, SB mode only
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
    # Label / pred_x0 computation in SB mode
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
    # Network calls
    # ----------------------------------------------------------

    def run_network(self, xt, step, cond, log_w=None):
        "SB mode: UNet(xt ⊕ cond [⊕ log_w], t) → score."
        parts = [xt, cond]
        if log_w is not None:
            parts.append(log_w)
        x_in = torch.cat(parts, dim=1)
        t = self.noise_levels[step]
        return self.model(x_in, t)

    def run_network_det(self, desi: torch.Tensor) -> torch.Tensor:
        """
        Deterministic UNet mode: UNet(desi[:, 1:]) → pred_x0.

        The caller supplies the already sliced desi[:, 1:, :, :] tensor,
        retaining the r/z channels for x1_mode='rz'. A zero time embedding
        preserves compatibility with the time-conditioned UNetModel interface.
        """
        B = desi.shape[0]
        t = torch.zeros(B, device=desi.device, dtype=torch.long)
        return self.model(desi, t)

    # ----------------------------------------------------------
    # EMA context
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
    # Metrics and loss functions
    # ----------------------------------------------------------

    def get_reduced_chi2(self, gt, recon, euclid_err):
        "Pixelwise reduced χ², used as a monitoring metric in both modes."
        if euclid_err is None:
            return torch.tensor(0.0, device=gt.device)
        chi2 = ((gt - recon) ** 2 / euclid_err.clamp(min=1e-10) ** 2).sum()
        dof = gt.numel()
        return chi2 / dof if dof > 0 else chi2
    def get_reduced_chi2_seg(self, gt, recon, euclid_err):
        "Pixelwise reduced χ², used as a monitoring metric in both modes."
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
        SB diffusion loss: regression on score/epsilon or x₀.

        Parameters
        ----------
        pred, label : (B, 1, H, W), network prediction and target
        pixel_mask  : (B, 1, H, W) bool/float, optional
        x0          : (B, 1, H, W), optional transformed and normalized Euclid GT
            If self.sb_bright_weight>0, apply per-pixel weights
            w = 1 / (|x0| + sb_bright_weight).
        euclid_err  : (B, 1, H, W) or None
            If self.sb_chi2_loss=True and this is provided, use a χ² form:
            (pred-label)²/σ², where σ=euclid_err. For x₀ prediction this is
            the pixel error; for epsilon prediction the same map is used
            approximately as a per-pixel scaling weight.

        self.sb_loss_type selects "mse" or "huber". self.sb_chi2_loss
        controls χ² weighting and can be combined with either loss.
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

        # χ² weighting: divide by per-pixel σ²; takes precedence over sb_bright_weight.
        if self.sb_chi2_loss and euclid_err is not None:
            sigma2 = euclid_err.clamp(min=1e-10).pow(2)
            err = err / sigma2
            w = None  # χ² already includes weights; do not add bright-source weighting.
        elif self.sb_bright_weight > 0.0 and x0 is not None:
            # Bright-source weights approximate a signal-dependent variance correction.
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
        Pixel-space loss for deterministic UNet mode.

        Parameters
        ----------
        pred        : (B, 1, H, W), network prediction
        target      : (B, 1, H, W), ground truth x0 in transformed space
        euclid_err  : (B, 1, H, W) or None, required only for chi2
        pixel_mask  : (B, 1, H, W) bool or None

        loss_type
        ---------
        "l2"   : MSE(pred, target)
        "l1"   : MAE(pred, target)
        "chi2" : sum((pred - target)² / σ²) / N_valid
                 Falls back to l2 with a warning if euclid_err is None.
        """
        # Construct mask weights.
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
                    "[det_loss/chi2] euclid_err is None; falling back to l2. "
                    "Check the data pipeline or use det_loss_type=l2."
                )
                err = (pred - target).pow(2)
            else:
                sigma2 = euclid_err.clamp(min=1e-10).pow(2)
                err = (pred - target).pow(2) / sigma2

        else:
            raise ValueError(f"Unknown det_loss_type='{loss_type}'")

        if self.det_loss_reduction == "mean":
            return (err * mask_f).sum() / n_valid
        else:  # "sum"
            return (err * mask_f).sum()

    # ----------------------------------------------------------
    # GAN losses, available in both modes
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
            raise ValueError(f"Unknown disc_loss_type='{self.disc_loss_type}'")
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
            raise ValueError(f"Unknown disc_loss_type='{self.disc_loss_type}'")

    # ----------------------------------------------------------
    # Section 3: Training steps dispatched by mode
    # ----------------------------------------------------------

    def on_train_start(self, *args, **kwargs):
        if self.use_i2sb:
            _ = self.diffusion

    def training_step(self, batch: Dict, batch_idx: int):
        if self.use_i2sb:
            return self._training_step_sb(batch, batch_idx)
        else:
            return self._training_step_det(batch, batch_idx)

    # ---- SB training step, original logic ----

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

        # -- Generator --
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

        # -- Discriminator --
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

        # Return a scalar loss without adversarial training; Lightning optimizes automatically.
        return gen_loss if not self.adversarial else None

    # ---- Deterministic UNet training step ----

    def _training_step_det(self, batch: Dict, batch_idx: int):
        """
        Training step for direct deterministic UNet regression.

        Forward: pred_x0 = UNet(desi)
        Loss: det_loss(pred_x0, euclid, euclid_err, pixel_mask)
              + optional [EXT-3] adv_weight * adv_loss
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

        # Forward using only desi[:, 1:], consistent with x1_mode='rz'.
        pred_x0 = self.run_network_det(desi[:, 1:, :, :])

        # -- Generator --
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

        # -- Discriminator --
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

        # Return a scalar loss without adversarial training; Lightning optimizes automatically.
        return gen_loss if not self.adversarial else None

    # ----------------------------------------------------------
    # Validation steps dispatched by mode
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
    # Helper: advance the scheduler
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
    # EMA updates
    # ----------------------------------------------------------

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    # ----------------------------------------------------------
    # Inference sampling in SB mode
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
        SB mode: full reverse trajectory x₁ → x₀.
        Returns (xs, pred_x0s), each with shape (B, log_count, 1, H, W).
        """
        assert self.use_i2sb, "sample() is available only when use_i2sb=True; use predict() in deterministic mode."
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
        # print("[DEBUG sample] log_w is None:", hetero["log_w"] is None)  # Optional diagnostic
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
    # [SPEEDUP] Efficient inference: shared-prefix forking and parallel sample batches
    # ------------------------------------------------------------
    # Differences from sample():
    #   1. No ema_scope(): the inference script should swap EMA weights once outside
    #      the loop, avoiding repeated store/copy_to operations for every batch.
    #   2. For num_samples > 1, use shared-prefix forking (see SBDiffusion.
    #      ddpm_sampling_forked), returning (num_samples, B, 1, H, W).
    #   3. Compatible with torch.compile(model.model): UNet forward sees only
    #      two batch sizes, B in the prefix and num_samples*B in the suffix,
    #      which favors reuse of the compile cache.
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
        Parameters
        ----------
        num_samples : Number of repeated samples, parallel after a shared prefix
        split_ratio : Shared-prefix fraction in [0, 1]
            0.0: fully independent, equivalent to num_samples independent samples
            0.7: suggested starting point; share 70% of steps, then fork for 30%
            1.0: fully shared, equivalent to one sample with std=0; avoid this setting

        Returns
        -------
        final_x0 : (num_samples, B, 1, H, W), final x0 estimates in transformed space
        """
        assert self.use_i2sb, "sample_forked() is available only when use_i2sb=True."
        assert num_samples >= 1
        assert 0.0 <= split_ratio <= 1.0

        nfe   = nfe or self.nfe
        steps = space_indices(self.interval, nfe + 1)
        log_count = min(len(steps) - 1, self.log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        # Fork after sharing the first split_ratio fraction of step pairs.
        n_pairs = len(steps) - 1  # = nfe
        split_step = int(round(split_ratio * n_pairs))

        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        # ---- Heteroscedastic weights: one copy, broadcast for the suffix ----
        hetero = self._build_hetero(
            x1=x1,
            desi_err=desi_err.to(self.device) if desi_err is not None else None,
        )
        sqrt_w, log_w = hetero["sqrt_w"], hetero["log_w"]

        # ---- Broadcast cond / sqrt_w / log_w for the forked suffix ----
        # Expand (B, ...) to (num_samples*B, ...) only when num_samples>1.
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

        # ---- Separate pred_x0 closures for the two shapes to reuse compiled graphs ----
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
        Deterministic UNet inference: desi → pred_x0 in transformed space.
        Call pixel_transform.inverse() to restore physical units.
        """
        assert not self.use_i2sb, "predict() is available only when use_i2sb=False; use sample() in SB mode."
        with self.ema_scope():
            self.model.eval()
            return self.run_network_det(desi.to(self.device)[:, 1:, :, :])

    # ----------------------------------------------------------
    # Image logging
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
            # x1 is one band extracted by build_x1; select it from the inverse-transformed desi_vis.
            _x1_band = 1 if self.x1_mode == "rz" else 0  # rz → r (index 1); desi_mean → approximate with the first channel
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
    # Optimizers / schedulers
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