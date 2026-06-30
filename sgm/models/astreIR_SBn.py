# astreIR_SBn.py — Multimodal DESI -> Euclid generative model
#
# Operating mode: Image-to-Image Schrodinger Bridge (I2SB), reference NVlabs/I2SB
#
# [EXT-1] Pixel stretch (pixel_transform) -- dynamic range compression on data side
#
# Training data flow:
#   batch -> get_input() -> euclid, desi, err (transform domain)
#     -> build_x1 / build_cond
#     -> diffusion.q_sample() -> xt
#     -> run_network(xt, step, cond) -> pred
#     -> sb_loss(pred, label)

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

logpy = logging.getLogger(__name__)


# ============================================================
# §0  Constants and utilities (I2SB diffusion)
# ============================================================

def _compute_gaussian_product_coef(sigma1: np.ndarray, sigma2: np.ndarray):
    """
    Product coefficients of two Gaussians (I2SB eq. 10).

    Given p1 = N(x_t | x_0, sigma1^2) and p2 = N(x_t | x_1, sigma2^2),
    computes p1*p2 = N(x_t | coef1*x0 + coef2*x1, var).
    """
    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom
    return coef1, coef2, var


def _unsqueeze_xdim(z: torch.Tensor, xdim) -> torch.Tensor:
    """Broadcast (B,) tensor to (B, 1, 1, ...) for per-pixel ops."""
    bc = (...,) + (None,) * len(xdim)
    return z[bc]


def make_sb_betas(n_timestep: int = 1000, linear_end: float = 2e-2) -> np.ndarray:
    """Symmetric beta schedule used by I2SB (equivalent to NVlabs/I2SB make_beta_schedule)."""
    linear_start = 1e-10
    assert linear_end >= linear_start, (
        f"linear_end={linear_end:.2e} < linear_start={linear_start:.2e}. "
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
    """Uniformly select count indices from [0, num_steps-1]."""
    assert count <= num_steps
    frac = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur, taken = 0.0, []
    for _ in range(count):
        taken.append(round(cur))
        cur += frac
    return taken


# ============================================================
# §1  Schrodinger Bridge diffusion process
# ============================================================

class SBDiffusion:
    """I2SB Schrodinger Bridge diffusion process."""

    def __init__(self, betas: np.ndarray, device: torch.device):
        self.device = device

        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = _compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        to_t = partial(torch.tensor, dtype=torch.float32)
        self.std_fwd = to_t(std_fwd).to(device)
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
    ) -> torch.Tensor:
        assert x0.shape == x1.shape
        _, *xdim = x0.shape
        mu0  = _unsqueeze_xdim(self.mu_x0[step], xdim)
        mu1  = _unsqueeze_xdim(self.mu_x1[step], xdim)
        s_sb = _unsqueeze_xdim(self.std_sb[step], xdim)
        xt = mu0 * x0 + mu1 * x1
        if not ot_ode:
            xt = xt + s_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(
        self,
        nprev: int,
        n: int,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n ** 2 - std_nprev ** 2).sqrt()
        mu_x0, mu_xn, var = _compute_gaussian_product_coef(std_nprev, std_delta)
        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
        return xt_prev

    def ddpm_sampling(
        self,
        steps: List[int],
        pred_x0_fn,
        x1: torch.Tensor,
        ot_ode: bool = False,
        log_steps: Optional[List[int]] = None,
        verbose: bool = True,
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
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
        stack = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack(xs), stack(pred_x0s)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shared-prefix forked sampling for efficient multi-sample inference.

        num_samples trajectories share the first split_step reverse steps
        (deterministic-dominated phase), then the batch dimension is tiled
        to num_samples*B and the remaining steps run independently.

        Total UNet forwards: split_step*B + (n_pairs - split_step)*num_samples*B
        vs. num_samples*n_pairs*B for fully independent sampling.
        Example: N=50, K=35, ns=3 -> 80B vs 150B (saves 47%).

        Args:
            pred_x0_fn_shared: (xt[B,...], step_int) -> pred_x0[B,...]  (shared phase)
            pred_x0_fn_forked: (xt[ns*B,...], step_int) -> pred_x0[ns*B,...] (forked phase)
            split_step: number of reverse pairs to run in shared mode before forking
        Returns:
            final_x0  : (num_samples, B, 1, H, W)
            bridge_xs : (B, log_count, 1, H, W)  for visualisation
        """
        B = x1.shape[0]
        device = self.device
        xt = x1.detach().to(device)
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        rev = steps[::-1]
        pairs = list(zip(rev[1:], rev[:-1]))
        n_pairs = len(pairs)
        split_step = max(0, min(n_pairs, int(split_step)))

        xs_bridge = []
        log_steps_set = set(log_steps)

        # Phase 1: shared prefix (batch=B)
        iter1 = pairs[:split_step]
        if verbose and len(iter1) > 0:
            iter1 = tqdm(iter1, desc=f"SB shared prefix (B={B})", total=len(iter1), leave=False)
        for prev_step, step in iter1:
            pred_x0 = pred_x0_fn_shared(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            if prev_step in log_steps_set:
                xs_bridge.append(xt.detach().cpu())

        # Fork: tile batch dimension to num_samples*B
        if num_samples > 1 and split_step < n_pairs:
            xt = xt.unsqueeze(0).expand(num_samples, *xt.shape).contiguous()
            xt = xt.reshape(num_samples * B, *xt.shape[2:])

        # Phase 2: forked suffix (batch=ns*B)
        iter2 = pairs[split_step:]
        if verbose and len(iter2) > 0:
            iter2 = tqdm(iter2, desc=f"SB forked suffix (B={xt.shape[0]})", total=len(iter2), leave=False)
        for prev_step, step in iter2:
            pred_x0 = pred_x0_fn_forked(xt, step) if (num_samples > 1 and split_step < n_pairs) \
                      else pred_x0_fn_shared(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            if prev_step in log_steps_set and num_samples > 1 and split_step < n_pairs:
                xs_bridge.append(xt[:B].detach().cpu())
            elif prev_step in log_steps_set:
                xs_bridge.append(xt.detach().cpu())

        # Reshape output
        if num_samples > 1 and split_step < n_pairs:
            final_x0 = xt.reshape(num_samples, B, *xt.shape[1:])
        else:
            final_x0 = xt.unsqueeze(0).expand(num_samples, *xt.shape).contiguous() \
                       if num_samples > 1 else xt.unsqueeze(0)

        if len(xs_bridge) > 0:
            bridge_xs = torch.flip(torch.stack(xs_bridge, dim=1), dims=(1,))
        else:
            bridge_xs = xt[:B].detach().cpu().unsqueeze(1)

        return final_x0, bridge_xs


# ============================================================
# §2  Main Lightning module
# ============================================================

class MultiModalSBDiffusion(pl.LightningModule):
    """Multimodal DESI -> Euclid I2SB Schrodinger Bridge model."""

    def __init__(
        self,
        # Network
        network_config: Dict,

        # Original I2SB parameters (cf. NVlabs/I2SB)
        interval: int = 1000,
        beta_max: float = 0.3,
        ot_ode: bool = False,
        nfe: int = 100,
        log_count: int = 10,

        # [EXT-1] Pixel stretch
        pixel_transform_config: Union[None, Dict, ListConfig, OmegaConf] = None,

        # Optimizer / scheduler
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,

        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.9999,

        # Checkpoint
        ckpt_path: Union[None, str] = None,

        # Input key names
        input_key_euclid: str = "euclid_img",
        input_key_desi: str = "desi_img",
        input_key_desi_error: str = "desi_error",
        input_key_euclid_error: str = "euclid_error",
        input_key_pixel_mask: str = "pixel_mask",

        **kwargs,
    ):
        super().__init__()

        self.input_key_euclid       = input_key_euclid
        self.input_key_desi         = input_key_desi
        self.input_key_desi_error   = input_key_desi_error
        self.input_key_euclid_error = input_key_euclid_error
        self.input_key_pixel_mask   = input_key_pixel_mask

        self.interval  = interval
        self._beta_max = beta_max
        self.ot_ode    = ot_ode
        self.nfe       = nfe
        self.log_count    = log_count

        # [EXT-1] Pixel stretch
        if pixel_transform_config is not None:
            self.pixel_transform: BasePixelTransform = instantiate_from_config(pixel_transform_config)
        else:
            self.pixel_transform = IdentityTransform()
        logpy.info(f"[EXT-1] pixel transform: {self.pixel_transform.__class__.__name__}")

        # Dynamic in_channels: xt(1) + cond_z(1)
        _in_ch = 2

        if isinstance(network_config, dict):
            network_config = dict(network_config)
            if "params" in network_config:
                network_config["params"] = dict(network_config["params"])
                network_config["params"]["in_channels"] = _in_ch
        else:
            from omegaconf import OmegaConf as _OC
            network_config = _OC.to_container(network_config, resolve=True)
            network_config["params"]["in_channels"] = _in_ch

        self.model: nn.Module = instantiate_from_config(network_config)

        noise_levels = torch.linspace(1e-4, 1.0, interval) * interval
        self.register_buffer("noise_levels", noise_levels)

        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay)
            logpy.info(f"[EMA] tracking {len(list(self.model_ema.buffers()))} buffers")

        self._sb: Optional[SBDiffusion] = None

        self.optimizer_config = default(optimizer_config, {"target": "torch.optim.AdamW"})
        self.scheduler_config = scheduler_config

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    # ----------------------------------------------------------
    # SB diffusion object property
    # ----------------------------------------------------------

    @property
    def diffusion(self) -> SBDiffusion:
        if self._sb is None:
            betas = make_sb_betas(
                n_timestep=self.interval,
                linear_end=self._beta_max / self.interval,
            )
            self._sb = SBDiffusion(betas, self.device)
        return self._sb

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------

    def init_from_ckpt(self, path: str) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError(f"Unsupported checkpoint format: {path}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        logpy.info(f"Restored from {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # ----------------------------------------------------------
    # Data input and transform
    # ----------------------------------------------------------

    def get_input(self, batch: Dict):
        """
        Extract data from batch and apply [EXT-1] pixel stretch + normalisation.

        Returns
        -------
        euclid     : (B, 1, H, W)      transform domain (SB target x0)
        desi       : (B, C_desi, H, W) transform domain (SB source)
        euclid_err : (B, 1, H, W) or None
        pixel_mask : (B, 1, H, W) bool or None
        """
        euclid_raw     = batch.get(self.input_key_euclid)
        desi_raw       = batch.get(self.input_key_desi)
        if desi_raw is None:
            desi_raw = batch.get("images")
        euclid_err_raw = batch.get(self.input_key_euclid_error, None)
        pixel_mask     = batch.get(self.input_key_pixel_mask, None)

        pt = self.pixel_transform
        euclid = pt.forward(euclid_raw, source="euclid") if euclid_raw is not None else None
        desi   = pt.forward(desi_raw, source="desi")

        euclid_err = None
        if euclid_err_raw is not None and euclid_raw is not None:
            euclid_err = pt.transform_error(euclid_err_raw, euclid_raw, source="euclid")

        norm_eu = pt._get_norm("euclid")

        if euclid is not None:
            euclid = pt.normalize(euclid, source="euclid")
            if euclid_err is not None and norm_eu is not None:
                euclid_err = euclid_err / norm_eu[1]

        desi = pt.normalize(desi, source="desi")

        return euclid, desi, euclid_err, pixel_mask

    # ----------------------------------------------------------
    # Build bridge endpoints and conditioning
    # ----------------------------------------------------------

    def build_x1(self, x0: torch.Tensor, desi: torch.Tensor) -> torch.Tensor:
        """x1 = r-band (index 1)"""
        return desi[:, 1, :, :].unsqueeze(1)

    def build_cond(self, desi: torch.Tensor) -> torch.Tensor:
        """cond = z-band (index 2)"""
        return desi[:, 2, :, :].unsqueeze(1)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # Score matching helpers
    # ----------------------------------------------------------

    def compute_label(self, step, x0, xt):
        """Score target: (xt - x0) / std_fwd[step]"""
        std = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        return ((xt - x0) / std).detach()

    def compute_pred_x0(self, step, xt, net_out):
        """Recover x0 from network score output: xt - std_fwd * net_out"""
        std = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        return xt - std * net_out

    # ----------------------------------------------------------
    # Network forward
    # ----------------------------------------------------------

    def run_network(self, xt, step, cond):
        """UNet(xt || cond, t) -> score = (xt - x0) / std_fwd"""
        x_in = torch.cat([xt, cond], dim=1)
        t = self.noise_levels[step]
        return self.model(x_in, t)

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
        """Per-pixel reduced chi-squared (monitoring metric)."""
        if euclid_err is None:
            return torch.tensor(0.0, device=gt.device)
        chi2 = ((gt - recon) ** 2 / euclid_err.clamp(min=1e-10) ** 2).sum()
        dof = gt.numel()
        return chi2 / dof if dof > 0 else chi2

    def sb_loss(self, pred, label, pixel_mask=None):
        """MSE on score prediction. pred, label: (B, 1, H, W)"""
        err = (pred - label).pow(2)
        if pixel_mask is None:
            return err.mean()
        mask_f = pixel_mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        return (err * mask_f).sum() / denom

    # ----------------------------------------------------------
    # §3  Training step
    # ----------------------------------------------------------

    def on_train_start(self, *args, **kwargs):
        _ = self.diffusion

    def training_step(self, batch: Dict, batch_idx: int):
        euclid, desi, euclid_err, pixel_mask = self.get_input(batch)
        x0, desi = euclid.to(self.device), desi.to(self.device)
        x1, cond = self.build_x1(x0, desi), self.build_cond(desi)

        if euclid_err is not None: euclid_err = euclid_err.to(self.device)
        if pixel_mask is not None: pixel_mask = pixel_mask.to(self.device)

        B     = x0.shape[0]
        step  = torch.randint(0, self.interval, (B,), device=self.device)
        xt    = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode)
        label = self.compute_label(step, x0, xt)
        pred  = self.run_network(xt, step, cond)
        loss  = self.sb_loss(pred, label, pixel_mask=pixel_mask)

        with torch.no_grad():
            pred_x0      = self.compute_pred_x0(step, xt, pred)
            reduced_chi2 = self.get_reduced_chi2(x0, pred_x0.detach(), euclid_err)

        self.log("train/diff_loss",    loss,         prog_bar=True)
        self.log("train/reduced_chi2", reduced_chi2, prog_bar=True)

        return loss

    # ----------------------------------------------------------
    # Validation step
    # ----------------------------------------------------------

    def validation_step(self, batch: Dict, batch_idx: int):
        euclid, desi, euclid_err, pixel_mask = self.get_input(batch)
        x0   = euclid.to(self.device)
        desi = desi.to(self.device)
        x1   = self.build_x1(x0, desi)
        cond = self.build_cond(desi)

        if euclid_err is not None: euclid_err = euclid_err.to(self.device)
        if pixel_mask is not None: pixel_mask = pixel_mask.to(self.device)

        B     = x0.shape[0]
        step  = torch.randint(0, self.interval, (B,), device=self.device)
        xt    = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode)
        label = self.compute_label(step, x0, xt)

        with self.ema_scope():
            pred = self.run_network(xt, step, cond)

        val_loss     = self.sb_loss(pred, label, pixel_mask=pixel_mask)
        pred_x0      = self.compute_pred_x0(step, xt, pred)
        reduced_chi2 = self.get_reduced_chi2(x0, pred_x0, euclid_err)

        self.log("val/loss",         val_loss,     on_epoch=True, sync_dist=True)
        self.log("val/reduced_chi2", reduced_chi2, on_epoch=True, sync_dist=True)
        return val_loss

    # ----------------------------------------------------------
    # EMA update
    # ----------------------------------------------------------

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    # ----------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
        nfe: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full reverse trajectory x1 -> x0.
        Returns (xs, pred_x0s) shaped (B, log_count, 1, H, W).
        """
        nfe   = nfe or self.nfe
        steps = space_indices(self.interval, nfe + 1)
        log_count = min(len(steps) - 1, self.log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        with self.ema_scope():
            self.model.eval()

            def pred_x0_fn(xt, step_int):
                step_t = torch.full(
                    (xt.shape[0],), step_int, device=self.device, dtype=torch.long
                )
                net_out = self.run_network(xt, step_t, cond)
                return self.compute_pred_x0(step_t, xt, net_out)

            xs, pred_x0s = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1,
                ot_ode=self.ot_ode,
                log_steps=log_steps,
                verbose=verbose,
            )

        return xs, pred_x0s

    @torch.no_grad()
    def sample_forked(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
        num_samples: int = 1,
        split_ratio: float = 0.7,
        nfe: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Efficient multi-sample inference using shared-prefix forked sampling.

        Does not enter ema_scope (caller should swap EMA weights once before the loop).
        When num_samples > 1, uses ddpm_sampling_forked to share the first
        split_ratio fraction of steps, returning (num_samples, B, 1, H, W).

        Args:
            num_samples  : number of repeated samples (run in parallel via batch tiling)
            split_ratio  : shared-prefix fraction in [0, 1]
                           0.0 = fully independent baseline (for std calibration)
                           0.7 = recommended production value

        Returns:
            final_x0 : (num_samples, B, 1, H, W) in transform domain
        """
        assert num_samples >= 1
        assert 0.0 <= split_ratio <= 1.0

        nfe   = nfe or self.nfe
        steps = space_indices(self.interval, nfe + 1)
        log_count = min(len(steps) - 1, self.log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        n_pairs = len(steps) - 1
        split_step = int(round(split_ratio * n_pairs))

        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        def _expand_ns(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None or num_samples == 1:
                return t
            return t.unsqueeze(0).expand(num_samples, *t.shape).reshape(
                num_samples * t.shape[0], *t.shape[1:]
            ).contiguous()

        cond_forked = _expand_ns(cond)

        self.model.eval()

        def pred_x0_fn_shared(xt, step_int):
            step_t = torch.full((xt.shape[0],), step_int, device=self.device, dtype=torch.long)
            net_out = self.run_network(xt, step_t, cond)
            return self.compute_pred_x0(step_t, xt, net_out)

        def pred_x0_fn_forked(xt, step_int):
            step_t = torch.full((xt.shape[0],), step_int, device=self.device, dtype=torch.long)
            net_out = self.run_network(xt, step_t, cond_forked)
            return self.compute_pred_x0(step_t, xt, net_out)

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
        )

        return final_x0  # (num_samples, B, 1, H, W)

    # ----------------------------------------------------------
    # Image logging
    # ----------------------------------------------------------

    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict:
        log = {}
        euclid, desi, euclid_err, pixel_mask = self.get_input(batch)
        x0   = euclid.to(self.device)
        desi = desi.to(self.device)
        pt   = self.pixel_transform

        def to3(t): return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t[:, :3]

        x1   = self.build_x1(x0, desi)
        cond = self.build_cond(desi)
        xs, _ = self.sample(x1, cond, verbose=False)
        generated = xs[:, 0].to(self.device)

        x0_vis        = pt.inverse(pt.denormalize(x0,        source="euclid"), source="euclid")
        generated_vis = pt.inverse(pt.denormalize(generated, source="euclid"), source="euclid")
        desi_vis = pt.inverse(pt.denormalize(desi[:, :3], source="desi"), source="desi")
        x1_vis   = to3(desi_vis[:, 1:2, :, :])  # r-band
        eu_vis   = to3(x0_vis)
        gen_vis  = to3(generated_vis)
        residual = eu_vis - gen_vis

        log["comparison"] = torch.cat([desi_vis, x1_vis, eu_vis, gen_vis, residual], dim=-1)

        B, L, C, H, W = xs.shape
        xs_vis = pt.inverse(
            pt.denormalize(xs.to(self.device).reshape(B * L, C, H, W), source="euclid"),
            source="euclid",
        )
        log["bridge_trajectory"] = make_grid(to3(xs_vis), nrow=L)

        return log

    # ----------------------------------------------------------
    # Optimizer / scheduler
    # ----------------------------------------------------------

    def _build_optimizer(self, params, cfg: Dict):
        return get_obj_from_str(cfg["target"])(params, **cfg.get("params", {}))

    def configure_optimizers(self):
        params_g = list(self.model.parameters())
        opt_g    = self._build_optimizer(params_g, self.optimizer_config)

        if self.scheduler_config is not None:
            sched_fn = instantiate_from_config(self.scheduler_config)
            sched = {
                "scheduler": LambdaLR(opt_g, lr_lambda=sched_fn.schedule),
                "interval":  "step",
                "frequency": 1,
            }
            return [opt_g], [sched]

        return opt_g
