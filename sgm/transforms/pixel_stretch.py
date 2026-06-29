import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

import numpy as np
import torch

logpy = logging.getLogger(__name__)


class BasePixelTransform(ABC):
    """Abstract base class for nonlinear pixel value transforms."""

    def __init__(
        self,
        a_euclid: float = 0.001,
        a_desi: float = 0.001,
        x_max: float = 10.0,
        scale: Optional[float] = None,
        norm_mean_transformed_euclid: Optional[float] = None,
        norm_std_transformed_euclid: Optional[float] = None,
        norm_mean_transformed_desi: Optional[float] = None,
        norm_std_transformed_desi: Optional[float] = None,
        clip_x_max_euclid: Optional[float] = None,
        clip_x_max_desi: Optional[float] = None,
        clip_x_min_euclid: Optional[float] = None,
        clip_x_min_desi: Optional[float] = None,
    ):
        self.a_euclid = a_euclid
        self.a_desi = a_desi
        self.x_max = x_max

        a_for_scale = min(a_euclid, a_desi)
        self.scale = scale if scale is not None else self._compute_scale(x_max, a_for_scale)

        self._clip_max = {"euclid": clip_x_max_euclid, "desi": clip_x_max_desi}
        self._clip_min = {"euclid": clip_x_min_euclid, "desi": clip_x_min_desi}

        self._norm: Dict[str, Optional[Tuple[float, float]]] = {"euclid": None, "desi": None}
        for src, mu_t, sig_t in [
            ("euclid", norm_mean_transformed_euclid, norm_std_transformed_euclid),
            ("desi",   norm_mean_transformed_desi,   norm_std_transformed_desi),
        ]:
            if mu_t is not None and sig_t is not None:
                if sig_t <= 0:
                    raise ValueError(f"[{src}] norm_std_transformed must be > 0, got {sig_t}")
                self._norm[src] = (float(mu_t), float(sig_t))
            elif (mu_t is None) != (sig_t is None):
                raise ValueError(f"[{src}] norm_mean and norm_std must both be provided or both be None")

    def _get_a(self, source: str) -> float:
        if source == "euclid":
            return self.a_euclid
        elif source == "desi":
            return self.a_desi
        else:
            raise ValueError(f"Unknown source='{source}'; expected 'euclid' or 'desi'")

    @abstractmethod
    def _compute_scale(self, x_max: float, a: float) -> float: ...

    @staticmethod
    @abstractmethod
    def _forward_torch(x: torch.Tensor, a: float, scale: float) -> torch.Tensor: ...

    @staticmethod
    @abstractmethod
    def _inverse_torch(y: torch.Tensor, a: float, scale: float) -> torch.Tensor: ...

    @staticmethod
    @abstractmethod
    def _jacobian_torch(x: torch.Tensor, a: float, scale: float) -> torch.Tensor: ...

    def _clip_raw_torch(self, x: torch.Tensor, source: str) -> torch.Tensor:
        lo = self._clip_min.get(source)
        hi = self._clip_max.get(source)
        if lo is None and hi is None:
            return x
        return x.clamp(min=lo, max=hi)

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        x = self._clip_raw_torch(x, source)
        return self._forward_torch(x, self._get_a(source), self.scale)

    def inverse(self, y: torch.Tensor, source: str) -> torch.Tensor:
        return self._inverse_torch(y, self._get_a(source), self.scale)

    def transform_error(self, err: torch.Tensor, x: torch.Tensor, source: str) -> torch.Tensor:
        jac = self._jacobian_torch(x, self._get_a(source), self.scale)
        return jac.abs() * err

    def _get_norm(self, source: str) -> Optional[Tuple[float, float]]:
        return self._norm.get(source)

    def normalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        ns = self._get_norm(source)
        if ns is None:
            return y
        mu, sigma = ns
        return (y - mu) / sigma

    def denormalize(self, y_norm: torch.Tensor, source: str) -> torch.Tensor:
        ns = self._get_norm(source)
        if ns is None:
            return y_norm
        mu, sigma = ns
        return y_norm * sigma + mu


class IdentityTransform(BasePixelTransform):
    """Identity transform (no stretching). Fallback when pixel_transform_config is None."""

    def _compute_scale(self, x_max: float, a: float) -> float:
        return 1.0

    @staticmethod
    def _forward_torch(x, a, scale): return x

    @staticmethod
    def _inverse_torch(y, a, scale): return y

    @staticmethod
    def _jacobian_torch(x, a, scale): return torch.ones_like(x)


class BandwiseArcsinhTransform:
    """
    Per-band standardized arcsinh stretch.

    Each band independently applies:
        1. clip(x, lower_bound[c], upper_bound[c])
        2. out = arcsinh((clipped - bg_median[c]) / bg_std[c])
        3. Linear normalization to [-1, 1]
    """

    def __init__(self, euclid: dict, desi: dict):
        self._params = {}
        for src, cfg in [("euclid", euclid), ("desi", desi)]:
            lb  = np.array(cfg["lower_bound"], dtype=np.float64)
            ub  = np.array(cfg["upper_bound"],  dtype=np.float64)
            mu  = np.array(cfg["bg_median"],    dtype=np.float64)
            sig = np.array(cfg["bg_std"],       dtype=np.float64)
            vmin = np.arcsinh((lb - mu) / sig)
            vmax = np.arcsinh((ub - mu) / sig)
            span = vmax - vmin
            self._params[src] = dict(lb=lb, ub=ub, mu=mu, sig=sig,
                                     vmin=vmin, vmax=vmax, span=span)
            logpy.info(f"[BandwiseArcsinh/{src}] vmin={vmin}, vmax={vmax}, span={span}")

    def _p(self, source: str):
        if source not in self._params:
            raise ValueError(f"Unknown source '{source}', expected 'euclid' or 'desi'")
        return self._params[source]

    def _broadcast_torch(self, x: torch.Tensor, vec: np.ndarray) -> torch.Tensor:
        c = x.shape[1] if x.ndim == 4 else x.shape[0]
        t = torch.tensor(vec, dtype=x.dtype, device=x.device)
        if t.numel() == 1:
            return t
        if t.numel() != c:
            raise ValueError(f"Parameter channel count {t.numel()} does not match image channel count {c}")
        if x.ndim == 4:
            return t.view(1, -1, 1, 1)
        return t.view(-1, 1, 1)

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        p = self._p(source)
        lb   = self._broadcast_torch(x, p["lb"])
        ub   = self._broadcast_torch(x, p["ub"])
        mu   = self._broadcast_torch(x, p["mu"])
        sig  = self._broadcast_torch(x, p["sig"])
        vmin = self._broadcast_torch(x, p["vmin"])
        span = self._broadcast_torch(x, p["span"])
        clipped = x.clamp(min=lb, max=ub)
        out = torch.arcsinh((clipped - mu) / sig)
        return (out - vmin) / span * 2.0 - 1.0

    def inverse(self, y: torch.Tensor, source: str) -> torch.Tensor:
        p = self._p(source)
        mu   = self._broadcast_torch(y, p["mu"])
        sig  = self._broadcast_torch(y, p["sig"])
        vmin = self._broadcast_torch(y, p["vmin"])
        span = self._broadcast_torch(y, p["span"])
        out = (y + 1.0) / 2.0 * span + vmin
        return torch.sinh(out) * sig + mu

    def transform_error(self, err: torch.Tensor, x: torch.Tensor, source: str) -> torch.Tensor:
        p = self._p(source)
        lb   = self._broadcast_torch(x, p["lb"])
        ub   = self._broadcast_torch(x, p["ub"])
        mu   = self._broadcast_torch(x, p["mu"])
        sig  = self._broadcast_torch(x, p["sig"])
        span = self._broadcast_torch(x, p["span"])
        x_c = x.clamp(min=lb, max=ub)
        z = (x_c - mu) / sig
        jac = (1.0 / sig) / torch.sqrt(z ** 2 + 1.0) * (2.0 / span)
        return jac.abs() * err

    def _get_norm(self, source: str):
        return None

    def normalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        return y

    def denormalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        return y
