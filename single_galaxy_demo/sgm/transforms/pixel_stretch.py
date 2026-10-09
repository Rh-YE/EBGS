# ---------------------------------------------------------------
# pixel_stretch.py — General framework for nonlinear pixel-value transforms
#
# Compress astronomical-image dynamic range before I2SB training so that
# pixels with different brightnesses contribute comparably to the MSE loss.
#
# Design principles:
#   1. All bands share the same scale normalization constant.
#   2. Set a (softening / transition scale) separately for Euclid and DESI.
#   3. forward and inverse must be mutually inverse.
#   4. transform_error propagates errors using the analytic Jacobian.
#
# YAML usage:
#   pixel_transform:
#     target: sgm.transforms.pixel_stretch.SqrtTransform
#     params:
#       a_euclid: 0.001
#       a_desi: 0.001
#       x_max: 10.0
#
#   Set target to IdentityTransform to disable stretching and restore the original behavior.
#
# Architecture:
#   BasePixelTransform, the abstract base class
#     ├── forward(x, source) → x_norm      Forward transform
#     ├── inverse(x_norm, source) → x      Inverse transform
#     └── transform_error(err, x, source)  Error propagation
#          │
#     ┌────┼────────┬───────────┬──────────┐
#     ▼    ▼        ▼           ▼          ▼
#  Identity Arcsinh  Sqrt     Lupton    Power
# ---------------------------------------------------------------

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

import numpy as np
import torch

logpy = logging.getLogger(__name__)


class BasePixelTransform(ABC):
    """
    Abstract base class for nonlinear pixel-value transforms.

    Subclasses implement three static methods:
        _forward_np: NumPy forward transform for preprocessing scripts
        _inverse_np: NumPy inverse transform
        _jacobian_np: Derivative df/dx for error propagation

    The base class provides torch forward / inverse / transform_error methods
    and convenience interfaces for NumPy arrays.

    Parameters
    ----------
    a_euclid : float
        Softening parameter for Euclid images.
    a_desi : float
        Softening parameter for DESI images.
    x_max : float
        Reference value used to calculate scale so forward(x_max) ≈ 1.0.
    scale : float or None
        Explicit normalization constant, overriding x_max when supplied.
        All bands and data sources must share the same scale.
    """


    def __init__(
        self,
        a_euclid: float = 0.001,
        a_desi: float = 0.001,
        x_max: float = 10.0,
        scale: Optional[float] = None,
        # [EXT-NORM] Transformed-space statistics measured directly with mean/std after forward.
        norm_mean_transformed_euclid: Optional[float] = None,
        norm_std_transformed_euclid: Optional[float] = None,
        norm_mean_transformed_desi: Optional[float] = None,
        norm_std_transformed_desi: Optional[float] = None,
        # [VMAX-ALIGN] Symmetrically clip pixels in the original space before stretching.
        # This aligns maximum values at t=0 (Euclid) and t=T (DESI),
        # addressing depressed bright cores in the convex combination μ_t and dark intermediate residuals.
        # None disables clipping, preserving the old behavior; a percentile bound such as 99.9% can be used.
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

        # [EXT-NORM] Store transformed-space statistics without Jacobian propagation.
        self._norm: Dict[str, Optional[Tuple[float, float]]] = {"euclid": None, "desi": None}
        for src, mu_t, sig_t in [
            ("euclid", norm_mean_transformed_euclid, norm_std_transformed_euclid),
            ("desi",   norm_mean_transformed_desi,   norm_std_transformed_desi),
        ]:
            if mu_t is not None and sig_t is not None:
                if sig_t <= 0:
                    raise ValueError(
                        f"[{src}] norm_std_transformed must be > 0, got {sig_t}"
                    )
                self._norm[src] = (float(mu_t), float(sig_t))
                logpy.info(
                    f"[Normalization/{src}] Using transformed-space statistics: "
                    f"mu_t={mu_t:.6f}, sigma_t={sig_t:.6f}"
                )
            elif (mu_t is None) != (sig_t is None):
                raise ValueError(
                    f"[{src}] norm_mean and norm_std must both be supplied or both be None."
                )
            else:
                logpy.info(f"[Normalization/{src}] Not configured; normalize/denormalize will be identity operations.")

        logpy.info(
            f"[Pixel transform] {self.__class__.__name__}: "
            f"a_euclid={a_euclid}, a_desi={a_desi}, "
            f"x_max={x_max}, scale={self.scale:.6f}"
        )
    def _get_a(self, source: str) -> float:
        "Return the softening parameter a for the specified data source."
        if source == "euclid":
            return self.a_euclid
        elif source == "desi":
            return self.a_desi
        else:
            raise ValueError(f"Unknown source='{source}'; expected 'euclid' or 'desi'.")

    # ----------------------------------------------------------
    # Methods that subclasses must implement
    # ----------------------------------------------------------

    @abstractmethod
    def _compute_scale(self, x_max: float, a: float) -> float:
        "Calculate the normalization constant from x_max and a."
        ...

    @staticmethod
    @abstractmethod
    def _forward_np(x: np.ndarray, a: float, scale: float) -> np.ndarray:
        "NumPy forward transform from original to transformed space."
        ...

    @staticmethod
    @abstractmethod
    def _inverse_np(y: np.ndarray, a: float, scale: float) -> np.ndarray:
        "NumPy inverse transform from transformed to original space."
        ...

    @staticmethod
    @abstractmethod
    def _jacobian_np(x: np.ndarray, a: float, scale: float) -> np.ndarray:
        "NumPy derivative df/dx of the forward transform for error propagation."
        ...

    # ----------------------------------------------------------
    # Torch implementations derived from the NumPy versions
    # ----------------------------------------------------------

    @staticmethod
    @abstractmethod
    def _forward_torch(x: torch.Tensor, a: float, scale: float) -> torch.Tensor:
        "Torch forward transform."
        ...

    @staticmethod
    @abstractmethod
    def _inverse_torch(y: torch.Tensor, a: float, scale: float) -> torch.Tensor:
        "Torch inverse transform."
        ...

    @staticmethod
    @abstractmethod
    def _jacobian_torch(x: torch.Tensor, a: float, scale: float) -> torch.Tensor:
        "Torch derivative."
        ...

    # ----------------------------------------------------------
    # Public NumPy interface
    # ----------------------------------------------------------

    def _clip_raw_np(self, x: np.ndarray, source: str) -> np.ndarray:
        "[VMAX-ALIGN] Symmetric clipping in original space before stretching."
        lo = self._clip_min.get(source)
        hi = self._clip_max.get(source)
        if lo is None and hi is None:
            return x
        return np.clip(x, lo, hi)

    def _clip_raw_torch(self, x: torch.Tensor, source: str) -> torch.Tensor:
        lo = self._clip_min.get(source)
        hi = self._clip_max.get(source)
        if lo is None and hi is None:
            return x
        return x.clamp(min=lo, max=hi)

    def forward_np(self, x: np.ndarray, source: str) -> np.ndarray:
        "NumPy forward transform; source is 'euclid' or 'desi'."
        x = self._clip_raw_np(x, source)
        return self._forward_np(x, self._get_a(source), self.scale)

    def inverse_np(self, y: np.ndarray, source: str) -> np.ndarray:
        "NumPy inverse transform."
        return self._inverse_np(y, self._get_a(source), self.scale)

    def transform_error_np(self, err: np.ndarray, x: np.ndarray, source: str) -> np.ndarray:
        """
        NumPy error propagation.

        σ_transformed = |df/dx| * σ_original

        Parameters
        ----------
        err : Original uncertainty σ.
        x : Original pixel values at which to evaluate the Jacobian.
        """
        jac = self._jacobian_np(x, self._get_a(source), self.scale)
        return np.abs(jac) * err

    # ----------------------------------------------------------
    # Public torch interface
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        "Torch forward transform; source is 'euclid' or 'desi'."
        x = self._clip_raw_torch(x, source)
        return self._forward_torch(x, self._get_a(source), self.scale)

    def inverse(self, y: torch.Tensor, source: str) -> torch.Tensor:
        "Torch inverse transform."
        return self._inverse_torch(y, self._get_a(source), self.scale)

    def transform_error(self, err: torch.Tensor, x: torch.Tensor, source: str) -> torch.Tensor:
        "Torch error propagation: σ_transformed = |df/dx| * σ_original."
        jac = self._jacobian_torch(x, self._get_a(source), self.scale)
        return jac.abs() * err
    
    def _get_norm(self, source: str) -> Optional[Tuple[float, float]]:
        "Return transformed-space (mu, sigma), or None if not configured."
        return self._norm.get(source)

    def normalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        "Normalize in transformed space: (y - mu) / sigma."
        ns = self._get_norm(source)
        if ns is None:
            return y
        mu, sigma = ns
        return (y - mu) / sigma

    def denormalize(self, y_norm: torch.Tensor, source: str) -> torch.Tensor:
        "Denormalize: y * sigma + mu."
        ns = self._get_norm(source)
        if ns is None:
            return y_norm
        mu, sigma = ns
        return y_norm * sigma + mu

# ============================================================
# Concrete implementations
# ============================================================

class IdentityTransform(BasePixelTransform):
    """
    Identity transform without pixel stretching.

    Use this to disable stretching and recover the original I2SB behavior.
    """

    def _compute_scale(self, x_max: float, a: float) -> float:
        return 1.0

    @staticmethod
    def _forward_np(x, a, scale):
        return x

    @staticmethod
    def _inverse_np(y, a, scale):
        return y

    @staticmethod
    def _jacobian_np(x, a, scale):
        return np.ones_like(x)

    @staticmethod
    def _forward_torch(x, a, scale):
        return x

    @staticmethod
    def _inverse_torch(y, a, scale):
        return y

    @staticmethod
    def _jacobian_torch(x, a, scale):
        return torch.ones_like(x)


class ArcsinhTransform(BasePixelTransform):
    """
    Arcsinh stretching for astronomical-image dynamic-range compression.

    Forward: f(x) = arcsinh(x / a) / scale
    Inverse: f⁻¹(y) = a * sinh(y * scale)
    Derivative: df/dx = 1 / (sqrt(x² + a²) * scale)

    Properties:
    - Odd function that naturally handles negative sky-subtracted values.
    - Approximately linear for |x| << a, preserving noise statistics.
    - Approximately logarithmic for |x| >> a, compressing bright pixels.
    - a controls the linear-to-logarithmic transition; a value near the
      background noise standard deviation is a suggested starting point.
    """

    def _compute_scale(self, x_max, a):
        return float(np.arcsinh(x_max / a))

    @staticmethod
    def _forward_np(x, a, scale):
        return np.arcsinh(x / a) / scale

    @staticmethod
    def _inverse_np(y, a, scale):
        return a * np.sinh(y * scale)

    @staticmethod
    def _jacobian_np(x, a, scale):
        return 1.0 / (np.sqrt(x ** 2 + a ** 2) * scale)

    @staticmethod
    def _forward_torch(x, a, scale):
        return torch.arcsinh(x / a) / scale

    @staticmethod
    def _inverse_torch(y, a, scale):
        return a * torch.sinh(y * scale)

    @staticmethod
    def _jacobian_torch(x, a, scale):
        return 1.0 / (torch.sqrt(x ** 2 + a ** 2) * scale)


class SqrtTransform(BasePixelTransform):
    """
    Square-root stretching: an Anscombe-type transform for Poisson noise.

    Forward: f(x) = sign(x) * sqrt(|x| + a) / scale
    Inverse: f⁻¹(y) = sign(y) * ((|y| * scale)² - a)
    Derivative: df/dx = 1 / (2 * sqrt(|x| + a) * scale)

    Properties described by the original implementation:
    - Approximately stabilizes variance when Poisson noise dominates.
    - Moderate compression, between identity and arcsinh.
    - Reported noise amplification of about 6x versus about 12x for arcsinh;
      these values depend on the data and transform parameters.
    - a prevents a divergent derivative at x=0; suggested range: 0.0001–0.01.
    """

    def _compute_scale(self, x_max, a):
        return float(np.sqrt(x_max + a))

    @staticmethod
    def _forward_np(x, a, scale):
        return np.sign(x) * np.sqrt(np.abs(x) + a) / scale

    @staticmethod
    def _inverse_np(y, a, scale):
        return np.sign(y) * np.clip((np.abs(y) * scale) ** 2 - a, 0.0, None)

    @staticmethod
    def _jacobian_np(x, a, scale):
        return 1.0 / (2.0 * np.sqrt(np.abs(x) + a) * scale)

    @staticmethod
    def _forward_torch(x, a, scale):
        return x.sign() * torch.sqrt(x.abs() + a) / scale

    @staticmethod
    def _inverse_torch(y, a, scale):
        return y.sign() * ((y.abs() * scale) ** 2 - a).clamp(min=0.0)

    @staticmethod
    def _jacobian_torch(x, a, scale):
        return 1.0 / (2.0 * torch.sqrt(x.abs() + a) * scale)


class LuptonTransform(BasePixelTransform):
    """
    Lupton et al. (2004) stretching, used for SDSS color images.

    Forward: f(x) = arcsinh(Q * x / a) / (Q * scale)
    Inverse: f⁻¹(y) = a * sinh(y * Q * scale) / Q
    Derivative: df/dx = 1 / (sqrt((Q*x)² + a²) * scale)

    Properties:
    - A parameterized form of arcsinh stretching.
    - Q controls compression strength independently of a; larger Q is stronger.
    - Q=1 recovers ordinary arcsinh stretching.
    - Useful when finer control of compression strength is needed.

    Parameters
    ----------
    Q : float
        Compression parameter; Q=8 is strong, Q=2 is milder.
    """

    def __init__(self, Q: float = 8.0, **kwargs):
        self.Q = Q
        super().__init__(**kwargs)
        logpy.info(f"[Lupton] Q={Q}")

    def _compute_scale(self, x_max, a):
        return float(np.arcsinh(self.Q * x_max / a) / self.Q)

    @staticmethod
    def _forward_np_inner(x, a, scale, Q):
        return np.arcsinh(Q * x / a) / (Q * scale)

    @staticmethod
    def _inverse_np_inner(y, a, scale, Q):
        return a * np.sinh(y * Q * scale) / Q

    @staticmethod
    def _jacobian_np_inner(x, a, scale, Q):
        return 1.0 / (np.sqrt((Q * x) ** 2 + a ** 2) * scale)

    # Override the public interface to handle the extra Lupton parameter Q.
    def forward_np(self, x, source):
        a = self._get_a(source)
        x = self._clip_raw_np(x, source)
        return self._forward_np_inner(x, a, self.scale, self.Q)

    def inverse_np(self, y, source):
        a = self._get_a(source)
        return self._inverse_np_inner(y, a, self.scale, self.Q)

    def transform_error_np(self, err, x, source):
        a = self._get_a(source)
        jac = self._jacobian_np_inner(x, a, self.scale, self.Q)
        return np.abs(jac) * err

    def forward(self, x, source):
        a = self._get_a(source)
        x = self._clip_raw_torch(x, source)
        return torch.arcsinh(self.Q * x / a) / (self.Q * self.scale)

    def inverse(self, y, source):
        a = self._get_a(source)
        return a * torch.sinh(y * self.Q * self.scale) / self.Q

    def transform_error(self, err, x, source):
        a = self._get_a(source)
        jac = 1.0 / (torch.sqrt((self.Q * x) ** 2 + a ** 2) * self.scale)
        return jac.abs() * err

    # Placeholders for abstract methods; use the overridden public methods above.
    @staticmethod
    def _forward_np(x, a, scale):
        raise NotImplementedError("Use forward_np instead.")

    @staticmethod
    def _inverse_np(y, a, scale):
        raise NotImplementedError("Use inverse_np instead.")

    @staticmethod
    def _jacobian_np(x, a, scale):
        raise NotImplementedError("Use transform_error_np instead.")

    @staticmethod
    def _forward_torch(x, a, scale):
        raise NotImplementedError("Use forward instead.")

    @staticmethod
    def _inverse_torch(y, a, scale):
        raise NotImplementedError("Use inverse instead.")

    @staticmethod
    def _jacobian_torch(x, a, scale):
        raise NotImplementedError("Use transform_error instead.")


class BandwiseArcsinhTransform:
    """
    Per-band normalized arcsinh stretching, matching the complete stretch() pipeline.

    Each band is processed independently:
        1. clip(x, lower_bound[c], upper_bound[c])
        2. out = arcsinh((clipped - bg_median[c]) / bg_std[c])
        3. Linearly normalize to [-1, 1]:
               vmin[c] = arcsinh((lower_bound[c] - bg_median[c]) / bg_std[c])
               vmax[c] = arcsinh((upper_bound[c] - bg_median[c]) / bg_std[c])
               out_norm[c] = (out[c] - vmin[c]) / (vmax[c] - vmin[c]) * 2 - 1

    Inverse:
        out[c] = (y + 1) / 2 * (vmax[c] - vmin[c]) + vmin[c]
        x[c]   = sinh(out[c]) * bg_std[c] + bg_median[c]

    Jacobian for error propagation:
        d(out_norm)/dx = 1/bg_std[c] / sqrt(((x-bg_median[c])/bg_std[c])^2 + 1)
                         * 2 / (vmax[c] - vmin[c])

    YAML example:
        pixel_transform_config:
          target: sgm.transforms.pixel_stretch.BandwiseArcsinhTransform
          params:
            euclid:
              lower_bound: [-0.015434673279337585]
              upper_bound: [2.0928584933740315]
              bg_median:   [0.00031603622]
              bg_std:      [0.017898018]
            desi:
              lower_bound: [-0.004908290165057406, -0.0067522825711406765, -0.02320259420014918]
              upper_bound: [9.20013931569946,       10.000000953674316,    10.00001049041748]
              bg_median:   [0.0001529879,            0.00022485493,         0.0005653821]
              bg_std:      [0.06832892,              0.11801922,            0.13742818]
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
            span = vmax - vmin          # (vmax - vmin) > 0 always
            self._params[src] = dict(lb=lb, ub=ub, mu=mu, sig=sig,
                                     vmin=vmin, vmax=vmax, span=span)
            logpy.info(
                f"[BandwiseArcsinh/{src}] vmin={vmin}, vmax={vmax}, span={span}"
            )

    # ------------------------------------------------------------------
    # numpy helpers
    # ------------------------------------------------------------------

    def _p(self, source: str):
        if source not in self._params:
            raise ValueError(f"Unknown source '{source}', expected 'euclid' or 'desi'")
        return self._params[source]

    def _broadcast_np(self, arr: np.ndarray, vec: np.ndarray) -> np.ndarray:
        "Broadcast per-channel parameters of shape (C,) to (C, H, W) or (1, H, W)."
        c = arr.shape[0]
        if len(vec) == 1:
            return vec[0]
        if len(vec) != c:
            raise ValueError(
                f"Parameter channel count {len(vec)} does not match image channel count {c}."
            )
        return vec.reshape(-1, *([1] * (arr.ndim - 1)))

    def _broadcast_torch(self, x: torch.Tensor, vec: np.ndarray) -> torch.Tensor:
        c = x.shape[1] if x.ndim == 4 else x.shape[0]
        t = torch.tensor(vec, dtype=x.dtype, device=x.device)
        if t.numel() == 1:
            return t
        if t.numel() != c:
            raise ValueError(
                f"Parameter channel count {t.numel()} does not match image channel count {c}."
            )
        # (C,) → (1, C, 1, 1) for BCHW or (C, 1, 1) for CHW
        if x.ndim == 4:
            return t.view(1, -1, 1, 1)
        return t.view(-1, 1, 1)

    # ------------------------------------------------------------------
    # Public NumPy interface
    # ------------------------------------------------------------------

    def forward_np(self, x: np.ndarray, source: str) -> np.ndarray:
        p = self._p(source)
        lb  = self._broadcast_np(x, p["lb"])
        ub  = self._broadcast_np(x, p["ub"])
        mu  = self._broadcast_np(x, p["mu"])
        sig = self._broadcast_np(x, p["sig"])
        vmin = self._broadcast_np(x, p["vmin"])
        span = self._broadcast_np(x, p["span"])
        clipped = np.clip(x, lb, ub)
        out = np.arcsinh((clipped - mu) / sig)
        return (out - vmin) / span * 2.0 - 1.0

    def inverse_np(self, y: np.ndarray, source: str) -> np.ndarray:
        p = self._p(source)
        mu   = self._broadcast_np(y, p["mu"])
        sig  = self._broadcast_np(y, p["sig"])
        vmin = self._broadcast_np(y, p["vmin"])
        span = self._broadcast_np(y, p["span"])
        out = (y + 1.0) / 2.0 * span + vmin
        return np.sinh(out) * sig + mu

    def transform_error_np(self, err: np.ndarray, x: np.ndarray, source: str) -> np.ndarray:
        p = self._p(source)
        lb  = self._broadcast_np(x, p["lb"])
        ub  = self._broadcast_np(x, p["ub"])
        mu  = self._broadcast_np(x, p["mu"])
        sig = self._broadcast_np(x, p["sig"])
        span = self._broadcast_np(x, p["span"])
        x_c = np.clip(x, lb, ub)
        z = (x_c - mu) / sig
        jac = (1.0 / sig) / np.sqrt(z ** 2 + 1.0) * (2.0 / span)
        return np.abs(jac) * err

    # ------------------------------------------------------------------
    # Public torch interface with method names compatible with BasePixelTransform
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # normalize / denormalize: values are already in [-1, 1]; no additional statistics are needed.
    # ------------------------------------------------------------------

    def _get_norm(self, source: str):
        return None

    def normalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        return y

    def denormalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        return y


class PowerTransform(BasePixelTransform):
    """
    Power-law stretching: f(x) = sign(x) * |x|^γ / scale.

    Properties:
    - γ < 1 compresses bright values; γ=0.5 gives a square root, γ=0.25 is stronger.
    - Strong compression, but the unregularized derivative diverges at x=0,
      leading to substantial noise amplification.
    - a provides numerical regularization: the implementation uses (|x| + a)^γ.

    Parameters
    ----------
    gamma : float
        Power-law exponent; values below 1 compress the bright end.
    """

    def __init__(self, gamma: float = 0.25, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)
        logpy.info(f"[Power] gamma={gamma}")

    def _compute_scale(self, x_max, a):
        return float((x_max + a) ** self.gamma)

    @staticmethod
    def _forward_np(x, a, scale):
        # gamma is supplied through a closure or class attribute.
        raise NotImplementedError("Use forward_np instead.")

    @staticmethod
    def _inverse_np(y, a, scale):
        raise NotImplementedError

    @staticmethod
    def _jacobian_np(x, a, scale):
        raise NotImplementedError

    @staticmethod
    def _forward_torch(x, a, scale):
        raise NotImplementedError

    @staticmethod
    def _inverse_torch(y, a, scale):
        raise NotImplementedError

    @staticmethod
    def _jacobian_torch(x, a, scale):
        raise NotImplementedError

    def forward_np(self, x, source):
        a = self._get_a(source)
        x = self._clip_raw_np(x, source)
        return np.sign(x) * (np.abs(x) + a) ** self.gamma / self.scale

    def inverse_np(self, y, source):
        a = self._get_a(source)
        return np.sign(y) * np.clip((np.abs(y) * self.scale) ** (1.0 / self.gamma) - a, 0, None)

    def transform_error_np(self, err, x, source):
        a = self._get_a(source)
        jac = self.gamma * (np.abs(x) + a) ** (self.gamma - 1.0) / self.scale
        return np.abs(jac) * err

    def forward(self, x, source):
        a = self._get_a(source)
        return x.sign() * (x.abs() + a) ** self.gamma / self.scale

    def inverse(self, y, source):
        a = self._get_a(source)
        return y.sign() * ((y.abs() * self.scale) ** (1.0 / self.gamma) - a).clamp(min=0.0)

    def transform_error(self, err, x, source):
        a = self._get_a(source)
        jac = self.gamma * (x.abs() + a) ** (self.gamma - 1.0) / self.scale
        return jac.abs() * err
