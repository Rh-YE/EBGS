# ---------------------------------------------------------------
# pixel_stretch.py — 通用像素值非线性拉伸框架
#
# 用于在 I2SB 训练前压缩天文图像的动态范围，使得不同亮度
# 的像素在 MSE 损失中有可比的贡献。
#
# 设计原则：
#   1. 所有波段共享同一个 scale（归一化常数）
#   2. a（softening / 转折点）可按 Euclid / DESI 分别设置
#   3. forward / inverse 必须严格互逆
#   4. transform_error 通过解析 Jacobian 自动传播
#
# 使用方式（YAML）：
#   pixel_transform:
#     target: sgm.transforms.pixel_stretch.SqrtTransform
#     params:
#       a_euclid: 0.001
#       a_desi: 0.001
#       x_max: 10.0
#
#   target 设为 IdentityTransform 即关闭拉伸，回退到原始行为。
#
# 架构：
#   BasePixelTransform (抽象基类)
#     ├── forward(x, source) → x_norm      正变换
#     ├── inverse(x_norm, source) → x      反变换
#     └── transform_error(err, x, source)  误差传播
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
    像素值非线性变换的抽象基类。

    子类需实现三个静态方法：
        _forward_np  : numpy 正变换（用于预处理脚本）
        _inverse_np  : numpy 反变换
        _jacobian_np : 正变换对 x 的导数 df/dx（用于误差传播）

    本基类自动提供 torch 版本的 forward / inverse / transform_error，
    以及 numpy 版本的便捷接口。

    参数
    ----------
    a_euclid : float
        Euclid 图像的 softening 参数。
    a_desi : float
        DESI 图像的 softening 参数。
    x_max : float
        用于计算归一化常数 scale，使 forward(x_max) ≈ 1.0。
    scale : float or None
        若指定，则直接使用该值作为归一化常数（忽略 x_max）。
        多波段 / 多数据源必须共用同一个 scale。
    """


    def __init__(
        self,
        a_euclid: float = 0.001,
        a_desi: float = 0.001,
        x_max: float = 10.0,
        scale: Optional[float] = None,
        # [EXT-NORM] 变换域统计量（由测量脚本在 forward 后直接 mean/std 得到）
        norm_mean_transformed_euclid: Optional[float] = None,
        norm_std_transformed_euclid: Optional[float] = None,
        norm_mean_transformed_desi: Optional[float] = None,
        norm_std_transformed_desi: Optional[float] = None,
        # [VMAX-ALIGN] 拉伸前在 *原始空间* 对像素做对称 clip，
        # 强制 t=0 (Euclid) 与 t=T (DESI) 的最大值严格一致，
        # 解决"凸组合 μ_t 在亮源处把核心拉低"问题（残差中间帧出现暗斑）。
        # None = 不 clip（旧行为）。建议设为某个 percentile 上界 (e.g. 99.9%)。
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

        # [EXT-NORM] 存储变换域统计量；不做任何 Jacobian 传播
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
                    f"[标准化/{src}] 使用变换域统计量: "
                    f"mu_t={mu_t:.6f}, sigma_t={sig_t:.6f}"
                )
            elif (mu_t is None) != (sig_t is None):
                raise ValueError(
                    f"[{src}] norm_mean 和 norm_std 必须同时提供或同时为 None"
                )
            else:
                logpy.info(f"[标准化/{src}] 未配置，normalize/denormalize 将为恒等操作")

        logpy.info(
            f"[像素拉伸] {self.__class__.__name__}: "
            f"a_euclid={a_euclid}, a_desi={a_desi}, "
            f"x_max={x_max}, scale={self.scale:.6f}"
        )
    def _get_a(self, source: str) -> float:
        """根据数据来源返回对应的 a 参数。"""
        if source == "euclid":
            return self.a_euclid
        elif source == "desi":
            return self.a_desi
        else:
            raise ValueError(f"未知数据来源 source='{source}'，应为 'euclid' 或 'desi'")

    # ----------------------------------------------------------
    # 子类必须实现的方法
    # ----------------------------------------------------------

    @abstractmethod
    def _compute_scale(self, x_max: float, a: float) -> float:
        """从 x_max 和 a 计算归一化常数。"""
        ...

    @staticmethod
    @abstractmethod
    def _forward_np(x: np.ndarray, a: float, scale: float) -> np.ndarray:
        """numpy 正变换：原始 → 变换域。"""
        ...

    @staticmethod
    @abstractmethod
    def _inverse_np(y: np.ndarray, a: float, scale: float) -> np.ndarray:
        """numpy 反变换：变换域 → 原始。"""
        ...

    @staticmethod
    @abstractmethod
    def _jacobian_np(x: np.ndarray, a: float, scale: float) -> np.ndarray:
        """正变换对 x 的导数 df/dx（numpy 版，用于误差传播）。"""
        ...

    # ----------------------------------------------------------
    # torch 版本（自动从 numpy 版本派生）
    # ----------------------------------------------------------

    @staticmethod
    @abstractmethod
    def _forward_torch(x: torch.Tensor, a: float, scale: float) -> torch.Tensor:
        """torch 正变换。"""
        ...

    @staticmethod
    @abstractmethod
    def _inverse_torch(y: torch.Tensor, a: float, scale: float) -> torch.Tensor:
        """torch 反变换。"""
        ...

    @staticmethod
    @abstractmethod
    def _jacobian_torch(x: torch.Tensor, a: float, scale: float) -> torch.Tensor:
        """torch 导数。"""
        ...

    # ----------------------------------------------------------
    # 公共接口（numpy）
    # ----------------------------------------------------------

    def _clip_raw_np(self, x: np.ndarray, source: str) -> np.ndarray:
        """[VMAX-ALIGN] 原始空间对称 clip：在拉伸前限制 |x| 上下界。"""
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
        """正变换（numpy 版）。source = 'euclid' 或 'desi'。"""
        x = self._clip_raw_np(x, source)
        return self._forward_np(x, self._get_a(source), self.scale)

    def inverse_np(self, y: np.ndarray, source: str) -> np.ndarray:
        """反变换（numpy 版）。"""
        return self._inverse_np(y, self._get_a(source), self.scale)

    def transform_error_np(self, err: np.ndarray, x: np.ndarray, source: str) -> np.ndarray:
        """
        误差传播（numpy 版）。

        σ_transformed = |df/dx| · σ_original

        参数
        ----------
        err : 原始误差 σ
        x   : 原始像素值（用于计算 Jacobian）
        """
        jac = self._jacobian_np(x, self._get_a(source), self.scale)
        return np.abs(jac) * err

    # ----------------------------------------------------------
    # 公共接口（torch）
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        """正变换（torch 版）。source = 'euclid' 或 'desi'。"""
        x = self._clip_raw_torch(x, source)
        return self._forward_torch(x, self._get_a(source), self.scale)

    def inverse(self, y: torch.Tensor, source: str) -> torch.Tensor:
        """反变换（torch 版）。"""
        return self._inverse_torch(y, self._get_a(source), self.scale)

    def transform_error(self, err: torch.Tensor, x: torch.Tensor, source: str) -> torch.Tensor:
        """误差传播（torch 版）。σ_transformed = |df/dx| · σ_original"""
        jac = self._jacobian_torch(x, self._get_a(source), self.scale)
        return jac.abs() * err
    
    def _get_norm(self, source: str) -> Optional[Tuple[float, float]]:
        """返回变换域的 (mu, sigma)，若未配置则返回 None。"""
        return self._norm.get(source)

    def normalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        """标准化：在变换域执行 (y - mu) / sigma。"""
        ns = self._get_norm(source)
        if ns is None:
            return y
        mu, sigma = ns
        return (y - mu) / sigma

    def denormalize(self, y_norm: torch.Tensor, source: str) -> torch.Tensor:
        """反标准化：y * sigma + mu。"""
        ns = self._get_norm(source)
        if ns is None:
            return y_norm
        mu, sigma = ns
        return y_norm * sigma + mu

# ============================================================
# 具体实现
# ============================================================

class IdentityTransform(BasePixelTransform):
    """
    恒等变换（不做任何拉伸）。

    用途：关闭像素拉伸，完全回退到原始 I2SB 行为。
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
    arcsinh 拉伸：天文学标准动态范围压缩。

    正变换：f(x) = arcsinh(x / a) / scale
    反变换：f⁻¹(y) = a · sinh(y · scale)
    导数：  df/dx = 1 / (sqrt(x² + a²) · scale)

    特性：
    - 奇函数，天然处理 BGSUB 后的负值
    - |x| << a 时近似线性（保留噪声统计）
    - |x| >> a 时近似对数（压缩高端）
    - a 控制线性→对数的转折点，建议设为背景噪声 RMS 附近
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
    sqrt 拉伸：泊松噪声的方差稳定化变换 (Anscombe)。

    正变换：f(x) = sign(x) · sqrt(|x| + a) / scale
    反变换：f⁻¹(y) = sign(y) · ((|y| · scale)² − a)
    导数：  df/dx = 1 / (2 · sqrt(|x| + a) · scale)

    特性：
    - 对泊松噪声主导的数据，变换后噪声方差近似均匀
    - 压缩力度中等（比 arcsinh 弱，比 identity 强）
    - 噪声放大温和（~6x vs arcsinh 的 ~12x）
    - a 防止 x=0 处导数发散，建议 0.0001 ~ 0.01
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
    Lupton+2004 拉伸：SDSS 彩色图像标准方法。

    正变换：f(x) = arcsinh(Q · x / a) / (Q · scale)
    反变换：f⁻¹(y) = a · sinh(y · Q · scale) / Q
    导数：  df/dx = 1 / (sqrt((Q·x)² + a²) · scale)

    特性：
    - 本质是 arcsinh 的参数化版本
    - Q 独立控制压缩激进程度（Q 大→压缩强），与 a 解耦
    - Q=1 退化为标准 arcsinh
    - 适合需要精细调控压缩强度的场景

    参数
    ----------
    Q : float
        压缩强度参数。Q=8 压缩很强，Q=2 温和。
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

    # 由于 Lupton 多一个 Q 参数，需要覆盖公共接口
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

    # 基类抽象方法的占位实现（实际通过上面的覆盖接口调用）
    @staticmethod
    def _forward_np(x, a, scale):
        raise NotImplementedError("使用 forward_np 代替")

    @staticmethod
    def _inverse_np(y, a, scale):
        raise NotImplementedError("使用 inverse_np 代替")

    @staticmethod
    def _jacobian_np(x, a, scale):
        raise NotImplementedError("使用 transform_error_np 代替")

    @staticmethod
    def _forward_torch(x, a, scale):
        raise NotImplementedError("使用 forward 代替")

    @staticmethod
    def _inverse_torch(y, a, scale):
        raise NotImplementedError("使用 inverse 代替")

    @staticmethod
    def _jacobian_torch(x, a, scale):
        raise NotImplementedError("使用 transform_error 代替")


class BandwiseArcsinhTransform:
    """
    逐波段标准化 arcsinh 拉伸，对应 stretch() 函数的完整流程。

    每个波段独立地做：
        1. clip(x, lower_bound[c], upper_bound[c])
        2. out = arcsinh((clipped - bg_median[c]) / bg_std[c])
        3. 线性归一化到 [-1, 1]：
               vmin[c] = arcsinh((lower_bound[c] - bg_median[c]) / bg_std[c])
               vmax[c] = arcsinh((upper_bound[c] - bg_median[c]) / bg_std[c])
               out_norm[c] = (out[c] - vmin[c]) / (vmax[c] - vmin[c]) * 2 - 1

    反变换：
        out[c] = (y + 1) / 2 * (vmax[c] - vmin[c]) + vmin[c]
        x[c]   = sinh(out[c]) * bg_std[c] + bg_median[c]

    Jacobian（误差传播）：
        d(out_norm)/dx = 1/bg_std[c] / sqrt(((x-bg_median[c])/bg_std[c])^2 + 1)
                         * 2 / (vmax[c] - vmin[c])

    YAML 示例：
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
        """将形如 (C,) 的逐通道参数广播到 (C, H, W) 或 (1, H, W)。"""
        c = arr.shape[0]
        if len(vec) == 1:
            return vec[0]
        if len(vec) != c:
            raise ValueError(
                f"参数通道数 {len(vec)} 与图像通道数 {c} 不匹配"
            )
        return vec.reshape(-1, *([1] * (arr.ndim - 1)))

    def _broadcast_torch(self, x: torch.Tensor, vec: np.ndarray) -> torch.Tensor:
        c = x.shape[1] if x.ndim == 4 else x.shape[0]
        t = torch.tensor(vec, dtype=x.dtype, device=x.device)
        if t.numel() == 1:
            return t
        if t.numel() != c:
            raise ValueError(
                f"参数通道数 {t.numel()} 与图像通道数 {c} 不匹配"
            )
        # (C,) → (1, C, 1, 1) for BCHW or (C, 1, 1) for CHW
        if x.ndim == 4:
            return t.view(1, -1, 1, 1)
        return t.view(-1, 1, 1)

    # ------------------------------------------------------------------
    # numpy 公共接口
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
    # torch 公共接口（与 BasePixelTransform 兼容的接口名）
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
    # normalize / denormalize — 变换已到 [-1, 1]，无需额外统计量
    # ------------------------------------------------------------------

    def _get_norm(self, source: str):
        return None

    def normalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        return y

    def denormalize(self, y: torch.Tensor, source: str) -> torch.Tensor:
        return y


class PowerTransform(BasePixelTransform):
    """
    幂律拉伸：f(x) = sign(x) · |x|^γ / scale。

    特性：
    - γ < 1 压缩高端（γ=0.5 即 sqrt，γ=0.25 压缩更强）
    - 压缩最激进，但导数在 x=0 处发散 → 噪声放大严重
    - a 在此作为数值稳定项：实际计算 (|x| + a)^γ

    参数
    ----------
    gamma : float
        幂指数，< 1 压缩高端。
    """

    def __init__(self, gamma: float = 0.25, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)
        logpy.info(f"[Power] gamma={gamma}")

    def _compute_scale(self, x_max, a):
        return float((x_max + a) ** self.gamma)

    @staticmethod
    def _forward_np(x, a, scale):
        # 注意：gamma 通过闭包或类属性传入
        raise NotImplementedError("使用 forward_np 代替")

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
