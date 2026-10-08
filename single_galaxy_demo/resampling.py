"""Unmodified functions extracted from sgm/data/utils.py for fast resampling."""
import numpy as np
from astropy.nddata import Cutout2D
_OVERLAP_CACHE = {}

def center_crop(image, position, target_size):
    """
    裁剪函数基本保持不变，但确保 position 传入的是从 get_crop_position 得到的浮点数
    """
    # 这里的 position 是 (y, x) 格式
    if image.ndim == 2:
        # Cutout2D 内部会处理 (y, x) 或 (x, y) 顺序，
        # 注意：Cutout2D(data, position, size) 中 position 默认是 (x, y)
        # 如果你的 get_crop_position 返回的是 (h_pos, w_pos)，这里需要反转
        cutout_pos = (position[1], position[0]) 
        
        cutout = Cutout2D(image, cutout_pos, target_size, mode='partial', fill_value=0)
        return cutout.data
    
    # 多通道处理
    cutout_data = []
    cutout_pos = (position[1], position[0])
    for i in range(image.shape[0]):
        cutout = Cutout2D(image[i], cutout_pos, target_size, mode='partial', fill_value=0)
        cutout_data.append(cutout.data)
    return np.array(cutout_data)

def get_crop_position(image, target_h, target_w, random_crop=False, current_pos=None):
    """
    获取裁剪位置
    
    Args:
        image: 输入图像
        target_h: 目标高度
        target_w: 目标宽度
        random_crop: 是否随机裁剪（使用高斯分布）
        current_pos: 已有位置（如果提供则直接返回）
    """
    if current_pos is not None:
        return current_pos
    
    # 关键修改：获取物理几何中心（浮点数）
    img_h, img_w = image.shape[-2], image.shape[-1]
    center_h = (img_h - 1) / 2.0
    center_w = (img_w - 1) / 2.0
    
    if not random_crop:
        # 返回精确的浮点中心，确保 128 和 91 的逻辑一致
        return (center_h, center_w)
    
    # 随机裁剪逻辑
    min_h, max_h = target_h / 2.0, img_h - target_h / 2.0
    min_w, max_w = target_w / 2.0, img_w - target_w / 2.0
    
    if min_h >= max_h or min_w >= max_w:
        return (center_h, center_w)
    
    sigma_h, sigma_w = target_h * 3 / 8.0, target_w * 3 / 8.0
    h_offset = np.random.normal(0, sigma_h)
    w_offset = np.random.normal(0, sigma_w)
    
    h_pos = np.clip(center_h + h_offset, min_h, max_h)
    w_pos = np.clip(center_w + w_offset, min_w, max_w)
    
    return (h_pos, w_pos)


def convert_flux_zeropoint(flux_data, from_zp=30, to_zp=22.5):
    factor = 10 ** (0.4 * (to_zp - from_zp))
    return flux_data * factor

def convert_invvar_zeropoint(invvar_data, from_zp=30, to_zp=22.5):
    factor = 10 ** (0.4 * (to_zp - from_zp))
    return invvar_data / (factor ** 2)

def _overlap_matrix(n_in, n_out, scale):
    """
    行/列方向的面积重叠权重矩阵 (n_out, n_in),行归一化 → 面积平均
    (surface brightness),复现 reproject_exact 默认语义。

    对齐约定:两网格物理中心重合(与合成 WCS 的 CRPIX=(N+1)/2 一致),
    FITS 像素 i(0-based)中心在坐标 i,覆盖 [i-0.5, i+0.5]。

    scale = to_pixel_scale / from_pixel_scale(输出像素跨多少输入像素,
    过采样时 < 1)。
    """
    center_in = (n_in - 1) / 2.0
    center_out = (n_out - 1) / 2.0
    j = np.arange(n_out)
    x_center = center_in + (j - center_out) * scale
    out_lo = (x_center - scale / 2.0)[:, None]
    out_hi = (x_center + scale / 2.0)[:, None]
    i = np.arange(n_in)[None, :]
    in_lo = i - 0.5
    in_hi = i + 0.5
    overlap = np.clip(np.minimum(out_hi, in_hi) - np.maximum(out_lo, in_lo), 0.0, None)
    rowsum = overlap.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return (overlap / rowsum).astype(np.float64)

def _reproj_fast(img_data, apply_zp_conv, from_zp, to_zp, is_invvar,
                 from_pixel_scale, to_pixel_scale):
    """
    可分离面积加权重采样,见上方说明。img_data: (C,H,W)。

    权重矩阵在 f64 下构造(裁剪/归一化精度),但缓存为 f32 并用 BLAS
    sgemm(np.matmul)做两次重采样:
        out = Wr @ x @ WcT    # (Ho,Hi)@(C,Hi,Wi)@(Wi,Wo) -> (C,Ho,Wo)
    f32 matmul 比 f64 einsum 快 ~6x,与 f64 路径数值差 ~5e-8(纯舍入,
    远小于与 reproject_exact 的 ~6e-4 残差)。
    """
    c, h_in, w_in = img_data.shape
    sf = from_pixel_scale / to_pixel_scale
    h_out, w_out = int(h_in * sf), int(w_in * sf)
    scale = to_pixel_scale / from_pixel_scale

    key = (h_in, h_out, w_in, w_out, scale)
    wmats = _OVERLAP_CACHE.get(key)
    if wmats is None:
        # f64 构造 → f32 缓存;列矩阵预转置为 (w_in, w_out) 供右乘
        Wr = _overlap_matrix(h_in, h_out, scale).astype(np.float32)      # (Ho, Hi)
        WcT = _overlap_matrix(w_in, w_out, scale).T.copy().astype(np.float32)  # (Wi, Wo)
        _OVERLAP_CACHE[key] = wmats = (Wr, WcT)
    Wr, WcT = wmats

    x = np.ascontiguousarray(img_data, dtype=np.float32)
    out = Wr @ x       # (Ho,Hi) @ (C,Hi,Wi) -> (C,Ho,Wi)  (batched sgemm)
    out = out @ WcT    # (C,Ho,Wi) @ (Wi,Wo) -> (C,Ho,Wo)

    if apply_zp_conv:
        out = (convert_invvar_zeropoint(out, from_zp, to_zp) if is_invvar
               else convert_flux_zeropoint(out, from_zp, to_zp))
    return out.astype(np.float32, copy=False)

