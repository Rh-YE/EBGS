import logging
import math
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ...modules.attention import SpatialTransformer
# from ...modules.video_attention import SpatialVideoTransformer
from ...modules.diffusionmodules.util import (avg_pool_nd, conv_nd, linear,
                                              normalization,
                                              timestep_embedding, zero_module)
from ...util import exists

logpy = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, _ = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: th.Tensor, emb: th.Tensor):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
        self,
        x: th.Tensor,
        emb: th.Tensor,
        context: Optional[th.Tensor] = None,
        image_only_indicator: Optional[th.Tensor] = None,
        time_context: Optional[int] = None,
        num_video_frames: Optional[int] = None,
    ):
        # from ...modules.diffusionmodules.video_model import VideoResBlock

        for layer in self:
            module = layer

            # if isinstance(module, TimestepBlock) and not isinstance(
            #     module, VideoResBlock
            # ):
            #     x = layer(x, emb)
            # elif isinstance(module, VideoResBlock):
            #     x = layer(x, emb, num_video_frames, image_only_indicator)
            # elif isinstance(module, SpatialVideoTransformer):
            #     x = layer(
            #         x,
            #         context,
            #         time_context,
            #         num_video_frames,
            #         image_only_indicator,
            #     )
            if isinstance(module, TimestepBlock):
                x = layer(x, emb)          # Layers such as ResBlock that require emb.
            elif isinstance(module, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_up: bool = False,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        self.scale_factor = scale_factor
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, kernel_size, padding=padding
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels

        if self.dims == 3:
            t_factor = 1 if not self.third_up else self.scale_factor
            x = F.interpolate(
                x,
                (
                    t_factor * x.shape[2],
                    x.shape[3] * self.scale_factor,
                    x.shape[4] * self.scale_factor,
                ),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            logpy.info(f"Building a Downsample layer with {dims} dims.")
            logpy.info(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            if dims == 3:
                logpy.info(f"  --> Downsampling third axis (time): {third_down}")
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels

        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            logpy.info(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): # Reshape the embedding so it can be added to h.
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: th.Tensor, **kwargs) -> th.Tensor:
        return checkpoint(self._forward, x)

    def _forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class Timestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: th.Tensor) -> th.Tensor:
        return timestep_embedding(t, self.dim)

class UNetModel(nn.Module):
    """
    Full UNet with attention and timestep embeddings.

    :param in_channels: Number of input tensor channels.
    :param model_channels: Base channel count.
    :param out_channels: Number of output tensor channels.
    :param num_res_blocks: Residual blocks per downsampling stage.
    :param attention_resolutions: Downsampling factors at which to apply
        attention, as a set, list, or tuple. Including 4 enables attention
        at a downsampling factor of 4.
    :param dropout: Dropout probability.
    :param channel_mult: Channel multipliers at each UNet level.
    :param conv_resample: Use convolution for up/downsampling when True.
    :param dims: Signal dimensionality: 1D, 2D, or 3D.
    :param num_classes: If specified as an integer, enable class conditioning
        with this number of classes.
    :param use_checkpoint: Use gradient checkpointing to reduce memory use.
    :param num_heads: Number of attention heads per attention layer.
    :param num_head_channels: Fixed width per attention head, overriding
        num_heads when specified.
    :param num_heads_upsample: Attention heads during upsampling; deprecated.
    :param use_scale_shift_norm: Use FiLM-style modulation.
    :param resblock_updown: Use residual blocks for up/downsampling.
    :param use_new_attention_order: Alternative attention ordering that may
        improve efficiency.
    """

    def __init__(
        self,
        in_channels: int,  # Input channels
        model_channels: int,  # Base model channels
        out_channels: int,  # Output channels
        num_res_blocks: int,  # Residual blocks per level
        attention_resolutions: int,  # Attention resolutions
        dropout: float = 0.0,  # Dropout probability
        channel_mult: Union[List, Tuple] = (1, 2, 4, 8),  # Channel multipliers
        conv_resample: bool = True,  # Use convolutional resampling
        dims: int = 2,  # Signal dimensionality: 1D, 2D, or 3D
        num_classes: Optional[Union[int, str]] = None,  # Number of classes
        use_checkpoint: bool = False,  # Use gradient checkpointing
        num_heads: int = -1,  # Number of attention heads
        num_head_channels: int = -1,  # Channels per attention head
        num_heads_upsample: int = -1,  # Attention heads during upsampling; deprecated
        use_scale_shift_norm: bool = False,  # Use scale-shift normalization
        resblock_updown: bool = False,  # Use residual blocks for up/downsampling
        transformer_depth: int = 1,  # Transformer depth
        context_dim: Optional[int] = None,  # Context dimension
        disable_self_attentions: Optional[List[bool]] = None,  # Disable self-attention
        num_attention_blocks: Optional[List[int]] = None,  # Number of attention blocks
        disable_middle_self_attn: bool = False,  # Disable middle-block self-attention
        disable_middle_transformer: bool = False,  # Disable the middle transformer
        use_linear_in_transformer: bool = False,  # Use linear projection in the spatial transformer
        spatial_transformer_attn_type: str = "softmax",  # Spatial-transformer attention type
        adm_in_channels: Optional[int] = None,  # ADM input channels
    ):
        super().__init__()

        # Default num_heads_upsample to num_heads when unspecified.
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # Require either num_heads or num_head_channels.
        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels must be set."

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels must be set."

        self.in_channels = in_channels  # Input channels
        self.model_channels = model_channels  # Base model channels
        self.out_channels = out_channels  # Output channels

        # Expand an integer transformer_depth to match the length of channel_mult.
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]  # Transformer depth in the middle block.

        # Expand an integer num_res_blocks to match the length of channel_mult.
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "num_res_blocks must be an integer shared across levels or a list matching the length of channel_mult."
                )
            self.num_res_blocks = num_res_blocks

        # Check the length of disable_self_attentions.
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        # Check the length of num_attention_blocks.
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            logpy.info(
                f"UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This setting has lower priority than attention_resolutions={attention_resolutions}: "
                f"if num_attention_blocks[i] > 0 but 2**i is absent from attention_resolutions, "
                f"attention is still disabled at that level."
            )

        self.attention_resolutions = attention_resolutions  # Attention resolutions
        self.dropout = dropout  # Dropout probability
        self.channel_mult = channel_mult  # Channel multipliers
        self.conv_resample = conv_resample  # Use convolutional resampling
        self.num_classes = num_classes  # Number of classes
        self.use_checkpoint = use_checkpoint  # Use gradient checkpointing
        self.num_heads = num_heads  # Number of attention heads
        self.num_head_channels = num_head_channels  # Channels per attention head
        self.num_heads_upsample = num_heads_upsample  # Attention heads during upsampling

        # Timestep embedding dimension
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Class embedding layer
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                logpy.info("Setting up a linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError

        # Input blocks
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels  # Feature size
        input_block_chans = [model_channels]  # Input-block channel list
        ch = model_channels  # Current channel count
        ds = 1  # Current downsampling factor

        # Build residual and attention blocks at each level.
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if context_dim is not None and exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle block
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                attn_type=spatial_transformer_attn_type,
                use_checkpoint=use_checkpoint,
            )
            if not disable_middle_transformer
            else th.nn.Identity(),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # Final output layer
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: th.Tensor,  # Input tensor
        timesteps: Optional[th.Tensor] = None,  # Timestep tensor
        context: Optional[th.Tensor] = None,  # Context tensor
        y: Optional[th.Tensor] = None,  # Class labels
        **kwargs,
    ) -> th.Tensor:
        """
        Apply the model to an input batch.

        :param x: Input tensor with shape [N x C x ...].
        :param timesteps: One-dimensional batch of timesteps.
        :param context: Conditioning supplied through cross-attention.
        :param y: Labels with shape [N] for a class-conditioned model.
        :return: Output tensor with shape [N x C x ...].
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "y must be specified if and only if the model is class-conditioned."
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # timesteps contains t; t_emb is its timestep embedding.
        emb = self.time_embed(t_emb) # Project to a higher-dimensional, more expressive embedding emb.

        if self.num_classes is not None: # Add class information to the embedding when required.
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks: # UNet downsampling
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context) # UNet middle block
        for module in self.output_blocks: # UNet upsampling
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)

        return self.out(h)




