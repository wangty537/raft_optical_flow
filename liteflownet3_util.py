import logging
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F


class _InputPadder:
    """Pads images such that dimensions are divisible by stride."""

    def __init__(
        self,
        dims,
        stride=8,
        two_side_pad=True,
        pad_mode="replicate",
        pad_value=0.0,
        size=None,
    ):
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        ht, wd = dims[-2:]
        if size is None:
            pad_ht = (((ht // stride) + 1) * stride - ht) % stride
            pad_wd = (((wd // stride) + 1) * stride - wd) % stride
        else:
            pad_ht = size[0] - ht
            pad_wd = size[1] - wd
        if two_side_pad:
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, x):
        in_shape = x.shape
        if len(in_shape) > 4:
            x = x.view(-1, *in_shape[-3:])
        x = F.pad(x, self._pad, mode=self.pad_mode, value=self.pad_value)
        if len(in_shape) > 4:
            x = x.view(*in_shape[:-2], *x.shape[-2:])
        return x

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
class InputPadder(_InputPadder):
    """Pads images such that dimensions are divisible by stride.

    This is just a wrapper for ptlflow.utils.external.raft.InputPadder.
    """

    def __init__(
        self,
        dims: Sequence[int],
        stride: int,
        size: Optional[Tuple[int, int]] = None,
        two_side_pad: bool = True,
        pad_mode: str = "replicate",
        pad_value: float = 0.0,
    ) -> None:
        """Initialize InputPadder.

        Parameters
        ----------
        dims : Sequence[int]
            The shape of the original input. It must have at least two elements. It is assumed that the last two dimensions
            are (height, width).
        stride : int
            The number to compute the amount of padding. The padding will be applied so that the input size is divisible
            by stride.
        size : Optional[Tuple[int, int]], optional
            The desired size after scaling defined as (height, width). If not provided, then scale_factor will be used instead.
        two_side_pad : bool, default True
            If True, half of the padding goes to left/top and the rest to right/bottom. Otherwise, all the padding goes to the bottom right.
        pad_mode : str, default "replicate"
            How to pad the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.pad.
        pad_value : float, default 0.0
            Used if pad_mode == "constant". The value to fill in the padded area.
        """
        super().__init__(
            dims,
            stride=stride,
            size=size,
            two_side_pad=two_side_pad,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )
        if size is None:
            self.tgt_size = (
                int(math.ceil(float(dims[-2]) / stride)) * stride,
                int(math.ceil(float(dims[-1]) / stride)) * stride,
            )
        else:
            self.tgt_size = size

    def fill(self, x):
        return self.pad(x)

    def unfill(self, x):
        if x.shape[-2] == self.tgt_size[0] and x.shape[-1] == self.tgt_size[1]:
            x = self.unpad(x)
        return x


class InputScaler(object):
    """Scale 2D torch.Tensor input to a target size, and then rescale it back to the original size."""

    def __init__(
        self,
        orig_shape: Tuple[int, int],
        stride: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = 1.0,
        interpolation_mode: str = "bilinear",
        interpolation_align_corners: bool = False,
    ) -> None:
        """Initialize InputScaler.

        Parameters
        ----------
        orig_shape : Tuple[int, int]
            The shape of the input tensor before the scale. I.e., the shape to which it will be rescaled back.
        stride : Optional[int], optional
            If provided, the input will be resized to the closest larger multiple of stride.
        size : Optional[Tuple[int, int]], optional
            The desired size after scaling defined as (height, width). If not provided, then scale_factor will be used instead.
        scale_factor : Optional[float], default 1.0
            This value is only used if stride and size are None. The multiplier that will be applied to the original shape to scale
            the input.
        interpolation_mode : str, default 'bilinear'
            How to perform the interpolation. It must be a value accepted by the 'mode' argument from
            torch.nn.functional.interpolate function.
        interpolation_align_corners : bool, default False
            Whether the interpolation keep the corners aligned. As defined in torch.nn.functional.interpolate.

        See Also
        --------
        torch.nn.functional.interpolate : The function used to scale the inputs.
        """
        super().__init__()
        self.orig_height, self.orig_width = orig_shape[-2:]
        if stride is not None:
            assert size is None, "only stride OR size can be provided, NOT BOTH."
            self.tgt_height = int(math.ceil(float(self.orig_height) / stride)) * stride
            self.tgt_width = int(math.ceil(float(self.orig_width) / stride)) * stride
        elif size is not None:
            assert stride is None, "only stride OR size can be provided, NOT BOTH."
            self.tgt_height, self.tgt_width = size
        else:
            self.tgt_height = int(self.orig_height * scale_factor)
            self.tgt_width = int(self.orig_width * scale_factor)

        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

    def fill(self, x: torch.Tensor, is_flow: bool = False) -> torch.Tensor:
        """Scale the input to the target size specified during initialization.

        Parameters
        ----------
        x : torch.Tensor
            The input to be scaled. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The scaled input.
        """
        return self._scale_keep_dims(x, (self.tgt_height, self.tgt_width), is_flow)

    def unfill(self, x: torch.Tensor, is_flow: bool = False) -> torch.Tensor:
        """Scale the input to back to the original size defined during initialization.

        Parameters
        ----------
        x : torch.Tensor
            The input to be rescaled back. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The rescaled input.
        """
        return self._scale_keep_dims(x, (self.orig_height, self.orig_width), is_flow)

    def _scale_keep_dims(
        self, x: torch.Tensor, size: Tuple[int, int], is_flow: bool
    ) -> torch.Tensor:
        """Scale the input to a given size while keeping the other dimensions intact.

        Parameters
        ----------
        x : torch.Tensor
            The input to be rescaled back. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        size : Tuple[int, int]
            The target size to scale the input.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The rescaled input.
        """
        x_shape = x.shape
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = F.interpolate(
            x,
            size=size,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
        )

        if is_flow:
            x[:, 0] = x[:, 0] * (float(x.shape[-1]) / x_shape[-1])
            x[:, 1] = x[:, 1] * (float(x.shape[-2]) / x_shape[-2])

        new_shape = list(x_shape)
        new_shape[-2], new_shape[-1] = x.shape[-2], x.shape[-1]
        x = x.view(new_shape)
        return x


def bgr_val_as_tensor(
    bgr_values: Union[float, List[float]], 
    reference_tensor: torch.Tensor, 
    bgr_tensor_shape_position: int = -3
) -> torch.Tensor:
    """
    将BGR值转换为与参考张量具有相同设备和数据类型的张量
    
    Args:
        bgr_values: BGR值，可以是单个浮点数或长度为3的列表，顺序为[B, G, R]
        reference_tensor: 参考张量，用于确定输出张量的设备和数据类型
        bgr_tensor_shape_position: BGR通道在张量中的位置，默认为-3
    
    Returns:
        torch.Tensor: 包含BGR值的张量，形状与reference_tensor兼容
    """
    if isinstance(bgr_values, (int, float)):
        bgr_values = [bgr_values, bgr_values, bgr_values]
    
    # 创建BGR张量
    bgr_tensor = torch.tensor(bgr_values, dtype=reference_tensor.dtype, device=reference_tensor.device)
    
    # 根据参考张量的形状调整BGR张量的形状
    target_shape = [1] * len(reference_tensor.shape)
    target_shape[bgr_tensor_shape_position] = 3
    
    return bgr_tensor.view(target_shape)
