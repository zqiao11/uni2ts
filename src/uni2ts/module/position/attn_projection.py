#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import abc
import math
from functools import cached_property
from typing import Any, Optional

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int
from torch import nn
import torch.nn.functional as F

class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int, num_heads: int, num_groups: int, **kwargs: Any):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

    @abc.abstractmethod
    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]: ...


class IdentityProjection(Projection):
    def __init__(self, *, proj_width: int, num_heads: int, num_groups: int, **kwargs):
        super().__init__(proj_width, num_heads, num_groups)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]] = None,
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        return x


class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(  # QZ: Eq 15    (16, )
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)  # (512, 32)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],  # (32, 6, 1, 46, 32)
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],  # (32, 1, 1, 46)
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id]  # (32, 1, 1, 46, 32)
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)  # QZ: Eq 34 in the paper


class MultiScaleRotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
        base: int = 10000,
        token_idx_per_scale: Optional[list] = None,  # ToDo: list of time index of each scale [[], [], []]
        base_ctx_token_idx: Optional[list] = None
    ):
        super().__init__(proj_width, num_heads, num_groups)

        self.token_idx_per_scale = token_idx_per_scale
        self.base_ctx_token_idx = base_ctx_token_idx

        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(  # QZ: Eq 15
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

        self.max_len = max_len

    def post_init(self, token_idx_per_scale, base_ctx_token_idx):
        self.token_idx_per_scale = token_idx_per_scale
        self.base_ctx_token_idx = base_ctx_token_idx
        self.num_scales = len(token_idx_per_scale)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)  # (512, 32)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],  # (32, 6, 1, 46, 32)
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:

        # Create tensors to store multi-scale rot_cos/sin
        rot_shape = (*seq_id.shape, self.cos.shape[-1])
        rot_cos = torch.empty(rot_shape, device=seq_id.device, dtype=torch.float)
        rot_sin = torch.empty(rot_shape, device=seq_id.device, dtype=torch.float)

        for i in range(self.num_scales):
            idx_scale_i = self.token_idx_per_scale[i]
            mapped_seq_id = seq_id[..., :, :, idx_scale_i]  # (bs, 1, 1, len)

            # Directly use original time_id to obtain sin/cos for base scale
            if i == 0:
                mapped_seq_id = mapped_seq_id.to(torch.int)
                rot_cos[..., :, :, idx_scale_i, :] = self.cos[mapped_seq_id]  # (bs, 1, 1, len0, proj_width)
                rot_sin[..., :, :, idx_scale_i, :] = self.sin[mapped_seq_id]

            # For new scales, compute the theta for their float mapped id. And their cos/sin.
            else:
                m_theta = einsum(mapped_seq_id.squeeze(), self.theta, "bs length, width -> bs length width")
                m_theta = repeat(m_theta, "bs length width -> bs length (width 2)")
                rot_cos[..., :, :, idx_scale_i, :] = torch.cos(m_theta).unsqueeze(1).unsqueeze(2)
                rot_sin[..., :, :, idx_scale_i, :] = torch.sin(m_theta).unsqueeze(1).unsqueeze(2)

        return rot_cos * x + rot_sin * self._rotate(x)  # QZ: Eq 34 in the paper


# class MultiScaleRotaryProjection(Projection):
#     def __init__(
#             self,
#             *,
#             proj_width: int,
#             num_heads: int,
#             num_groups: int,
#             max_len: int = 512,
#             base: int = 10000,
#             token_idx_per_scale: Optional[list] = None  # ToDo: list of time index of each scale [[], [], []]
#     ):
#         super().__init__(proj_width, num_heads, num_groups)
#
#         self.token_idx_per_scale = token_idx_per_scale
#
#         assert (
#                 self.proj_width % 2 == 0
#         ), f"proj_width must be even, got {self.proj_width}"
#         self.register_buffer(  # QZ: Eq 15
#             "theta",
#             1.0
#             / torch.pow(
#                 base,
#                 torch.arange(0, self.proj_width, 2, dtype=torch.float)
#                 / self.proj_width,
#             ),
#             persistent=False,
#         )
#
#         self.max_len = max_len
#
#     def post_init(self, token_idx_per_scale):
#         self.token_idx_per_scale = token_idx_per_scale
#         self.num_scales = len(token_idx_per_scale)
#
#         for i in range(self.num_scales):
#             self.register_buffer(f"cos{i}", None, persistent=False)
#             self.register_buffer(f"sin{i}", None, persistent=False)
#         self._init_freq(max_len=self.max_len)
#
#     def _init_freq(self, max_len: int):
#
#         for i in range(self.num_scales):
#
#             if i == 0:
#                 position = torch.arange(
#                     max_len, device=self.theta.device, dtype=self.theta.dtype
#                 )
#                 m_theta = einsum(position, self.theta, "length, width -> length width")
#                 m_theta = repeat(m_theta, "length width -> length (width 2)")
#                 self.register_buffer("cos0", torch.cos(m_theta), persistent=False)
#                 self.register_buffer("sin0", torch.sin(m_theta), persistent=False)
#
#             else:
#                 start, step = sum(list(range(0, 2 ** i))) / 2 ** i, 2 * i
#                 position = torch.arange(
#                     start=start, end=max_len, step=step, device=self.theta.device, dtype=self.theta.dtype
#                 )
#                 m_theta = einsum(position, self.theta, "length, width -> length width")
#                 m_theta = repeat(m_theta, "length width -> length (width 2)")
#                 self.register_buffer(f"cos{i}", torch.cos(m_theta), persistent=False)
#                 self.register_buffer(f"sin{i}", torch.sin(m_theta), persistent=False)
#
#     @staticmethod
#     def _rotate(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
#         x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
#         return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa
#
#     def forward(
#             self,
#             x: Float[torch.Tensor, "*batch group hpg seq dim"],
#             seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
#     ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
#
#         out = torch.empty_like(x, device=x.device, dtype=x.dtype)
#
#         for i in range(self.num_scales):
#             idx_scale_i = self.token_idx_per_scale[i]
#
#             rot_cos = getattr(self, f"cos{i}")[seq_id[..., :, :, idx_scale_i]]
#             rot_sin = getattr(self, f"sin{i}")[seq_id[..., :, :, idx_scale_i]]
#             out[..., :, :, idx_scale_i, :] = rot_cos * x[..., :, :, idx_scale_i, :] + rot_sin * self._rotate(
#                 x[..., :, :, idx_scale_i, :])
#
#         return out  # QZ: Eq 34 in the paper


class LearnedProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        self.max_len = max_len
        self.weight = nn.Parameter(
            torch.empty((max_len, self.proj_width, self.proj_width))
        )
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_len):
            nn.init.kaiming_uniform_(self.weight[idx], a=math.sqrt(5))

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        weight = self.weight[seq_id]
        return einsum(weight, x, "... out inp, ... inp -> ... out")


class QueryKeyProjection(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        proj_layer: type[Projection],
        kwargs: Optional[dict[str, Any]] = None,
        key_proj_layer: Optional[type[Projection]] = None,
        key_kwargs: Optional[dict[str, Any]] = None,
        partial_factor: Optional[
            tuple[float, float]
        ] = None,  # QZ: Only rotate part of embedding dimension
    ):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            num_groups=num_groups,
            **(kwargs or {}),
        )
        if key_proj_layer is None:
            self.key_proj = self.query_proj
        else:
            self.key_proj = key_proj_layer(
                proj_width=self.proj_width,
                num_heads=num_heads,
                num_groups=num_groups,
                **(key_kwargs or {}),
            )

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self) -> tuple[int, int, int]:
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        kv_id: Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
    ) -> tuple[
        Float[torch.Tensor, "*batch group hpg seq dim"],
        Float[torch.Tensor, "*batch group hpg seq dim"],
    ]:
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key
