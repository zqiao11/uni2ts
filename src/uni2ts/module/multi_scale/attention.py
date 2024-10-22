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

import math
from collections.abc import Callable
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int
from torch import nn

from ..position import AttentionBias, QueryKeyProjection

# TODO: Support returning weights
# TODO: Support caching (return past_key_value)


def native_scaled_dot_product_attention(
    query: Float[torch.Tensor, "*batch group hpg q_len dim"],
    key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
    value: Float[torch.Tensor, "*batch group hpg kv_len dim"],
    attn_mask: Optional[
        Bool[torch.Tensor, "*batch #group #hpg q_len kv_len"]
        | Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]
    ] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
):
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_weight)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask
        attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        bias: bool = True,
        norm_layer: Optional[type[nn.Module] | partial[nn.Module]] = nn.LayerNorm,
        softmax_scale: Optional[float] = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        time_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        var_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
        time_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.heads_per_group = num_heads // num_groups
        self.var_attn_bias = var_attn_bias() if var_attn_bias is not None else None
        self.time_attn_bias = time_attn_bias() if time_attn_bias is not None else None
        self.var_qk_proj = var_qk_proj() if var_qk_proj is not None else None
        self.time_qk_proj = time_qk_proj() if time_qk_proj is not None else None

        self.softmax_scale = softmax_scale or 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.q_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
        )
        self.attn_dropout_p = attn_dropout_p
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        # Todo: Create these mudules based on num_new_scales & ds_factor & seq_len
        # base = 24  # 24 for ETTH1, 12 for ETTm1
        # self.query_filmed_generator = nn.ModuleList(
        #     [
        #         nn.Linear(in_features=dim, out_features=2 * base),
        #         nn.Linear(in_features=dim, out_features=2 * base // 2),
        #         nn.Linear(in_features=dim, out_features=2 * base // 4),
        #     ]
        # )
        #
        # self.key_filmed_generator = nn.ModuleList(
        #     [
        #         nn.Linear(in_features=dim, out_features=2 * base),
        #         nn.Linear(in_features=dim, out_features=2 * base // 2),
        #         nn.Linear(in_features=dim, out_features=2 * base // 4),
        #     ]
        # )
        #
        # # self.value_filmed_generator = nn.ModuleList(
        # #     [
        # #         nn.Linear(in_features=dim, out_features=2 * 12),  # each scale's length
        # #         nn.Linear(in_features=dim, out_features=2 * 6)
        # #     ]
        # # )

    def _get_var_id(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_var_id: Optional[Int[torch.Tensor, "*batch q_len"]],
        kv_var_id: Optional[Int[torch.Tensor, "*batch kv_len"]],
    ) -> tuple[
        Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
    ]:
        if self.var_attn_bias is not None or self.var_qk_proj is not None:
            if query_var_id is None:
                query_var_id = repeat(
                    torch.zeros((), device=query.device, dtype=torch.long),
                    f" -> {' '.join(map(str, query.shape[:-4]))} 1 1 {query.shape[-2]}",
                )
            else:
                query_var_id = rearrange(query_var_id, "... q_len -> ... 1 1 q_len")

            if kv_var_id is None:
                kv_var_id = repeat(
                    torch.zeros((), device=key.device, dtype=torch.long),
                    f" -> {' '.join(map(str, key.shape[:-4]))} 1 1 {key.shape[-2]}",
                )
            else:
                kv_var_id = rearrange(kv_var_id, "... kv_len -> ... 1 1 kv_len")

        return query_var_id, kv_var_id

    def _get_time_id(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_time_id: Optional[Int[torch.Tensor, "*batch q_len"]],
        kv_time_id: Optional[Int[torch.Tensor, "*batch kv_len"]],
    ) -> tuple[
        Optional[Int[torch.Tensor, "*batch 1 1 q_len"]],
        Optional[Int[torch.Tensor, "*batch 1 1 kv_len"]],
    ]:
        if self.time_attn_bias is not None or self.time_qk_proj is not None:
            if query_time_id is None:
                query_time_id = repeat(
                    torch.arange(
                        query.shape[-2], device=query.device, dtype=torch.long
                    ),
                    f"q_len -> {' '.join(map(str, query.shape[:-4]))} 1 1 q_len",
                )
            else:
                query_time_id = rearrange(query_time_id, "... q_len -> ... 1 1 q_len")

            if kv_time_id is None:
                kv_time_id = repeat(
                    torch.arange(key.shape[-2], device=key.device, dtype=torch.long),
                    f"kv_len -> {' '.join(map(str, key.shape[:-4]))} 1 1 kv_len",
                )
            else:
                kv_time_id = rearrange(kv_time_id, "... kv_len-> ... 1 1 kv_len")

        return query_time_id, kv_time_id

    def _update_attn_mask(
        self,
        attn_mask: Optional[Bool[torch.Tensor, "*batch q_len kv_len"]],
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_var_id: Optional[Int[torch.Tensor, "*batch 1 1 q_len"]] = None,
        kv_var_id: Optional[Int[torch.Tensor, "*batch 1 1 kv_len"]] = None,
        query_time_id: Optional[Int[torch.Tensor, "*batch 1 1 q_len"]] = None,
        kv_time_id: Optional[Int[torch.Tensor, "*batch 1 1 kv_len"]] = None,
    ) -> Optional[
        Bool[torch.Tensor, "*batch #group #hpg q_len kv_len"]
        | Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]
    ]:
        if attn_mask is not None:
            attn_mask = rearrange(
                attn_mask,
                "... q_len kv_len -> ... 1 1 q_len kv_len",
            )

        attn_bias = 0

        # Bias scalars are different in different groups.
        if self.var_attn_bias is not None:
            attn_bias = (
                attn_bias
                + self.var_attn_bias(  # 2 scales for same-variate and different-variate positions
                    query,
                    key,
                    query_id=query_var_id,
                    kv_id=kv_var_id,
                )
            )

        if self.time_attn_bias is not None:
            attn_bias = attn_bias + self.time_attn_bias(
                query,
                key,
                query_id=query_time_id,
                kv_id=kv_time_id,
            )

        attn_mask = (
            attn_mask
            if isinstance(attn_bias, int)
            else (
                attn_bias
                if attn_mask is None
                else attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
                # Mask out positions from different samples
            )
        )
        return attn_mask

    def _qk_proj(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_var_id: Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        kv_var_id: Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
        query_time_id: Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        kv_time_id: Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
    ) -> tuple[
        Float[torch.Tensor, "*batch group hpg q_len dim"],
        Float[torch.Tensor, "*batch group hpg kv_len dim"],
    ]:
        if self.var_qk_proj is not None:
            query, key = self.var_qk_proj(
                query, key, query_id=query_var_id, kv_id=kv_var_id
            )

        if self.time_qk_proj is not None:
            query, key = self.time_qk_proj(
                query, key, query_id=query_time_id, kv_id=kv_time_id
            )

        return query, key

    def get_token_index_by_variate(
        self,
        variate_id: Int[torch.Tensor, "*batch q_len"],
    ):

        # batch中所有的variate_id和attn_mask是一样的
        variate_id = variate_id[0]
        max_variate_id = variate_id.max().item()
        indices_by_variate = []
        for vid in range(max_variate_id + 1):
            indices = torch.nonzero(variate_id == vid, as_tuple=True)[0]
            indices_by_variate.append(indices)

        return indices_by_variate

    def forward(
        self,
        query: Float[torch.Tensor, "*batch q_len dim"],
        key: Float[torch.Tensor, "*batch kv_len dim"],
        value: Float[torch.Tensor, "*batch kv_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch q_len kv_len"]] = None,
        query_var_id: Optional[Int[torch.Tensor, "*batch q_len"]] = None,
        kv_var_id: Optional[Int[torch.Tensor, "*batch kv_len"]] = None,
        query_time_id: Optional[Int[torch.Tensor, "*batch q_len"]] = None,
        kv_time_id: Optional[Int[torch.Tensor, "*batch kv_len"]] = None,
    ) -> Float[torch.Tensor, "*batch q_len dim"]:
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # ToDo: Plan B: Directly apply different Film on query / key to different scales. W.o revising RoPE
        # index_by_variate = self.get_token_index_by_variate(query_var_id)
        #
        # for scale in range(3):  # ToDO: number_of scales:
        #     assert torch.equal(query_var_id, kv_var_id), "query_var_id is different from kv_var_id"
        #     index = index_by_variate[scale+1]
        #     query_scale = query[..., index, :]
        #     query_film_out = self.query_filmed_generator[scale](torch.mean(query_scale, dim=1))
        #     query_weight, query_bias = query_film_out[:, :int(query_film_out.size(-1) / 2)], query_film_out[:, int(query_film_out.size(-1) / 2):]
        #     query[..., index, :] = query_weight.unsqueeze(-1) * query_scale + query_bias.unsqueeze(-1)
        #
        #     key_scale = key[..., index, :]
        #     key_film_out = self.key_filmed_generator[scale](torch.mean(key_scale, dim=1))
        #     key_weight, key_bias = key_film_out[:, :int(key_film_out.size(-1) / 2)], key_film_out[:,
        #                                                                                      int(key_film_out.size(
        #                                                                                          -1) / 2):]
        #     key[..., index, :] = key_weight.unsqueeze(-1) * key_scale + key_bias.unsqueeze(-1)
        #
        #     # value_i = value[..., index, :]
        #     # value_film_out = self.value_filmed_generator[scale](torch.mean(value_i, dim=1))
        #     # value_weight, value_bias = value_film_out[:, :int(value_film_out.size(-1) / 2)], value_film_out[:,
        #     #                                                                                  int(value_film_out.size(
        #     #                                                                                      -1) / 2):]
        #     # value[..., index, :] = value_weight.unsqueeze(-1) * value_i + value_bias.unsqueeze(-1)

        query = self.q_norm(
            rearrange(
                query,
                "... q_len (group hpg dim) -> ... group hpg q_len dim",  # (bs, 6, 1, 512, 64)
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        key = self.k_norm(
            repeat(
                key,
                "... kv_len (group dim) -> ... group hpg kv_len dim",
                group=self.num_groups,
                hpg=self.heads_per_group,
            )
        )
        value = repeat(
            value,
            "... kv_len (group dim) -> ... group hpg kv_len dim",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )

        query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
        query_time_id, kv_time_id = self._get_time_id(
            query,
            key,
            query_time_id,
            kv_time_id,
        )

        # Add attn_bias
        attn_mask = self._update_attn_mask(
            attn_mask,
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        # RoPE
        query, key = self._qk_proj(
            query,
            key,
            query_var_id=query_var_id,
            kv_var_id=kv_var_id,
            query_time_id=query_time_id,
            kv_time_id=kv_time_id,
        )

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            scale=self.softmax_scale,
        )
        out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
        return self.out_proj(out)


class MultiQueryAttention(GroupedQueryAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        norm_layer: Optional[type[nn.Module] | partial[nn.Module]] = nn.LayerNorm,
        softmax_scale: Optional[float] = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        time_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        var_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
        time_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_groups=1,
            bias=bias,
            norm_layer=norm_layer,
            softmax_scale=softmax_scale,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )


class MultiHeadAttention(GroupedQueryAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        norm_layer: Optional[type[nn.Module] | partial[nn.Module]] = nn.LayerNorm,
        softmax_scale: Optional[float] = None,
        attn_dropout_p: float = 0.0,
        var_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        time_attn_bias: Optional[Callable[[], AttentionBias]] = None,
        var_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
        time_qk_proj: Optional[Callable[[], QueryKeyProjection]] = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_groups=num_heads,
            bias=bias,
            norm_layer=norm_layer,
            softmax_scale=softmax_scale,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
