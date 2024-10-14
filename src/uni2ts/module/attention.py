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

from .position import AttentionBias, QueryKeyProjection

# TODO: Support returning weights
# TODO: Support caching (return past_key_value)


def split_consecutive_indices(indices):
    if not indices:
        return []
    split_result = [[indices[0]]]
    for idx in indices[1:]:
        if idx == split_result[-1][-1] + 1:  # 如果当前索引与前一个连续
            split_result[-1].append(idx)
        else:
            split_result.append([idx])  # 否则开一个新的分组
    return split_result


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

        self.query_filmed_generator = nn.ModuleList(
            [
                nn.Linear(in_features=dim, out_features=2 * 12),  # each scale's length
                nn.Linear(in_features=dim, out_features=2 * 6)
            ]
        )

        self.key_filmed_generator = nn.ModuleList(
            [
                nn.Linear(in_features=dim, out_features=2 * 12),  # each scale's length
                nn.Linear(in_features=dim, out_features=2 * 6)
            ]
        )

        # self.value_filmed_generator = nn.ModuleList(
        #     [
        #         nn.Linear(in_features=dim, out_features=2 * 12),  # each scale's length
        #         nn.Linear(in_features=dim, out_features=2 * 6)
        #     ]
        # )


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
            attn_bias = attn_bias + self.var_attn_bias(  # 2 scales for same-variate and different-variate positions
                query,
                key,
                query_id=query_var_id,
                kv_id=kv_var_id,
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

    def get_token_index_of_target_variate_per_sample(
        self,
        variate_id: Int[torch.Tensor, "*batch q_len"],
        attn_mask: Bool[torch.Tensor, "*batch q_len kv_len"],
        target_variate: int = 1  # Default to variate_id = 1
    ):

        # ToDo: 当前假设batch中所有的variate_id和attn_mask是一样的
        #    不同samples的variate_id是一样的，但不同item的sample的个数有可能不一样吗？
        #    有的batch是不是并没装满max_seq_len? 比如val如果不fill的话？

        variate_id = variate_id[0]
        attn_mask = attn_mask[0]

        # Step 1: 计算当前 batch 中最大的 variate_id
        max_variate_id = variate_id.max().item()

        # Step 2: 使用无序组合，确保 (variate_1, variate_2) 和 (variate_2, variate_1) 的映射相同
        # 扩展维度以广播到 q_len x q_len 矩阵
        variate_id_min = torch.minimum(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2))
        variate_id_max = torch.maximum(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2))

        # 使用偏移量生成唯一组合值，确保无序组合的对称性
        variate_pair_matrix = variate_id_min * (max_variate_id + 1) + variate_id_max

        # Step 3: 找到 variate_id = target_variate 的组合
        target_combination_value = target_variate * (max_variate_id + 1) + target_variate
        variate_target_mask = (variate_pair_matrix == target_combination_value)

        # Step 4: 用 attn_mask 进行 AND 运算，筛选出每个 sample 内 variate_id = target_variate 的组合
        final_mask = variate_target_mask & attn_mask

        # Step 5: 映射回 len 维度的 Tensor
        # 沿 kv_len 维度使用 torch.any 找到 q_len 中哪些位置为 True
        mapped_mask = final_mask.any(dim=-1)

        # Step 6: 使用 torch.diff 或 torch.split 进行分组
        # 利用 torch.nonzero 和 torch.diff 找到分组边界
        indices = torch.nonzero(mapped_mask).squeeze().tolist()

        # 调用切分函数
        index_per_sample = split_consecutive_indices(indices)

        return index_per_sample

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
        # var_id0 = query_var_id[0].to('cpu').numpy()
        # attn_mask0 = attn_mask[0].to('cpu').numpy()

        # for scale in range(2):  # number_of scales:
        #     assert torch.equal(query_var_id, kv_var_id), "query_var_id is different from kv_var_id"
        #     index_per_sample = self.get_token_index_of_target_variate_per_sample(query_var_id, attn_mask, target_variate=scale+1)
        #     for index in index_per_sample:
        #         query_i = query[..., index, :]
        #
        #         out = self.query_filmed_generator[scale](torch.mean(query_i, dim=1))
        #         query_weight, query_bias = out[:, :int(out.size(-1) / 2)], out[:, int(out.size(-1) / 2):]
        #         query[..., index, :] = query_weight.unsqueeze(-1) * query_i + query_bias.unsqueeze(-1)
        #
        #         key_i = key[..., index, :]
        #         key_film_out = self.key_filmed_generator[scale](torch.mean(key_i, dim=1))
        #         key_weight, key_bias = key_film_out[:, :int(key_film_out.size(-1) / 2)], key_film_out[:,
        #                                                                                          int(key_film_out.size(
        #                                                                                              -1) / 2):]
        #         key[..., index, :] = key_weight.unsqueeze(-1) * key_i + key_bias.unsqueeze(-1)
        #
        #         # value_i = value[..., index, :]
        #         # value_film_out = self.value_filmed_generator[scale](torch.mean(value_i, dim=1))
        #         # value_weight, value_bias = value_film_out[:, :int(value_film_out.size(-1) / 2)], value_film_out[:,
        #         #                                                                                  int(value_film_out.size(
        #         #                                                                                      -1) / 2):]
        #         # value[..., index, :] = value_weight.unsqueeze(-1) * value_i + value_bias.unsqueeze(-1)


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


class FilmedGroupedQueryAttention(nn.Module):
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

        # self.filmed_generator = nn.ModuleList(
        #     [
        #         nn.Linear(in_features=dim, out_features=2*length)
        #         for _ in range(num_new_scales)
        #     ]
        # )

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
            attn_bias = attn_bias + self.var_attn_bias(  # 2 scales for same-variate and different-variate positions
                query,
                key,
                query_id=query_var_id,
                kv_id=kv_var_id,
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
            # Mask out positions from differnet samples
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

        # ToDo: Plan B: Directly apply different Film on query / key to different scales. W.o revising RoPE

        query_var_id, kv_var_id = self._get_var_id(query, key, query_var_id, kv_var_id)
        query_time_id, kv_time_id = self._get_time_id(
            query,
            key,
            query_time_id,
            kv_time_id,
        )

        # Add attn_bias
        attn_mask = self._update_attn_mask(  # ... group hpg q_len kv_len
            attn_mask,
            query,
            key,
            query_var_id=query_var_id,  # ... 1 1 q_len
            kv_var_id=kv_var_id,  # ... 1 1 kv_len
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