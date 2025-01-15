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

        self.dim = dim
        self.num_new_scales = None


    def init_multi_scale_modules(self, num_new_scales, r, alpha):
        self.num_new_scales = num_new_scales

        self.r = r
        self.alpha = alpha

        print(" ******** r={}, alpha={} *********".format(r, alpha))

        # Initialize parameter lists
        self.q_A = nn.ParameterList()
        self.q_B = nn.ParameterList()
        self.k_A = nn.ParameterList()
        self.k_B = nn.ParameterList()
        self.v_A = nn.ParameterList()
        self.v_B = nn.ParameterList()

        # freeze q k v
        self.q_proj.requires_grad_(False)
        self.k_proj.requires_grad_(False)
        self.v_proj.requires_grad_(False)

        for _ in range(1+num_new_scales):
            # Append the new parameters for the current scale
            self.q_A.append(nn.Parameter(torch.randn((r, self.dim), dtype=torch.float) * 0.01))
            self.k_A.append(nn.Parameter(torch.randn((r, self.dim), dtype=torch.float) * 0.01))
            self.v_A.append(nn.Parameter(torch.randn((r, self.dim), dtype=torch.float) * 0.01))

            self.q_B.append(nn.Parameter(torch.zeros((self.dim, r), dtype=torch.float)))
            self.k_B.append(nn.Parameter(torch.zeros((self.dim, r), dtype=torch.float)))
            self.v_B.append(nn.Parameter(torch.zeros((self.dim, r), dtype=torch.float)))

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

    def apply_lora(self,
                   input: torch.Tensor,
                   layer: nn.Linear,
                   A: nn.Parameter,
                   B: nn.Parameter,
                   ):
        """
        在给定的线性层上应用 LoRA。
        """
        W_no_grad = layer.weight.detach()
        lora_update = (self.alpha / self.r) * (B @ A)  # (in_features, out_features)
        W_lora = W_no_grad + lora_update  # (in_features, out_features)
        out = torch.matmul(input, W_lora.T)

        return out

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

        # query = self.q_proj(query)
        # key = self.k_proj(key)
        # value = self.v_proj(value)

        # ToDO: Apply lora for each scale
        updated_query = query.clone()
        updated_key = key.clone()
        updated_value = value.clone()

        if self.num_new_scales is not None:
            index_by_variate = self.get_token_index_by_variate(query_var_id)
            assert torch.equal(query_var_id, kv_var_id), "query_var_id is different from kv_var_id"

            for i in range(1 + self.num_new_scales):
                index = index_by_variate[i]
                query_scale = query[..., index, :]
                key_scale = key[..., index, :]
                value_scale = value[..., index, :]

                updated_query[..., index, :] = self.apply_lora(query_scale, self.q_proj, self.q_A[i], self.q_B[i])
                updated_key[..., index, :] = self.apply_lora(key_scale, self.k_proj, self.k_A[i], self.k_B[i])
                updated_value[..., index, :] = self.apply_lora(value_scale, self.v_proj, self.v_A[i], self.v_B[i])

        query = updated_query
        key = updated_key
        value = updated_value

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
        attn_mask = self._update_attn_mask(     # (bs, 6, 1, len, len)
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


    #     # For query of each scale, map the all the kv_time_id to the same scale.
    #     out = torch.zeros_like(query)
    #     for i in range(self.num_new_scales+1):
    #         index = index_by_variate[i]
    #         query_i = query[..., :, :, index, :]               # ... group hpg q_len dim
    #         query_i_time_id = query_time_id[..., :, :, index]  # "*batch #group #hpg q_len"
    #         attn_mask_i = attn_mask[..., :, :, index, :]
    #
    #         mapped_kv_time_id = self.time_id_mapping(kv_time_id, index_by_variate, target_scale=i)
    #
    #         query_i, key = self._qk_proj(
    #             query_i,
    #             key,
    #             query_var_id=query_var_id,
    #             kv_var_id=kv_var_id,
    #             query_time_id=query_i_time_id,
    #             kv_time_id=mapped_kv_time_id,
    #         )
    #
    #         out_i = F.scaled_dot_product_attention(
    #             query_i,
    #             key,
    #             value,
    #             attn_mask=attn_mask_i,
    #             dropout_p=self.attn_dropout_p,
    #             scale=self.softmax_scale,
    #         )
    #
    #         out[..., :, :, index, :] = out_i
    #     out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
    #     return self.out_proj(out)
    #
    # def time_id_mapping(self, time_id,  index_by_variate, target_scale, factor=2):
    #     """
    #     Map time_id to target_scale
    #      time_id: "*batch #group #hpg kv_len"
    #     """
    #
    #     for i in range(0, self.num_new_scales+1):
    #         if i == target_scale:
    #             continue
    #         else:
    #             idx_scale_i = index_by_variate[i]  # 包含ctx和pred
    #             time_id = time_id.to(torch.float)
    #             time_id[..., :, :, idx_scale_i] = time_id[..., :, :, idx_scale_i] * (factor ** (i-target_scale))
    #             # time_id[..., :, :, idx_scale_i] = time_id[..., :, :, idx_scale_i] + (2 ** i-1) * torch.sigmoid(self.time_id_bias[i])
    #
    #     return time_id



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
