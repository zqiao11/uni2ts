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

import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import nn


class AttentionBias(nn.Module, abc.ABC):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
    ):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]: ...


class RelativeAttentionBias(AttentionBias):
    def __init__(self, num_buckets: int, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)
        self.emb = nn.Embedding(
            num_embeddings=num_buckets, embedding_dim=self.num_heads
        )

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]:
        raise NotImplementedError


class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)  # QZ: Each head has a scalr

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]:
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = rearrange(  # try to avoid advanced indexing
            ~ind * weight[:1] + ind * weight[1:],
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        return bias


class CrossVariateAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int, num_vars: int):
        super().__init__(dim, num_heads, num_groups)
        # QZ: Initialize a learnable embedding for each variate
        # Each embedding should contain num_heads embeddings with d dimension?
        # Each head has num_vars embeddings with d dimension
        self.num_vars = num_vars
        self.emb = nn.ModuleList(
            [nn.Embedding(num_embeddings=self.num_heads, embedding_dim=dim//4) for _ in range(num_vars)]
        )

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]:
        # Create empty tensors for query and kv embeddings
        bs = query.size(0)
        q_len, kv_len = query.size(-2), kv_id.size(-2)
        q_emb = torch.empty((q_len, self.num_heads, self.dim//4), device=query.device)
        kv_emb = torch.empty((kv_len, self.num_heads, self.dim//4), device=key.device)
        index_by_variate = self.get_token_index_by_variate(query_id)

        # Insert the emb based on variate_id
        for i in range(self.num_vars):
            index = index_by_variate[i]
            q_emb[index, :, :] = self.emb[i].weight
            kv_emb[index, :, :] = self.emb[i].weight

        # Matrix multiplication
        bias = einsum(q_emb, kv_emb, "q_len n_heads dim , kv_len n_heads dim -> n_heads q_len kv_len")
        bias = rearrange(
            bias,
            "(group hpg) q_len kv_len -> bs group hpg q_len kv_len",
            bs=bs,
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        return bias

    def get_token_index_by_variate(
        self,
        variate_id: Int[torch.Tensor, "*batch q_len"],
    ):
        # batch中所有的variate_id是一样的
        variate_id = variate_id[0]
        max_variate_id = variate_id.max().item()
        indices_by_variate = []
        for vid in range(max_variate_id + 1):
            indices = torch.nonzero(variate_id == vid, as_tuple=True)[0]
            indices_by_variate.append(indices)

        return indices_by_variate

class LinearAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)
        m = 0.5 ** ((1 + torch.arange(self.num_heads)) * (8 / self.num_heads))
        m = rearrange(
            m,
            "(group hpg) -> group hpg 1 1",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        self.register_buffer("m", m)

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]:
        ind = kv_id.unsqueeze(-2) - query_id.unsqueeze(-1)
        return self.m * ind
