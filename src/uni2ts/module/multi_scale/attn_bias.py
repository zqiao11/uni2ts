import abc

import torch
from einops import einsum, rearrange, repeat
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


class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)  # QZ: Each head has a scalr

    def post_init(self, num_scales: int):
        self.num_scales = num_scales
        self.scale_emb = nn.ModuleList(
            [nn.Embedding(num_embeddings=self.num_heads, embedding_dim=1) for _ in range(num_scales)]
        )
        # 初始化权重为零
        for embedding in self.scale_emb:
            nn.init.zeros_(embedding.weight)
        self.emb.weight.requires_grad = False  # Fix pretrained var_bias

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

        # Create empty tensors for query and kv embeddings
        bs = query_id.size(0)
        q_len, kv_len = query_id.size(-1), kv_id.size(-1)
        q_emb = torch.empty((q_len, self.num_heads, 1), device=query_id.device)
        kv_emb = torch.empty((kv_len, self.num_heads, 1), device=kv_id.device)
        index_by_variate = self.get_token_index_by_variate(query_id.squeeze())

        # Insert the emb based on variate_id
        for i in range(self.num_scales):
            index = index_by_variate[i]
            q_emb[index, :, :] = self.scale_emb[i].weight
            kv_emb[index, :, :] = self.scale_emb[i].weight

        # Matrix multiplication
        new_bias = einsum(q_emb, kv_emb, "q_len n_heads dim , kv_len n_heads dim -> n_heads q_len kv_len")
        new_bias = rearrange(
            new_bias,
            "(group hpg) q_len kv_len -> group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )

        new_bias = repeat(
            new_bias,
            "group hpg q_len kv_len -> bs group hpg q_len kv_len",
            bs=bs
        )

        bias = bias + 10000 * new_bias  # Amplify lr  5e-3

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