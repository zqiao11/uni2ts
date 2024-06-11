"""Resampler-based projectors
"""

import torch
from einops import rearrange, repeat
from torch import einsum, nn


class LinearProjector(nn.Module):
    def __init__(
        self, in_features, out_features, inner_features=512, dropout=0, bias=True
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, inner_features, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_features, out_features, bias=bias),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights"""
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, x):
        return self.model(x)


"""" Perceiver Resampler from Flamingo 
https://github.com/lucidrains/flamingo-pytorch/blob/main/setup.py """


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, queries):
        """
        einstein notation
        b - batch
        t - time / sequence length
        d - dimension
        h - head
        """
        x = self.norm_x(x)
        queries = self.norm_queries(queries)

        b, t, h = *x.shape[:2], self.heads

        q = self.to_q(queries)

        # the paper differs from Perceiver in which they also concat the key / values derived from the queries to be attended to
        kv_input = torch.cat((x, queries), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = (
            rearrange(q, "b t (h d) -> b h t d", h=h),
            rearrange(k, "b t (h d) -> b h t d", h=h),
            rearrange(v, "b t (h d) -> b h t d", h=h),
        )

        q = q * self.scale

        # attention

        sim = torch.einsum("b h i d, b h j d  -> b h i j", q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_queries=64,
        max_seq_len=512,
        ff_mult=4,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.pos_emb = nn.Parameter(
            torch.randn(max_seq_len, dim)
        )  # Positional embeddings

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x has the shape [batch, seq_len, dim]
        b, t, d = x.shape
        x = x + self.pos_emb[:t]  # add pos_emb to different frames
        queries = repeat(self.queries, "n d -> b n d", b=b)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        return self.norm(queries)  # (batch, num_queries, dim)
