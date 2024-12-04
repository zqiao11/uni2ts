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

from functools import partial

import torch
import torch.nn.functional as F
from einops import reduce
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
from torch.utils._pytree import tree_map

from uni2ts.common.torch_util import mask_fill, packed_attention_mask
from uni2ts.distribution import DistributionOutput
from uni2ts.module.multi_scale.transformer import TransformerEncoder
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    BinaryAttentionBias,
    CrossVariateAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
    MultiScaleRotaryProjection
)
from uni2ts.module.ts_embed import MultiInSizeLinear

import copy

def encode_distr_output(
    distr_output: DistributionOutput,
) -> dict[str, str | float | int]:
    """Serialization function for DistributionOutput"""

    def _encode(val):
        if not isinstance(val, DistributionOutput):
            return val

        return {
            "_target_": f"{val.__class__.__module__}.{val.__class__.__name__}",
            **tree_map(_encode, val.__dict__),
        }

    return _encode(distr_output)


def decode_distr_output(config: dict[str, str | float | int]) -> DistributionOutput:
    """Deserialization function for DistributionOutput"""
    return instantiate(config, _convert_="all")


class MoiraiModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    Subclasses huggingface_hub.PyTorchModelHubMixin to support loading from HuggingFace Hub.
    """

    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],  # tuple[int, ...] | list[int]
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
    ):
        """
        :param distr_output: distribution output object
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_sizes: sequence of patch sizes
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(
                BinaryAttentionBias
                # CrossVariateAttentionBias,
                # num_vars=4   # ToDo: 这个num_vars得提供外部接口
            ),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=MultiScaleRotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),  # 之前的partial factor是0-0.5

                # QueryKeyProjection,
                # proj_layer=RotaryProjection,  # ToDo: 可以改
                # kwargs=dict(max_len=max_seq_len),
                # partial_factor=(0.0, 0.5),  # 之前的partial factor是0-0.5
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,   # True by default
            d_ff=None,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

        self.time_id_q_proj = nn.ParameterList()
        self.time_id_k_proj = nn.ParameterList()

        self.in_proj_new_scales = nn.ParameterList()

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        """
        Defines the forward pass of MoiraiModule.
        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to distribution parameters
        6. Return distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :return: predictive distribution
        """

        # Map time id for new scales.
        # Key: base scale context tokens; Value: base scale time id
        time_id = time_id.to(torch.float)
        idx_kv = self.base_ctx_token_idx
        key = target[..., idx_kv, :self.ps].clone()  # (bs, len0, dim)
        value = time_id[..., idx_kv].clone().unsqueeze(-1).to(dtype=torch.float)  # (bs, len0, 1)

        for i in range(1, self.num_scales):
            idx_scale_i = self.token_idx_per_scale[i]

            query = target[..., idx_scale_i, :self.ps].clone()   # (bs, leni, dim)
            query = self.time_id_q_proj[i - 1](query)
            key = self.time_id_k_proj[i - 1](key)  # (bs, len0, dim)

            # Generate attn_mask. Make sure each query only attend to the keys in its down-sampling range.
            attn_mask = self.generate_segmented_attn_mask(query, key, 2**i)

            # mapped_time_id is float time id on the original scale. (bs, len_i, 1)
            mapped_time_id = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
            )

            time_id[..., idx_scale_i] = mapped_time_id.squeeze()

        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale    # ToDo: If use conv for DS, consider to modify here?



        # reprs = self.in_proj(scaled_target, patch_size)

        reprs_all_scales = []
        for i in range(0, self.num_scales):
            idx_scale_i = self.token_idx_per_scale[i]

            if i == 0:
                reprs_base = self.in_proj(scaled_target[..., idx_scale_i, :], patch_size[..., idx_scale_i])
                reprs_all_scales.append(reprs_base)

            else:
                reprs_new_scale = self.in_proj_new_scales[i - 1](scaled_target[..., idx_scale_i, :], patch_size[..., idx_scale_i])
                reprs_all_scales.append(reprs_new_scale)

        reprs = torch.cat(reprs_all_scales, dim=-2)


        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

        reprs = self.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )  # (bs, seq_len, max_patch)
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr

    def post_init(self, token_idx_per_scale, base_ctx_token_idx, patch_size):
        self.token_idx_per_scale = token_idx_per_scale
        self.base_ctx_token_idx = base_ctx_token_idx

        self.num_scales = len(token_idx_per_scale)

        self.ps = patch_size

        # Assign Q and K for each new scale
        for scale in range(1, self.num_scales):
            self.time_id_q_proj.append(nn.Linear(self.ps, self.ps))
            self.time_id_k_proj.append(nn.Linear(self.ps, self.ps))

            self.in_proj_new_scales.append(copy.deepcopy(self.in_proj))

    def generate_segmented_attn_mask(self, query, key, k):
        """
        生成一个 attention mask，使得 query 的位置 i 只能注意到 key 的范围 [k*i, k*(i+1)-1]。

        参数：
        bs: batch size
        len_q: query 的序列长度
        len_k: key 的序列长度
        k: 每个 query 索引范围内 key 的跨度

        返回：
        attn_mask: BoolTensor，shape = (bs, len_q, len_k)
        """
        bs, len_q, len_k = query.shape[0], query.shape[1], key.shape[1]
        # 创建基础的 mask
        attn_mask = torch.zeros(len_q, len_k, dtype=torch.bool)

        for i in range(len_q):
            # 定义 query 的位置 i 对应的 key 范围
            start = i * k
            end = min((i + 1) * k, len_k)  # 防止超出 len_k
            attn_mask[i, start:end] = True  # 允许注意的范围

        # 扩展到 batch 维度
        attn_mask = attn_mask.unsqueeze(0).expand(bs, -1, -1)

        return attn_mask.to(query.device)
