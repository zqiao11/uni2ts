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
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.multi_scale.transformer import TransformerEncoder
from uni2ts.module.ts_embed import MultiInSizeLinear


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
        num_new_scales: int = 2,
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

        # self.new_scale_encoding = nn.ModuleList(
        #     [
        #         nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        #         for _ in range(num_new_scales)
        #     ]
        # )

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
            var_attn_bias_layer=partial(BinaryAttentionBias),  # ToDo: 这个var attn bias可以改
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

        self.num_new_scales = num_new_scales

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
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        reprs = self.in_proj(scaled_target, patch_size)
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

        # # ToDo: Plan 1. Add learnable variate embedding to the tokens of each new scales
        # masked_reprs = self.add_new_scale_embedding(
        #     masked_reprs, sample_id=sample_id, variate_id=variate_id
        # )

        reprs = self.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )  # (bs, seq_len, max_patch)
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr

    def add_new_scale_embedding(
        self,
        reprs: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ):

        # Each item has the same number of samples.
        num_samples_per_item = torch.max(sample_id).item()

        # Each sample has the same seq_len
        sample_mask = torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2))
        sample_seq_len = sample_mask[0][0].int().sum().item()

        variate_diff = torch.diff(variate_id[0], dim=-1)
        variate_change_points = torch.nonzero(variate_diff).flatten() + 1
        variate_change_points = torch.cat(
            [
                variate_change_points,
                torch.tensor([variate_id.shape[1]]).to(variate_id.device),
            ]
        )

        for i in range(self.num_new_scales):
            index_scale_i = []  # index of new scale i in all the samples

            for sample in range(num_samples_per_item):
                variate_start_idx = (
                    variate_change_points[i].item() + sample * sample_seq_len
                )
                variate_end_idx = (
                    variate_change_points[i + 1].item() + sample * sample_seq_len
                )
                index_scale_i.append((variate_start_idx, variate_end_idx))

            for start, end in index_scale_i:
                reprs[..., start:end, :] += self.new_scale_encoding[i].weight

        return reprs
