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
import re
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Generator, Optional

import lightning as L
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from gluonts.model import Input, InputSpec
from gluonts.torch import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    ExpandDimArray,
    TestSplitSampler,
    Transformation,
)
from gluonts.transform.split import TFTInstanceSplitter
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from uni2ts.common.torch_util import safe_div
from uni2ts.loss.packed import PackedNLLLoss as _PackedNLLLoss

from .module import MoiraiModule
from uni2ts.module.multi_scale.attention import GroupedQueryAttention


from uni2ts.module.position import (
    LearnedEmbedding,
    LearnedProjection,
    MultiScaleRotaryProjection
)

from uni2ts.module.multi_scale.attn_bias import BinaryAttentionBias

from peft import LoraConfig, LoraModel
from torch import nn

class SampleNLLLoss(_PackedNLLLoss):
    def reduce_loss(
        self,
        loss: Float[torch.Tensor, "batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "batch seq_len #dim"]],
        sample_id: Optional[Int[torch.Tensor, "batch seq_len"]],
        variate_id: Optional[Int[torch.Tensor, "batch seq_len"]],
    ) -> Float[torch.Tensor, "batch"]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loss = safe_div(loss, tobs)
        return (loss * mask).sum(dim=(-1, -2))


class MoiraiForecast(L.LightningModule):

    DOWN_SAMPLE_FACTOR = {
        "S": 60,  # Seconds to Minutes
        "T": 60,  # Minutes to Hours
        "H": 24,  # Hours to Days
        "D": 7,  # Days to Weeks
        "W": 4,  # Weeks to Months
        "M": 4,  # Months to Quarters
        "Q": 3,  # Quarters to Years
    }

    FREQ_ORDER = ["S", "T", "H", "D", "W", "M", "Q"]

    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiModule] = None,
        patch_size: int | str = "auto",
        num_samples: int = 100,
        pretrained_checkpoint_path: str = None,
        num_new_scales: int = 1,
        ds_factor: int = 2,
        r: int = 16,
        alpha: int = 16,
        use_lora: bool = False,
        lora_kwargs: Optional[dict[str, Any]] = None,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = MoiraiModule(**module_kwargs) if module is None else module  # module is None. Initialized by module_kwargs
        self.per_sample_loss_func = SampleNLLLoss()
        self.num_new_scales = num_new_scales

        self.ds_factor = ds_factor
        self.r = r
        self.alpha = alpha

        self.strict_loading = False

        self.token_idx_per_scale, self.base_ctx_token_idx = self._get_token_idx_per_scale()

        # Set Lora for Moirai
        if use_lora:
            self.lora_config = LoraConfig(**lora_kwargs)
            self.module = LoraModel(self.module, self.lora_config, "default")

        self.scale_weights = nn.Parameter(torch.ones(1 + num_new_scales))
        self.post_init()



    def post_init(self):
        """
        Initialize the new params added for Multi Scale.
        """
        # # ToDo: for time id & in_proj
        self.module.post_init(self.token_idx_per_scale, self.base_ctx_token_idx, self.hparams.patch_size)

        for layer in self.module.encoder.layers:
            # Check if the layer has an attribute named `self_attn` and if it is an instance of GroupedQueryAttention
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, GroupedQueryAttention):
                # Call post_init() method of the GroupedQueryAttention object
                layer.self_attn.init_multi_scale_modules(self.num_new_scales, self.r, self.alpha)

        # # Post init BinaryAttentionBias
        # for module in self.module.encoder.modules():
        #     if isinstance(module, BinaryAttentionBias):
        #         module.post_init(self.num_new_scales+1)

        # ToDo: for time id
        # for module in self.module.encoder.modules():
        #     if isinstance(module, MultiScaleRotaryProjection):
        #         module.post_init(self.token_idx_per_scale, self.base_ctx_token_idx)

        pass


    def _get_token_idx_per_scale(self):
        base_token_len = math.ceil(self.hparams.context_length / self.hparams.patch_size) + math.ceil(self.hparams.prediction_length / self.hparams.patch_size)
        ctx_len = self.hparams.context_length
        pred_len = self.hparams.prediction_length
        new_scale_token_len = []

        # New scales only include context part.
        for i in range(self.num_new_scales):
            # ctx_len = math.ceil(ctx_len / self.ds_factor)
            # ctx_token_len = math.ceil(ctx_len / self.hparams.patch_size)
            #
            # new_scale_token_len.append(ctx_token_len)

            ctx_len = math.ceil(ctx_len / self.ds_factor)
            ctx_token_len = math.ceil(ctx_len / self.hparams.patch_size)
            pred_len = math.ceil(pred_len / self.ds_factor)
            pred_token_len = math.ceil(pred_len / self.hparams.patch_size)
            new_scale_token_len.append(ctx_token_len+pred_token_len)

        token_idx_per_scale = [list(range(base_token_len))]

        for i in range(self.num_new_scales):
            start = base_token_len if i == 0 else end
            end = start + new_scale_token_len[i]

            index = list(range(start, end))
            token_idx_per_scale.append(index)

        base_ctx_token_len = math.ceil(self.hparams.context_length / self.hparams.patch_size)
        base_ctx_token_idx = list(range(base_ctx_token_len))

        return token_idx_per_scale, base_ctx_token_idx


    @contextmanager
    def hparams_context(
        self,
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        feat_dynamic_real_dim: Optional[int] = None,
        past_feat_dynamic_real_dim: Optional[int] = None,
        context_length: Optional[int] = None,
        patch_size: Optional[int | str] = None,
        num_samples: Optional[int] = None,
    ) -> Generator["MoiraiForecast", None, None]:
        kwargs = {
            "prediction_length": prediction_length,
            "target_dim": target_dim,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "context_length": context_length,
            "patch_size": patch_size,
            "num_samples": num_samples,
        }
        old_hparams = deepcopy(self.hparams)
        for kw, arg in kwargs.items():
            if arg is not None:
                self.hparams[kw] = arg

        yield self

        for kw in kwargs:
            self.hparams[kw] = old_hparams[kw]

    def create_predictor(
        self,
        batch_size: int,
        device: str = "auto",
    ) -> PyTorchPredictor:

        print("scale_weights: {}".format(torch.softmax(self.scale_weights, dim=0)))

        ts_fields = []
        if self.hparams.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")
        past_ts_fields = []
        if self.hparams.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")
        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
            future_length=self.hparams.prediction_length,
            observed_value_field="observed_target",
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )
        return PyTorchPredictor(
            input_names=self.prediction_input_names,
            prediction_net=self,
            batch_size=batch_size,
            prediction_length=self.hparams.prediction_length,
            input_transform=self.get_default_transform() + instance_splitter,
            device=device,
        )

    def describe_inputs(self, batch_size: int = 1) -> InputSpec:
        data = {
            "past_target": Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.target_dim,
                ),
                dtype=torch.float,
            ),
            "past_observed_target": Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.target_dim,
                ),
                dtype=torch.bool,
            ),
            "past_is_pad": Input(
                shape=(batch_size, self.past_length),
                dtype=torch.bool,
            ),
        }
        if self.hparams.feat_dynamic_real_dim > 0:
            data["feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        if self.hparams.past_feat_dynamic_real_dim > 0:
            data["past_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["past_observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        return InputSpec(data=data, zeros_fn=torch.zeros)

    @property
    def prediction_input_names(self) -> list[str]:
        return list(self.describe_inputs())

    @property
    def training_input_names(self):
        return self.prediction_input_names + ["future_target", "future_observed_values"]

    @property
    def past_length(self) -> int:
        return (
            self.hparams.context_length + self.hparams.prediction_length
            if self.hparams.patch_size == "auto"
            else self.hparams.context_length
        )

    # def context_token_length(self, patch_size: int) -> int:
    #     return math.ceil(self.hparams.context_length / patch_size)
    #
    # def prediction_token_length(self, patch_size) -> int:
    #     return math.ceil(self.hparams.prediction_length / patch_size)

    def context_token_length(self, patch_size: int, context_length: int) -> int:
        return math.ceil(context_length / patch_size)

    def prediction_token_length(self, patch_size: int, prediction_length: int) -> int:
        return math.ceil(prediction_length / patch_size)

    @property
    def max_patch_size(self) -> int:
        return max(self.module.patch_sizes)

    def forward(
        self,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        num_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        if self.hparams.patch_size == "auto":
            val_loss = []
            preds = []
            for patch_size in self.module.patch_sizes:
                val_loss.append(
                    self._val_loss(
                        patch_size=patch_size,
                        target=past_target[..., : self.past_length, :],
                        observed_target=past_observed_target[
                            ..., : self.past_length, :
                        ],
                        is_pad=past_is_pad[..., : self.past_length],
                        feat_dynamic_real=(
                            feat_dynamic_real[..., : self.past_length, :]
                            if feat_dynamic_real is not None
                            else None
                        ),
                        observed_feat_dynamic_real=(
                            observed_feat_dynamic_real[..., : self.past_length, :]
                            if observed_feat_dynamic_real is not None
                            else None
                        ),
                        past_feat_dynamic_real=(
                            past_feat_dynamic_real[
                                ..., : self.hparams.context_length, :
                            ]
                            if past_feat_dynamic_real is not None
                            else None
                        ),
                        past_observed_feat_dynamic_real=(
                            past_observed_feat_dynamic_real[
                                ..., : self.hparams.context_length, :
                            ]
                            if past_observed_feat_dynamic_real is not None
                            else None
                        ),
                    )
                )
                distr = self._get_distr(
                    patch_size,
                    past_target[..., -self.hparams.context_length :, :],
                    past_observed_target[..., -self.hparams.context_length :, :],
                    past_is_pad[..., -self.hparams.context_length :],
                    (
                        feat_dynamic_real[..., -self.past_length :, :]
                        if feat_dynamic_real is not None
                        else None
                    ),
                    (
                        observed_feat_dynamic_real[..., -self.past_length :, :]
                        if observed_feat_dynamic_real is not None
                        else None
                    ),
                    (
                        past_feat_dynamic_real[..., -self.hparams.context_length :, :]
                        if past_feat_dynamic_real is not None
                        else None
                    ),
                    (
                        past_observed_feat_dynamic_real[
                            ..., -self.hparams.context_length :, :
                        ]
                        if past_observed_feat_dynamic_real is not None
                        else None
                    ),
                )
                preds.append(
                    self._format_preds(
                        patch_size,
                        distr.sample(
                            torch.Size((num_samples or self.hparams.num_samples,))
                        ),
                        past_target.shape[-1],
                    )
                )
            val_loss = torch.stack(val_loss)
            preds = torch.stack(preds)
            idx = val_loss.argmin(dim=0)
            return preds[idx, torch.arange(len(idx), device=idx.device)]
        else:
            distr, distr_param = self._get_distr(
                self.hparams.patch_size,
                past_target,
                past_observed_target,
                past_is_pad,
                feat_dynamic_real,
                observed_feat_dynamic_real,
                past_feat_dynamic_real,
                past_observed_feat_dynamic_real,
            )
            preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
            return self._format_preds(
                self.hparams.patch_size, preds, past_target.shape[-1]
            )

    def _val_loss(
        self,
        patch_size: int,
        target: Float[torch.Tensor, "batch time tgt"],
        observed_target: Bool[torch.Tensor, "batch time tgt"],
        is_pad: Bool[torch.Tensor, "batch time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> Float[torch.Tensor, "batch"]:
        # convert format
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            past_target=target[..., : self.hparams.context_length, :],
            past_observed_target=observed_target[..., : self.hparams.context_length, :],
            past_is_pad=is_pad[..., : self.hparams.context_length],
            future_target=target[..., self.hparams.context_length :, :],
            future_observed_target=observed_target[
                ..., self.hparams.context_length :, :
            ],
            future_is_pad=is_pad[..., self.hparams.context_length :],
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )
        # get predictions
        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            torch.ones_like(time_id, dtype=torch.long) * patch_size,
        )
        val_loss = self.per_sample_loss_func(
            pred=distr,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed_mask,
            sample_id=sample_id,
            variate_id=variate_id,
        )
        return val_loss

    def _get_distr(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> Distribution:
        # convert format
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        # get predictions
        distr, distr_param = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            torch.ones_like(time_id, dtype=torch.long) * patch_size,
        )
        return distr, distr_param

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        prediction_length: int,
        past_observed_target: Bool[torch.Tensor, "batch past_seq tgt"],
    ) -> tuple[
        Int[torch.Tensor, "batch past_token"], Int[torch.Tensor, "batch future_token"]
    ]:
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(past_seq_id.cumsum(dim=-1) - 1, min=0)
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size, prediction_length),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> tuple[
        Float[torch.Tensor, "batch combine_seq patch"],  # target
        Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
        Int[torch.Tensor, "batch combine_seq"],  # sample_id
        Int[torch.Tensor, "batch combine_seq"],  # time_id
        Int[torch.Tensor, "batch combine_seq"],  # variate_id
        Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target_all_scales = []
        observed_mask_all_scales = []
        sample_id_all_scales = []
        time_id_all_scales = []
        variate_id_all_scales = []
        prediction_mask_all_scales = []
        dim_count = 0

        for i in range(self.num_new_scales + 1):

            target = []
            observed_mask = []
            sample_id = []
            time_id = []
            variate_id = []
            prediction_mask = []

            if i == 0:
                past_target = past_target
                past_observed_target = past_observed_target
                past_is_pad = past_is_pad
                future_target = future_target
                future_observed_target = future_observed_target
                future_is_pad = future_is_pad

                prediction_length = self.hparams.prediction_length
                context_length = self.hparams.context_length

            else:
                # Downsample
                past_target = self._downsample(past_target, left=True)
                past_observed_target = self._downsample(past_observed_target, left=True)
                past_is_pad = self._downsample(
                    past_is_pad.bool(), ds_factor=self.ds_factor, left=True
                ).int()
                future_target = self._downsample(future_target, left=False)
                future_observed_target = self._downsample(future_observed_target, left=False)
                future_is_pad = self._downsample(
                    future_is_pad.bool(), ds_factor=self.ds_factor, left=False
                ).int()
                context_length = math.ceil(context_length / self.ds_factor)
                prediction_length = math.ceil(prediction_length / self.ds_factor)

            if future_target is None:
                future_target = torch.zeros(
                    batch_shape
                    + (
                        prediction_length,
                        past_target.shape[-1],
                    ),
                    dtype=past_target.dtype,
                    device=device,
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size, past_target, -2, left=True
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size, future_target, -2, left=False
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )

            if future_observed_target is None:
                future_observed_target = torch.ones(
                    batch_shape
                    + (
                        prediction_length,
                        past_observed_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )

            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size, past_observed_target, -2, left=True
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size, future_observed_target, -2, left=False
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )

            if future_is_pad is None:
                future_is_pad = torch.zeros(
                    batch_shape + (prediction_length,),
                    dtype=torch.long,
                    device=device,
                )

            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True, value=1
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=past_target.shape[-1],
                    ),
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size,
                                    future_is_pad,
                                    -1,
                                    left=False,
                                    value=1,
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=past_target.shape[-1],
                    ),
                ]
            )

            past_seq_id, future_seq_id = (
                self._generate_time_id(  # (bs, num_pas_patch)  (bs, num_future_patch)
                    patch_size, prediction_length, past_observed_target
                )
            )

            time_id.extend(
                [past_seq_id] * past_target.shape[-1]
                + [future_seq_id] * past_target.shape[-1]
            )

            variate_id.extend(
                [
                    repeat(
                        torch.arange(past_target.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size, context_length),
                    ),
                    repeat(
                        torch.arange(past_target.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(
                            patch_size, prediction_length
                        ),
                    ),
                ]
            )
            dim_count += past_target.shape[-1]

            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size, context_length)
                            * past_target.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(
                                patch_size, prediction_length
                            )
                            * past_target.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

            # else:
            #     # Downsample
            #     past_target = self._downsample(past_target, left=True)
            #     past_observed_target = self._downsample(past_observed_target, left=True)
            #     past_is_pad = self._downsample(
            #         past_is_pad.bool(), ds_factor=self.ds_factor, left=True
            #     ).int()
            #
            #     target.extend(
            #         [
            #             torch.nn.functional.pad(
            #                 rearrange(
            #                     self._patched_seq_pad(
            #                         patch_size, past_target, -2, left=True
            #                     ),
            #                     "... (seq patch) dim -> ... (dim seq) patch",
            #                     patch=patch_size,
            #                 ),
            #                 (0, self.max_patch_size - patch_size),
            #             )
            #         ]
            #     )
            #
            #     observed_mask.extend(
            #         [
            #             torch.nn.functional.pad(
            #                 rearrange(
            #                     self._patched_seq_pad(
            #                         patch_size, past_observed_target, -2, left=True
            #                     ),
            #                     "... (seq patch) dim -> ... (dim seq) patch",
            #                     patch=patch_size,
            #                 ),
            #                 (0, self.max_patch_size - patch_size),
            #             ),
            #         ]
            #     )
            #
            #     sample_id.extend(
            #         [
            #             repeat(
            #                 reduce(
            #                     (
            #                         self._patched_seq_pad(
            #                             patch_size, past_is_pad, -1, left=True, value=1
            #                         )
            #                         == 0
            #                     ).int(),
            #                     "... (seq patch) -> ... seq",
            #                     "max",
            #                     patch=patch_size,
            #                 ),
            #                 "... seq -> ... (dim seq)",
            #                 dim=past_target.shape[-1],
            #             ),
            #         ]
            #     )
            #
            #     past_seq_id, future_seq_id = (
            #         self._generate_time_id(  # (bs, num_pas_patch)  (bs, num_future_patch)
            #             patch_size, prediction_length, past_observed_target
            #         )
            #     )
            #
            #     time_id.extend([past_seq_id] * past_target.shape[-1])
            #
            #     variate_id.extend(
            #         [
            #             repeat(
            #                 torch.arange(past_target.shape[-1], device=device)
            #                 + dim_count,
            #                 f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
            #                 past=self.context_token_length(patch_size, context_length),
            #             ),
            #         ]
            #     )
            #     dim_count += past_target.shape[-1]
            #
            #     prediction_mask.extend(
            #         [
            #             torch.zeros(
            #                 batch_shape
            #                 + (
            #                     self.context_token_length(patch_size, context_length)
            #                     * past_target.shape[-1],
            #                 ),
            #                 dtype=torch.bool,
            #                 device=device,
            #             ),
            #         ]
            #     )
            target = torch.cat(target, dim=-2)
            observed_mask = torch.cat(observed_mask, dim=-2)
            sample_id = torch.cat(sample_id, dim=-1)
            time_id = torch.cat(time_id, dim=-1)
            variate_id = torch.cat(variate_id, dim=-1)
            prediction_mask = torch.cat(prediction_mask, dim=-1)

            target_all_scales.append(target)
            observed_mask_all_scales.append(observed_mask)
            sample_id_all_scales.append(sample_id)
            time_id_all_scales.append(time_id)
            variate_id_all_scales.append(variate_id)
            prediction_mask_all_scales.append(prediction_mask)

        target = torch.cat(target_all_scales, dim=-2)
        observed_mask = torch.cat(observed_mask_all_scales, dim=-2)
        sample_id = torch.cat(sample_id_all_scales, dim=-1)
        time_id = torch.cat(time_id_all_scales, dim=-1)
        variate_id = torch.cat(variate_id_all_scales, dim=-1)
        prediction_mask = torch.cat(prediction_mask_all_scales, dim=-1)

        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    def _downsample(
        self, arr: torch.Tensor, ds_factor: int = 2, left: bool = True
    ) -> torch.Tensor:
        # Check if the input tensor is 2D (bs, time) or 3D (*bs, time, feature)
        if arr.ndim == 2:
            # 2D case: arr is (bs, time) without feature dimension
            *bs, time = arr.shape
            feature = None
            dim_time = -1
        elif arr.ndim == 3:
            # 3D case: arr is (*bs, time, feature)
            *bs, time, feature = arr.shape
            dim_time = -2
        else:
            raise ValueError(
                "Input tensor must be either 2D (bs, time) or 3D (*bs, time, feature)"
            )

        # Determine padding value based on tensor's dtype (False for Bool, NaN for float)
        if arr.dtype == torch.bool or arr.dtype == torch.int:
            pad_value = False  # Use False for Bool tensors
        else:
            pad_value = float("nan")  # Use NaN for float tensors

        # Apply padding to make the time dimension divisible by ds_factor
        arr = self._patched_seq_pad(
            ds_factor, arr, dim=dim_time, left=left, value=pad_value
        )

        new_arr_length = math.ceil(
            time / ds_factor
        )  # Compute the expected new time length after downsampling

        # If the tensor is 2D, add a singleton feature dimension to match the 3D case
        if arr.ndim == 2:
            arr = arr.unsqueeze(-1)  # Add feature dimension: (bs, time, 1)

        # Reshape the array to group time dimension into windows of size ds_factor
        arr = arr.reshape(*bs, -1, ds_factor, 1 if feature is None else feature)

        # Downsample based on tensor type
        if arr.dtype == torch.bool:
            arr_new = torch.any(
                arr, dim=-2
            )  # For Bool tensors, check if any value in the window is True
        else:
            arr_new = torch.nanmean(
                arr, dim=-2
            )  # For float tensors, compute the mean, ignoring NaN values

        # If the input was 2D, remove the singleton feature dimension
        if feature is None:
            arr_new = arr_new.squeeze(-1)

        assert (
            arr_new.shape[dim_time] == new_arr_length
        ), "Error occurs during downsampling!"

        return arr_new

    # def _format_preds(
    #     self,
    #     patch_size: int,
    #     preds: Float[torch.Tensor, "sample batch combine_seq patch"],
    #     target_dim: int,
    # ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
    #     start = target_dim * self.context_token_length(
    #         patch_size, self.hparams.context_length
    #     )
    #     end = start + target_dim * self.prediction_token_length(
    #         patch_size, self.hparams.prediction_length
    #     )
    #     preds = preds[..., start:end, :patch_size]
    #     preds = rearrange(
    #         preds,
    #         "sample ... (dim seq) patch -> ... sample (seq patch) dim",
    #         dim=target_dim,
    #     )[..., : self.hparams.prediction_length, :]
    #     return preds.squeeze(-1)


    def _format_preds(
        self,
        patch_size: int,
        preds: Float[torch.Tensor, "sample batch combine_seq patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:

        preds_all_scales = []

        for i in range(self.num_new_scales+1):
            if i == 0:
                start = target_dim * self.context_token_length(
                    patch_size, self.hparams.context_length
                )
                end = start + target_dim * self.prediction_token_length(
                    patch_size, self.hparams.prediction_length
                )
                preds_i = preds[..., start:end, :patch_size]
                preds_i = rearrange(
                    preds_i,
                    "sample ... (dim seq) patch -> ... sample (seq patch) dim",
                    dim=target_dim,
                )[..., : self.hparams.prediction_length, :]
                preds_all_scales.append(preds_i)

            else:
                context_length = math.ceil(self.hparams.context_length/(self.ds_factor**i))
                prediction_length = math.ceil(self.hparams.prediction_length / (self.ds_factor ** i))
                start = end + target_dim * self.context_token_length(
                    patch_size, context_length
                )
                end = start + target_dim * self.prediction_token_length(
                    patch_size, prediction_length
                )
                preds_i = preds[..., start:end, :patch_size]
                preds_i = rearrange(
                    preds_i,
                    "sample ... (dim seq) patch -> ... sample (seq patch) dim",
                    dim=target_dim,
                )[..., : prediction_length, :]
                preds_all_scales.append(preds_i)

        preds = None
        weight = torch.softmax(self.scale_weights, dim=0)

        for i in range(self.num_new_scales+1):
            preds_i = preds_all_scales[i]
            bs, num_samples, seq_len, channels = preds_i.shape
            scale_factor = self.hparams.prediction_length // seq_len

            if preds is None:
                preds = preds_i.repeat_interleave(scale_factor, dim=2) * weight[i]
            else:
                preds += preds_i.repeat_interleave(scale_factor, dim=2) * weight[i]

        return preds.squeeze(-1)

    def get_default_transform(self) -> Transformation:
        transform = AsNumpyArray(
            field="target",
            expected_ndim=1 if self.hparams.target_dim == 1 else 2,
            dtype=np.float32,
        )
        if self.hparams.target_dim == 1:
            transform += ExpandDimArray(field="target", axis=0)
        transform += AddObservedValuesIndicator(
            target_field="target",
            output_field="observed_target",
            dtype=bool,
        )

        if self.hparams.feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="feat_dynamic_real",
                output_field="observed_feat_dynamic_real",
                dtype=bool,
            )

        if self.hparams.past_feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="past_feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="past_feat_dynamic_real",
                output_field="past_observed_feat_dynamic_real",
                dtype=bool,
            )
        return transform
