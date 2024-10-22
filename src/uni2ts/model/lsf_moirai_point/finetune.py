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
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution

from uni2ts.distribution import StudentTOutput
from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
)
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddSampleIndex,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    EvalCrop,
    EvalMaskedPrediction,
    EvalPad,
    ExtendMask,
    FinetunePatchCrop,
    FixedPatchSizeConstraints,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    Identity,
    ImputeTimeSeries,
    MaskedPrediction,
    MaskedPredictionGivenFixedConfig,
    MaskOutRangePaddedTokens,
    PackFields,
    PatchCrop,
    PatchCropGivenFixedConfig,
    Patchify,
    SelectFields,
    SequencifyField,
    Transformation,
)

from .module import MoiraiModule


class MoiraiFinetune(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }

    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        context_length: Optional[int | list[int]] = None,
        prediction_length: Optional[int | list[int]] = None,
        patch_size: Optional[int] = None,
        finetune_pattern: str | list[str] = "full",
        # full
        # in_proj
        # param_proj
        # norm: norm1 norm2
        # mask_encoding
        # self_attn: q_proj, k_proj, v_proj, q_norm, k_norm, var_attn_bias, out_proj
        # ffn: 2 * fc + 1 fc_gating.
        # No PE, implicitly included in q_proj & k_proj as RoPE
        # Except in_proj & param_poj, other params only have weight, without bias.
    ):
        # assert (module is not None) or (
        #     module_kwargs is not None
        # ), "if module is not provided, module_kwargs is required"
        # assert (
        #     num_warmup_steps <= num_training_steps
        # ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = MoiraiModule(**module_kwargs) if module is None else module

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.finetune_pattern = finetune_pattern

        self.criterion = torch.nn.MSELoss()

    def replace_forecast_head(self):
        seq_len = math.ceil(self.context_length / self.patch_size) + math.ceil(
            self.prediction_length / self.patch_size
        )
        self.module.replace_forecast_head(
            seq_len=seq_len, pred_len=self.prediction_length
        )

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> torch.Tensor:
        # QZ: Directly returns the predicted TS

        pred = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return pred

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        pred = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )

        label = self.get_prediction_label(batch)
        loss = self.criterion(pred, label)

        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/MSE",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        pred = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )

        label = self.get_prediction_label(batch)
        val_loss = self.criterion(pred, label)

        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/MSE",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        return val_loss

    def get_prediction_label(self, batch):
        target = batch["target"]
        prediction_mask = batch["prediction_mask"]
        observed_mask = batch["observed_mask"]

        mask = torch.logical_and(prediction_mask.unsqueeze(-1), observed_mask)
        pred_patches = target[mask].view(target.size(0), -1)

        return pred_patches

    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        if "full" in self.finetune_pattern:
            pass
        else:
            for param in self.parameters():
                param.requires_grad = False

        if "head" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "head" in pn:
                    p.requires_grad = True

        # Unfreeze the corresponding params
        if "param_proj" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "param_proj" in pn:
                    p.requires_grad = True

        if "in_proj" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "in_proj" in pn:
                    p.requires_grad = True

        if "norm" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "norm1" in pn or "norm2" in pn:
                    p.requires_grad = True

        if "mask" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "mask_encoding" in pn:
                    p.requires_grad = True

        if "ffn" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "ffn" in pn:
                    p.requires_grad = True

        if "q_proj" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "q_proj" in pn:
                    p.requires_grad = True

        if "k_proj" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "k_proj" in pn:
                    p.requires_grad = True

        if "v_proj" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "v_proj" in pn:
                    p.requires_grad = True

        if "attn_norm" in self.finetune_pattern:  #
            for pn, p in self.named_parameters():
                if "self_attn.q_norm" in pn or "self_attn.k_norm" in pn:
                    p.requires_grad = True

        if "var_attn_bias" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "var_attn_bias" in pn:
                    p.requires_grad = True

        if "out_proj" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if "out_proj" in pn:
                    p.requires_grad = True

        if "studentT" in self.finetune_pattern:
            for pn, p in self.named_parameters():
                if (
                    "param_proj.proj.components.0" in pn
                    or "param_proj.proj.weights_logits" in pn
                ):
                    p.requires_grad = True

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        self.trainable_params = param_dict

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.CONSTANT,  # Use constant lr scheduler
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    @property
    def train_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_train_transform(
            distance: int,
            prediction_length: int,
            context_length: int,
            patch_size: int,
        ):
            return (
                GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                    offset=True,
                )
                + FinetunePatchCrop(
                    distance,
                    prediction_length,
                    context_length,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                )
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + EvalPad(
                    prediction_pad=-prediction_length % patch_size,
                    context_pad=-context_length % patch_size,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + AddSampleIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    sample_id_field="sample_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(
                        "variate_id",
                        "time_id",
                        "observed_mask",
                        "sample_id",
                    ),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="sample_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_train_transform)

    @property
    def val_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_val_transform(
            offset: int,
            distance: int,
            prediction_length: int,
            context_length: int,
            patch_size: int,
        ):
            return (
                GetPatchSize(
                    min_time_patches=2,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                    offset=True,
                )
                + EvalCrop(
                    offset,
                    distance,
                    prediction_length,
                    context_length,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                )
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + EvalPad(
                    prediction_pad=-prediction_length % patch_size,
                    context_pad=-context_length % patch_size,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + AddSampleIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    sample_id_field="sample_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(
                        "variate_id",
                        "time_id",
                        "observed_mask",
                        "sample_id",
                    ),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="sample_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_val_transform)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        Modify state_dict to only save trainable params.
        Note the default state_dict saved by PL converts all params to require_grads=False
        """
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        filtered_state = {
            name: tensor
            for name, tensor in state.items()
            if name in self.trainable_params
        }
        return filtered_state
