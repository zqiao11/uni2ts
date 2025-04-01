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

from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
    PackedMSELoss,
    PackedMAELoss,
    PackedMAPELoss,
    PackedSMAPELoss
)
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from .module import MoiraiModule
from .finetune import MoiraiOnline


class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, gating_init=0.01, var_wise=True):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        if var_wise:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.weight.data.zero_()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(n_var, window_len))

    def forward(self, x):
        if self.var_wise:
            x = x + torch.tanh(self.gating).unsqueeze(1) * (torch.einsum('vi,iov->vo', x, self.weight) + self.bias)
        else:
            x = x + torch.tanh(self.gating).unsqueeze(1) * (torch.einsum('vi,io-> vo', x, self.weight) + self.bias)
        return x


class TafasMoiraiOnline(MoiraiOnline):

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
        zero_shot: bool = False,
        num_variates: int = None,
    ):
        super().__init__(
            min_patches=min_patches,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            max_dim=max_dim,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            module_kwargs=module_kwargs,
            module=module,
            num_samples=num_samples,
            beta1=beta1,
            beta2=beta2,
            loss_func=loss_func,
            val_metric=val_metric,
            lr=lr,
            weight_decay=weight_decay,
            log_on_step=log_on_step,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_size=patch_size,
            finetune_pattern=finetune_pattern,
            zero_shot=zero_shot,
        )

        self.in_cali = GCM(window_len=self.context_length, n_var=num_variates, var_wise=True)  # 先试下var
        self.out_cali = GCM(window_len=self.prediction_length, n_var=num_variates, var_wise=True)

    def _cali_target(self, target):
        # Apply Input Adapter (in_cali)
        context_series_with_pad = target[:, :self.num_ctx_patch, :self.patch_size].reshape(target.size(0), -1)
        context_series = context_series_with_pad[:, -self.context_length:]
        cali_context_series = self.in_cali(context_series)

        # Restore padding
        cali_context_series_with_pad = torch.cat([
            context_series_with_pad[:, :-self.context_length],
            cali_context_series,
        ], dim=1)

        # Restore shape (bs, num_ctx_patch, patch_size)
        cali_context_series_patched = cali_context_series_with_pad.reshape(target.size(0), self.num_ctx_patch,
                                                                        self.patch_size)
        cali_target = target.clone()
        cali_target[:, :self.num_ctx_patch, :self.patch_size] = cali_context_series_patched

        return cali_target

    def _cali_pred(self, pred):
        # Apply Output Adapter (out_cali)
        pred_series_with_pad = pred[:, -self.num_pred_patch:, :self.patch_size].reshape(pred.size(0), -1)
        pred_series = pred_series_with_pad[:, :self.prediction_length]
        cali_pred_series = self.out_cali(pred_series)

        # Restore padding
        cali_pred_series_with_pad = torch.cat([
            cali_pred_series,
            pred_series_with_pad[:, self.prediction_length:],
        ], dim=1)

        # Restore shape (bs, num_pred_patch, patch_size)
        cali_pred_patch = cali_pred_series_with_pad.view(pred.size(0), self.num_pred_patch, self.patch_size)

        cali_pred = pred.clone()
        cali_pred[:, -self.num_pred_patch:, :self.patch_size] = cali_pred_patch  # Replace only the modified part

        return cali_pred

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

        cali_target = self._cali_target(target)

        distr = self.module(
            target=cali_target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return distr

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        if batch_idx % self.prediction_length != 0:  # Skip the samples with overlapped horizon
            return None

        self.online_val_step(batch, batch_idx)

        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )

        pred = distr.rsample(torch.Size((self.hparams.num_samples,)))  # ToDo: rsample保证梯度可以反传
        pred = torch.median(pred, dim=0).values  # ToDo: median不可导
        cali_pred = self._cali_pred(pred)

        loss = self.hparams.loss_func(
            pred=cali_pred,  # distr --> pred
            **{
                field: batch[field]
                for field in [
                    "target",
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
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

    @torch.no_grad()
    def online_val_step(self, batch, batch_idx):
        self.eval()
        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )

        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )

        val_metrics = (
            self.hparams.val_metric
            if isinstance(self.hparams.val_metric, list)
            else [self.hparams.val_metric]
        )

        pred = distr.sample(torch.Size((self.hparams.num_samples,)))
        pred = torch.median(pred, dim=0).values
        pred = self._cali_pred(pred)

        for metric_func in val_metrics:
            metric = metric_func(
                pred=pred,
                **{
                    field: batch[field]
                    for field in [
                        "target",
                        "prediction_mask",
                        "observed_mask",
                        "sample_id",
                        "variate_id",
                    ]
                },
            )
            if isinstance(metric_func, PackedMAELoss):
                self.online_metrics['mae'].append(metric.item())
            if isinstance(metric_func, PackedMSELoss):
                self.online_metrics['mse'].append(metric.item())
            if isinstance(metric_func, PackedMAPELoss):
                self.online_metrics['mape'].append(metric.item())
            if isinstance(metric_func, PackedSMAPELoss):
                self.online_metrics['smape'].append(metric.item())

            self.log(
                f"val/{metric_func.__class__.__name__}",
                metric,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        self.train()

    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        cali_params = set()

        if self.finetune_pattern == 'full':
            pass
        elif self.finetune_pattern == 'freeze_ffn':
            for pn, p in self.named_parameters():
                if "ffn" in pn:
                    p.requires_grad = False
        elif self.finetune_pattern == 'head_only':
            for pn, p in self.named_parameters():
                if "param_proj" not in pn:
                    p.requires_grad = False
        elif self.finetune_pattern == 'adapter_only':
            for pn, p in self.named_parameters():
                if "cali" not in pn:
                    p.requires_grad = False
        else:
            raise ValueError("Unsupported finetune pattern {}".format(self.finetune_pattern))

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
                if "cali" in fpn:
                    cali_params.add(fpn)


                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        self.updated_params = param_dict

        inter_params = decay & no_decay & cali_params
        union_params = decay | no_decay | cali_params
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
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(cali_params)]
                ),
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.lr,  # Set different learning rate for cali parameters
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
