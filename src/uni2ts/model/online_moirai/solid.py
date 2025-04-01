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
    PackedMAELoss
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
    Transformation
)

from .module import MoiraiModule
from .finetune import MoiraiOnline


def get_period(dataset_name):
    if "etth" in dataset_name:
        period = 24
    elif "ettm" in dataset_name:
        period = 96
    elif "electricity" in dataset_name:
        period = 24
    elif "ECL" in dataset_name:
        period = 24
    elif "traffic" in dataset_name.lower():
        period = 24
    elif "illness" in dataset_name.lower():
        period = 52.142857
    elif "weather" in dataset_name.lower():
        period = 144
    elif "Exchange" in dataset_name:
        period = 1
    elif "WTH_informer" in dataset_name:
            period = 24
    else:
        period = 1
    return period


class SolidMoiraiOnline(MoiraiOnline):

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
        dataset_name: str = None,
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

        self.period = get_period(dataset_name)
        self.lambda_T = 500  # Number of saved past samples
        self.lambda_N = 10  # Top-k to retrieve
        self.lambda_period = 0.1

        self.buffer_idx = 0
        self.buffer_filled = False

    @property
    def flag_proactive_adaption(self):
        if self.buffer_idx < self.lambda_T and not self.buffer_filled:
            return False
        else:
            return True

    def _update_buffer_with_cur_sample(self, batch):
        self.buffer[self.buffer_idx] = batch["target"].clone()
        self.buffer_om[self.buffer_idx] = batch["observed_mask"].clone()
        self.buffer_idx = (self.buffer_idx + 1) % self.lambda_T
        if not self.buffer_filled and self.buffer_idx == 0:
            self.buffer_filled = True  # 当 buffer 写满一次后，标记为已满

    @torch.no_grad()
    def _retrieve_ctx_batch(self, batch, batch_idx):
        real_buffer = torch.cat([
            self.buffer[self.buffer_idx:],
            self.buffer[:self.buffer_idx]
        ], dim=0)[0:self.lambda_T]

        real_buffer_om = torch.cat([
            self.buffer_om[self.buffer_idx:],
            self.buffer_om[:self.buffer_idx]
        ], dim=0)[0:self.lambda_T]

        # 从 buffer 选取相似的样本
        indices = []
        threshold = self.period * self.lambda_period
        t = batch_idx * self.prediction_length + self.context_length
        phase = t % self.period

        # 计算符合条件的历史索引
        for i in range(self.lambda_T):
            buffer_t = t - (i + 1)  # 计算历史时间戳
            if abs(phase - (buffer_t % self.period)) < threshold:
                indices.append((self.lambda_T - i - 1) % self.lambda_T)  # 计算 buffer 中的实际索引

        # 构造 batch
        candidates = real_buffer[indices]
        distances = ((candidates - batch['target'].unsqueeze(0)) ** 2).sum(dim=(1, 2, 3))
        topk_values, topk_indices = torch.topk(-distances, k=self.lambda_N)
        ctx_target = candidates[topk_indices]
        ctx_observed_mask = real_buffer_om[indices][topk_indices]
        ctx_target = ctx_target.reshape(-1, *ctx_target.shape[2:])
        ctx_observed_mask = ctx_observed_mask.reshape(-1, *ctx_observed_mask.shape[2:])
        new_bs = ctx_target.size(0)

        from einops import repeat
        ctx_batch = {
            field: repeat(batch[field][0], "l ... -> new_bs l ...", new_bs=new_bs)
            for field in [
                "prediction_mask",
                "sample_id",
                "time_id",
                "variate_id",
                "patch_size"
            ]
        }
        ctx_batch['target'] = ctx_target
        ctx_batch['observed_mask'] = ctx_observed_mask
        return ctx_batch

    def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Initialize ring buffers, each has (lambda_T + pred_len) slots. Only the first lambda_T slots are retrieved.
        if self.buffer_idx==0 and not self.buffer_filled:
            buffer = torch.empty(
                (self.lambda_T + self.prediction_length, *batch["target"].shape),
                dtype=batch["target"].dtype,
                device=batch["target"].device,
            )
            buffer_om = torch.empty(
                (self.lambda_T + self.prediction_length, *batch["observed_mask"].shape),
                dtype=batch["observed_mask"].dtype,
                device=batch["observed_mask"].device,
            )
            self.register_buffer("buffer", buffer)
            self.register_buffer("buffer_om", buffer_om)

        if batch_idx % self.prediction_length != 0:  # Skip the samples with overlapped horizon
            return None

        if self.flag_proactive_adaption:
            # Retrieve past samples with no overlapping horizon with the current sample. Eval after finetune.
            ctx_batch = self._retrieve_ctx_batch(batch, batch_idx)
        else:
            # Finetune with current sample solely if there's no sufficient past samples. Eval before finetune.
            ctx_batch = batch
            self.online_val_step(batch, batch_idx)

        distr = self(
            **{field: ctx_batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        loss = self.hparams.loss_func(
            pred=distr,
            **{
                field: ctx_batch[field]
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
            ctx_batch["sample_id"].max(dim=1).values.sum() if "sample_id" in ctx_batch else None
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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % self.prediction_length == 0 and self.flag_proactive_adaption:
            self.online_val_step(batch, batch_idx)
        self._update_buffer_with_cur_sample(batch)  # Save samples into buffers