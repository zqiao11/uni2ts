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

from uni2ts.module.multi_scale.attention import GroupedQueryAttention
from peft import LoraConfig, LoraModel

from collections import deque


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


class SolidMoiraiOnline(L.LightningModule):
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
        dataset_name: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = MoiraiModule(**module_kwargs) if module is None else module

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.finetune_pattern = finetune_pattern

        self.period = get_period(dataset_name)

        self.lambda_T = 1
        self.lambda_N = 1
        # self.lambda_T = 500
        # self.lambda_N = 10
        self.lambda_period = 0.1

        self.buffer = None
        self.buffer_observed_mask = None
        self.buffer_idx = 0
        self.buffer_filled = False

        self.online_metrics = {'mae': [], 'mse': []}


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
        distr = self.module(
            target=target,
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
        with torch.no_grad():
            if self.buffer is None:
                # 在第一次 training_step 时，根据 batch["target"] 进行初始化
                self.buffer = torch.empty(
                    (self.lambda_T + self.prediction_length, *batch["target"].shape),
                    dtype=batch["target"].dtype,
                    device=batch["target"].device,
                    requires_grad=False
                )
                self.buffer_observed_mask = torch.empty(
                    (self.lambda_T + self.prediction_length, *batch["observed_mask"].shape),
                    dtype=batch["observed_mask"].dtype,
                    device=batch["observed_mask"].device,
                    requires_grad=False
                )

            # 跳过不训练/测试的样本
            if batch_idx % self.prediction_length != 0 or not self.buffer_filled:
                self.buffer[self.buffer_idx] = batch["target"].clone()
                self.buffer_observed_mask[self.buffer_idx] = batch["observed_mask"].clone()
                self.buffer_idx = (self.buffer_idx + 1) % self.lambda_T

                if not self.buffer_filled and self.buffer_idx == 0:
                    self.buffer_filled = True  # 当 buffer 写满一次后，标记为已满

                return None

            else:
                real_buffer = torch.cat([
                    self.buffer[self.buffer_idx:],
                    self.buffer[:self.buffer_idx]
                ], dim=0)[0:-self.prediction_length]

                real_buffer_observed_mask = torch.cat([
                    self.buffer_observed_mask[self.buffer_idx:],
                    self.buffer_observed_mask[:self.buffer_idx]
                ], dim=0)[0:-self.prediction_length]

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
                subset = real_buffer[indices]
                distances = ((subset - batch['target'].unsqueeze(0)) ** 2).sum(dim=(1, 2, 3))
                topk_values, topk_indices = torch.topk(-distances, k=self.lambda_N)
                ctx_target = subset[topk_indices]
                ctx_observed_mask = real_buffer_observed_mask[indices][topk_indices]
                ctx_target = ctx_target.reshape(-1, *ctx_target.shape[2:])
                ctx_observed_mask = ctx_observed_mask.reshape(-1, *ctx_observed_mask.shape[2:])
                new_bs = ctx_target.size(0)

                # ctx_target = real_buffer[:self.lambda_T]
                # ctx_observed_mask = real_buffer_observed_mask[:self.lambda_T]
                # ctx_target = ctx_target.reshape(-1, *ctx_target.shape[2:])
                # ctx_observed_mask = ctx_observed_mask.reshape(-1, *ctx_observed_mask.shape[2:])
                # new_bs = ctx_target.size(0)

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

                self.buffer[self.buffer_idx] = batch["target"].clone()
                self.buffer_observed_mask[self.buffer_idx] = batch["observed_mask"].clone()
                self.buffer_idx = (self.buffer_idx + 1) % self.lambda_T

                if not self.buffer_filled and self.buffer_idx == 0:
                    self.buffer_filled = True  # 当 buffer 写满一次后，标记为已满


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
        if batch_idx % self.prediction_length == 0:
            self.online_val_step(batch, batch_idx)

    @torch.no_grad()
    def online_val_step(self, batch, batch_idx):
        self.eval()
        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        val_loss = self.hparams.loss_func(
            pred=distr,
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
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                if isinstance(metric_func, PackedPointLoss):
                    pred = distr.sample(torch.Size((self.hparams.num_samples,)))
                    pred = torch.median(pred, dim=0).values
                elif isinstance(metric_func, PackedDistributionLoss):
                    pred = distr
                else:
                    raise ValueError(f"Unsupported loss function: {metric_func}")

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
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        self.updated_params = param_dict

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
            offset: int,
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
                # + InterpolateToPeriod(
                #
                # )
                + EvalPad(
                    prediction_pad=-prediction_length % patch_size,
                    context_pad=-context_length % patch_size,
                    # prediction_pad=-128 % patch_size,
                    # context_pad=-4000 % patch_size,
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
                    # mask_length=math.ceil(128 / patch_size),
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
            if name in self.updated_params
        }
        return filtered_state