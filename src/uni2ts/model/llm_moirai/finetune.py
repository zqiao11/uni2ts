
from collections import defaultdict
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    BertConfig,
    BertModel,
    BertTokenizer
)
import math
from uni2ts.model.moirai import MoiraiModule
from collections.abc import Callable, Sequence
from typing import Any, Optional
import lightning as L
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import nn
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    EvalCrop,
    EvalMaskedPrediction,
    EvalPad,
    ExtendMask,
    FixedPatchSizeConstraints,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    SpecifiedPatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
    AddSampleIndex,
    PadOutRangeTokens
)
from einops import repeat
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, LoraModel
from uni2ts.model.llm_moirai.resampler import PerceiverResampler, LinearProjector


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def get_data_description(data: str):
    if 'ETT' in data:
        file = 'ETT'
    else:
        file = data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


def calculate_lags(target):
    # x_enc: (Num channels, Time, 1)  Now: (Num channels, Num patches, 128) --> has been flatten to 1d (5000)
    q_fft = torch.fft.rfft(target, dim=-1)
    k_fft = torch.fft.rfft(target, dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    _, lags = torch.topk(corr, 5, dim=-1)
    return lags


class LlmMoiraiFinetune(L.LightningModule):
    """
    No sequence packing! Each item in batch is a individual sample.
    Pass specific context_length, prediction_length and patch_size when finetuning.

    """
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "sample_id",
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
        module_kwargs: dict[str, Any],  # Already provided in checkpoints of Moirai classes
        llm_kwargs: dict[str, Any],
        proj_kwargs: dict[str, Any],
        lora_kwargs: dict[str, Any],
        task_kwargs: dict[str, Any],  # If not provided, follow MoiraiFinetune's training strategy
        data: str,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        moirai_opt_mode: str = 'freeze',
    ):

        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."

        super().__init__()
        self.save_hyperparameters()

        # Name moirai as 'module' to enable to load ckpt from pretrained moirai.
        self.module = MoiraiModule(**module_kwargs)

        # Set params related to LLM.
        self.d_llm = llm_kwargs['d_llm']
        self.llm_layers = llm_kwargs['llm_layers']

        # Type of projector
        self.proj_type = proj_kwargs['projector']

        # Load dataset description based on 'data'
        self.data_description = get_data_description(data)

        # Set params related to the forecasting task.
        if task_kwargs['use_specified_task_config']:
            self.patch_size = task_kwargs['patch_size']
            self.prediction_length = task_kwargs['prediction_length']
            self.context_length = task_kwargs['context_length']
        else:
            self.patch_size, self.prediction_length, self.context_length = None, None, None

        # Lora config
        if lora_kwargs['use_lora']:
            lora_kwargs.pop('use_lora')
            self.lora_config = LoraConfig(**lora_kwargs)

    def init_after_loading_moirai(self):
        # Initialize the pretrained LLM and tokenizer.
        self.llm_model = self._set_llm_model(self.hparams.llm_kwargs['llm_model'])  # LLM is frozen
        self.llm_tokenizer = self._set_llm_tokenizer(self.hparams.llm_kwargs['llm_model'])
        # See https://huggingface.co/docs/transformers/main/en/model_doc/llama3
        if self.hparams.llm_kwargs['llm_model'] == 'LLAMA3':
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id
            self.llm_model.embed_tokens = nn.Embedding(self.llm_model.config.vocab_size,
                                                       self.llm_model.config.hidden_size,
                                                       padding_idx=self.llm_model.config.pad_token_id)

        # Todo: By default, Moirai uses multiple patch size for projection. Randomly select patch size for samples.
        #  If finetune with multi patch size, we need multiple Linear layers to project r_txt to TS patch.
        #  But note that all the patches are padded to max_patch_size, which is padded in transformation.
        #  So we also need to pad the e_txt to max_patch_size after projection.
        #  If patch_size is specified, then it is easier: Only train 1 Linear layer.
        #  For now, we only consider passing specified configs and using 1 Linear layer for simplicity.

        # Initialize projector
        if self.proj_type == 'Linear':
            self.projector = LinearProjector(in_features=self.d_llm,
                                             out_features=self.patch_size,
                                             dropout=self.hparams.proj_kwargs['dropout'],
                                             bias=self.hparams.proj_kwargs['bias'])

        elif self.proj_type == 'Resampler':
            self.projector = nn.Sequential(PerceiverResampler(dim=self.d_llm,
                                                              depth=3,
                                                              dim_head=64,
                                                              heads=8,
                                                              num_queries=self.hparams.proj_kwargs['num_queries'],
                                                              max_seq_len=512,
                                                              ff_mult=2),
                                           LinearProjector(in_features=self.d_llm,
                                                           out_features=self.patch_size,
                                                           dropout=self.hparams.proj_kwargs['dropout'],
                                                           bias=self.hparams.proj_kwargs['bias'])
                                           )

        elif self.proj_type == 'Honeybee':
            pass  # ToDo: D-abstractor
        else:
            raise ValueError("Unknown projector type")

        #  Use Lora for Moirai
        if self.lora_config is not None:
            self.module = LoraModel(self.module, self.lora_config, "default")
            print_trainable_parameters(self.module)

    def forward(
            self,
            target: Float[torch.Tensor, "*batch seq_len max_patch"],
            observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
            sample_id: Int[torch.Tensor, "*batch seq_len"],
            time_id: Int[torch.Tensor, "*batch seq_len"],
            variate_id: Int[torch.Tensor, "*batch seq_len"],
            prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
            patch_size: Int[torch.Tensor, "*batch seq_len"],
    ):

        # ToDo: ======== Now it is only for uni-variate. Not flatten multi-variate. =======

        prompt = self._get_sample_prompt(target, observed_mask, prediction_mask)

        #  Get LLM reprs of prompt.
        #  ToDo: Need to process prompt_len. max_length=2048 is not compatible with Moirai's max_length.
        prompt = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        prompt = prompt.to(self.llm_model.device)
        prompt_reprs = self.llm_model(input_ids=prompt.input_ids, #  (bs, num_prompt_patches, d_llm)
                                      attention_mask=prompt.attention_mask).last_hidden_state

        # ToDo: Add a Q-former/Some module to reduce prompt length to a fixed length!
        #  Samples from different batches have different prompt length.
        #  num_prompt_patches is larger than num_ts_patches

        # Use projector to map r_t to p_t. Prompt tokens are in the same space of TS patches.
        if isinstance(self.patch_size, int):
            prompt_tokens = self.projector(prompt_reprs)  # (bs, num_prompt_patches, patch_size)
        else:
            # Todo: Use the specific patch_size for each sample...
            prompt_tokens = self.projector[patch_size](prompt_reprs)


        # ToDo: Make prompt tokens have the same loc and std of context ts
        prompt_tokens = self.rescale(prompt_tokens, target, observed_mask, prediction_mask, sample_id, variate_id)


        batch_size = prompt_tokens.size(0)
        num_prompt_tokens = prompt_tokens.size(1)

        # Modify the masks and ids before prepending:
        # Pad last dim of prompt_tokens to max_patch_size.
        padded_prompt_prefix = torch.zeros((prompt_tokens.size(0), prompt_tokens.size(1), max(self.module.patch_sizes)),
                                           dtype=prompt_tokens.dtype,
                                           device=target.device)
        padded_prompt_prefix[:, :, :prompt_tokens.size(2)] = prompt_tokens  # (bs, num_prompt_patches, max_patch_size)

        # Create observed_mask for prompt tokens.
        prompt_observed_mask = torch.zeros((prompt_tokens.size(0), prompt_tokens.size(1), max(self.module.patch_sizes)),
                                           dtype=observed_mask.dtype,
                                           device=observed_mask.device)
        prompt_observed_mask[:, :, :prompt_tokens.size(2)] = True

        # Prepend prompt tokens to TS patches.
        target = torch.cat([padded_prompt_prefix, target], dim=1)
        observed_mask = torch.cat([prompt_observed_mask, observed_mask], dim=1)

        # Create sample_id for prompt and prepend. Not using packing, so all sample_id are 1.
        sample_id = torch.cat(
            [torch.ones((batch_size, num_prompt_tokens), dtype=sample_id.dtype, device=sample_id.device), sample_id],
            dim=1
        )

        # Todo: For uni-channel are as below. How to deal with flatten multi-channel? prompt as a new variate?
        # Treat prompt tokens as TS patches, so we need to add original time_id by num_prompt_tokens.
        # Then prepend it with prompt time_id: [0,..., num_prompt_tokens].
        # No sequence packing, so no worry about the ending patches are padded and with time id of zeros.
        time_id = torch.cat(
            [torch.arange(0, num_prompt_tokens, dtype=time_id.dtype, device=time_id.device).repeat(time_id.size(0), 1),
             time_id + num_prompt_tokens],
            dim=1
        )

        # ToDo: For uni-channel, duplicate. For flatten multi-channel, create a new variate.
        #  Cannot be the same as the ones in existing variates. Need to be in the max_dim range.
        variate_id = repeat(
            variate_id[:, 0],
            'batch -> batch seq_len',
            seq_len=variate_id.shape[1] + num_prompt_tokens
        )

        prediction_mask = torch.cat(
            [torch.zeros((batch_size, num_prompt_tokens), dtype=prediction_mask.dtype, device=prediction_mask.device),
             prediction_mask],
            dim=1
        )

        patch_size = repeat(
            patch_size[:, 0],
            'batch -> batch seq_len',
            seq_len=patch_size.shape[1] + num_prompt_tokens
        )

        prompt_batch = {"target": target,
                        "prediction_mask": prediction_mask,
                        "observed_mask": observed_mask,
                        "sample_id": sample_id,
                        "time_id": time_id,
                        "variate_id": variate_id,
                        "patch_size": patch_size}

        distr = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )

        loss = self.hparams.loss_func(
                pred=distr,
                target=target,
                prediction_mask=prediction_mask,
                observed_mask=observed_mask,
                sample_id=sample_id,
                variate_id=variate_id,
            )

        return distr, loss, prompt_batch

    #     preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
    #     return rearrange(preds, "n b ... -> b n ...")

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.llm_model.eval()

        distr, loss, prompt_batch = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )

        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            self.hparams.loss_func.__class__.__name__,
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

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        distr, val_loss, prompt_batch = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
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
                        field: prompt_batch[field]
                        for field in [
                            "target",
                            "prediction_mask",
                            "observed_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

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

        return val_loss

    def configure_optimizers(self) -> dict:
        """
        Set the optimizer:
            - Freeze params in LLM.
            - Always update params in Projector
            - Handle params in Moirai based on  moirai_opt_mode
        Follow the same optimizer setup as MoiraiFinetune.
        """
        # ToDo: Partial finetuning is not a good practice? Use Lora.
        # Set all params except the ones in projector to Non-trainable
        for pn, p in self.named_parameters():
            if not pn.startswith('projector'):
                p.requires_grad = False

        # 1. Freeze Moirai
        if self.hparams.moirai_opt_mode == 'freeze':
            pass
        # 2. Fully finetune Moirai
        elif self.hparams.moirai_opt_mode == 'full':
            for pn, p in self.named_parameters():
                if pn.startswith('module'):
                    p.requires_grad = True
        # 3. Partially finetune Moirai
        else:
            for mn, m in self.named_modules():
                if mn.startswith('module'):
                    # Finetune Norm layers
                    if 'Norm' in self.hparams.moirai_opt_mode:
                        if isinstance(m, RMSNorm):
                            for pn, p in m.named_parameters():
                                p.requires_grad = True
                    # Finetune Input Projection layers
                    if 'InProject' in self.hparams.moirai_opt_mode:
                        if isinstance(m, MultiInSizeLinear):
                            for pn, p in m.named_parameters():
                                p.requires_grad = True
                    # Finetune Output Projection layers
                    if 'OutProject' in self.hparams.moirai_opt_mode:
                        if isinstance(m, MultiOutSizeLinear):
                            for pn, p in m.named_parameters():
                                p.requires_grad = True

        decay = set()
        no_decay = set()

        # Decay
        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )

        # No decay
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
                if pn.endswith("bias"):  # All bias no decay
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):  # Weights in white decay
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):  # Weights in black no decay
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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
            SchedulerType.COSINE_WITH_RESTARTS,
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

    def rescale(self, prompt_tokens, target, observed_mask, prediction_mask, sample_id, variate_id):
        with torch.no_grad():
            loc_, scale_ = self.module.scaler(
                target,
                observed_mask * ~prediction_mask.unsqueeze(-1),  # Observed and not in prediction range
                sample_id,
                variate_id,
            )
            loc, scale = loc_.mean(dim=1, keepdim=False), scale_.mean(dim=1, keepdim=False)
        # (bs, n_promt, ps)
        # Example shapes
        bs, num_patch, patch_size = prompt_tokens.shape
        flattened_size = num_patch * patch_size

        # Step 1: Flatten the prompt_tokens tensor
        flattened_tokens = prompt_tokens.view(bs, flattened_size)

        # Step 2: Calculate the current mean and standard deviation
        current_mean = flattened_tokens.mean(dim=1, keepdim=True)
        current_std = flattened_tokens.std(dim=1, keepdim=True)

        # Step 3: Normalize the flattened tensor to have a mean of 0 and std of 1
        normalized_tokens = (flattened_tokens - current_mean) / current_std

        # Step 4: Scale and shift the normalized tensor to match the desired loc and scale
        processed_tokens = normalized_tokens * scale + loc

        # Reshape back to the original shape if needed
        processed_tokens = processed_tokens.view(bs, num_patch, patch_size)

        return processed_tokens




    def _set_llm_model(self, llm_model):
        """
        Adapt from https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
        """
        if llm_model == 'LLAMA3':
            self.llama_config = AutoConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            self.llama_config.num_hidden_layers = self.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            llm_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    config=self.llama_config,
            )

        if llm_model == 'LLAMA2':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = self.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    attn_implementation="eager",
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    attn_implementation="eager",
                    # load_in_4bit=True
                )

        elif llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = self.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

        elif llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = self.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
        else:
            raise Exception('LLM model is not defined')

        # Freeze LLM's parameters
        for param in llm_model.parameters():
            param.requires_grad = False

        return llm_model

    def _set_llm_tokenizer(self, llm_model):
        """
        Adapt from https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
        """
        if llm_model == 'LLAMA3':
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

        elif llm_model == 'LLAMA2':
            try:
                tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif llm_model == 'GPT2':
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif llm_model == 'BERT':
            try:
                tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token})
            tokenizer.pad_token = pad_token

        return tokenizer

    @torch.no_grad()
    def _get_sample_prompt(self, target, observed_mask, prediction_mask):
        prompt = []
        for b in range(target.size(0)):
            # Dataset Description
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.data_description};"
            )

            # Task description if task params are known.
            if self.prediction_length and self.context_length:
                prompt_ += f"Task description: forecast the next {str(self.prediction_length)} steps given the previous {str(self.context_length)} steps information; "

            # Todo: Compute the statistics for each sample.
            # In Moirai, each sample is a MTS? --> Compute channel-wise statistics.
            # It has been flattened. Need to use variate_id to compute.
            # Do we need to consider pad/missing values when computing statistics?

            # if self.hparams.prompt_statistics:

            # Mask indicating the observed tokens in context range for a sample b.
            mask = observed_mask[b] & ~prediction_mask[b].unsqueeze(-1).expand_as(observed_mask[b])
            valid_target = target[b][mask]  # A 1D tensor with observed context time steps
            min_value = torch.min(valid_target).item()
            max_value = torch.max(valid_target).item()
            median = torch.median(valid_target).item()
            lags = calculate_lags(valid_target)
            trend = valid_target.diff().sum()

            min_value_str = str(round(min_value, 4))
            max_value_str = str(round(max_value, 4))
            median_value_str = str(round(median, 4))
            lags_values_str = str(lags.tolist())
            prompt_ += (f"Input statistics: "
                        f"min value {min_value_str}, "
                        f"max value {max_value_str}, "
                        f"median value {median_value_str}, "
                        f"the trend of input is {'upward' if trend > 0 else 'downward'}, "
                        f"top 5 lags are : {lags_values_str};")

            prompt_ += "<|<end_prompt>|>"
            prompt.append(prompt_)

        return prompt

    @property
    def num_time_patches(self):
        return math.ceil(self.context_length / self.patch_size) + math.ceil(self.prediction_length / self.patch_size)

    @property
    def is_specified_all_config(self):
        return self.context_length and self.prediction_length and self.patch_size

    @property
    def train_transform_map(self,) -> dict[str | type, Callable[..., Transformation]]:
        """
        Transformation per sample for train dataset.
        Called in cli/train.py to process the training dataset.
        If 'wide', each data_entry is the entire record of a channel.
        If 'wide_multivariate', each data_entry is the entire records of all the channels.
        Initially, each 'target' is [(L, ), ..., (L, )], as _pa_column_to_numpy of HuggingFaceDatasetIndexer.
        """
        def default_train_transform():
            return (
                GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=FixedPatchSizeConstraints(self.patch_size) if self.patch_size else DefaultPatchSizeConstraints(),
                    offset=True,
                )
                + SpecifiedPatchCrop(
                    min_time_patches=self.hparams.min_patches if not self.is_specified_all_config else self.num_time_patches,
                    max_time_patches=None if not self.is_specified_all_config else self.num_time_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
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
                # Set the first tokens in context (1st patch) and last tokens in prediction (the last patch) as Nan.
                + PadOutRangeTokens(
                    prediction_pad=-self.prediction_length % self.patch_size,
                    context_pad=-self.context_length % self.patch_size,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                # Add a new field 'observed_mask'. Observed or missing: nan are False.
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
                # Patchify TS record into patches, based on its patch_size field.
                # No matter used patch_size, pad all the patches to max_patch_size.
                # Shape of patchified fields of a sample: (1, n_patch, max_patch_size)
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
                    randomize=True,
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
                # Add a new field "prediction_mask". Random mask patches in the end for prediction.
                # Mask ratio is uniformly sampled from [min_mask_ratio, max_mask_ratio]
                # For truncate_fields, truncate the part corresponding to prediction patches.
                # If no specific prediction length, different samples have different prediction masks?
                + (MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "sample_id", "observed_mask"),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                ) if self.prediction_length is None else EvalMaskedPrediction(
                    mask_length=math.ceil(self.prediction_length / self.patch_size),
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "sample_id", "observed_mask"),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                ))
                # Extend prediction_mask for "past_feat_dynamic_real" (If it exists)
                # set another prediction mask with all False for it in field "prediction_mask".
                # So there will be 2 items in field "prediction_mask".
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                # Turn item in field into nparray. Flat along time dimension then pack.
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
    def val_transform_map(self,) -> dict[str | type, Callable[..., Transformation]]:
        def default_val_transform(
            offset: int,               # Offset to val split. Range after offset is used.
            distance: int,             # distance bt prediction windows, equal to prediction_length
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
                    randomize=True,
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
                    truncate_fields=("variate_id", "time_id", "observed_mask"),
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
