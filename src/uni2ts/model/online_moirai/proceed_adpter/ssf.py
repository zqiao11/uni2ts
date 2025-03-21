
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Adaptation


def add_ssf_(parent_module: nn.Module, module_name: str, freeze_weight: bool,
             merge_weights=True, load_weights=True, **kwargs):
    old_module = getattr(parent_module, module_name)
    if isinstance(old_module, nn.Linear):
        new_module = Linear(in_features=old_module.in_features, out_features=old_module.out_features,
                            bias=old_module.bias is not None,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    else:
        raise NotImplementedError
    if load_weights:
        new_module.load_state_dict(old_module.state_dict(), strict=False)
    setattr(parent_module, module_name, new_module)


class SSF(Adaptation):
    def __init__(self, out_features: int, fan_in_fan_out: bool = False, flag_adapt_bias: bool = True,
                 merge_weights: bool = True, freeze_weight: bool = True, freeze_bias: bool = True, **kwargs):
        super().__init__(flag_adapt_bias=flag_adapt_bias, merge_weights=merge_weights,
                         freeze_weight=freeze_weight, freeze_bias=freeze_bias)
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        self.register_buffer('scale', None, persistent=False)
        self.register_buffer('shift', None, persistent=False)

    def assign_adaptation(self, adaptation):
        if adaptation is None:
            self.scale, self.shift = None, None
        else:
            self.scale = adaptation[..., :-self.out_features] + 1
            if self.scale.dim() == 2:
                self.scale = self.scale.unsqueeze(1)
            if self.flag_adapt_bias:
                self.shift = adaptation[..., -self.out_features:]
                self.shift = self.shift.view_as(self.scale)

    def _merge(self, weight, bias):
        if weight is not None and self.flag_adapt_weight:
            scale = self.scale.squeeze()
            if self.fan_in_fan_out:
                weight = weight * scale.reshape((1, scale.shape[-1]) + (1,) * (weight.dim() - 2))
            else:
                weight = weight * scale.reshape(scale.shape[-1:] + (1,) * (weight.dim() - 1))
        if bias is not None:
            if self.flag_adapt_weight:
                bias = bias * scale
            if self.flag_adapt_bias:
                bias = bias + self.shift.squeeze()
        return weight, bias

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-1]
        res = res.view(self.scale.shape[0], -1, res.shape[-1])
        if self.shift is not None:
            res = res * self.scale + self.shift
        else:
            res = res * self.scale
        return res.view(*batch_size, res.shape[-1])


class Linear(nn.Linear, SSF):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None, dtype=None,
            merge_weights: bool = True, freeze_weight: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        SSF.__init__(self, out_features=out_features, flag_adapt_bias=bias,
                     merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if self.scale is None:
            return super().forward(x)
        if self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.linear(x, weight, bias=bias)
        else:
            return self._ssf(F.linear(x, self.weight, bias=self.bias))
