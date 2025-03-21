import torch
import torch.nn as nn
import torch.nn.functional as F

from .up import Adaptation_Up
from .ssf import Linear as ssf_Linear

def add_down_up_(parent_module: nn.Module, module_name: str, freeze_weight: bool,
             merge_weights=False, load_weights=True, **kwargs):
    old_module = getattr(parent_module, module_name)
    if isinstance(old_module, nn.Linear):
        new_module = Linear(in_features=old_module.in_features, out_features=old_module.out_features,
                            bias=old_module.bias is not None, freeze_weight=freeze_weight,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            merge_weights=merge_weights, **kwargs)

    else:
        raise NotImplementedError
    if load_weights:
        new_module.load_state_dict(old_module.state_dict(), strict=False)
    setattr(parent_module, module_name, new_module)


class Down_Up(Adaptation_Up):
    def __init__(self, in_features: int, *args, **kwargs):
        Adaptation_Up.__init__(self, *args, **kwargs)
        self.in_features = in_features
        self.register_buffer('scale2', None, persistent=False)

    def assign_adaptation(self, adaptation):
        if adaptation is None:
            self.scale, self.scale2, self.shift = None, None, None
        else:
            self.scale = adaptation[..., :self.out_features] + 1
            self.scale2 = adaptation[..., self.out_features:self.out_features + self.in_features] + 1
            if self.flag_adapt_bias:
                self.shift = adaptation[..., -self.out_features:]
            if self.scale.dim() == 2:
                self.scale = self.scale.unsqueeze(1)
                self.scale2 = self.scale2.unsqueeze(1)
                self.shift = self.shift.unsqueeze(1) if self.shift is not None else None

    def _merge(self, weight, bias):
        weight, bias = super()._merge(weight, bias)
        if weight is not None and self.flag_adapt_weight:
            scale2 = self.scale2.squeeze()
            if self.fan_in_fan_out:
                weight = weight * scale2.reshape(scale2.shape[-1:] + (1,) * (weight.dim() - 1))
            else:
                weight = weight * scale2.reshape((1, scale2.shape[-1]) + (1,) * (weight.dim() - 2))
        return weight, bias

    def _ssf_input(self, x: torch.Tensor):
        batch_size = x.size()[:-1]
        x = x.reshape(self.scale2.shape[0], -1, x.shape[-1])
        return (x * self.scale2).view(*batch_size, x.shape[-1])

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-1]
        res = res.view(self.scale.shape[0], -1, res.shape[-1])
        if self.bias is not None:
            res = res * self.scale + (self.shift + self.bias)
        else:
            res = res * self.scale
        return res.view(*batch_size, res.shape[-1])


class Linear(Down_Up, ssf_Linear):
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
        Down_Up.__init__(self, in_features=in_features, out_features=out_features, flag_adapt_bias=bias,
                         merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if not hasattr(self, 'scale') or self.scale is None:
            return nn.Linear.forward(self, x)
        if self.merged:
            return F.linear(x, self.weight, bias=self.bias)
        elif self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.linear(x, weight, bias=bias)
        else:
            return self._ssf(F.linear(self._ssf_input(x), self.weight, bias=None))
