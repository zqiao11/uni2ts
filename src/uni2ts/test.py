# import torch
# import torch.nn as nn
#
#
# def apply_lora(
#                input: torch.Tensor,
#                layer: nn.Linear,
#                A: nn.Parameter,
#                B: nn.Parameter,
#                alpha: float = 1.0,
#                train_bias: bool = False):
#     """
#     在给定的线性层上应用 LoRA。
#     """
#     # 获取线性层的权重和偏置
#     W_no_grad = layer.weight.detach()  # 冻结权重
#     bias = layer.bias.detach() if not train_bias else layer.bias  # 冻结或训练偏置
#
#     # LoRA 更新部分
#     lora_update = alpha * (B @ A)  # (in_features, out_features)
#
#     # 合成 LoRA 后的权重
#     W_lora = W_no_grad + lora_update  # 最终的权重 (in_features, out_features)
#
#     # 计算输出
#     out = torch.matmul(input, W_lora.T) + bias  # 加入偏置
#
#     return out
#
# # 示例：假设输入的线性层为 Wq，并且已有 A 和 B
# bs, seq_len, dim, rank = 2, 10, 8, 4  # rank 是低秩矩阵的秩
# x = torch.randn(bs, seq_len, dim)
#
# # 预训练的 Linear Layer（含 W 和 b）
# Wq = nn.Linear(dim, dim)
# Wq.weight.data.normal_()
# Wq.bias.data.zero_()
#
# # LoRA 参数：A 和 B
# A = nn.Parameter(torch.randn((rank, dim), dtype=torch.float) * 0.01)
# B = nn.Parameter(torch.zeros((dim, rank), dtype=torch.float))
#
# y = Wq(x)
#
# y_ = torch.matmul(x, Wq.weight.T) + Wq.bias
#
# y_lora = apply_lora(x, Wq, A, B)
#
# end = 1