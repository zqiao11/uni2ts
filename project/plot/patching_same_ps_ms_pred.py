import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def downsample(context, factor):
    """
    对上下文序列进行降采样
    :param context: 输入的上下文序列
    :param factor: 下采样因子
    :return: 降采样后的序列
    """
    new_len = len(context) // factor
    downsampled = np.array([np.mean(context[i * factor:(i + 1) * factor]) for i in range(new_len)])
    return downsampled


# 参数
start = 0
ctx_len = 96
pred_len = 24

# 读取数据
file = '/home/zhongzheng/datasets/TSLib/long_term_forecast/ETT-small/ETTh2.csv'
df = pd.read_csv(file, index_col=0, parse_dates=True)
target = df.iloc[:, 1].to_numpy()
target = target[start: start + ctx_len + pred_len]


# 使用down sampling factor = 2 和 4 进行降采样
target_ds2 = downsample(target, factor=2)
target_ds4 = downsample(target, factor=4)

# 动态调整宽度
width_per_step = 0.1  # 每个时间步的宽度
height = 3  # 图的高度
full_width = width_per_step * len(target)
context_width = width_per_step * ctx_len

# 定义窗口大小
window_size = 12
step_size = 12
gap_size = 0.75  # 微小的间隔，单位是时间步

tick_fontsize = 20  # x轴刻度字体大小
label_fontsize = 32  # x轴标题字体大小


# 绘制完整时间序列
plt.figure(figsize=(full_width, height))
# 绘制完整时间序列，并设置预测部分的透明度
plt.plot(np.arange(len(target[:ctx_len])), target[:ctx_len], label="Target Series", color="blue", linewidth=4)  # 绘制上下文部分
plt.plot(np.arange(ctx_len-1, len(target)), target[ctx_len-1:], label="Prediction Area", color="blue", alpha=0.3, linewidth=4)  # 绘制预测部分，透明度0.3

# 为预测区域添加灰色填充（透明度调整）
plt.axvspan(ctx_len, len(target), color='gray', alpha=0.3)  # alpha 控制透明度，数值越小越透明

# 添加虚线方框（浅灰色，虚线，粗线）
for start_idx in range(0, len(target), step_size):
    end_idx = start_idx + window_size
    bias = 0.25 if start_idx == 0 else 0
    if end_idx <= len(target):
        # 微调上下边框的位置，给每个窗口的上下边框留些间隙
        plt.axvspan(start_idx + bias, end_idx - gap_size,  ymin=0.01, ymax=0.99, facecolor='none', edgecolor='gray', linestyle='--', linewidth=4)

plt.xlim(0, len(target))
plt.yticks([])
# plt.title("Full Time Series")
plt.xlabel("Time Steps", fontsize=label_fontsize)
plt.xticks([])
# plt.xticks(fontsize=tick_fontsize)
# plt.ylabel("Target Value")
# plt.legend()
plt.tight_layout()
plt.show()


# plt.figure(figsize=(full_width, height))
# plt.plot(np.arange(ctx_len//2+1), target_ds2[:ctx_len//2+1], label="Context Series", color="green", linewidth=4)
# plt.plot(np.arange(ctx_len//2, len(target_ds2)), target_ds2[ctx_len//2:], label="Prediction Area", color="green", alpha=0.5, linewidth=4)
#
# # 为预测区域添加灰色填充（透明度调整）
# plt.axvspan(ctx_len//2, len(target_ds2), color='gray', alpha=0.3)  # alpha 控制透明度，数值越小越透明
#
# # 添加虚线方框（浅灰色，虚线，粗线）
# for start_idx in range(0, len(target_ds2), step_size):
#     end_idx = start_idx + window_size
#     bias = 0.25 if start_idx == 0 else 0
#     if end_idx <= len(target_ds2):
#         # 微调上下边框的位置，给每个窗口的上下边框留些间隙
#         plt.axvspan(start_idx + bias/2, end_idx - gap_size/2,  ymin=0.01, ymax=0.99, facecolor='none', edgecolor='gray', linestyle='--', linewidth=4)
#
# plt.xlim(0, len(target_ds2))
# plt.yticks([])
# plt.xlabel("Time Steps", fontsize=label_fontsize)
# plt.xticks([])
# # plt.xticks(fontsize=tick_fontsize)
# plt.tight_layout()
# plt.show()
#
#
# plt.figure(figsize=(context_width * 1.5, height))
# plt.plot(np.arange(ctx_len//4+1), target_ds4[:ctx_len//4+1], label="Context Series", color="orange", linewidth=4)
# plt.plot(np.arange(ctx_len//4, len(target_ds4)), target_ds4[ctx_len//4:], label="Prediction Area", color="orange", alpha=0.6, linewidth=4)
# plt.plot(np.arange(len(target_ds4), len(target_ds4)+6), np.zeros(6), label="Padded Area", color='gray', alpha=0.3)
#
# # 为预测区域添加灰色填充（透明度调整）
# plt.axvspan(ctx_len//4, len(target_ds4)+6, color='gray', alpha=0.3)  # alpha 控制透明度，数值越小越透明
#
# # 添加虚线方框（浅灰色，虚线，粗线）
# for start_idx in range(0, len(target_ds4)+6, step_size):
#     end_idx = start_idx + window_size
#     bias = 0.25 if start_idx == 0 else 0
#     if end_idx > len(target_ds4)+6:
#         end_idx = len(target_ds4)+6
#         # 微调上下边框的位置，给每个窗口的上下边框留些间隙
#     plt.axvspan(start_idx + bias/4, end_idx - gap_size/4,  ymin=0.01, ymax=0.99, facecolor='none', edgecolor='gray', linestyle='--', linewidth=4)
#
# plt.xlim(0, len(target_ds4)+6)
# plt.yticks([])
# plt.xlabel("Time Steps", fontsize=label_fontsize)
# plt.xticks([])
# # plt.xticks(fontsize=tick_fontsize)
# plt.tight_layout()
# plt.show()

