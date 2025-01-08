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

# 分割数据
context = target[:ctx_len]
pred = target[ctx_len:]

# 使用down sampling factor = 2 和 4 进行降采样
pred_ds2 = downsample(pred, factor=2)
pred_ds4 = downsample(pred, factor=4)

# 动态调整宽度
width_per_step = 0.1  # 每个时间步的宽度
height = 3  # 图的高度
pred_width = width_per_step * len(pred)
pred_ds2_width = width_per_step * len(pred_ds2)
pred_ds4_width = width_per_step * len(pred_ds4)

# 定义窗口大小
window_size = 12
step_size = 12
gap_size = 0.75  # 微小的间隔，单位是时间步

tick_fontsize = 20  # x轴刻度字体大小
label_fontsize = 24  # x轴标题字体大小


# ---------- 开始绘制 ----------
fig, ax = plt.subplots(figsize=(pred_width, height))

# 绘制上下文部分
ax.plot(
    np.arange(pred_len),
    pred,
    color="blue",
    linewidth=4
)


# 隐藏坐标轴、ticks和边框
ax.set_xlim(0, len(pred))
ax.set_ylim(min(target), max(target))
plt.xlabel("Time Steps", fontsize=32)
# 关键：关闭四周边框和所有刻度
ax.axis('off')
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(pred_width, height))

# 绘制上下文部分
ax.plot(
    np.arange(len(pred_ds2)),
    pred_ds2,
    color="green",
    linewidth=4
)


# 隐藏坐标轴、ticks和边框
ax.set_xlim(0, len(pred_ds2))
ax.set_ylim(min(target), max(target))
plt.xlabel("Time Steps", fontsize=32)
# 关键：关闭四周边框和所有刻度
ax.axis('off')
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(pred_width, height))

# 绘制上下文部分
ax.plot(
    np.arange(len(pred_ds4)),
    pred_ds4,
    color="orange",
    linewidth=4
)


# 隐藏坐标轴、ticks和边框
ax.set_xlim(0, len(pred_ds4))
ax.set_ylim(min(target), max(target))
plt.xlabel("Time Steps", fontsize=32)
# 关键：关闭四周边框和所有刻度
ax.axis('off')
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()


preds_us2 = np.repeat(pred_ds2, 2, axis=0)  # 将每个值重复 2 次
preds_us4 = np.repeat(pred_ds4, 4, axis=0)  # 将每个值重复 4 次
agg_pred = 1/3 * (pred + preds_us2 + preds_us4)

fig, ax = plt.subplots(figsize=(pred_width, height))

# 绘制上下文部分
ax.plot(
    np.arange(len(pred)),
    agg_pred,
    color="black",
    linewidth=4
)


# 隐藏坐标轴、ticks和边框
ax.set_xlim(0, len(pred))
ax.set_ylim(min(target), max(target))
plt.xlabel("Time Steps", fontsize=32)
# 关键：关闭四周边框和所有刻度
ax.axis('off')
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()