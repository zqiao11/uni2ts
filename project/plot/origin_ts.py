import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





# 参数
start = 0
ctx_len = 96
pred_len = 24

# 读取数据
file = '/home/zhongzheng/datasets/TSLib/long_term_forecast/ETT-small/ETTh2.csv'
df = pd.read_csv(file, index_col=0, parse_dates=True)
target = df.iloc[:, 1].to_numpy()
target = target[start: start + ctx_len + pred_len]

# 一些辅助参数
width_per_step = 0.1  # 每个时间步的宽度
height = 3           # 图像高度
full_width = width_per_step * len(target)

# 窗口划分
window_size = 12
step_size = 12
gap_size = 0.75

# ---------- 开始绘制 ----------
fig, ax = plt.subplots(figsize=(full_width, height))

# 绘制上下文部分
ax.plot(
    np.arange(ctx_len+pred_len),
    target[:ctx_len+pred_len],
    label="Target Series",
    color="blue"
)


# 隐藏坐标轴、ticks和边框
ax.set_xlim(0, len(target))
ax.set_ylim(min(target), max(target))
plt.xlabel("Time Steps", fontsize=32)
# 关键：关闭四周边框和所有刻度
ax.axis('off')
for spine in ax.spines.values():
    spine.set_visible(False)

# 如果不需要图例，注释或删除此行
# ax.legend()

plt.tight_layout()
plt.show()
