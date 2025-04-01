

import matplotlib.pyplot as plt

tick_fontsize = 20  # x轴刻度字体大小
label_fontsize = 24  # x轴标题字体大小
legend_fontsize = 24
title_fontsize = 30

# Data
k = [0, 1, 2, 3, 4, 5]

Ettm1_MSE = [0.367, 0.358, 0.357, 0.357, 0.357, 0.359]
Ettm1_MAE = [0.382, 0.378, 0.378, 0.378, 0.378, 0.379]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot MSE and MAE curves
plt.plot(k, Ettm1_MSE, label="MSE", color="blue", marker="o")
plt.plot(k, Ettm1_MAE, label="MAE", color="orange", marker="s")

# Adjust y-axis to display data effectively
y_min = min(min(Ettm1_MSE), min(Ettm1_MAE)) - 0.01
y_max = max(max(Ettm1_MSE), max(Ettm1_MAE)) + 0.01
plt.ylim(y_min, y_max)

# Labels and title
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize-4)
plt.xlabel("Number of New Scales", fontsize=label_fontsize)
plt.ylabel("Error", fontsize=label_fontsize)
# plt.title("Ettm1", fontsize=title_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


Ettm2_MSE = [0.273, 0.269, 0.254, 0.251, 0.25, 0.250]
Ettm2_MAE = [0.316, 0.309, 0.305, 0.303, 0.3025, 0.302]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot MSE and MAE curves
plt.plot(k, Ettm2_MSE, label="MSE", color="blue", marker="o")
plt.plot(k, Ettm2_MAE, label="MAE", color="orange", marker="s")

# Adjust y-axis to display data effectively
y_min = min(min(Ettm2_MSE), min(Ettm2_MAE)) - 0.01
y_max = max(max(Ettm2_MSE), max(Ettm2_MAE)) + 0.01
plt.ylim(y_min, y_max)

# Labels and title
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize-4)
plt.xlabel("Number of New Scales", fontsize=label_fontsize)
plt.ylabel("Error", fontsize=label_fontsize)
# plt.title("ETTm2", fontsize=title_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


Weather_MSE = [0.228, 0.224, 0.219, 0.219,  0.222, 0.222]
Weather_MAE = [0.254, 0.250, 0.249, 0.251, 0.250, 0.251]

# Create the plot
# Create the plot
plt.figure(figsize=(8, 6))

# Plot MSE and MAE curves
plt.plot(k, Weather_MSE, label="MSE", color="blue", marker="o")
plt.plot(k, Weather_MAE, label="MAE", color="orange", marker="s")

# Adjust y-axis to display data effectively
y_min = min(min(Weather_MSE), min(Weather_MAE)) - 0.01
y_max = max(max(Weather_MSE), max(Weather_MAE)) + 0.01
plt.ylim(y_min, y_max)

# Labels and title
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize-4)
plt.xlabel("Number of New Scales", fontsize=label_fontsize)
plt.ylabel("Error", fontsize=label_fontsize)
# plt.title("Weather", fontsize=title_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()