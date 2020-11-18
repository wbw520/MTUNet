import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(6, 6), dpi=80)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
ax5 = plt.subplot(325)
ax6 = plt.subplot(326)

linewidth = 2.5
font = 15
markers = ['o', 's', '']
colors = ['#edb03d', "#4dbeeb", "#77ac41"]
index = [0, 1, 2, 3, 4, 5, 6, 7]
index2 = [0, 1, 2, 3, 4, 5, 6]
x_mini = ["7", "8", "10", "13", "16", "22", "32", "64"]
x_tiered = ["36", "44", "51", "71", "88", "117", "176", "351"]
y_mini_res_1 = [55.03, 55.05, 55.31, 55.17, 54.59, 54.86, 55.17, 55.41]
y_mini_res_5 = [71.22, 70.08, 71.39, 70.88, 70.04, 69.84, 71.15, 71.44]
y_mini_wrn_1 = [56.72, 56.53, 56.23, 56.43, 56.28, 56.24, 56.06, 56.24]
y_mini_wrn_5 = [72.88, 73.01, 72.53, 72.22, 71.89, 72.24, 72.00, 72.47]
y_tiered_res_1 = [61.57, 61.65, 61.23, 61.40, 60.96, 61.27, 61.01]
y_tiered_res_5 = [78.82, 79.03, 78.56, 78.28, 78.01, 78.02, 78.19]
y_tiered_wrn_1 = [62.97, 62.88, 63.02, 62.70, 62.39, 62.37, 62.48]
y_tiered_wrn_5 = [80.05, 79.86, 80.43, 79.24, 78.99, 79.20, 79.43]
y_cifar_res_1 = [66.31, 66.99, 67.49, 67.08, 67.45, 67.31, 67.89, 67.12]
y_cifar_res_5 = [80.16, 80.59, 81.23, 81.77, 81.84, 81.30, 81.75, 81.29]
y_cifar_wrn_1 = [68.34, 69.43, 70.05, 69.78, 69.84, 69.56, 69.42, 69.78]
y_cifar_wrn_5 = [82.13, 83.49, 84.28, 83.36, 84.08, 83.88, 83.03, 83.58]

ax1.set_xticks(index)
ax1.set_xticklabels(x_mini)
ax2.set_xticks(index)
ax2.set_xticklabels(x_mini)
ax3.set_xticks(index)
ax3.set_xticklabels(x_tiered)
ax4.set_xticks(index)
ax4.set_xticklabels(x_tiered)
ax5.set_xticks(index)
ax5.set_xticklabels(x_mini)
ax6.set_xticks(index)
ax6.set_xticklabels(x_mini)

ax1.axis(ymin=54, ymax=60)
ax2.axis(ymin=69, ymax=76)
ax2.set_yticks(np.linspace(69, 78, 4, endpoint=True))
ax3.axis(ymin=60, ymax=66)
ax3.set_yticks(np.linspace(60, 66, 4, endpoint=True))
ax4.axis(ymin=77, ymax=83)
ax4.set_yticks(np.linspace(77, 83, 4, endpoint=True))
ax5.axis(ymin=65, ymax=75)
ax5.set_yticks(np.linspace(65, 74, 4, endpoint=True))
ax6.axis(ymin=79, ymax=89)
ax6.set_yticks(np.linspace(79, 88, 4, endpoint=True))

ax1.tick_params(labelsize=font-2)
ax2.tick_params(labelsize=font-2)
ax3.tick_params(labelsize=font-2)
ax4.tick_params(labelsize=font-2)
ax5.tick_params(labelsize=font-2)
ax6.tick_params(labelsize=font-2)

ax1.set_title("mini-ImageNet 1-shot", fontsize=font+1)
ax2.set_title("mini-ImageNet 5-shot", fontsize=font+1)
ax3.set_title("tiered-ImageNet 1-shot", fontsize=font+1)
ax4.set_title("tiered-ImageNet 5-shot", fontsize=font+1)
ax5.set_title("CIFAR-FS 1-shot", fontsize=font+1)
ax6.set_title("CIFAR-FS 5-shot", fontsize=font+1)

ax1.plot(index, y_mini_res_1, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
ax1.plot(index, y_mini_wrn_1, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax1.legend(loc='upper right', fontsize=font-4.5, ncol=1)

ax2.plot(index, y_mini_res_5, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
ax2.plot(index, y_mini_wrn_5, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax2.legend(loc='upper right', fontsize=font-4.5, ncol=1)

ax3.plot(index2, y_tiered_res_1, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
ax3.plot(index2, y_tiered_wrn_1, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax3.legend(loc='upper right', fontsize=font-4.5, ncol=1)

ax4.plot(index2, y_tiered_res_5, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
ax4.plot(index2, y_tiered_wrn_5, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax4.legend(loc='upper right', fontsize=font-4.5, ncol=1)

ax5.plot(index, y_cifar_res_1, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
ax5.plot(index, y_cifar_wrn_1, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax5.legend(loc='upper right', fontsize=font-4.5, ncol=1)

ax6.plot(index, y_cifar_res_5, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
ax6.plot(index, y_cifar_wrn_5, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax6.legend(loc='upper right', fontsize=font-4.5, ncol=1)

plt.tight_layout()
plt.savefig("ablation.pdf")
plt.show()