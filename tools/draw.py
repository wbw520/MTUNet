import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(6, 6), dpi=80)
ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
ax5 = plt.subplot(325)
ax6 = plt.subplot(326)

linewidth = 2.5
font = 11
markers = ['o', 's', '']
colors = ['#edb03d', "#4dbeeb", "#77ac41"]
index = [0, 1, 2, 3, 4, 5, 6, 7]
x_mini = ["7", "8", "10", "13", "16", "22", "32", "64"]
x_tiered = ["36", "44", "51", "71", "88", "117", "176", "351"]
y_res_1 = [55.69, 55.05, 55.35, 55.17, 55.13, 54.86, 12, 55.41]
y_res_5 = [71.22, 72.08, 71.39, 70.88, 69.64, 69.94, 69.64, 72.14]
y_wrn_1 = [57.32, 57.13, 56.89, 57.58, 56.28, 56.34, 56.02, 56.18]
y_wrn_5 = []

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

ax1.axis(ymin=54, ymax=59)
ax2.axis(ymin=58, ymax=73)
ax3.axis(ymin=54, ymax=60)
ax4.axis(ymin=54, ymax=60)
ax5.axis(ymin=60, ymax=60)
ax6.axis(ymin=54, ymax=60)

ax1.tick_params(labelsize=font-2)
ax2.tick_params(labelsize=font-2)
ax3.tick_params(labelsize=font-2)
ax4.tick_params(labelsize=font-2)
ax5.tick_params(labelsize=font-2)
ax6.tick_params(labelsize=font-2)

ax1.set_title("mini-ImageNet 1-shot", fontsize=font)
ax2.set_title("mini-ImageNet 5-shot", fontsize=font)
ax3.set_title("tiered-ImageNet 1-shot", fontsize=font)
ax4.set_title("tiered-ImageNet 5-shot", fontsize=font)
ax5.set_title("CIFAR-FS 1-shot", fontsize=font)
ax6.set_title("CIFAR-FS 5-shot", fontsize=font)

ax1.plot(index, y_res_1, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
# ax1.plot(index, y_wrn_1, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax1.legend(loc='upper right', fontsize=font-3)

ax2.plot(index, y_res_1, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
# ax2.plot(index, y_wrn_1, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
ax2.legend(loc='upper right', fontsize=font-3)

# ax3.plot(index, y_res_1, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="ResNet-18")
# ax3.plot(index, y_wrn_1, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="WRN")
# ax3.legend(loc='upper right', fontsize=font-3)

plt.tight_layout()
plt.show()