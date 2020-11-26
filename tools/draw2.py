import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from tools.calculate_tool import compute_confidence_interval

# mini-ImageNet
# test = [0.5465, 0.5432, 0.5458, 0.5437, 0.5512, 0.5412, 0.5478, 0.5387, 0.5483, 0.5545, 0.5483, 0.5478, 0.5524, 0.537, 0.5537, 0.551, 0.5259,
#         0.5347, 0.5471, 0.5415, 0.5433, 0.5474, 0.5413, 0.5485, 0.5404, 0.547, 0.555,
#         0.5508, 0.5447, 0.5447, 0.5441, 0.5543, 0.5432, 0.5508, 0.5445, 0.5399, 0.551,
#         0.5514, 0.5494, 0.5457, 0.5472, 0.5554, 0.5461, 0.55, 0.5486, 0.5422, 0.5492]
# val = [0.5723, 0.5732, 0.5786, 0.5645, 0.5731, 0.5690, 0.5713, 0.5655, 0.5712, 0.5731, 0.5720, 0.5656, 0.5783, 0.5612, 0.5701, 0.5758, 0.5510,
#        0.5734, 0.5746, 0.5728, 0.5573, 0.5712, 0.571, 0.5708, 0.5714, 0.5713, 0.5799,
#        0.5752, 0.5716, 0.5708, 0.5684, 0.5757, 0.568, 0.5768, 0.5758, 0.5742, 0.576,
#        0.5745, 0.5704, 0.5733, 0.5791, 0.5741, 0.5736, 0.5774, 0.5736, 0.5735, 0.5744]

# tiered-ImageNet
# test = [0.6128, 0.6086, 0.6102, 0.6099, 0.6217, 0.6238, 0.6211, 0.6186, 0.6118, 0.6088,
#         0.6172, 0.6182, 0.6128, 0.6163, 0.6192, 0.6188, 0.6203, 0.6199, 0.6171, 0.6101]
# val = [0.5681, 0.5576, 0.561, 0.5596, 0.5701, 0.5728, 0.5713, 0.5689, 0.5599, 0.5578,
#        0.5667, 0.5677, 0.5648, 0.5667, 0.5667, 0.5643, 0.5671, 0.5627, 0.5598, 0.5571]

# cifar100
test = [0.7, 0.6858, 0.6949, 0.6936, 0.6829, 0.6949, 0.6996, 0.7032, 0.6873, 0.6932, 0.6928, 0.6914, 0.6867, 0.6964, 0.6956, 0.688, 0.686, 0.6879, 0.6969, 0.6915, 0.6969, 0.6922, 0.6925, 0.6928, 0.6969, 0.6918, 0.685, 0.6917, 0.6922, 0.7004, 0.6926, 0.6958,
0.6872, 0.6824, 0.6929, 0.6962, 0.6992, 0.6944, 0.7038, 0.694, 0.6969, 0.6925, 0.6944, 0.6848, 0.6923, 0.6887, 0.7011, 0.6885, 0.6952]
val = [0.6322, 0.6236, 0.6313, 0.6266, 0.6228, 0.6325, 0.6326, 0.6328, 0.6255, 0.6308, 0.6216, 0.6271, 0.6274, 0.6203, 0.63, 0.6265, 0.6213, 0.6272, 0.6351, 0.6198, 0.6374, 0.6315, 0.6184, 0.6252, 0.6287, 0.6282,
0.6252, 0.6352, 0.6268, 0.6391, 0.6321, 0.6226, 0.619, 0.6272, 0.6243, 0.6256, 0.6289, 0.6288, 0.6386, 0.6303, 0.6193, 0.6239, 0.6254, 0.6237, 0.6255, 0.6255, 0.6312, 0.6267, 0.6345]

test = np.array(test)
val = np.array(val)
print(test.shape)
print(val.shape)
# regr = linear_model.LinearRegression()
# regr.fit(test.reshape(-1, 1), val)
# a, b = regr.coef_, regr.intercept_

plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.axis(xmin=55, xmax=58)
plt.xticks(np.linspace(55, 58, 6, endpoint=True))
plt.axis(ymin=60, ymax=63)
plt.yticks(np.linspace(60, 63, 6, endpoint=True))
plt.tick_params(labelsize=15)
# plt.plot(100*test, 100*regr.predict(test.reshape(-1, 1)), color="red", linewidth=3, zorder=1)
plt.scatter(val*100, test*100, color="b", zorder=3)
plt.scatter([57.00], [61.94], color="g", zorder=5, marker='s', s=100)
plt.xlabel('Validation Accuracy (%)', fontsize=18)
plt.ylabel('Test Accuracy (%)', fontsize=18)
plt.grid()
# plt.title("tiered-ImageNet", fontsize=18)
plt.tight_layout()
plt.savefig("scatter_tiered.pdf")
plt.show()
print(compute_confidence_interval(test))


g_val = pd.Series(val)
g_test = pd.Series(test)
print(round(g_test.corr(g_val), 4))