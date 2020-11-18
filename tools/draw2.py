import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from tools.calculate_tool import compute_confidence_interval

# mini-ImageNet
test = [0.5465, 0.5432, 0.5458, 0.5437, 0.5512, 0.5412, 0.5478, 0.5387, 0.5483, 0.5545, 0.5483, 0.5478, 0.5524, 0.537, 0.5537, 0.551, 0.5259,
        0.5347, 0.5471, 0.5415, 0.5433, 0.5474, 0.5413, 0.5485, 0.5404, 0.547, 0.555,
        0.5508, 0.5447, 0.5447, 0.5441, 0.5543, 0.5432, 0.5508, 0.5445, 0.5399, 0.551,
        0.5514, 0.5494, 0.5457, 0.5472, 0.5554, 0.5461, 0.55, 0.5486, 0.5422, 0.5492]
val = [0.5723, 0.5732, 0.5786, 0.5645, 0.5731, 0.5690, 0.5713, 0.5655, 0.5712, 0.5731, 0.5720, 0.5656, 0.5783, 0.5612, 0.5701, 0.5758, 0.5510,
       0.5734, 0.5746, 0.5728, 0.5573, 0.5712, 0.571, 0.5708, 0.5714, 0.5713, 0.5799,
       0.5752, 0.5716, 0.5708, 0.5684, 0.5757, 0.568, 0.5768, 0.5758, 0.5742, 0.576,
       0.5745, 0.5704, 0.5733, 0.5791, 0.5741, 0.5736, 0.5774, 0.5736, 0.5735, 0.5744]

# tiered-ImageNet
# test = []
# val = []

# cifar100
# test = []
# val = []

test = np.array(test)
val = np.array(val)

# regr = linear_model.LinearRegression()
# regr.fit(test.reshape(-1, 1), val)
# a, b = regr.coef_, regr.intercept_

plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.axis(xmin=54, xmax=59)
plt.xticks(np.linspace(54, 59, 6, endpoint=True))
plt.axis(ymin=52, ymax=56)
plt.yticks(np.linspace(52, 56, 5, endpoint=True))
plt.tick_params(labelsize=15)
# plt.plot(100*test, 100*regr.predict(test.reshape(-1, 1)), color="red", linewidth=3, zorder=1)
plt.scatter(val*100, test*100, color="b", zorder=3)
plt.scatter([57.28], [55.03], color="g", zorder=5, marker='s', s=100)
plt.xlabel('Validation Accuracy (%)', fontsize=18)
plt.ylabel('Test Accuracy (%)', fontsize=18)
plt.grid()
plt.tight_layout()
plt.savefig("scatter.pdf")
plt.show()
print(compute_confidence_interval(test))


g_val = pd.Series(val)
g_test = pd.Series(test)
print(round(g_test.corr(g_val), 4))