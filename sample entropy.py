from sampen import sampen2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator
from copy import copy, deepcopy
from PyEMD import CEEMDAN, Visualisation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

data = pd.read_csv(r'../2122ukdata/21-22+STL.csv', header=0, index_col=0)
print(data)
IImfs = data.values
IImfs = IImfs.transpose()

print(IImfs)
print(IImfs.shape)
sampen = []
# 计算样本熵 m=1、2, r=0.1、0.2
for i in IImfs:
    #for j in (0.1, 0.2):
    sample_entropy = sampen2(list(i), mm=2, r=0.1, normalize=True)
    print(sample_entropy)
    sampen.append(sample_entropy)

print(sampen)
# 分离
entropy_r1m1 = []  # r=0.1、m=1
entropy_r1m2 = []  # r=0.1、m=2
entropy_r2m1 = []  # r=0.2、m=1
entropy_r2m2 = []  # r=0.2、m=2
for i in range(len(sampen)):
    #if (i % 2) == 0:  # r = 0.1
        # m = 2
    entropy_r1m2.append(sampen[i][2][1])

print(entropy_r1m2)

# 可视化
#设置画布的尺寸，单位为英寸，根据最终输出的图像大小调整一下数字
plt.figure(figsize=(5, 4))
myfont = FontProperties(fname='/home/jiyongqiang/.conda/envs/tf/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Times New Roman.ttf')

# 设置坐标轴标题字体及其大小
plt.rcParams.update({'font.size': 12})

x = list(range(1, len(IImfs) + 1, 1))
plt.plot(x, entropy_r1m2, 'r:H')

plt.xlabel('Component',fontsize=12,fontproperties=myfont)
plt.ylabel('SampEn',fontsize=12,fontproperties=myfont)
plt.xticks([1, 2, 3, 4], [ 'data','trend', 'seasonal', 'residual'],fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.tick_params(direction='in')
#plt.savefig(r'../17results/17data+STLSampEn.tiff', dpi=300, bbox_inches='tight')
plt.show()
