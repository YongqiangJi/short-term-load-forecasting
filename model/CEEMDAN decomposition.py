import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np
from copy import copy, deepcopy
from PyEMD import CEEMDAN, Visualisation
from sampen import sampen2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data = pd.read_csv('../2122ukdata/21-22+STL.csv', index_col=0)
print(data)
data = (data.values).astype('float32')

data1 = copy(data[:, -1])

print(data1,data1.shape)
ceemdan = CEEMDAN()
ceemdan.ceemdan(data1)
imfs, res = ceemdan.get_imfs_and_residue()
print(imfs,imfs.shape)
print(res,res.shape)
IImfs = imfs.transpose()
print(IImfs)
print(IImfs.shape)
np.savetxt('../2122ukresults/2122STLResid-CEEMDAN-IImfs2.csv', np.round_(IImfs, 6), delimiter=',')
'''
imfs = pd.read_csv('../results/ukSTLResid-CEEMDAN-IImfs.csv', header=None, index_col=None)
imfs = imfs.values.transpose()
print(imfs,imfs.shape)

#设置画布的尺寸，单位为英寸，根据最终输出的图像大小调整一下数字
plt.figure(figsize=(12, 9))
myfont = FontProperties(fname='/home/jiyongqiang/.conda/envs/tf/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Times New Roman.ttf')

# 设置坐标轴标题字体及其大小
#plt.rcParams.update({'font.size': 12})

plt.subplots_adjust(hspace=0.2)
plt.subplot(imfs.shape[0]+1, 1, 1)

plt.plot(data1,color = '#508AB2')
plt.ylabel("Resid",fontsize=10,fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)

for i in range(imfs.shape[0]):
    plt.subplot(imfs.shape[0]+1,1,i+2)
    plt.plot(imfs[i], color = '#719F85')
    plt.ylabel("IMF %i" %(i+1),fontsize=10,fontproperties=myfont)
    plt.xticks(fontproperties=myfont)
    plt.yticks(fontproperties=myfont)
    plt.locator_params(axis='x', nbins=10)

plt.xlabel('timestep/30 mins',fontsize=12,fontproperties=myfont)

#plt.savefig(r'../results/ukSTLResid-CEEMDAN-IImfs.tiff', dpi=300, bbox_inches='tight')
plt.show()

np.savetxt('../2122ukresults/2122STLResid-CEEMDAN-IImfs.csv', np.round_(IImfs, 6), delimiter=',')
#np.savetxt('../17results/17STLResid-CEEMDAN-res.csv', np.round_(res, 6), delimiter=',')


sampen = []
# 计算样本熵 m=1、2, r=0.1、0.2
for i in imfs:
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

x = []
for i in range(1, len(entropy_r1m2) + 1, 1):
    x.append(i)
print(x)

plt.plot(x, entropy_r1m2, 'r:H')

plt.xlabel('IMFs',fontsize=12,fontproperties=myfont)
plt.ylabel('SampEn',fontsize=12,fontproperties=myfont)
plt.xticks(x, fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.tick_params(direction='in')
#plt.savefig(r'../17results/17STLResid-CEEMDAN-IImfs-SampEn.tiff', dpi=300, bbox_inches='tight')
plt.show()

'''