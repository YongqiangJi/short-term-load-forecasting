from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np
from copy import copy, deepcopy
from PyEMD import CEEMDAN, Visualisation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

turkeydata = pd.read_csv('../2122ukdata/21-22.csv', index_col=0)
print(turkeydata)

decomposition = STL(turkeydata, period=48).fit()
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
np.savetxt('../2122ukresults/2122STLtrend.csv', np.round_(trend, 6), delimiter=',')
np.savetxt('../2122ukresults/2122STLseasonal.csv', np.round_(seasonal, 6), delimiter=',')
np.savetxt('../2122ukresults/2122STLresidual.csv', np.round_(residual, 6), delimiter=',')


#设置画布的尺寸，单位为英寸，根据最终输出的图像大小调整一下数字
plt.figure(figsize=(10, 8))
myfont = FontProperties(fname='/home/jiyongqiang/.conda/envs/tf/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Times New Roman.ttf')

# 设置坐标轴标题字体及其大小
plt.rcParams.update({'font.size': 12})

# 设置坐标轴线条粗细，也就是下面代码中的 cu= 的数值，一般设置为 0.6 左右即可
plt.subplot(411)
plt.plot(turkeydata.values[2832:4320], linewidth = 1.5, color = 'k')
plt.ylabel('Data',fontsize=12,fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.tick_params(direction='in')

plt.subplot(412)
plt.plot(trend.values[2832:4320], linewidth = 1.5, color = 'k')
plt.ylabel('Trend',fontsize=12,fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.tick_params(direction='in')

plt.subplot(413)
plt.plot(seasonal.values[2832:4320], linewidth = 1.5, color = 'k')
plt.ylabel('Season',fontsize=12,fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.tick_params(direction='in')

plt.subplot(414)
plt.plot(residual.values[2832:4320], linewidth = 1.5, color = 'k')
plt.xlabel('timestep/30 mins',fontsize=12,fontproperties=myfont)
plt.ylabel('Resid',fontsize=12,fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.tick_params(direction='in')

#plt.savefig(r'../2122ukresults/2122STLdecomposition.tiff', dpi=300, bbox_inches='tight')
plt.show()