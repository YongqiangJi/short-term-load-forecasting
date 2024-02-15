from scipy.io import savemat,loadmat
import numpy as np
import pandas as pd
#from tslearn.utils import to_time_series
#from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw, dtw_path
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data = pd.read_csv('../2122ukdata/21-22IMFs.csv', index_col=0)
data = data.values.transpose()
print(data)
print(data.shape)

n = data.shape[0]
score = []
distance_matrix = np.zeros((n, n))
for i in range(0,data.shape[0]):
    for j in range(i + 1,data.shape[0]):
        dtw_score = dtw(data[i], data[j])
        distance_matrix[i, j] = distance_matrix[j, i] = dtw_score
        score.append(dtw_score)
print(score)
print(distance_matrix)

#保存矩阵
save_dict = {'name':'matrix','data':distance_matrix}
#  test.mat是保存路径，save_dict必须是dict类型，他就这么定义的！
savemat('../2122ukresults/17resid-IMFs-distance_matrix.mat',save_dict)

'''
#加载矩阵
distance_matrix = loadmat('../17results/17resid-IMFs-distance_matrix.mat')
distance_matrix = distance_matrix["data"]
print(distance_matrix)
'''

# 使用KMeans算法进行聚类
for j in range(2,6,1):
    kmeans = KMeans(n_clusters=j).fit(distance_matrix)
    score = silhouette_score(distance_matrix, kmeans.labels_)
    print(j,'类KMeans结果:',kmeans.labels_,"KMeans轮廓系数:",score)

# 使用层次聚类算法进行聚类
for j in range(2, 6, 1):
    kmeans = AgglomerativeClustering(n_clusters=j).fit(distance_matrix)
    score = silhouette_score(distance_matrix, kmeans.labels_)
    print(j,'类层次聚类结果:',kmeans.labels_,"层次聚类轮廓系数:",score)

'''
# 使用kmeans算法进行聚类
for j in range(2,6,1):
    centroids, _ = kmeans(distance_matrix, j)
    labels, _ = vq(distance_matrix, centroids)
    score = silhouette_score(distance_matrix, labels)
    print(j,"类kmeans结果:",labels,"kmeans轮廓系数:", score)
'''
# 设置坐标轴标题字体及其大小
myfont = FontProperties(fname='/home/jiyongqiang/.conda/envs/tf/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Times New Roman.ttf')
plt.rcParams.update({'font.size': 12})

# 使用层次聚类算法进行聚类
for j in range(2,6,1):
    Z = linkage(distance_matrix, 'ward')
    labels = fcluster(Z, j, criterion='maxclust')
    score = silhouette_score(distance_matrix, labels)
    print(j,'类层次聚类结果:', labels,"层次聚类轮廓系数为：",score)
    x = []
    for i in range(1, labels.shape[0] + 1, 1):
        x.append(i)
    print(x)
    # 建立树状图
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=x)
    plt.xlabel('IMF number', fontsize=12, fontproperties=myfont)
    plt.ylabel('Distance', fontsize=12, fontproperties=myfont)
    plt.xticks(fontsize=12,fontproperties=myfont)
    plt.yticks(fontsize=12,fontproperties=myfont)
    plt.savefig(r'../2122ukresults/2122resid-IMFs-cluster.tiff', dpi=300, bbox_inches='tight')
    plt.show()
