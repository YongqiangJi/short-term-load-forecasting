import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras.models import save_model, load_model
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval
import pickle
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

window_size = 192  # 窗口大小
out_size = 48 #输出数量
batch_size = 48  # 训练批次大小
epochs = 200  # 训练epoch
CAP = 60000
horizon1 = 11712
horizon2 = 5856

time_1 = time.time()

'''
space = {

    'gamma': hp.choice('gamma', [0.5,0.1,0.05,0.01]),
    'epsilon': hp.choice('epsilon', [0.5,0.1,0.05,0.01,0.001]),
}
'''

def get_dataset():
    train,test,train_X,train_label,test_X,test_label = [], [], [], [], [], []

    df = pd.read_csv('../2122ukdata/21-22+STL.csv', index_col=0)
    print(df)
    # 归一化
    scaler = MinMaxScaler()
    df['trend'] = scaler.fit_transform(df['trend'].values.reshape(-1, 1)).reshape(-1)

    #划分数据集
    train, test = df.iloc[:-horizon2, 1:2], df.iloc[-horizon2:, 1:2]
    testforpred = df.iloc[-horizon2 - window_size:, 1:2]
    print(train)

    # 滑动窗口重构训练集和验证集
    for i in range(0, train.shape[0] - window_size - out_size + 1):
        a = train.iloc[i:(i + window_size), -1]
        train_X.append(a)
        b = train.iloc[(i + window_size):(i + window_size + out_size), -1]
        train_label.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_label = np.array(train_label, dtype='float64')
    train_X = train_X.reshape(train_X.shape[0],window_size)
    train_label = train_label.reshape(train_label.shape[0],out_size)
    print(train_X.shape)
    print(train_label.shape)

    # 滑动窗口用来预测的测试集
    for i in range(0, testforpred.shape[0] - window_size - out_size + 1,out_size):
        e = testforpred.iloc[i:(i + window_size), -1]
        test_X.append(e)
        f = testforpred.iloc[(i + window_size):(i + window_size + out_size), -1]
        test_label.append(f)
    test_X = np.array(test_X, dtype='float64')
    test_label = np.array(test_label, dtype='float64')
    test_X = test_X.reshape(test_X.shape[0], window_size)
    test_label = test_label.reshape(test_label.shape[0], out_size)
    print(test_X.shape)
    print(test_label.shape)
    print(test.shape)
    print(test)
    print(test_label)
    return train,test,train_X,train_label,test_X,test_label,scaler

def RMSE(pred, true):
    rs = []
    for i in range(1, 5857):
        if i % 48 == 0:
            rmse31 = np.sqrt(np.mean(np.square(pred[i - 48:i] - true[i - 48:i])))
            rs.append(rmse31)

    rmse = np.mean(rs)

    rs1 = np.array(rs)  # 列表转数组
    rs1 = np.round(rs1, 2)  # 对数组中的元素保留两位小数
    rs_new = list(rs1)

    print("rmse31:", rs_new)
    print("rmse：{:.2f}".format(rmse))
    return rmse

def MAE(y_test, predict_save):
    ma = []
    result = abs(y_test-predict_save)
    for i in range(1, 5857):
        if i % 48 == 0:
            mae1 = np.mean(result[i-48:i])
            ma.append(mae1)

    mae = np.mean(ma)

    ma1 = np.array(ma)  # 列表转数组
    ma1 = np.round(ma1, 2)  # 对数组中的元素保留两位小数
    ma_new = list(ma1)

    print("mae1:", ma_new)
    print("mae：{:.2f}".format(mae))
    return mae

def accuracy_passrate(y_test, predict):
    a = []
    b = []
    c = []
    d = []
    for i in range(0, 5856):
        a1 = (abs(y_test[i,] - predict[i,])) / CAP
        a.append(a1)
        if (1 - a[i]) >= 0.9:
            flag = 1
        else:
            flag = 0
        c.append(flag)
    for i in range(1, 5857):
        a2 = np.square(a)
        if i % 48 == 0:
            b1 = (1 - np.sqrt(np.mean(a2[i-48:i])))*100
            c1 = ((sum(c[i-48:i]))/48)*100
            b.append(b1)
            d.append(c1)
    accu = np.mean(b)
    pr = np.mean(d)
    print('the accuracy of test:{:.2f}%'.format(accu))
    print('the pass_rate of test:{:.2f}%'.format(pr))
    return accu, pr

def plot(pred, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(pred)), pred , label = 'predict')
    ax.plot(range(len(true)), true , label = 'true')
    plt.show()
'''
def build_model(params):
    train,test,train_X,train_label,test_X,test_label,scaler = get_dataset()
    model = MultiOutputRegressor(SVR(C=100, kernel='rbf', epsilon=params['epsilon'], gamma=params['gamma']))
    model.fit(train_X, train_label)
    score = model.score(test_X, test_label)
    print(score,params['epsilon'],params['gamma'])
    #0.9954089919861842 0.01 0.05
    return score

best = fmin(fn=build_model,
            space=space,
            algo=tpe.suggest,
            max_evals=10)

print(best)
Best = space_eval(space, best)
# 输出最优参数


epsion = Best['epsilon']
gamma = Best['gamma']


print("epsion:", epsion)
print("gamma:", gamma)
'''


train,test,train_X,train_label,test_X,test_label,scaler = get_dataset()
#best_model = MultiOutputRegressor(SVR(C=100, kernel='rbf',gamma=gamma,epsilon=epsion))
best_model = MultiOutputRegressor(SVR(C=100, kernel='rbf', gamma=0.02, epsilon=0.01))
best_model.fit(train_X,train_label)
# 计算训练时间
time_2 = time.time()
time_2_1 = time_2 - time_1  # 训练所花时间
print('time cost:%s' % time_2_1)

# 保存模型
with open('../2122ukresults/fakeuktrendSVR2.pickle', 'wb') as f:
    pickle.dump(best_model, f)

# 加载模型
#with open('../results/2017STLtrendSVR.pickle', 'rb') as f:
    #best_model = pickle.load(f)
# 窗口滑动测试集预测
prediction = best_model.predict(test_X)
P = scaler.inverse_transform(prediction)
P = P.reshape(-1, 1)
print(P.shape)
np.savetxt('../2122ukresults/fakeuktrendSVRresult2.csv', np.round_(P, 2), delimiter=',')

# 测试集的真实值反归一化
test = scaler.inverse_transform(np.array(test).reshape(-1, 1))
test_label = scaler.inverse_transform(np.array(test_label).reshape(-1, 1))

MAE(test, P)
RMSE(P, test)
plot(P, test)
accuracy_passrate(test, P)