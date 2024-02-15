import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

window_size = 192  # 窗口大小
out_size = 48  # 输出数量
batch_size = 48  # 训练批次大小
epochs = 200  # 训练epoch
CAP = 60000
horizon1 = 11760
horizon2 = 5856

'''
space = {
    'units1': hp.choice('units1', [12,16, 24, 32, 36, 48, 60, 64,72, 76, 80,84, 96, 108,112, 120, 128, 132,144, 160, 168,172,176,184, 192, 216, 240, 264, 288]),
    #'units2': hp.choice('units2', [12,16, 24, 32, 36, 48, 60, 64,72, 76, 80,84, 96, 108,112, 120, 128, 132,144, 160, 168,172,176,184, 192, 216, 240, 264, 288]),
    #'units3': hp.choice('units3', [32, 64, 96]),
    #'units4': hp.choice('units4', [32, 64, 96]),
    'units0': hp.choice('units0', [12,16, 24, 32, 36, 48, 60, 64,72, 76, 80,84, 96, 108,112, 120, 128, 132,144, 160, 168,172,176,184, 192, 216, 240, 264, 288]),
    'learning_rate': hp.choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4])
}
'''
time_1 = time.time()

def get_dataset():
    train_X, train_label, val_X, val_label, pred_test = [], [], [], [], []

    df = pd.read_csv('../2122ukdata/21-22tre+sea+com3-6+com2.csv', index_col=0)
    print(df)
    # 归一化
    scaler = MinMaxScaler()
    df['com3-6'] = scaler.fit_transform(df['com3-6'].values.reshape(-1, 1)).reshape(-1)

    # 划分数据集
    train, val, test = df.iloc[:-horizon1, 2:3], df.iloc[-horizon1:-horizon2, 2:3], df.iloc[-horizon2:, 2:3]
    testforpred = df.iloc[-horizon2 - window_size:, 2:3]
    print(train)
    # 滑动窗口重构训练集和验证集
    for i in range(0, train.shape[0] - window_size - out_size + 1):
        a = train.iloc[i:(i + window_size), -1]
        train_X.append(a)
        b = train.iloc[(i + window_size):(i + window_size + out_size), -1]
        train_label.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_label = np.array(train_label, dtype='float64')
    train_X = train_X.reshape(-1, window_size, 1)
    train_label = train_label.reshape(-1, out_size, 1)
    print(train_X.shape)
    print(train_label.shape)

    for i in range(0, val.shape[0] - window_size - out_size + 1):
        c = val.iloc[i:(i + window_size), -1]
        val_X.append(c)
        d = val.iloc[(i + window_size):(i + window_size + out_size), -1]
        val_label.append(d)
    val_X = np.array(val_X, dtype='float64')
    val_label = np.array(val_label, dtype='float64')
    val_X = val_X.reshape(-1, window_size, 1)
    val_label = val_label.reshape(-1, out_size, 1)
    print(val_X.shape)
    print(val_label.shape)

    # 滑动窗口用来预测的测试集
    for i in range(0, testforpred.shape[0] - window_size - out_size + 1, out_size):
        e = testforpred.iloc[i:(i + window_size), -1]
        pred_test.append(e)
    pred_test = np.array(pred_test, dtype='float64')
    pred_test = pred_test.reshape(-1, window_size, 1)
    print(pred_test.shape)

    return test, train_X, train_label, val_X, val_label, pred_test, scaler

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
    result = abs(y_test - predict_save)
    for i in range(1, 5857):
        if i % 48 == 0:
            mae1 = np.mean(result[i - 48:i])
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
        a1 = (abs(y_test[i] - predict[i])) / CAP
        a.append(a1)
        if (1 - a[i]) >= 0.9:
            flag = 1
        else:
            flag = 0
        c.append(flag)
    for i in range(1, 5857):
        a2 = np.square(a)
        if i % 48 == 0:
            b1 = (1 - np.sqrt(np.mean(a2[i - 48:i]))) * 100
            c1 = ((sum(c[i - 48:i])) / 48) * 100
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
    ax.plot(range(len(pred)), pred, label='predict')
    ax.plot(range(len(true)), true, label='true')
    plt.show()
'''
def build_model(params):
    test, train_X, train_label, val_X, val_label, pred_test, scaler = get_dataset()

    model = Sequential()
    model.add(GRU(units=int(params['units1']), input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    #model.add(GRU(units=int(params['units2']), return_sequences=True))
    #model.add(GRU(units=int(params['units3']), return_sequences=True))
    #model.add(GRU(units=int(params['units4']), return_sequences=True))
    model.add(GRU(units=int(params['units0']), return_sequences=False))
    #model.add(Dropout(0.1))
    model.add(Dense(48))
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(loss='mse', optimizer=optimizer)
    return model

def objective(params):
    test, train_X, train_label, val_X, val_label, pred_test, scaler = get_dataset()
    model = build_model(params)
    earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    history = model.fit(train_X, train_label, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_X, val_label),
                        verbose=2,
                        validation_freq=1, callbacks=[earlystop],
                        shuffle=True)
    loss = history.history['val_loss'][-1]
    print(loss)
    return loss

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)

print(best)
Best = space_eval(space, best)
# 输出最优参数

units1 = int(Best['units1'])
#units2 = int(Best['units2'])
#units3 = int(Best['units3'])
#units4 = int(Best['units4'])
units0 = int(Best['units0'])
learning_rate = Best['learning_rate']

print("units1:",units1)
#print("units2:",units2)
#print("units3:",units3)
#print("units4:",units4)
print("units0:",units0)
print("learning_rate:",learning_rate)
#补充添加：  'units2': units2, 'units3': units3, 'units4': units4,
# 保存最优模型
'''

test, train_X, train_label, val_X, val_label, pred_test, scaler = get_dataset()
#best_model = build_model({'units1': units1, 'units0': units0, 'learning_rate': learning_rate})

best_model = Sequential()
best_model.add(LSTM(units=48, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
#best_model.add(GRU(units=36, return_sequences=True))
#model.add(GRU(units=int(params['units3']), return_sequences=True))
#model.add(GRU(units=int(params['units4']), return_sequences=True))
best_model.add(LSTM(units=72, return_sequences=False))
best_model.add(Dropout(0.1))
best_model.add(Dense(48))
optimizer = Adam(learning_rate=0.0001)
best_model.compile(loss='mse', optimizer=optimizer)
best_model.summary()
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
best_history = best_model.fit(train_X, train_label, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_label),
                verbose=2,
                validation_freq=1, callbacks=[earlystop],
                shuffle=True)
#保存最优模型
best_model.save('../2122ukresults/fakeukbest2layersLSTMcom3-6.h5')
'''
#加载最优模型
best_model = load_model('../results/best2layersGRUseason.h5')
'''
time_2 = time.time()
time_2_1 = time_2 - time_1  # 训练所花时间
print('time cost:%s' % time_2_1)

#设置画布的尺寸，单位为英寸，根据最终输出的图像大小调整一下数字
plt.figure(figsize=(5, 4))
myfont = FontProperties(fname='/home/jiyongqiang/.conda/envs/tf/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Times New Roman.ttf')

# 设置坐标轴标题字体及其大小
plt.rcParams.update({'font.size': 12})

plt.plot(best_history.history['loss'], linewidth = 1.5,label='train loss')
plt.plot(best_history.history['val_loss'], linewidth = 1.5,label='val loss')
plt.xlabel('Epoch',fontsize=12,fontproperties=myfont)
plt.ylabel('Loss',fontsize=12,fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.tick_params(direction='in')
plt.legend(fontsize=12,prop=myfont)
plt.savefig(r'../2122ukresults/fakeukbest2layersLSTMcom3-6LOSS.tiff', dpi=300, bbox_inches='tight')
plt.show()

# 使用最优模型进行预测并输出mae
prediction = best_model.predict(pred_test)
inv_yhat = scaler.inverse_transform(prediction)
P = inv_yhat.reshape(-1, 1)

np.savetxt('../2122ukresults/fakeukbest2layersLSTMcom3-6result.csv', np.round_(P, 2), delimiter=',')

# 测试集的真实值反归一化
scaled_test_label = scaler.inverse_transform(np.array(test).reshape(-1, 1))

MAE(scaled_test_label, P)
RMSE(P, scaled_test_label)
accuracy_passrate(scaled_test_label, P)
plot(P, scaled_test_label)