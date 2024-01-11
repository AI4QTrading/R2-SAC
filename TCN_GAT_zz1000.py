

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from stock_dataset_havkes_zz1000 import LoadData
from utils import Evaluation
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import TCN

from sklearn.preprocessing import MinMaxScaler

import scipy.signal as signal

from gat import GATNet

import warnings
warnings.filterwarnings('ignore')

def split_windows(data,labely,size):
    X=[]
    Y=[]
    for i in range(len(data)-size):
        X.append(data[i:i+size,:])
        Y.append(labely[i+size])
    return np.array(X),np.array(Y)


def process(data,bs):
    l=len(data)
    tmp=[]
    for i in range(0,l,bs):
        if i+bs >l:
            tmp.append(data[i:].tolist())
        else:
            tmp.append(data[i:i+bs].tolist())

    tmp=np.array(tmp)
    return tmp



def main(a,b):

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    train_data = LoadData(
        data_path=["/data/hxyz/graph/impact_causality_zz1000.npy", "/data/hxyz/TCN/tezheng_zz1000.npy"], num_nodes=705,
        divide_days=[1206, 250],
        time_interval=1, history_length=7,
        train_mode="train")

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)  # num_workers是加载数据（batch）的线程数目


    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    channel_sizes = [256]*5

    kernel_size = 5
    dropout = .0


    model_params = {
        'input_size': 4,
        'output_size': 1,
        'num_channels': channel_sizes,
        'kernel_size': kernel_size,
        'dropout': dropout
    }
    model = TCN(**model_params)
    training_loss = []
    my_net = GATNet(in_c=7*1 , hid_c=32, out_c=1, n_heads=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_net = my_net.to(device)

    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()


    optimizer = optim.Adam(params=my_net.parameters(),lr=1e-3)


    Epoch = 2000

    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        count = 0
        model_x=[]
        model_y=[]
        net_data=[]
        for batch_id, data in enumerate(a):
            label = b[batch_id]
            x_train = torch.tensor(data=data).float()
            y_train = torch.tensor(data=label).float()
            x_train = x_train.transpose(1, 2)
            model_x.append(x_train)
            model_y.append(y_train)
        for data in train_loader:
            net_data.append(data)

        start_time = time.time()

        for i in range(len(net_data)):
            model.zero_grad()
            prediction = model(model_x[i])
            loss1 = criterion1(prediction, model_y[i])

            my_net.zero_grad()
            count +=1
            predict_value = my_net(net_data[i], device).to(torch.device("cpu"))

            loss2 = criterion2(predict_value, net_data[i]["flow_y"])

            loss=loss1+loss2
            epoch_loss += loss.item()



            loss.backward()

            optimizer.step()
        training_loss.append(epoch_loss)
        end_time = time.time()
        if (epoch + 1) % 20 == 0:
            torch.save(my_net.state_dict(), '/data/hxyz/zz1000/net_params_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), '/data/hxyz/zz1000/model_params_' + str(epoch) + '.pth')

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss,
                                                                          (end_time - start_time) / 60))


    plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(training_loss, label='train')

    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()




def compute_performance(prediction, target, data):  # 计算模型性能
    # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
    # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
    try:
        dataset = data.dataset  # 数据为dataloader型，通过它下面的属性.dataset类变成dataset型数据
    except:
        dataset = data  # 数据为dataset型，直接赋值

    # 下面就是对预测和目标数据进行逆归一化，recover_data()函数在上一小节的数据处理中
    #  flow_norm为归一化的基，flow_norm[0]为最大值，flow_norm[1]为最小值
    # prediction.numpy()和target.numpy()是需要逆归一化的数据，转换成numpy型是因为 recover_data()函数中的数据都是numpy型，保持一致
    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）


if __name__ == '__main__':
    torch.set_num_threads(4)
    #df = pd.read_pickle('/data/hxyz/hushen300.pkl')
    df = pd.read_pickle('/data/hxyz/mindgo/zz1000_zhibiao.pkl')
    df = df.set_index('date')
    all_data = df.values
    all_dataT = all_data.T
    max_index = signal.argrelextrema(all_dataT[0, :], np.greater, order=3)
    min_index = signal.argrelextrema(all_dataT[0, :], np.less, order=3)
    index = np.append(max_index, min_index)

    y_label = []

    for i in range(len(df)):
        # if i in min_index[0]:
        if i in index:

            y_label.append(1)
        else:
            y_label.append(0)

    # train_len=850
    train_len = 1200
    train_data = all_data[:train_len, :]
    test_data = all_data[train_len:, :]
    y_train = y_label[:train_len]
    y_test = y_label[train_len:]

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(train_data.shape[0]), train_data[:, 0], label='train data')
    plt.plot(np.arange(train_data.shape[0], train_data.shape[0] + test_data.shape[0]), test_data[:, 0],
             label='test data')
    plt.legend()
    plt.show()

    scaler = MinMaxScaler()
    scaler_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    window_size = 7
    train_X, train_Y = split_windows(scaler_train_data, y_train, size=window_size)
    test_X, test_Y = split_windows(scaled_test_data, y_test, size=window_size)
    print('train shape', train_X.shape, train_Y.shape)
    print('test shape', test_X.shape, test_Y.shape)
    train_X = train_X.astype('float32')
    train_Y = train_Y.astype('float32')
    test_X = test_X.astype('float32')
    test_Y = test_Y.astype('float32')
    train_Y = train_Y[:, np.newaxis]
    test_Y = test_Y[:, np.newaxis]


    x_test = torch.tensor(data=test_X).float()
    y_test = torch.tensor(data=test_Y).float()
    batch_size=512
    train_X = process(train_X, batch_size)
    train_Y = process(train_Y, batch_size)
    main(a=train_X,b=train_Y)

