

import time
import torch

import torch.nn as nn
from torch.utils.data import DataLoader

from stock_dataset_havkes_zz1000 import LoadData  # 这个就是上一小节处理数据自己写的的类，封装在traffic_dataset.py文件中
from utils import Evaluation  # 三种评价指标以及可视化类
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np

from model_tcn import TCN
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

    # 第一步：准备数据（上一节已经准备好了，这里只是调用而已，链接在最开头）
    train_data = LoadData(data_path=["./impact_causality_zz1000.npy", "./tezheng_zz1000.npy"], num_nodes=705, divide_days=[1206, 250],
                         time_interval=1, history_length=7,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)  # num_workers是加载数据（batch）的线程数目


    test_data = LoadData(data_path=["./impact_causality_zz1000.npy", "./tezheng_zz1000.npy"], num_nodes=705, divide_days=[1206, 250],
                         time_interval=1, history_length=7,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

    # 第二步：定义模型（这里其实只是加载模型，关于模型的定义在下面单独写了，先假设已经写好）
    # my_net = GCN(in_c=6, hid_c=6, out_c=1)  # 加载GCN模型
    # my_net = ChebNet(in_c=6, hid_c=6, out_c=1, K=2)   # 加载ChebNet模型
    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    channel_sizes = [32] * 4

    # convolution kernel size
    kernel_size = 3
    dropout = .0

    # 对时间序列进行差分处理

    # x:[N,features,4],y:[N,1,1]

    # [n_train,4,features]

    # [n_test,4,features]

    # [N,1]

    """device = torch.device("cuda")
    for x in [x_train,x_test,y_train,y_test]:
         x = x.to(device)"""

    # train_len = x_train.size()[0]

    model_params = {
        # 'input_size',C_in
        'input_size': 4,
        # 单步，预测未来一个时刻
        'output_size': 1,
        'num_channels': channel_sizes,
        'kernel_size': kernel_size,
        'dropout': dropout
    }
    model = TCN(**model_params)
    # model = model.to(device)

    training_loss = []

    my_net = GATNet(in_c=7 * 1, hid_c=32, out_c=1, n_heads=2)  # 加载GAT模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

    my_net = my_net.to(device)  # 模型送入设备

    # 第三步：定义损失函数和优化器
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()

    optimizer1 = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    optimizer2 = torch.optim.Adam(params=my_net.parameters(), lr=1e-3)  # 没写学习率，表示使用的是默认的，也就是lr=1e-3

    # 第四步：训练+测试
    # Train model
    Epoch = 6000  # 训练的次数

    my_net.train()  # 打开训练模式
    model.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0

        model_x = []
        model_y = []
        net_data = []
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
            my_net.zero_grad()
            prediction = model(model_x[i])

            loss1 = criterion1(prediction, model_y[i])
            # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]],一次把一个batch的训练数据取出来
            # 梯度清零

            predict_value = my_net(net_data[i], device).to(
                torch.device("cpu"))  # [B, N, 1, D],由于标签flow_y在cpu中，所以最后的预测值要放回到cpu中

            loss2 = criterion2(predict_value, net_data[i]["flow_y"])  # 计算损失，切记这个loss不是标量

            loss = loss1 + loss2
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            epoch_loss += loss.item()  # 这里是把一个epoch的损失都加起来，最后再除训练数据长度，用平均loss来表示

            # 反向传播
            loss.backward()
            optimizer1.step()  # 更新参数
            optimizer2.step()

        training_loss.append(epoch_loss)
        end_time = time.time()
        #if (epoch + 1) % 50 == 0:
            #torch.save(my_net.state_dict(), '/data/hxyz/gat_tcn_update/gat_loss_update' + str(epoch) + '.pth')
            #torch.save(model.state_dict(), '/data/hxyz/gat_tcn_update/tcn_loss_update' + str(epoch) + '.pth')

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss,
                                                                          (end_time - start_time) / 60))



    # Test Model
    # 对于测试:
    # 第一、除了计算loss之外，还需要可视化一下预测的结果（定性分析）
    # 第二、对于预测的结果这里我使用了 MAE, MAPE, and RMSE 这三种评价标准来评估（定量分析）
    #state_dict=torch.load('/data/hxyz/GAT/model/net_params_5.pth')
    #my_net.load_state_dict(state_dict)
    """plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(training_loss, label='train')

    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()  # 图例
    plt.show()"""



    #result_file = "GAT_result.h5"
    #file_obj = h5py.File(result_file, "w")  # 将预测值和目标值保存到文件中，因为要多次可视化看看结果

    #file_obj["predict"] = Predict  # [N, T, D]
    #file_obj["target"] = Target  # [N, T, D]


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
    torch.set_num_threads(5)
    #df = pd.read_pickle('/data/hxyz/hushen300.pkl')
    df = pd.read_pickle('/data/hxyz/mindgo/zz1000_zhibiao.pkl')
    #df=df.iloc[500:]
    df = df.set_index('date')
    df = df / df.max()
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
    train_len = len(df) - 250 - 7
    train_data = all_data[:train_len, :]
    test_data = all_data[train_len:, :]
    y_train = y_label[:train_len]
    y_test = y_label[train_len:]

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


    batch_size = 256
    train_X = process(train_X, batch_size)
    train_Y = process(train_Y, batch_size)
    main(a=train_X, b=train_Y)

