import paddle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stock_dataset_havkes_zz1000 import LoadData  # 这个就是上一小节处理数据自己写的的类，封装在traffic_dataset.py文件中
from utils import Evaluation  # 三种评价指标以及可视化类

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import TCN
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal

from sklearn.metrics import accuracy_score

from gat import GATNet

import warnings
warnings.filterwarnings('ignore')

from StockEnv_GAT_zz1000 import StockTradingEnv
from parl.algorithms import SAC
from StockModel import StockModel
from StockAgent import StockAgent





#TG-test
def split_windows(data,labely,size):
    X=[]
    Y=[]
    for i in range(len(data)-size):
        X.append(data[i:i+size,:])
        Y.append(labely[i+size])
    return np.array(X),np.array(Y)

"""def del_pred(x):
    z = np.mean(x) + 1.8 * np.std(x)
    y = np.mean(x) - 1.8 * np.std(x)
    for i in range(len(x)):
        if x[i] <= z and x[i] >= y:
            x[i] = 0
        else:
            x[i] = 1
    return x"""

def del_pred(x):

    for i in range(len(x)):
        if x[i]>=0.5:
            x[i]=1
        else:
            x[i]=0
    return x

def main(a,b):

    test_data = LoadData(
        data_path=["./impact_causality_zz1000.npy", "./tezheng_zz1000.npy"],
        num_nodes=705, divide_days=[1212, 250],
        time_interval=1, history_length=7,
        train_mode="test")

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
    channel_sizes = [32] * 4

    # convolution kernel size
    kernel_size = 3
    dropout = .0
    # 第二步：定义模型（这里其实只是加载模型，关于模型的定义在下面单独写了，先假设已经写好）
    # my_net = GCN(in_c=6, hid_c=6, out_c=1)  # 加载GCN模型
    # my_net = ChebNet(in_c=6, hid_c=6, out_c=1, K=2)   # 加载ChebNet模型
    my_net = GATNet(in_c=7*1 , hid_c=32, out_c=1, n_heads=2)  # 加载GAT模型

    device = torch.device("cpu")  # 定义设备
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

    my_net = my_net.to(device)  # 模型送入设备
    model = model.to(device)
    criterion = nn.MSELoss()

    state_dict=torch.load('./tg_models/gat_3999.pth')
    state_dict2 = torch.load('./tg_models/tcn_3999.pth')
    my_net.load_state_dict(state_dict)
    model.load_state_dict(state_dict2)
    my_net.eval()  # 打开测试模式
    model.eval()
    with torch.no_grad():  # 关闭梯度
        MAE, MAPE, RMSE = [], [], []# 定义三种指标的列表
        Target = np.zeros([705, 1, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
        Predict = np.zeros_like(Target)  # [N, T, D],T=1 # 预测数据的维度

        total_loss = 0.0
        for data in test_loader:  # 一次把一个batch的测试数据取出来

            # 下面得到的预测结果实际上是归一化的结果，有一个问题是我们这里使用的三种评价标准以及可视化结果要用的是逆归一化的数据
            predict_value = my_net(data, device) # [B, N, 1, D]，B是batch_size, N是节点数量,1是时间T=1, D是节点的流量特征

            loss = criterion(predict_value, data["flow_y"])  # 使用MSE计算loss

            total_loss += loss.item()  # 所有的batch的loss累加
            # 下面实际上是把预测值和目标值的batch放到第二维的时间维度，这是因为在测试数据的时候对样本没有shuffle，
            # 所以每一个batch取出来的数据就是按时间顺序来的，因此放到第二维来表示时间是合理的.
            predict_value = predict_value.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
            target_value = data["flow_y"].transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]

            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)  # 计算模型的性能，返回评价结果和恢复好的数据

            # 下面这个是每一个batch取出的数据，按batch这个维度进行串联，最后就得到了整个时间的数据，也就是
            # [N, T, D] = [N, T1+T2+..., D]
            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)


            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

            print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))
        tcn_prediction=del_pred(model(a.transpose(1,2)).numpy().flatten())

        true_cases = b.numpy().flatten()
        accuracy = accuracy_score(true_cases, tcn_prediction)
        print('准确率Accuracy:', accuracy)

        y_pred = pd.DataFrame([tcn_prediction[-250:]])
        y_pred = y_pred.T
        y_pred.to_pickle('./tcn_gat_zz1000.pkl')


    # 三种指标取平均
    print("Performance:  MAE {:2.2f}    MAPE{:2.2f}%    RMSE{:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))

    Predict = np.delete(Predict, 0, axis=1) # 将第0行的0删除，因为开始定义的时候用0填充，但是时间是从1开始的

    pre=pd.DataFrame(np.squeeze(Predict,axis=2))

    pre.to_pickle('./pre_youxian_zz1000.pkl')





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
    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy(),'pre')
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy(),'ture')

    # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data

EED=1
WARMUP_STEPS=640
EVAL_EPISODES =5
MEMORY_SIZE=int(1e5)
BATCH_SIZE=64
GAMMA=0.995
TAU=0.005
ACTOR_LR=1e-4
CRITIC_LR=1e-4
alpha=0.2
MAX_REWARD=-1e9
position_coe=60

def run_test_episodes(agent,env,y_pred):
    avg_reward = 0.
    avg_worth = 0.
    date = df.date.unique()
    code = df.code.unique()

    worth = []
    worth3 = []
    obs = env.reset()
    env.seed(0)
    done = False
    t = 0
    buy_sell = []
    cangwei = []
    balance = []
    day_sum = []
    while not done:

        if y_pred.iloc[t, 0] == 1:

                action = agent.predict(obs)
                action = (action + 1.) / 2.
                action = action.T

                buy = np.where(action[0] < 1 / 3)
                action[1][buy] = action[1][buy] * position_coe / np.sum(action[1][buy])
                action = action.T

        else:
            action = np.array([[0.5, 0] for _ in range(705)])
        obs, reward, done, info, chicang = env.step(action)
        avg_reward += reward
        buy_sell.append(chicang)

        avg_worth += info['profit']
        t += 1
        print(info['status'])
        worth.append(info['profit'] - 10000000)

        worth3.append(info['profit'])
        cangwei.append(info['cangwei'])
        balance.append(info['balance'])
        day_sum.append(info['day_buy'])


    avg_reward /= t
    avg_worth /= t
    worth2 = []
    for i in range(1, len(worth)):
        worth2.append(worth[i] - worth[i - 1])
    worth2 = [worth[0]] + worth2

    zhongzheng = pd.read_csv('./indexclosedf.csv', index_col=0)
    zhongzheng500 = pd.DataFrame([[zhongzheng.loc[i, '000852']] for i in date])
    earn = pd.DataFrame(np.array([worth2, cangwei, worth3, balance]).T,
                        columns=['worth2', 'chicang', 'worth3', 'blance'])
    earn['earn'] = (earn.iloc[:, 0] / (earn.iloc[:, 2]-earn.iloc[:, 0])) + 1
    earn['earnx'] = [0 for _ in range(len(earn.index))]

    earn['zhongzheng1000'] = zhongzheng500 / zhongzheng500.shift()

    earn['zhongzheng1000'][0] = 1

    earn['zhongzheng1000x'] = [0 for _ in range(len(earn.index))]

    for i in range(len(earn.index)):
        earn['earnx'][i] = earn['earn'][:i + 1].prod()
        earn['zhongzheng1000x'][i] = earn['zhongzheng1000'][:i + 1].prod()

    earn['chaoe'] = earn['earnx'] - earn['zhongzheng1000x']
    x = [i for i in range(len(date))]
    plt.plot(x, earn['earnx'] - 1)
    plt.plot(x, earn['zhongzheng1000x'] - 1)

    plt.xticks(x, list(date))
    plt.title("SAC_strategy_returns")
    plt.xlabel('date')
    plt.xticks(np.arange(0, len(x), step=30), rotation=270)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.legend(['SAC_stratery', 'zz1000'], loc='upper left')
    ax2 = plt.twinx()
    ax2.set_ylabel("amount/yuan")
    plt.bar(x, day_sum, label='Buy money', alpha=0.25)
    plt.legend(loc='lower right')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.show()

    plt.plot(x, earn['chaoe'])
    plt.xticks(x, list(date))
    plt.title("SAC_strategy_returns_chaoe")
    plt.xlabel('date')
    plt.xticks(np.arange(0, len(x), step=30), rotation=270)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.legend(['chaoe'], loc='upper left')
    plt.show()

    plt.bar(x, earn['chicang'])
    plt.xticks(x, list(date))
    plt.title("Average position")
    plt.xlabel('date')
    plt.ylabel('%')
    plt.xticks(np.arange(0, len(x), step=30), rotation=270)
    plt.gcf().subplots_adjust(bottom=0.3)

    plt.show()
    earn['date'] = date



    print(f'Evaluator:The average reward id {avg_reward:.3f} over {t} days.')
    print(f'Evaluator:The average worth id {avg_worth:.3f} over {t} days.')

    return avg_reward

if __name__ == '__main__':
    df = pd.read_pickle('./zz1000_zhibiao.pkl')

    df = df.set_index('date')
    all_data = df.values
    all_dataT = all_data.T
    max_index = signal.argrelextrema(all_dataT[0, :], np.greater, order=3)
    min_index = signal.argrelextrema(all_dataT[0, :], np.less, order=3)
    index = np.append(max_index, min_index)

    y_label = []

    for i in range(len(df)):

        if i in index:

            y_label.append(1)
        else:
            y_label.append(0)

    train_len = len(df)-250-7
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
    x_train = torch.tensor(data=train_X).float()
    y_train = torch.tensor(data=train_Y).float()

    x_test = torch.tensor(data=test_X).float()
    y_test = torch.tensor(data=test_Y).float()
    main(x_test,y_test)

    #优先级
    df = pd.read_csv('./zz1000.csv',index_col=0)

    df = df.iloc[-250 * 705:]


    data = pd.read_pickle('./pre_youxian_zz1000.pkl')

    df['youxianji'] = data.T.values.flatten()



    #test-tg-sac

    y_pred = pd.read_pickle('./tcn_gat_zz1000.pkl')

    env = StockTradingEnv(df)
    state_dim = 7
    action_dim = 2
    model = StockModel(state_dim, action_dim)
    algorithm = SAC(model, gamma=GAMMA, tau=TAU, alpha=alpha, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)

    agent = StockAgent(algorithm)
    agent.restore('./sac_models/SAC_100.ckpt')

    avg_reward = run_test_episodes(agent, env, y_pred)


