import random
import copy
import numpy as np
import pandas as pd
import gym
from gym import spaces

INITIAL_ACCOUNT_BALANCE = 10000000
#INITIAL_ACCOUNT_BALANCE =13279174.853655895# 初始的金钱
#INITIAL_ACCOUNT_BALANCE =14351536.360511404
BUY_RATE = 0.001  # 买入费率
SELL_RATE = 0.0015  # 买入费率


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.date = df.date.unique()
        self.df = df.set_index(['date'])
        self.take_action = pd.DataFrame([[[1]], [[0]], [[1]], [[0]]])
        self.s_closedf = pd.read_csv('/disk/ljq_104/1d_data/close_post.csv', index_col=0)
        self.close_pre = [[self.s_closedf.loc[self.date[i], j] if (j in self.s_closedf) else 0 for j in
                           self.df.code.loc[self.date[i - 1]]] for i in
                          range(1, len(self.date))]
        self.close_now = [[self.s_closedf.loc[self.date[i], j] if (j in self.s_closedf) else 0 for j in
                           self.df.code.loc[self.date[i]]] for i in
                          range(len(self.date))]
        # 动作的可能情况：买入x%, 卖出x%, 观望
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1, 50]), dtype=np.float32)

        # 环境状态的维度
        # self.observation_space = spaces.Box( shape=(15,), dtype=np.float32)

        self.current_step = 0

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    def add_step(self):
        self.current_step +=1

    # 处理状态
    def _next_observation(self):
        # 有些股票数据缺失一些数据，处理一下
        date = self.date
        df = self.df

        obs = np.array([
            list((df.loc[date[self.current_step], 'close'].fillna(0) - df.loc[
                date[self.current_step], 'close'].fillna(0).mean()) / (
                     df.loc[date[self.current_step], 'close'].fillna(0).std())),
            list((df.loc[date[self.current_step], 'open'].fillna(0) - df.loc[
                date[self.current_step], 'open'].fillna(0).mean()) / (
                     df.loc[date[self.current_step], 'open'].fillna(0).std())),
            list((df.loc[date[self.current_step], 'high'].fillna(0) - df.loc[
                date[self.current_step], 'high'].fillna(0).mean()) / (
                     df.loc[date[self.current_step], 'high'].fillna(0).std())),
            list((df.loc[date[self.current_step], 'low'].fillna(0) - df.loc[
                date[self.current_step], 'low'].fillna(0).mean()) / (
                     df.loc[date[self.current_step], 'low'].fillna(0).std())),
            list((df.loc[date[self.current_step], 'volume'].fillna(0) - df.loc[
                date[self.current_step], 'volume'].fillna(0).mean()) / (
                     df.loc[date[self.current_step], 'volume'].fillna(0).std())),
            list((df.loc[date[self.current_step], 'avg_price'].fillna(0) - df.loc[date[self.current_step], 'avg_price'].fillna(
                0).mean()) / (df.loc[date[self.current_step], 'avg_price'].fillna(0).std())),
            list((df.loc[date[self.current_step], 'turnover_rate'].fillna(0) - df.loc[
                date[self.current_step], 'turnover_rate'].fillna(0).mean()) / (
                     df.loc[date[self.current_step], 'turnover_rate'].fillna(0).std())),
        ])
        #print(list(df.loc[date[self.current_step],'minute']))
        return obs.T

    def _take_action(self, action):
        # 随机设置当前的价格，其范围上界为当前时间点的价格
        # current_price = random.uniform(self.df.loc[self.current_step,"low"],self.df.loc[self.current_step,"high"])
        current_price = list(self.df.loc[self.date[self.current_step], "close"])
        action_type = action[0]
        amount = action[1].tolist()
        balance_fix=self.balance

        for i in range(len(amount)):

            if action_type[i] < 1 / 3 and self.balance >= current_price[i]*100:  # 买入amount%
                #total_possible = int(self.balance / current_price[i])
                #shares_bought = int(total_possible * amount[i])
                amount[i]=np.clip(amount[i],0,0.1)
                total_possible = balance_fix*amount[i] # 买入amount%
                shares_bought = int(total_possible / (current_price[i]*100 * (1 + BUY_RATE)))
                if shares_bought != 0.:
                    prev_cost = self.cost_basis[i] * self.shares_held[i]
                    additional_cost = shares_bought * current_price[i]*100

                    self.balance -= additional_cost * (1 + BUY_RATE)
                    self.cost_basis[i] = (prev_cost + additional_cost * (1 + BUY_RATE)) / (
                                              self.shares_held[i] + shares_bought)
                    self.shares_held[i] += shares_bought

            elif action_type[i] > 2 / 3 and self.shares_held[i] != 0:  # 卖出amount%
                shares_sold = int(self.shares_held[i] * amount[i])
                self.balance = self.balance + shares_sold * current_price[i]*100 - shares_sold * current_price[i] *100* SELL_RATE
                self.shares_held[i] -= shares_sold
                self.total_shares_sold[i] += shares_sold
                self.total_sales_value[i] += shares_sold * current_price[i]*100

            else:
                pass
        chicang=self.shares_held+[self.balance]

        final_gushu=[]
        for i in self.shares_held:
            final_gushu.append(i*100)
        net_worth_before=self.net_worth
        # 计算出执行动作后的资产净值
        self.net_worth = self.balance +  sum(np.multiply(final_gushu, current_price))

        cangwei = (self.net_worth - self.balance) / self.net_worth
        day_buy = self.net_worth - self.balance
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == [0 for _ in range(705)]:
            self.cost_basis = [0 for _ in range(705)]

        return chicang,cangwei,day_buy,net_worth_before

    # 执行当前动作，并计算出当前的数据（如：资产等）



    # 与环境交互
    def step(self, action):
        # 在环境内执行动作

        #price_now = np.array(list(self.df.loc[self.date[self.current_step], 'pricenow']))

        action = action.T



        chicang,cangwei,day_buy,net_worth_before=self._take_action(action)

        done = False
        status = None

        reward = 0

        # 判断是否终止
        self.current_step += 1



        # delay_modifier = (self.current_step / MAX_STEPS)

        # reward += delay_modifier

        if self.current_step > len(self.date) - 1:
            status = f'[ENV] Loop training. Max worth was {self.max_net_worth}, final worth is {self.net_worth}.'
            # reward += (self.net_worth / INITIAL_ACCOUNT_BALANCE - max_predict_rate) / max_predict_rate
            reward += (10 * (self.net_worth - net_worth_before) / net_worth_before)
            self.current_step = 0  # loop training
            done = True

        if self.net_worth <= 0:
            status = f'[ENV] Failure at step {self.current_step}. Loss all worth. Max worth was {self.max_net_worth}'
            reward += -10
            # self.current_step = 0
            done = True

        else:
            reward += (5 * (self.net_worth - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE)

        status = f'TESTING: Max worth was {self.max_net_worth}, final worth is {self.net_worth}.'


        obs = self._next_observation()

        return obs, reward, done, {
            'profit': self.net_worth,
            'current_step': self.current_step,
            'status': status,
            'cangwei': cangwei,
            'balance': self.balance,
            'day_buy': day_buy

        },chicang

    # 重置环境
    def reset(self, new_df=None):
        # 重置环境的变量为初始值
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = [0 for _ in range(705)]
        self.cost_basis = [0 for _ in range(705)]
        self.total_shares_sold =[0 for _ in range(705)]
        self.total_sales_value = [0 for _ in range(705)]

        # 传入环境数据集
        if new_df:
            self.df = new_df
        # if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
        self.current_step = 0

        return self._next_observation()

    # 显示环境至屏幕
    def render(self, mode='human'):
        # 打印环境信息
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-' * 30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        return profit