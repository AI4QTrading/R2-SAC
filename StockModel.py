import paddle
import paddle.nn as nn

import paddle.nn.functional as F

import parl

LOG_SIG_MAX= 1
LOG_SIG_MIN= -1e9

class StockModel(parl.Model):
    def __init__(self,obs_dim,action_dim):
        super(StockModel,self).__init__()
        self.actor_model = Actor(obs_dim,action_dim)
        self.critic_model = Critic(obs_dim,action_dim)

    def policy(self,obs):
        return self.actor_model(obs)

    def value(self,obs,action):
        return self.critic_model(obs,action)

    def get_actor_params(self):
        return self.actor_model.parameters()


    def get_critic_params(self):
        return self.critic_model.parameters()


# 动作网络：输出连续的动作信号
class Actor(parl.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        #weight= paddle.ParamAttr(name='weight',)

        self.l1 = nn.Linear(state_dim, 512)




        #self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.mean_linear = nn.Linear(512,action_dim)
        self.std_linear = nn.Linear(512, action_dim)

    def forward(self, obs):


        a = F.relu(self.l1(obs))
        bn = nn.BatchNorm(a.shape[1])
        a=bn(a)

        #print(self.l1.bias)
        #a = F.relu(self.l2(a))
        a = F.sigmoid(self.l3(a))
        bn = nn.BatchNorm(a.shape[1])
        a = bn(a)

       # a = F.relu(self.l4(a))
    # 输出层激活函数采用tanh，将输出映射至[-1,1]
        act_mean =self.mean_linear(a)
        act_std = self.std_linear(a)
        act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        #print(act_mean)

        return act_mean,act_log_std


# 值函数网络：评价一个动作的价值
class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(obs_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)
       # self.l4 = nn.Linear(512, 512)


        self.l4 = nn.Linear(obs_dim + action_dim, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 1)


    def forward(self, obs, action):

        q1 = F.relu(self.l1(paddle.concat([obs, action], 2)))
        bn = nn.BatchNorm(q1.shape[1])
        q1 = bn(q1)
        q1 = F.relu(self.l2(q1))
        bn = nn.BatchNorm(q1.shape[1])
        q1 = bn(q1)
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(paddle.concat([obs, action], 2)))
        bn = nn.BatchNorm(q2.shape[1])
        q2 = bn(q2)
        q2 = F.relu(self.l5(q2))
        bn = nn.BatchNorm(q2.shape[1])
        q2 = bn(q2)
        q2 = self.l6(q2)
        #q1 = q1.sum(axis=1)z
        #q2=q2.sum(axis=1)

        return q1,q2

        