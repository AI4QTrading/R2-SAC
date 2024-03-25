import numpy as np
import paddle

from replay_memory_sacduiqi import ReplayMemory
from Sac import SAC
from StockModel import StockModel
from StockAgent import StockAgent

import StockEnv_zz1000
import pandas as pd



SEED=1
WARMUP_STEPS=640
EVAL_EPISODES =10
MEMORY_SIZE=int(1e5)
BATCH_SIZE=32
GAMMA=0.95
TAU=0.005
ACTOR_LR=1e-4
CRITIC_LR=1e-4
alpha=0.35
position_coe=60


df = pd.read_csv('./zz1000.csv',index_col=0)#两天的样例
df=df.iloc[:1250*705]#选择训练的日期


def run_train_episode(agent,env,rpm,episode_num):

    obs = env.reset()
    env.seed(SEED)
    done =False
    episode_reward =0
    episode_steps = 0

    while not done:
        episode_steps += 1

        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(0,1,size=(obs.shape[0],2))

        else:
            action = agent.sample(obs)
            action=(action+1.)/2.
            action = action.T

            buy = np.where(action[0] < 1 / 3)
            action[1][buy] = action[1][buy] * position_coe / np.sum(action[1][buy])
            action = action.T



        next_obs, reward, done, info,_ = env.step(action)

        terminal = float(done)

        rpm.append(obs,action,reward,next_obs,terminal)

        obs = next_obs
        episode_reward += reward

        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)


    print(f'Learner: Episode {episode_num} done. The reward is {episode_reward:.3f}.')
    print(info['status'])
    return episode_reward,episode_steps,info['profit']

def do_train(agent,env,rpm):
    save_freq =2
    total_steps =0
    train_total_steps =1e6
    episode_num =0

    while total_steps < train_total_steps:
        episode_num +=1

        episode_reward, episode_steps,worth = run_train_episode(agent,env,rpm,episode_num)
        total_steps +=episode_steps
        if (episode_num%save_freq==0):

            agent.save("./sac_models/SAC_"+str(episode_num)+".ckpt")




if __name__ == '__main__':


    env = StockEnv_zz1000.StockTradingEnv(df)

    env.seed(SEED)
    paddle.seed(SEED)
    np.random.seed(SEED)

    state_dim = 7
    action_dim = 2
    model = StockModel(state_dim,action_dim)
    algorithm = SAC(model,gamma=GAMMA,tau=TAU, alpha=alpha, actor_lr=ACTOR_LR,critic_lr=CRITIC_LR)
    agent=StockAgent(algorithm)
    rpm = ReplayMemory(705, max_size=MEMORY_SIZE, obs_dim=state_dim, act_dim=action_dim)


    do_train(agent,env,rpm)

