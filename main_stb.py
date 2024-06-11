import gym, optuna
import numpy as np
from stb3.evaluation import evaluate_policy
from utils import calculate_ndcg
from stb3.ppo import PPO
from stb3.a2c import A2C
from stb3.dqn import DQN
from stable_baselines3.common.env_util import make_vec_env
from env_stb import RecipeEnvironment
import torch
import utils
import time
import tqdm
from stb3.policy_stb import CustomActorCriticPolicy
from args import episode_length, top_k, device
import args
from stb3.dqn_policy import DQNPolicy

utils.set_seed()
log_dir = './data/panda_reach_v2_tensorboard/'
train_env = RecipeEnvironment(episode_length=episode_length)
rela_num = train_env.rela_num
train_env = make_vec_env(lambda: train_env, n_envs=1)
model = PPO(CustomActorCriticPolicy, 
            train_env, 
            learning_rate=0.001,
            batch_size=512,
            gamma=0.99,
            seed=0,
            device=device,
            verbose=1)

episodes = 32
vec_env = model.get_env()

test_step = 0
storage = []
result_file_path = './data/result/' + time.strftime(
            '%Y%m%d%H%M%S') + '_recipe_%f' % 0.0

def save_model():
    model.save("best_model")
       
   
if __name__ == "__main__": 
    model.learn(total_timesteps=10000)  
    



