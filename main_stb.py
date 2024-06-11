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
test_env = RecipeEnvironment(True, True, episode_length)
rela_num = train_env.rela_num
train_env = make_vec_env(lambda: train_env, n_envs=1)
model = PPO(CustomActorCriticPolicy, 
            train_env, 
            learning_rate=0.001,
            batch_size=512,
            gamma=0.99,
            seed=3407,
            device=device,
            verbose=1)



episodes = 100
vec_env = model.get_env()
boundary_userid = int(13696 * 0.8)

test_step = 0
storage = []
result_file_path = './data/result/' + time.strftime(
            '%Y%m%d%H%M%S') + '_recipe_%f' % 0.0

def save_model():
    model.save("best_model")

def eval_metrics(mean_reward, std_reward, tp_list):
    precision = np.array(tp_list) / episode_length
    recall = np.array(tp_list) / (rela_num[int(boundary_userid):] + 1e-20)
    f1 = (2 * precision * recall) / (precision + recall + 1e-20)

    test_ave_precision = np.mean(precision)
    test_ave_recall = np.mean(recall)
    test_ave_f1 = np.mean(f1)

    storage.append(
        [mean_reward, std_reward, test_ave_precision,
            test_ave_recall, test_ave_f1])
    utils.pickle_save(storage, result_file_path)

    print('\ttest  average reward over step: %2.4f, std reward: %2.4f, precision@%d: %.4f, recall@%d: %.4f, f1@%d: %.4f' % (
    mean_reward, std_reward, episode_length, test_ave_precision, episode_length, test_ave_recall, episode_length,
    test_ave_f1))

       
   
if __name__ == "__main__": 
    model.learn(total_timesteps=10000)  
    



