from typing import Optional, Tuple, Union
import gym
import numpy as np
import model_stb as model
import utils
from gym import spaces
import copy
import os
import torch
import torch.nn as nn
from args import top_k, linear_input_dim, episode_length
import random
import warnings
from args import device
cuda_i = 0

class RecipeEnvironment(gym.Env):
    def __init__(self, istest=False, istop_k=False, episode_length=32):
        self.top_k = istop_k
        self.istest = istest
        self.episode_length = episode_length
        self.boundary_rating = 3.0
        self.alpha = 0.0
        self.device = torch.device(device)
        self.user_num = 13696
        self.item_num = 240789
        self.cat_num = 287
        self.boundary_userid = int(self.user_num * 0.8)
        max_rating = 5.0
        min_rating = 0.0
        self.stop_count = 1000
        self.a = 2.0 / (float(max_rating) - float(min_rating))
        self.b = - (float(max_rating) + float(min_rating)) / (float(max_rating) - float(min_rating))
        self.positive = self.a * self.boundary_rating + self.b
        self.user_dict = utils.pickle_load('./data/recipe/users_dict.pkl')
        self.done = False
        self.user_embeddings_matrix = torch.nn.Embedding(self.user_num, 100)
        self.item_embeddings_matrix = torch.nn.Embedding(self.item_num, 100)
        self.cat_embeddings_matrix = torch.nn.Embedding(self.cat_num, 100)
        self.item_bias = torch.nn.Embedding(self.item_num, 1).to(self.device)
        self.memory = np.ones((self.user_num,10))*-1
        env_object_path = './data/run_time/recipe_env_objects'
        if os.path.exists(env_object_path):
            objects = utils.pickle_load(env_object_path)
            user_embedding = objects['user_embedding']
            item_embedding = objects['item_embedding']
            ibias_bias = objects['ibias_bias']
            self.rela_num = objects['rela_num']
        else:
            utils.get_envobjects(ratingfile='recipe')
            objects = utils.pickle_load(env_object_path)
            user_embedding = objects['user_embedding']
            item_embedding = objects['item_embedding']
            ibias_bias = objects['ibias_bias']
            self.rela_num = objects['rela_num']

        self.category = utils.pickle_load('./data/recipe/recipe_category.pkl')
        self.user_embeddings_matrix.weight.data.copy_(torch.from_numpy(user_embedding))
        self.item_embeddings_matrix.weight.data.copy_(torch.from_numpy(item_embedding))
        self.item_bias.weight.data.copy_(torch.from_numpy(ibias_bias))
        self.srm_ave = model.DRRAveStateRepresentation(embedding_dim=100,
                                                       user_embeddings=self.user_embeddings_matrix,
                                                       recipe_embeddings=self.item_embeddings_matrix,
                                                       category_embedding=self.cat_embeddings_matrix,
                                                       positive=self.positive,
                                                       episode_length=self.episode_length).to(device=device)  
        self.ibias_bias = self.item_bias
        self.episode_length = self.episode_length

        self.action_space = spaces.Discrete(240785)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,linear_input_dim + self.episode_length))
        self.reward_space = spaces.Box(low=-1, high=1,shape=(1,1))

        self.user_id = torch.tensor(0)
        self.test_flag = False
        self.num = self.boundary_userid
        
        

    def seed(self, seed=2432):
        np.random.seed(seed)
    
    def render(self):
        pass

    def close(self):
        pass

    def reset(self):
        if self.istest == False:
            self.user_id = random.randint(0, self.boundary_userid)
            self.user_data = self.user_dict[self.user_id]
            self.item_b, self.rating_b, self.userid_b, self.idx_b = list(self.user_data['item'][0:10]), \
                                                                    torch.tensor(self.user_data['rating'][0:10]), \
                                                                    torch.tensor(self.user_id),\
                                                                    torch.tensor(self.user_id)
        else:   
            self.user_id = self.num
            self.user_data = self.user_dict[self.user_id]
            self.item_b, self.rating_b, self.userid_b, self.idx_b = list(self.user_data['item'][0:10]), \
                                                                    torch.tensor(self.user_data['rating'][0:10]), \
                                                                    torch.tensor(self.user_id),\
                                                                    torch.tensor(self.user_id)
            self.num += 1
            if self.user_id == (self.user_num - 1):
                self.num = self.boundary_userid
        self.reward_rate = 0.9
        self.repetition = torch.tensor([0])
        self.last_gate = self.category[self.item_b[-1]]
        self.memory[self.idx_b] = [item for item in self.item_b]
        self.preds = set()
        self.cr_list = []
        self.step_count = 0
        self.con_neg_count = 0
        self.con_pos_count = 0
        self.con_zero_count = 0
        self.con_not_neg_count = 0
        self.con_not_pos_count = 0
        self.all_neg_count = 0
        self.all_pos_count = 0
        self.history_items = set()
        self.history_items_windows = self.item_b
        self.state = self.srm_ave(self.userid_b.to(self.device), 
                                  self.item_b, 
                                  self.idx_b.to(self.device), 
                                  self.history_items_windows,
                                  self.history_items,
                                  self.rating_b.to(self.device),
                                  self.preds,
                                  self.last_gate,
                                  self.repetition)
        self.done = False
        return self.state.cpu().detach().numpy()
    
    def calculate_mf_rating(self, action):
        warnings.filterwarnings("ignore")
        if torch.is_tensor(action):
            user_embedding = torch.nn.functional.embedding(torch.tensor(self.user_id), self.user_embeddings_matrix.weight).clone().detach()
            item_embedding = torch.nn.functional.embedding(torch.tensor(action), self.item_embeddings_matrix.weight).clone().detach()
            item_bias = torch.nn.functional.embedding(torch.tensor(action), self.ibias_bias.weight).clone().detach()
            item_embedding1 = item_embedding.squeeze(1)
            mf_rating = torch.mm(user_embedding.unsqueeze(0), item_embedding1.T) + item_bias.T
            return torch.tensor(mf_rating)
        else:
            user_embedding = torch.nn.functional.embedding(torch.tensor([self.user_id]).to(device), self.user_embeddings_matrix.weight).clone().detach()
            item_embedding = torch.nn.functional.embedding(torch.tensor([action]).to(device), self.item_embeddings_matrix.weight).clone().detach()
            item_bias = torch.nn.functional.embedding(torch.tensor([action]).to(device), self.ibias_bias.weight).clone().detach()
            item_embedding1 = item_embedding.squeeze(1)
            mf_rating = torch.mm(user_embedding, item_embedding1.T) + item_bias.T
            return mf_rating

    def step(self, action):
        if not self.top_k or (self.istest and top_k[0]==1):
            action = int(action)
            reward = 0.0
            r = self.calculate_mf_rating(action)
            if r == None:
                r = 0
            cr = r
            r = self.a * r + self.b
            sr = self.con_pos_count - self.con_neg_count
            self.step_count += 1
            if r < 0:
                self.con_neg_count += 1
                self.all_neg_count += 1
                self.con_not_pos_count += 1
                self.con_pos_count = 0
                self.con_not_neg_count = 0
                self.con_zero_count = 0
            elif r > 0:
                self.con_pos_count += 1
                self.all_pos_count += 1
                self.con_not_neg_count += 1
                self.con_neg_count = 0
                self.con_not_pos_count = 0
                self.con_zero_count = 0
            else:
                self.con_not_neg_count += 1
                self.con_not_pos_count += 1
                self.con_zero_count += 1
                self.con_pos_count = 0
                self.con_neg_count = 0
            self.history_items.add(action)
            self.history_items_windows = self.history_items_windows[1:] + [action]
            r += self.alpha * sr
            if r > self.positive:
                self.item_b = self.item_b[1:] + [torch.tensor([action])]
                self.preds.add(int(action))
            if self.step_count == self.episode_length or len(self.history_items) == self.item_num:
                self.done = True
            
            self.rating_b = self.rating_b[1:]
            self.rating_b = torch.cat((self.rating_b, torch.tensor([cr])), 0)

            self.last_gate = self.category[action]
            reward = r
            if action in self.preds:
                self.repetition = torch.tensor([1])
            self.state = self.srm_ave(self.userid_b.to(device), self.item_b, self.idx_b.to(device), self.history_items_windows, self.history_items, self.rating_b.to(device), \
                                    self.preds, self.last_gate, self.repetition)
            self.repetition = torch.tensor([0])
            self.cr_list.append(cr)
            info = {'preds':list(self.preds), 'cr': self.cr_list}
            return self.state.cpu().detach().numpy(), reward, self.done, info
        else:
            reward = -1
            r = self.calculate_mf_rating(torch.tensor(action))
            for i,ir in enumerate(r):
                if int(ir) > 5:
                    r[i] = 5
                if int(ir) < 0 or int(ir) == None:
                    r[i] = 0
            cr = r
            r = self.a * r + self.b
            for act in action:
                self.history_items.add(act)
                self.history_items_windows = self.history_items_windows[1:] + [act]
            for i,ir in enumerate(r):
                if ir >= self.positive:
                    self.preds.add(int(action[i]))
                    self.done = True
            
            reward = torch.mean(r)
            info = {'preds':list(self.preds)}
            self.state = torch.tensor([])
            return self.state, reward, self.done, info
    

