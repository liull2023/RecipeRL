import tensorflow as tf
from math import sqrt
from tensorflow import keras
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import itertools
import copy
from collections import namedtuple
from collections import Counter
from args import State_compare_flag, device
#####################  hyper parameters  ####################
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
gpu_use = 'cuda:0'
cuda_i = 0
RENDER = False
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
    
class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        
        # use mask
        if mask is not None:
            mask = mask.view((mask.shape[0], 1, 1, 1))
            attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention
    
class Multi_CrossAttention(nn.Module):
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       
        self.all_head_size  = all_head_size     
        self.num_heads      = head_num          
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,x,y,attention_mask=None):
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        if attention_mask is not None:
            attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output
    

class DRRAveStateRepresentation(nn.Module):
    def __init__(self, embedding_dim, user_embeddings, recipe_embeddings, category_embedding, positive, episode_length):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = nn.Conv1d(10, 1, 1)
        self.concat = torch.cat
        self.flatten = nn.Flatten()
        self.user_embeddings = user_embeddings
        self.recipe_embeddings = recipe_embeddings
        self.category_embedding = category_embedding
        self.positive = positive
        self.episode_length = episode_length
        self.attention = Multi_CrossAttention(100, 100, 5)
  
    def forward(self, user_ids, item_id, idx, history, global_history, rating, preds, last_category, repetition):
        user_num = idx
        preds = list(preds)
        if len(preds) < self.episode_length:
            temp = [0 for i in range(self.episode_length - len(preds))]
            for tm in temp:
                preds.append(tm)
        H = []
        mask = []
        item_embedding = []
        user_n_items = torch.tensor(item_id).to(device=device)
        user_embeddings = self.user_embeddings(user_ids, ).clone().detach()
 
        for i,item in enumerate(user_n_items):
            ui = user_embeddings * self.recipe_embeddings(item).clone().detach() 
            H.append(ui.unsqueeze(0))
            item_embedding.append(self.recipe_embeddings(item).clone().detach())
        item_embedding = torch.stack(item_embedding)
        Sui = torch.stack(H).view(1, -1)

        rating_mean = torch.mean(rating.float())
        rating_var = torch.var(rating.float())
        for i in rating:
            tmp = (5 - i) / 5
            noise = torch.normal(rating_mean/5, rating_var) / 5
            tmp += noise
            mask.append(tmp)
        history_embedding = []
        attention_mask = []
        item_count = Counter(global_history)
        for i,item in enumerate(torch.tensor(history).to(device=device)):
            re = self.recipe_embeddings(item).clone().detach()
            pi = torch.rand(1).to(device)
            if pi < mask[i]:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
            count = item_count[int(item)]
            discount_rate = count/10             
            history_embedding.append(re*mask[i]*(1-discount_rate))
        history_embedding = torch.stack(history_embedding)
        Sch = history_embedding.view(1, -1)
        Suc = user_embeddings * self.category_embedding(torch.tensor(last_category-1).to(device)).clone().detach()   
        Suc = Suc.unsqueeze(0)
        I = []
        I = []
        attn_weights = torch.matmul(history_embedding, item_embedding.transpose(0, 1))  # 10, 10
        attn_weights = F.softmax(attn_weights, dim=1)

        attended_state = torch.matmul(attn_weights.unsqueeze(1).to(device), item_embedding.unsqueeze(0))  # 10, 1, 100
        for att_state in attended_state:
            I.append(att_state.squeeze(1))

        SAch = torch.stack(I).view(1, -1)

        if State_compare_flag[0]:
            state= torch.cat((Sui, torch.tensor([preds]).to(device)), dim=-1)       
        if State_compare_flag[1]:
            state = torch.cat((Sch, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[2]:
            state= torch.cat((SAch, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[3]:  
            state = torch.cat((Sui, Sch, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[4]:  
            state = torch.cat((Sui, SAch, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[5]:  
            state = torch.cat((Sui, Suc, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[6]:  
            state = torch.cat((Sch, Suc, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[7]:  
            state = torch.cat((SAch, Suc, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[8]:  
            state = torch.cat((Sui, Sch, Suc, torch.tensor([preds]).to(device)), dim=-1)
        if State_compare_flag[9]:  
            state = torch.cat((Sui, SAch, Suc, torch.tensor([preds]).to(device)), dim=-1)   
        return state
        


class CFEmbedding(nn.Module):
    def __init__(self, max_user_id, max_item_id, emb_size, l2_factor):
        super(CFEmbedding, self).__init__()
        self.user_embeddings = nn.Embedding(max_user_id, emb_size)
        self.item_embeddings = nn.Embedding(max_item_id, emb_size)
        self.item_bias = nn.Embedding(max_item_id, 1)
        self.l2_factor = l2_factor

    def forward(self, user_ids, item_ids):
        user_embs = self.user_embeddings(user_ids)
        item_embs = self.item_embeddings(item_ids)
        ibias_embs = self.item_bias(item_ids).squeeze()
        dot_e = user_embs * item_embs
        ys_pre = torch.sum(dot_e, dim=1) + ibias_embs
        return ys_pre



