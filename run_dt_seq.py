import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_seq import GPT, GPTConfig
from mingpt.trainer_seq import Trainer, TrainerConfig
from mingpt.model_simulator import GPT as GPT_simu
from mingpt.model_simulator import GPTConfig as GPTConfig_simu
from mingpt.trainer_simulator import Trainer as Trainer_simu
from mingpt.trainer_simulator import TrainerConfig as TrainerConfig_simu
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions,actions_neg, actions_len, return_step, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = 5010
        # self.vocab_size = actions.shape[0] 
        self.data = data
        self.actions = actions
        self.actions_neg = actions_neg
        self.actions_len = actions_len
        self.return_step = return_step
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx and i>block_size: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        # states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.
        # states = torch.tensor(self.data[idx:done_idx], dtype=torch.long).unsqueeze(1)
        # actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        states = torch.tensor(self.data[idx:done_idx], dtype=torch.long)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long)
        actions_neg = torch.tensor(self.actions_neg[idx:done_idx], dtype=torch.long)
        actions_len = torch.tensor(self.actions_len[idx:done_idx], dtype=torch.long)
        return_step = torch.tensor(self.return_step[idx:done_idx], dtype=torch.float32)
        
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        return states, actions,actions_neg, actions_len, return_step, rtgs, timesteps



# 4Rec accuracy

# data_load_num
# 小于4893
idx_num=3000


#划分数据集
idx_num_train = int(0.8 * idx_num)
idx_num_test = idx_num-idx_num_train

user_retain=pd.read_csv('./WSDM/user_retain_seq.csv')
# obss=user_retain['obss'].values
rtgs=user_retain['rtg'].values
# actions=user_retain['actions'].values
actions_len=user_retain['actions_len'].values
return_step=user_retain['return_step'].values
timesteps=user_retain['timesteps'].values
done_idx_file=pd.read_csv('./WSDM/done_idx_seq.csv')
done_idxs=done_idx_file['done_idx'].values

action_seq=pd.read_csv('./WSDM/action_seq.csv')
actions=action_seq['actions'].values.reshape(-1,20)
state_seq=pd.read_csv('./WSDM/state_seq.csv')
obss=state_seq['obss'].values.reshape(-1,30)

action_seq_neg=pd.read_csv('./WSDM/action_seq_small.csv')
actions_neg=action_seq_neg['actions'].values.reshape(-1,20)
ran_pad = actions.shape[0]-actions_neg.shape[0]
actions_neg = np.concatenate((actions_neg,actions_neg[:ran_pad]),0)

action_seq_large=pd.read_csv('./WSDM/action_seq_large.csv')
actions_large=action_seq_large['actions'].values.reshape(-1,20)
ran_pad_large = actions.shape[0]-actions_large.shape[0]
actions_large = np.concatenate((actions_large,actions_large[:ran_pad_large]),0)
for i in range(actions_neg.shape[0]):
    if return_step[i]<5:
        actions_neg[i]=actions_large[i]
print('start training!')
# print(actions_neg)
# print(actions_neg.shape)
# print(actions_large)
# print(return_step)
# de



# # reward large
# user_retain=pd.read_csv('./WSDM/user_retain_seq_large.csv')
# # obss=user_retain['obss'].values
# rtgs=user_retain['rtg'].values
# # actions=user_retain['actions'].values
# actions_len=user_retain['actions_len'].values
# return_step=user_retain['return_step'].values
# timesteps=user_retain['timesteps'].values
# done_idx_file=pd.read_csv('./WSDM/done_idx_seq_large.csv')
# done_idxs=done_idx_file['done_idx'].values

# action_seq=pd.read_csv('./WSDM/action_seq_large.csv')
# actions=action_seq['actions'].values.reshape(-1,20)
# state_seq=pd.read_csv('./WSDM/state_seq_large.csv')
# obss=state_seq['obss'].values.reshape(-1,30)


# # reward small
# user_retain=pd.read_csv('./WSDM/user_retain_seq_small.csv')
# # obss=user_retain['obss'].values
# rtgs=user_retain['rtg'].values
# # actions=user_retain['actions'].values
# actions_len=user_retain['actions_len'].values
# return_step=user_retain['return_step'].values
# timesteps=user_retain['timesteps'].values
# done_idx_file=pd.read_csv('./WSDM/done_idx_seq_small.csv')
# done_idxs=done_idx_file['done_idx'].values

# action_seq=pd.read_csv('./WSDM/action_seq_small.csv')
# actions=action_seq['actions'].values.reshape(-1,20)
# state_seq=pd.read_csv('./WSDM/state_seq_small.csv')
# obss=state_seq['obss'].values.reshape(-1,30)


# print(actions[10:12])
# debug

def re_index(actions,obss):
    vocab_size=5010
    import random
    idx_list=list(range(vocab_size))
    random.shuffle(idx_list)
    action_dic={}
    action_flag=0
    action_new=[]
    obss_new=[]
    for i in range(actions.shape[0]):
        action_day=[]
        for j in range(len(actions[i])):
            if str(actions[i][j]) in action_dic.keys():
                action_day.append(action_dic[str(actions[i][j])])
            else:
                action_day.append(idx_list[action_flag])
                action_dic[str(actions[i][j])]=idx_list[action_flag]
                action_flag+=1
        action_new.append(action_day)          
    for i in range(obss.shape[0]):
        obss_day=[]
        for j in range(len(obss[i])):
            if str(obss[i][j]) in action_dic.keys():
                obss_day.append(action_dic[str(obss[i][j])])
            else:
                obss_day.append(idx_list[action_flag])
                action_dic[str(obss[i][j])]=idx_list[action_flag]
                action_flag+=1
        obss_new.append(obss_day)
    return action_new, obss_new, vocab_size

def timestep_paddle(timesteps_train):
    time_flag_train=0
    timesteps_list_train=list(timesteps_train)
    for i in range(len(timesteps_list_train)):
        if timesteps_list_train[i]==0:
            time_flag_train+=1
            if time_flag_train==2:
                timesteps_list_train.insert(i,timesteps_list_train[i-1]+1)
                break
    timesteps_train=np.array(timesteps_list_train)
    return timesteps_train
    
sample_num=done_idxs[idx_num]
actions=actions[:sample_num+1]
actions_neg=actions_neg[:sample_num+1]
actions_len=actions_len[:sample_num+1]
return_step=return_step[:sample_num+1]
obss=obss[:sample_num+1]
vocab_size=5013
# actions, obss, vocab_size = re_index(actions, obss)

# # shuffle
# import random
# idx_list=list(range(idx_num+1))
# random.shuffle(idx_list)
# obss_shuffle=[]
# rtgs_shuffle=[]
# actions_shuffle=[]
# timesteps_shuffle=[]
# done_idxs_shuffle=[]
# shuffle_idx_flag=0
# for i in idx_list:
#     idx_end=done_idxs[i]
#     if i==0:
#         idx_start=0
#     else:
#         idx_start=done_idxs[i-1]
#     obss_shuffle+=(list(obss[idx_start:idx_end]))
#     rtgs_shuffle+=(list(rtgs[idx_start:idx_end]))
#     actions_shuffle+=(list(actions[idx_start:idx_end]))
#     timesteps_shuffle+=(list(timesteps[idx_start:idx_end]))
#     shuffle_idx_flag+=(idx_end-idx_start)
#     done_idxs_shuffle.append(shuffle_idx_flag)
# obss=np.array(obss_shuffle)
# rtgs=np.array(rtgs_shuffle)
# actions=np.array(actions_shuffle)
# timesteps=np.array(timesteps_shuffle)
# done_idxs=np.array(done_idxs_shuffle)







#train_dataset
sample_num_train=done_idxs[idx_num_train]
obss_train=obss[:sample_num_train]
rtgs_train=rtgs[:sample_num_train]
actions_train=actions[:sample_num_train]
actions_neg_train=actions_neg[:sample_num_train]

actions_len_train=actions_len[:sample_num_train]
return_step_train=return_step[:sample_num_train]
timesteps_train=timesteps[:sample_num_train]
done_idxs_train=done_idxs[:idx_num_train+1]
timesteps_train=timestep_paddle(timesteps_train)

# print('state',obss_train[:100])
# print('rtg',rtgs_train[:100])
# print('action',actions_train[:100])
# print('timestep',timesteps_train[:100])
# print('done_idxs',done_idxs_train[:100])

# debug
# a=actions_train[10:20].tolist()[0]
# print(a)
# print(list(actions_train[0]))
# debug
# print(type(actions_train[0]))
# print(type(rtgs[0]))
# print(torch.tensor(actions_train[10:40], dtype=torch.long))
# debug

train_dataset = StateActionReturnDataset(obss_train, args.context_length*3, actions_train,actions_neg_train, actions_len_train, return_step_train, done_idxs_train, rtgs_train, timesteps_train)

#test_dataset
sample_num_test=done_idxs[idx_num]
print('interaction number is:',sample_num_test)
obss_test=obss[sample_num_train:sample_num_test]
rtgs_test=rtgs[sample_num_train:sample_num_test]
actions_test=actions[sample_num_train:sample_num_test]
actions_neg_test=actions_neg[sample_num_train:sample_num_test]
actions_len_test=actions_len[sample_num_train:sample_num_test]
return_step_test=return_step[sample_num_train:sample_num_test]
timesteps_test=timesteps[sample_num_train:sample_num_test]
done_idxs_test=done_idxs[idx_num_train+1:idx_num+1]-sample_num_train
timesteps_test=timestep_paddle(timesteps_test)

test_dataset = StateActionReturnDataset(obss_test, args.context_length*3, actions_test,actions_neg_test, actions_len_test, return_step_test, done_idxs_test, rtgs_test, timesteps_test)

# state_vocab_size = max(obss[:sample_num_test+1]) + 1
# action_vocab_size = max(actions[:sample_num_test+1]) + 1
# vocab_size = max(state_vocab_size,action_vocab_size)

print('item number is:',vocab_size)

mconf = GPTConfig(vocab_size, train_dataset.block_size,
                  n_layer=2, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=89)
model = GPT(mconf)

mconf_simu = GPTConfig_simu(vocab_size, train_dataset.block_size,
                  n_layer=2, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=89)
model_simu = GPT_simu(mconf_simu)

# initialize a trainer instance and kick off training
epochs = args.epochs


# tconf_simu = TrainerConfig_simu(max_epochs=epochs, batch_size=args.batch_size, learning_rate=0.01,
#                       lr_decay=False, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
#                       num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
# trainer_simu = Trainer_simu(model_simu, train_dataset, test_dataset, tconf_simu)

# model_simu=trainer_simu.train()
# PATH='./simulator/my_model.pth'
# model_simu.load_state_dict(torch.load(PATH))


tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=0.01,
                      lr_decay=False, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=89)
trainer = Trainer(model, train_dataset, test_dataset, tconf)

trainer.train()

# PATH='./simulator/my_model_eva.pth'
# torch.save(model.state_dict(),PATH)






# # DT user_retain
# sample_num=414
# idx_num=11

# user_retain=pd.read_csv('./WSDM/user_retain.csv')
# obss=user_retain['obss'].values[:sample_num]
# rtgs=user_retain['rtg'].values[:sample_num]
# actions=user_retain['actions'].values[:sample_num]
# timesteps=user_retain['timesteps'].values[:sample_num]
# done_idx_file=pd.read_csv('./WSDM/done_idx.csv')
# done_idxs=done_idx_file['done_idx'].values[:idx_num]
# time_flag=0
# timesteps_list=list(timesteps)
# for i in range(len(timesteps_list)):
#   if timesteps_list[i]==0:
#     time_flag+=1
#     if time_flag==2:
#       timesteps_list.insert(i,timesteps_list[i-1]+1)
#       break
# timesteps=np.array(timesteps_list)



# def re_index(actions):
#   action_dic={}
#   action_flag=0
#   action_new=[]
#   for i in range(actions.shape[0]):
#     if str(actions[i]) in action_dic.keys():
#       action_new.append(action_dic[str(actions[i])])
#     else:
#       action_new.append(action_flag)
#       action_dic[str(actions[i])]=action_flag
#       action_flag+=1
#   return action_new

# actions = re_index(actions)
# obss = re_index(obss)
# # obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# # set up logging
# logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
# )

# train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)


# mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
#                   n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
# model = GPT(mconf)

# # initialize a trainer instance and kick off training
# epochs = args.epochs
# tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
#                       lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
#                       num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
# trainer = Trainer(model, train_dataset, None, tconf)

# trainer.train()
