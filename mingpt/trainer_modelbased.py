"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image

from gpu_mem_track import MemTracker
import inspect
import time

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

# class Trainer:

#     def __init__(self, model, train_dataset, test_dataset, config):
#         self.model = model
#         self.train_dataset = train_dataset
#         self.test_dataset = test_dataset
#         self.config = config

#         # take over whatever gpus are on the system
#         self.device = 'cpu'
#         if torch.cuda.is_available():
#             self.device = torch.cuda.current_device()
#             self.model = torch.nn.DataParallel(self.model).to(self.device)

#     def save_checkpoint(self):
#         # DataParallel wrappers keep raw model object in .module attribute
#         raw_model = self.model.module if hasattr(self.model, "module") else self.model
#         logger.info("saving %s", self.config.ckpt_path)
#         # torch.save(raw_model.state_dict(), self.config.ckpt_path)

#     def train(self):
#         model, config = self.model, self.config
#         raw_model = model.module if hasattr(self.model, "module") else model
#         optimizer = raw_model.configure_optimizers(config)
        
#         # def evaluator(rankedlist, testlist,k):
#         #     data_shape=rankedlist.shape  # (batch_size,block_size,voc_size)
#         #     Hits_i = 0
#         #     Len_R = 0
#         #     Len_T = data_shape[0] * data_shape[1]
#         #     MRR_i = 0
#         #     HR_i = 0
#         #     NDCG_i = 0
            
#         #     for i in range(data_shape[0]):
#         #         for j in range(data_shape[1]):
#         #             rec_list=rankedlist[i,j,:]
#         #             values,topk_index=rec_list.topk(k, largest=True, sorted=True)
#         #             topk_index=list(topk_index)
#         #             for p in range(k):
#         #                 if testlist[i,j,0]==topk_index[p]:
#         #                     Hits_i+=1
#         #                     HR_i+=1
#         #                     # 注意j的取值从0开始
#         #                     MRR_i+=1/(p+1)   
#         #                     NDCG_i+=1/(math.log2(1+p+1))
#         #                     break
#         #     HR_i/=Len_T
#         #     MRR_i/=Len_T
#         #     NDCG_i/=Len_T
#         #     return MRR_i, HR_i, NDCG_i

#         def evaluator(rankedlist, testlist,k):
#             data_shape=rankedlist.shape  # (batch_size,block_size,voc_size)
#             Hits_i = 0
#             Len_R = 0
#             Len_T = data_shape[0] 
#             MRR_i = 0
#             HR_i = 0
#             NDCG_i = 0
            
#             for i in range(data_shape[0]):
#                 rec_list=rankedlist[i,-1,:]
#                 values,topk_index=rec_list.topk(k, largest=True, sorted=True)
#                 topk_index=list(topk_index)
#                 for p in range(k):
#                     if testlist[i,-1,0]==topk_index[p]:
#                         Hits_i+=1
#                         HR_i+=1
#                         # 注意j的取值从0开始
#                         MRR_i+=1/(p+1)   
#                         NDCG_i+=1/(math.log2(1+p+1))
#                         break
#             HR_i/=Len_T
#             MRR_i/=Len_T
#             NDCG_i/=Len_T
#             return MRR_i, HR_i, NDCG_i


#         def run_epoch(split, epoch_num=0):
#             is_train = split == 'train'
#             model.train(is_train)
#             data = self.train_dataset if is_train else self.test_dataset
#             loader = DataLoader(data, shuffle=True, pin_memory=True,
#                                 batch_size=config.batch_size,
#                                 num_workers=config.num_workers)

#             losses = []
#             pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

#             for it, (x, y, r, t) in pbar:

#                 # place data on the correct device
#                 x = x.to(self.device)
#                 y = y.to(self.device)
#                 r = r.to(self.device)
#                 t = t.to(self.device)
#                 # forward the model
#                 MRR=[]
#                 HR=[]
#                 NDCG=[]
#                 with torch.set_grad_enabled(is_train):
#                     # logits, loss = model(x, y, r)
#                     logits, loss = model(x, y, y, r, t)
#                     topk=20
#                     MRR_batch,HR_batch,NDCG_batch=evaluator(logits,y,topk)   
#                     MRR.append(MRR_batch)
#                     HR.append(HR_batch)
#                     NDCG.append(NDCG_batch)

#                     loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
#                     losses.append(loss.item())


#                 if is_train:

#                     # backprop and update the parameters
#                     model.zero_grad()
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
#                     optimizer.step()

#                     # decay the learning rate based on our progress
#                     if config.lr_decay:
#                         self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
#                         if self.tokens < config.warmup_tokens:
#                             # linear warmup
#                             lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
#                         else:
#                             # cosine learning rate decay
#                             progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
#                             lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
#                         lr = config.learning_rate * lr_mult
#                         for param_group in optimizer.param_groups:
#                             param_group['lr'] = lr
#                     else:
#                         lr = config.learning_rate

#                     # report progress
#                     pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f} MRR{topk} {MRR_batch:.5f} HR{topk} {HR_batch:.5f} NDCG{topk} {NDCG_batch:.5f}. lr {lr:e}")
#                     # pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

#             if not is_train:
#                 test_loss = float(np.mean(losses))
#                 MRR_mean=np.mean(MRR)
#                 HR_mean=np.mean(HR)
#                 NDCG_mean=np.mean(NDCG)
#                 print(f"epoch {epoch+1}: MRR{topk} {MRR_mean:.5f} HR{topk} {HR_mean:.5f} NDCG{topk} {NDCG_mean:.5f}.")

#                 logger.info(f"epoch {epoch+1}: MRR{topk} {MRR_mean:.5f} HR{topk} {HR_mean:.5f} NDCG{topk} {NDCG_mean:.5f}.")
#                 logger.info("test loss: %f", test_loss)
#                 return test_loss
        
#         # Rec accuracy eval
#         best_loss = float('inf')
        
#         # best_return = -float('inf')

#         self.tokens = 0 # counter used for learning rate decay

#         for epoch in range(config.max_epochs):

#             run_epoch('train', epoch_num=epoch)
#             if self.test_dataset is not None:
#                 test_loss = run_epoch('test')

#             # supports early stopping based on the test loss, or just save always if no test set is provided
#             good_model = self.test_dataset is None or test_loss < best_loss
#             if self.config.ckpt_path is not None and good_model:
#                 best_loss = test_loss
#                 self.save_checkpoint()
            

            
            
            
#             # # -- pass in target returns
#             # if self.config.model_type == 'naive':
#             #     eval_return = self.get_returns(0)
#             # elif self.config.model_type == 'reward_conditioned':
#             #     if self.config.game == 'Breakout':
#             #         eval_return = self.get_returns(40)
#             #     elif self.config.game == 'Seaquest':
#             #         eval_return = self.get_returns(1150)
#             #     elif self.config.game == 'Qbert':
#             #         eval_return = self.get_returns(14000)
#             #     elif self.config.game == 'Pong':
#             #         eval_return = self.get_returns(20)
#             #     else:
#             #         raise NotImplementedError()
#             # else:
#             #     raise NotImplementedError()

#网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Trainer:

    def __init__(self, model, model_simu, train_dataset, test_dataset, config):
        self.model = model
        self.model_simu = model_simu
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)
            self.model_simu = self.model_simu.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        model_simu=self.model_simu
        # def evaluator(rankedlist, testlist,k):
        #     data_shape=rankedlist.shape  # (batch_size,block_size,voc_size)
        #     Hits_i = 0
        #     Len_R = 0
        #     Len_T = data_shape[0] * data_shape[1]
        #     MRR_i = 0
        #     HR_i = 0
        #     NDCG_i = 0
            
        #     for i in range(data_shape[0]):
        #         for j in range(data_shape[1]):
        #             rec_list=rankedlist[i,j,:]
        #             values,topk_index=rec_list.topk(k, largest=True, sorted=True)
        #             topk_index=list(topk_index)
        #             for p in range(k):
        #                 if testlist[i,j,0]==topk_index[p]:
        #                     Hits_i+=1
        #                     HR_i+=1
        #                     # 注意j的取值从0开始
        #                     MRR_i+=1/(p+1)   
        #                     NDCG_i+=1/(math.log2(1+p+1))
        #                     break
        #     HR_i/=Len_T
        #     MRR_i/=Len_T
        #     NDCG_i/=Len_T
        #     return MRR_i, HR_i, NDCG_i

        def evaluator(rankedlist, testlist,k):
            data_shape=rankedlist.shape  # (batch_size,block_size,voc_size)
            Hits_i = 0
            Len_R = 0
            Len_T = data_shape[0] 
            MRR_i = 0
            HR_i = 0
            NDCG_i = 0
            
            for i in range(data_shape[0]):
                rec_list=rankedlist[i,-1,:]
                values,topk_index=rec_list.topk(k, largest=True, sorted=True)
                topk_index=list(topk_index)
                for p in range(k):
                    if testlist[i,-1,0]==topk_index[p]:
                        Hits_i+=1
                        HR_i+=1
                        # 注意j的取值从0开始
                        MRR_i+=1/(p+1)   
                        NDCG_i+=1/(math.log2(1+p+1))
                        break
            HR_i/=Len_T
            MRR_i/=Len_T
            NDCG_i/=Len_T
            return MRR_i, HR_i, NDCG_i


        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            model_simu.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            scores = []
            return_total=[]
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # frame=inspect.currentframe()
            # gpu_tracker=MemTracker(frame)
            gpu_tracker=MemTracker()
            for it, (x, y, y_neg, y_len, r_step, r, t) in pbar:
                # if is_train:
                #     for i in range(r.shape[0]):
                #         r[i,:,:]=r[i,:,:]-r[i,-1,:]
                # else:
                # r = r_step.unsqueeze(2)
                if not is_train: 
                    
                    r_batch=list(range(210,0,-7))
                    r_batch=torch.tensor(r_batch, dtype=torch.float32)
                    for i in range(r.shape[0]):
                        r[i,:,0]=r_batch
                    # r=torch.ones_like(r)*7
                    # r_simu = r_step.unsqueeze(2)
                    # r_simu = r_step.unsqueeze(2)
                    r_simu = torch.ones_like(r)*7
                    r_simu = r_simu.to(self.device)

                    

                # place data on the correct device
                
                
                # # random test
                # m=y.shape
                # y_pred=y.reshape(-1)
                # import random
                # # print(y_pred.shape[0])
                # idx_list=list(range(y_pred.shape[0]))
                # random.shuffle(idx_list)
                # y_pred=y_pred[idx_list]
                # y_pred=y_pred.reshape(m[0],30,-1).to(self.device)

                # print(y_pred.shape)
                
                x = x.to(self.device)
                y = y.to(self.device)
                y_neg = y_neg.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                y_len = y_len.to(self.device)
                r_step = r_step.to(self.device)
                
                # forward the model
                MRR=[]
                HR=[]
                NDCG=[]
                if is_train:
                    with torch.set_grad_enabled(is_train):
                        # logits, loss = model(x, y, r)
                        logits, loss = model(x, y,y_neg, y_len, y, r,r_step, t)
                        # topk=20
                        # MRR_batch,HR_batch,NDCG_batch=evaluator(logits,y,topk)   
                        # MRR.append(MRR_batch)
                        # HR.append(HR_batch)
                        # NDCG.append(NDCG_batch)

                        # loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if not is_train:
                    with torch.set_grad_enabled(is_train):
                        # score_values,y_pred=torch.topk(logits, 1, dim=2, largest=True, sorted=True)
                        
                        y_pred,return_batch=model.predict_seq2seq(x, y, y_len, y, r,r_step, t, 20, self.device)


                        
                        # para_num=get_parameter_number(model)
                        # print('para num',para_num)
                        # gpu_tracker.track()
                        # print(r_simu)

                        return_batch,loss=model_simu(x,y_pred,r_step,r_simu,t)
                        


                        # print(return_batch)
                        # simu_para_num=get_parameter_number(model)
                        # torch.cuda.empty_cache()
                        # gpu_tracker.track()
                        # print('simu para num',simu_para_num)
                        # print(return_batch)
                        # print(r)
                        # debug

                        return_batch_sum=torch.sum(return_batch, 1)
                        return_batch_mean=torch.mean(return_batch_sum)
                        return_total.append(return_batch_mean)
                        # return_total.append(return_batch)

                # if not is_train:
                #     score=model.predict_seq2seq(x, y, y_len, y, r, t, 20, self.device)
                #     scores.append(score)



                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    # pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f} MRR{topk} {MRR_batch:.5f} HR{topk} {HR_batch:.5f} NDCG{topk} {NDCG_batch:.5f}. lr {lr:e}")
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                
                
            if not is_train:
                # test_loss = float(np.mean(losses))
                # MRR_mean=np.mean(MRR)
                # HR_mean=np.mean(HR)
                # NDCG_mean=np.mean(NDCG)
                # print(f"epoch {epoch+1}: MRR{topk} {MRR_mean:.5f} HR{topk} {HR_mean:.5f} NDCG{topk} {NDCG_mean:.5f}.")

                # logger.info(f"epoch {epoch+1}: MRR{topk} {MRR_mean:.5f} HR{topk} {HR_mean:.5f} NDCG{topk} {NDCG_mean:.5f}.")
                
                # return_epochs_mean=sum(return_total)/len(return_total)
                # print('return_mean is:',return_epochs_mean)
                # return return_epochs_mean
                # return_epochs_mean=return_total[0]
                # for i in range(1,len(return_total)):
                #     for j in range(8):
                #         # print('i',return_total[i])
                #         return_epochs_mean[str(j)][0]+=return_total[i][str(j)][0]
                #         return_epochs_mean[str(j)][1]+=return_total[i][str(j)][1]
                # # print(return_epochs_mean)
                # return_score=0
                # for j in range(8):
                #     return_score=return_score+(j-4)*return_epochs_mean[str(j)][0]
                #     return_epochs_mean[str(j)][0]/=return_epochs_mean[str(j)][1]
                    
                        
                return_epochs_mean=sum(return_total)/len(return_total)
                print('return_mean is:',return_epochs_mean)
                # print('return_score is:',return_score)
                return return_epochs_mean



                # logger.info("test loss: %f", test_loss)
                # logger.info(f"epoch {epoch+1}: return_mean {return_epochs_mean:.3f}.")
                # model.predict_seq2seq(x, y, y, r, t, 20, self.device)
                # scores=sum(scores)/len(scores)
                # print('bleu score is:',scores)
                # return test_loss
            
        
        # Rec accuracy eval
        best_loss = float('inf')
        
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            if self.test_dataset is not None:
                time1=time.time()
                test_loss = run_epoch('test')
                time2=time.time()
                print(time2-time1)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss > best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()
            

            
            
            
            # -- pass in target returns
            # if self.config.model_type == 'naive':
            #     eval_return = self.get_returns(0)
            # elif self.config.model_type == 'reward_conditioned':
            #     if self.config.game == 'Breakout':
            #         eval_return = self.get_returns(70)
            #     elif self.config.game == 'Seaquest':
            #         eval_return = self.get_returns(1150)
            #     elif self.config.game == 'Qbert':
            #         eval_return = self.get_returns(14000)
            #     elif self.config.game == 'Pong':
            #         eval_return = self.get_returns(20)
            #     else:
            #         raise NotImplementedError()
            # else:
            #     raise NotImplementedError()

    
    def get_returns(self, ret):
        self.model.train(False)
        args=Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4


#     def get_returns(self, ret):
#         self.model.train(False)
#         args=Args(self.config.game.lower(), self.config.seed)
#         env = Env(args)
#         env.eval()

#         T_rewards, T_Qs = [], []
#         done = True
#         for i in range(10):
#             state = env.reset()
#             state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
#             rtgs = [ret]
#             # first state is from env, first rtg is target return, and first timestep is 0
#             sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
#                 rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
#                 timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

#             j = 0
#             all_states = state
#             actions = []
#             while True:
#                 if done:
#                     state, reward_sum, done = env.reset(), 0, False
#                 action = sampled_action.cpu().numpy()[0,-1]
#                 actions += [sampled_action]
#                 state, reward, done = env.step(action)
#                 reward_sum += reward
#                 j += 1

#                 if done:
#                     T_rewards.append(reward_sum)
#                     break

#                 state = state.unsqueeze(0).unsqueeze(0).to(self.device)

#                 all_states = torch.cat([all_states, state], dim=0)

#                 rtgs += [rtgs[-1] - reward]
#                 # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
#                 # timestep is just current timestep
#                 sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
#                     actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
#                     rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
#                     timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
#         env.close()
#         eval_return = sum(T_rewards)/10.
#         print("target return: %d, eval return: %d" % (ret, eval_return))
#         self.model.train(True)
#         return eval_return


# class Env():
#     def __init__(self, args):
#         self.device = args.device
#         self.ale = atari_py.ALEInterface()
#         self.ale.setInt('random_seed', args.seed)
#         self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
#         self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
#         self.ale.setInt('frame_skip', 0)
#         self.ale.setBool('color_averaging', False)
#         self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
#         actions = self.ale.getMinimalActionSet()
#         self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
#         self.lives = 0  # Life counter (used in DeepMind training)
#         self.life_termination = False  # Used to check if resetting only from loss of life
#         self.window = args.history_length  # Number of frames to concatenate
#         self.state_buffer = deque([], maxlen=args.history_length)
#         self.training = True  # Consistent with model training mode

#     def _get_state(self):
#         state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
#         return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

#     def _reset_buffer(self):
#         for _ in range(self.window):
#             self.state_buffer.append(torch.zeros(84, 84, device=self.device))

#     def reset(self):
#         if self.life_termination:
#             self.life_termination = False  # Reset flag
#             self.ale.act(0)  # Use a no-op after loss of life
#         else:
#             # Reset internals
#             self._reset_buffer()
#             self.ale.reset_game()
#             # Perform up to 30 random no-ops before starting
#             for _ in range(random.randrange(30)):
#                 self.ale.act(0)  # Assumes raw action 0 is always no-op
#                 if self.ale.game_over():
#                     self.ale.reset_game()
#         # Process and return "initial" state
#         observation = self._get_state()
#         self.state_buffer.append(observation)
#         self.lives = self.ale.lives()
#         return torch.stack(list(self.state_buffer), 0)

#     def step(self, action):
#         # Repeat action 4 times, max pool over last 2 frames
#         frame_buffer = torch.zeros(2, 84, 84, device=self.device)
#         reward, done = 0, False
#         for t in range(4):
#             reward += self.ale.act(self.actions.get(action))
#             if t == 2:
#                 frame_buffer[0] = self._get_state()
#             elif t == 3:
#                 frame_buffer[1] = self._get_state()
#             done = self.ale.game_over()
#             if done:
#                 break
#         observation = frame_buffer.max(0)[0]
#         self.state_buffer.append(observation)
#         # Detect loss of life as terminal in training mode
#         if self.training:
#             lives = self.ale.lives()
#             if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
#                 self.life_termination = not done  # Only set flag when not truly done
#                 done = True
#             self.lives = lives
#         # Return state, reward, done
#         return torch.stack(list(self.state_buffer), 0), reward, done

#     # Uses loss of life as terminal signal
#     def train(self):
#         self.training = True

#     # Uses standard terminal signal
#     def eval(self):
#         self.training = False

#     def action_space(self):
#         return len(self.actions)

#     def render(self):
#         cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
#         cv2.waitKey(1)

#     def close(self):
#         cv2.destroyAllWindows()

# class Args:
#     def __init__(self, game, seed):
#         self.device = torch.device('cuda')
#         self.seed = seed
#         self.max_episode_length = 108e3
#         self.game = game
#         self.history_length = 4
