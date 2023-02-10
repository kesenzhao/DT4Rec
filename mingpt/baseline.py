"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

import collections
from torch import nn
from d2l import torch as d2l
import time

from torch.nn.modules.activation import Sigmoid

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size// 3 + 1, config.block_size // 3 + 1))
                                     .view(1, 1, config.block_size// 3 + 1, config.block_size// 3 + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
  
        # reward_mask+=self.mask 
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # att = att.masked_fill(reward_mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Autodis(nn.Module):
    def __init__(self, config, bucket_number):
        super().__init__()
        self.bucket = nn.Sequential(nn.Linear(1, config.n_embd))
        self.ret_emb_score = nn.Sequential(nn.Linear(1, bucket_number, bias=False), nn.LeakyReLU())
        self.res = nn.Linear(bucket_number, bucket_number, bias=False)
        self.temp = nn.Sequential(
            nn.Linear(1, bucket_number, bias=False), 
            nn.LeakyReLU(), 
            nn.Linear(bucket_number, bucket_number, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x, layer_past=None):
        bucket_value = torch.arange(0, 700, 7).to(x.device).reshape(100,1).type(torch.float32)
        Meta_emb = self.bucket(bucket_value)
        t = self.temp(x)
        x = self.ret_emb_score(x)
        # x.shape [batch_size, timestep, bucket_value]
        # t.shape [bucket_value]
        x = x + self.res(x)
        max_value,_ = torch.max(x, dim=2, keepdim=True)
        x = torch.exp(x - max_value)
        soft_sum = torch.sum(x, dim=2).unsqueeze(2)
        x = x / soft_sum
        x = torch.einsum('nck,km->ncm', [x, Meta_emb])
        return x



class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state, logits_new):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = logits_new.unsqueeze(0).repeat(X.shape[0], 1, 1)
        # context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        state=state.contiguous()
        output, state = self.rnn(X_and_context, state)
        output_emb = output.permute(1, 0, 2)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state, output_emb

#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def bleu(pred_seq, label_seq, y_len):  #@save
    """计算BLEU"""
    retA=0
    for i in range(pred_seq.shape[0]):
        for j in range(y_len[i]):
            if pred_seq[i,j]==label_seq[i,j]:
                retA+=1
                break
    score=retA/sum(y_len)
    return score

def bleu_emb_pos(pred_seq, label_seq, y_len):  #@save
    """计算BLEU"""
    retA=0
    score_step=torch.abs(torch.cosine_similarity(pred_seq, label_seq,dim=-1))

    for i in range(pred_seq.shape[0]):
        score_batch=torch.sum(score_step[i,:y_len[i]])/y_len[i]
        retA+=score_batch
    score=retA/y_len.shape[0]
    return score


def bleu_emb(pred_seq, label_seq, y_len,return_step_one):  #@save
    """计算BLEU"""
    retA=0

    score_neg=torch.zeros([8,pred_seq.shape[0],pred_seq.shape[1]],device=y_len.device)
    for i in range(8):
        score_neg[i,:,:]=torch.abs(torch.cosine_similarity(pred_seq, label_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1)) * (15-2*i) / 8
        if i > 4:
            score_neg[i,:,:]=torch.abs(torch.cosine_similarity(pred_seq, label_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1)) * 0
    for i in range(pred_seq.shape[0]):
        score_batch=torch.sum(score_neg[:,i,:y_len[i]],dim=1)/y_len[i]
        score_batch_neg=(torch.sum(score_batch)-score_batch[int(return_step_one[i].item())])/7

        retA+=score_batch_neg
    score=retA/y_len.shape[0]
    return score

def InfoNCE(pred_seq, pos_seq, neg_seq, y_len,return_step_one):
    
    score_neg=torch.zeros([8,pred_seq.shape[0],pred_seq.shape[1]],device=y_len.device)
    for i in range(8):
        score_neg[i,:,:]=torch.cosine_similarity(pred_seq, neg_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1)
    l_neg = torch.zeros([pred_seq.shape[0],7],device=y_len.device)
    for i in range(pred_seq.shape[0]):
        score_batch=torch.sum(score_neg[:,i,:y_len[i]],dim=1)/y_len[i]
        index = list(set(range(8))-set([int(return_step_one[i].item())]))
        l_neg[i] = score_batch[index]
    pos_score_step=torch.cosine_similarity(pred_seq, pos_seq,dim=-1)
    l_pos = torch.zeros([pred_seq.shape[0],1],device=y_len.device)
    for i in range(pred_seq.shape[0]):
        pos_score_batch=torch.sum(pos_score_step[i,:y_len[i]])/y_len[i]
        l_pos[i] = pos_score_batch
    logits = torch.cat([l_pos, l_neg], dim=1)
    T = 0.07
    logits /= T
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    loss = criterion(logits, labels)
    return loss

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        self.action_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        bucket_number = 100
        self.ret_emb = Autodis(config, bucket_number)
        # my state_embedding
        self.state_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)



    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        for i in (param_dict.keys() - union_params):
            no_decay.add(str(i))
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, actions_neg, y_len, targets, rtgs,return_step, timesteps=None):
        
        action_embeddings=torch.zeros([actions.shape[0], actions.shape[1], 128])
        state_allstep=[]
        for i in range(actions.shape[1]):
            action_seq=actions[:,i,:].type(torch.long).squeeze(1)
            output, state = self.action_encoder(action_seq)
            # output=output.permute(1, 0, 2) 
            # action_embeddings[:,i,:]=output[:,-1,:]
            context=state.permute(1, 0, 2) 
            action_embeddings[:,i,:]=context[:,-1,:]
            state_allstep.append(state)
        token_embeddings = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        # position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        token_embeddings=token_embeddings.to(device)
        position_embeddings = all_global_pos_emb[:,::3,:]


        # x = self.drop(token_all + position_all)

        x = self.drop(token_embeddings + position_embeddings)
        logits = self.blocks(x)
        # logits=token_embeddings

        # if we are given some desired targets also calculate the loss
        loss_func = MaskedSoftmaxCELoss()    
        loss=[]

        for i in range(actions.shape[1]):
            logits_new=logits[:,i,:].squeeze(1)
            targets_seq=targets[:,i,:].type(torch.long).squeeze(1) 
            neg_seq=actions_neg[:,i,:].type(torch.long).squeeze(1) 
            pos_seq=actions[:,i,:].type(torch.long).squeeze(1) 
            bos = torch.tensor([5011] * targets_seq.shape[0]).reshape(-1, 1).to(device)
            dec_input = torch.cat([bos, targets_seq[:, :-1]], 1)
            logits_new_pos=logits_new[:actions.shape[0]]
            Y_hat,_,Y_emb = self.decoder(dec_input, state_allstep[i], logits_new_pos)
            y_len_step=y_len[:,i]
            loss_step1 = loss_func(Y_hat, targets_seq, y_len_step)
            loss_step1=loss_step1.mean()
            loss_step=loss_step1 

            loss.append(loss_step)

        loss_mean=sum(loss)/len(loss)
        return logits[:actions.shape[0]], loss_mean
    #@save

    def predict_seq2seq(self, states, actions, actions_len, targets, rtgs,r_step, timesteps, num_steps,
                        device, save_attention_weights=False):
        """序列到序列模型的预测"""
        # 在预测时将net设置为评估模式
        device=rtgs.device

        action_embeddings=torch.zeros([actions.shape[0], actions.shape[1], 128])
        state_allstep=[]
        for i in range(actions.shape[1]):
            action_seq=actions[:,i,:].type(torch.long).squeeze(1)
            # bos = torch.tensor([5011] * Y.shape[0]).reshape(-1, 1)
            output, state = self.action_encoder(action_seq)
            context=state.permute(1, 0, 2) 
            action_embeddings[:,i,:]=context[:,-1,:]
            state_allstep.append(state)
        token_embeddings = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        # position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        token_embeddings=token_embeddings.to(device)
        position_embeddings = all_global_pos_emb[:,::3,:]
        x = self.drop(token_embeddings + position_embeddings)
        logits = self.blocks(x)
        loss_func = MaskedSoftmaxCELoss()
        y_pred=torch.zeros_like(actions)
        for j in range(actions.shape[1]-1, actions.shape[1]):
            logits_new=logits[:,j,:].squeeze(1)
            targets_seq=targets[:,j,:].type(torch.long).squeeze(1)

            score=[]
            seq_len=actions_len[:,j]
            dec_state=state_allstep[j]
            output_seq, attention_weight_seq = [], []
            dec_X = torch.tensor([5011] * targets_seq.shape[0]).reshape(-1, 1).to(device)
            for k in range(num_steps):
                Y, dec_state, Y_emb = self.decoder(dec_X, dec_state, logits_new)
                # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
                dec_X = Y.argmax(dim=2)
                pred = dec_X.type(torch.int32)
                # 保存注意力权重（稍后讨论）
                if save_attention_weights:
                    attention_weight_seq.append(self.decoder.attention_weights)
               
                y_pred[:,j,k]=pred.squeeze(1)
        score = []
        for i in range(y_pred.shape[0]):
            score_i = bleu_seq(y_pred[i,-1,:],targets[i,-1,:])
            score.append(score_i)
        score = sum(score)/len(score)





        return score


def bleu_seq(y_pred,y):
    score_sum=0
    for i in range(y_pred.shape[0]):
        for j in range(y.shape[0]):
            if y_pred[i]==y[j]:
                score_sum+=1
                break
    score=score_sum/y_pred.shape[0]
    return score