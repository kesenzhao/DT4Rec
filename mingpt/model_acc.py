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
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
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
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        reward_mask = torch.ones_like(self.mask)
        # print(reward_mask)
        for i in range(reward_mask.shape[-1]):
            reward_mask[:,:,::3,i] = 0
            # reward_mask[:,:,i,::3] = 0
  
        reward_mask+=self.mask 
        att = att.masked_fill(reward_mask[:,:,:T,:T] == 0, float('-inf'))

        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
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
        # y = x/t - max_value
        # z = x/t
        # print(z[0,0,:])
        # print(max_value[0,0,:])
        # print(y[0,0,:])
        # print(t[1,1,:])
        x = torch.exp(x - max_value)
        # print(x[0,0,:])
        # de
        soft_sum = torch.sum(x, dim=2).unsqueeze(2)
        # print(x)
        # print(x[1,1,:])
        # print(soft_sum[1,1,:])
        x = x / soft_sum
        # print(x)
        # de
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
        # print(weights)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# def bleu(pred_seq, label_seq, k):  #@save
#     """计算BLEU"""
#     pred_tokens=[]
#     label_tokens=[]
#     for i in range(len(pred_seq)):
#         pred_tokens.append(str(pred_seq[i]))
#     for i in range(len(label_seq)):
#         label_tokens.append(str(label_seq[i]))
#     len_pred, len_label = len(pred_tokens), len(label_tokens)
#     score = math.exp(min(0, 1 - len_label / len_pred))
#     for n in range(1, k + 1):
#         num_matches, label_subs = 0, collections.defaultdict(int)
#         for i in range(len_label - n + 1):
#             label_subs[' '.join(label_tokens[i: i + n])] += 1
#         for i in range(len_pred - n + 1):
#             if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
#                 num_matches += 1
#                 label_subs[' '.join(pred_tokens[i: i + n])] -= 1
#         score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
#     return score

def bleu(pred_seq, label_seq, y_len):  #@save
    """计算BLEU"""
    retA=0
    for i in range(pred_seq.shape[0]):
        for j in range(y_len[i]):
            # print(pred_seq[i,j])
            # print(label_seq[i,j])
            if pred_seq[i,j]==label_seq[i,j]:
                retA+=1
                break
    score=retA/sum(y_len)
    return score



def bleu_emb_pos(pred_seq, label_seq, y_len):  #@save
    """计算BLEU"""
    retA=0

    # score_neg=torch.zeros([8,pred_seq.shape[0],pred_seq.shape[1]],device=y_len.device)
    # for i in range(8):
    #     score_neg[i,:,:]=torch.abs(torch.cosine_similarity(pred_seq, label_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1))
    score_step=torch.abs(torch.cosine_similarity(pred_seq, label_seq,dim=-1))

    for i in range(pred_seq.shape[0]):
        # score_step=torch.abs(torch.cosine_similarity(pred_seq[i], label_seq[i],dim=1))
        # score_step=torch.einsum('nc,nc->n', [pred_seq[i], label_seq[i]])
        score_batch=torch.sum(score_step[i,:y_len[i]])/y_len[i]
        # score_batch_neg=(torch.sum(score_batch)-score_batch[int(return_step_one[i].item())])/7

        retA+=score_batch
        # for j in range(y_len[i]):
            # print(pred_seq[i,j])
            # print(label_seq[i,j])
            # retA=retA
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

    # score_step=torch.abs(torch.cosine_similarity(pred_seq, label_seq,dim=-1))

    for i in range(pred_seq.shape[0]):
        # score_step=torch.abs(torch.cosine_similarity(pred_seq[i], label_seq[i],dim=1))
        # score_step=torch.einsum('nc,nc->n', [pred_seq[i], label_seq[i]])
        score_batch=torch.sum(score_neg[:,i,:y_len[i]],dim=1)/y_len[i]
        score_batch_neg=(torch.sum(score_batch)-score_batch[int(return_step_one[i].item())])/7

        retA+=score_batch_neg
        # for j in range(y_len[i]):
            # print(pred_seq[i,j])
            # print(label_seq[i,j])
            # retA=retA
    score=retA/y_len.shape[0]
    return score

def InfoNCE(pred_seq, pos_seq, neg_seq, y_len,return_step_one):
    
    score_neg=torch.zeros([8,pred_seq.shape[0],pred_seq.shape[1]],device=y_len.device)
    for i in range(8):
        score_neg[i,:,:]=torch.cosine_similarity(pred_seq, neg_seq[i*pred_seq.shape[0]:(i+1)*pred_seq.shape[0]],dim=-1)
    # score_step=torch.abs(torch.cosine_similarity(pred_seq, label_seq,dim=-1))
    l_neg = torch.zeros([pred_seq.shape[0],7],device=y_len.device)
    for i in range(pred_seq.shape[0]):
        # score_step=torch.abs(torch.cosine_similarity(pred_seq[i], label_seq[i],dim=1))
        # score_step=torch.einsum('nc,nc->n', [pred_seq[i], label_seq[i]])
        score_batch=torch.sum(score_neg[:,i,:y_len[i]],dim=1)/y_len[i]
        index = list(set(range(8))-set([int(return_step_one[i].item())]))
        l_neg[i] = score_batch[index]
    


        # score_batch_neg=(torch.sum(score_batch)-score_batch[int(return_step_one[i].item())])/7
    
    pos_score_step=torch.cosine_similarity(pred_seq, pos_seq,dim=-1)
    l_pos = torch.zeros([pred_seq.shape[0],1],device=y_len.device)
    for i in range(pred_seq.shape[0]):
        # score_step=torch.abs(torch.cosine_similarity(pred_seq[i], label_seq[i],dim=1))
        # score_step=torch.einsum('nc,nc->n', [pred_seq[i], label_seq[i]])
        pos_score_batch=torch.sum(pos_score_step[i,:y_len[i]])/y_len[i]
        l_pos[i] = pos_score_batch


    # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
	# # negative logits: NxK
    # l_neg = torch.einsum('nc,ck->nk', [q, v])
	
	# logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)
	
	# apply temperature
    T = 0.07
    logits /= T
	
	# labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    loss = criterion(logits, labels)
    return loss

# def InfoNCE(q,k,v):
#     l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
# 	# negative logits: NxK
#     l_neg = torch.einsum('nc,ck->nk', [q, v])
	
# 	# logits: Nx(1+K)
#     logits = torch.cat([l_pos, l_neg], dim=1)
	
# 	# apply temperature
#     logits /= self.T
	
# 	# labels: positive key indicators
#     labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#     criterion = nn.CrossEntropyLoss().cuda(args.gpu)
#     loss = criterion(output, target)



class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # self.state_encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # self.action_encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        self.action_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                        0.2)
        # self.net = d2l.EncoderDecoder(self.action_encoder, self.decoder)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        # self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
        #                          nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #                          nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #                          nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        # self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        bucket_number = 100
        self.ret_emb = Autodis(config, bucket_number)
        # bucket_number = 10
        # self.bucket = nn.Sequential(nn.Linear(1, config.n_embd))
        # self.ret_emb_score = Autodis(bucket_number)
        # self.ret_emb_score = nn.Sequential(nn.Linear(1, bucket_number, bias=False), nn.LeakyReLU())
        # self.res = nn.Linear(bucket_number, bucket_number, bias=False)


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
            # print(str(i))
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
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        # state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        # state_embeddings=torch.zeros([states.shape[0], states.shape[1], 128])
        # for i in range(states.shape[1]):
        #     states_seq=states[:,i,:].type(torch.long).squeeze(1)
        #     state_embeddings_seq = self.state_embeddings(states_seq)
        #     state_embeddings_seq = self.state_encoder(state_embeddings)
        #     state_embeddings[:,i,:]=state_embeddings_seq[:,-1,:]
        
        # action_embeddings=torch.zeros([actions.shape[0], actions.shape[1], 128])
        # for i in range(actions.shape[1]):
        #     action_seq=actions[:,i,:].type(torch.long).squeeze(1)
        #     action_embeddings_seq = self.action_embeddings(action_seq)
        #     action_embeddings_seq = self.action_encoder(state_embeddings)
        #     action_embeddings[:,i,:]=action_embeddings_seq[:,-1,:]
        device=rtgs.device
        state_embeddings=torch.zeros([states.shape[0], states.shape[1], 128])
        for i in range(states.shape[1]):
            states_seq=states[:,i,:].type(torch.long).squeeze(1)
            # bos = torch.tensor([5011] * Y.shape[0]).reshape(-1, 1)
            output, state = self.state_encoder(states_seq)
            # output=output.permute(1, 0, 2) 
            # state_embeddings[:,i,:]=output[:,-1,:]
            context=state.permute(1, 0, 2) 
            state_embeddings[:,i,:]=context[:,-1,:]
        
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

        
        # #my state embedding
        # state_embeddings = self.state_embeddings(states.type(torch.long).squeeze(-1))
        rtg_neg=torch.zeros([rtgs.shape[0]*8,rtgs.shape[1],rtgs.shape[2]],device=device)
        for i in range(8):
            # r_batch=list(range(i*30,0,-i))
            # r_batch=torch.tensor(r_batch, dtype=torch.float32)
            for j in range(rtgs.shape[0]):
                rtg_neg[i*rtgs.shape[0]+j,:-1,0]=rtgs[j,1:,0]+i
                rtg_neg[i*rtgs.shape[0]+j,-1,0]=rtgs[j,-1,0]
                

        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
          
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        token_embeddings=token_embeddings.to(device)


        token_neg_embeddings=torch.repeat_interleave(token_embeddings,8,0)
        rtg_neg_embeddings = self.ret_emb(rtg_neg.type(torch.float32))
        token_neg_embeddings[:,::3,:] = rtg_neg_embeddings
        token_all = torch.cat((token_embeddings, token_neg_embeddings), 0)
        position_all = torch.repeat_interleave(position_embeddings,9,0)


        x = self.drop(token_all + position_all)

        # x = self.drop(token_embeddings + position_embeddings)
        logits = self.blocks(x)
        # logits=token_embeddings




        # x = self.ln_f(x)
        # logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

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
            
            
            # dec_input = torch.repeat_interleave(dec_input,9,0)
            # state_neg = torch.repeat_interleave(state_allstep[i],9,1)
            # Y_hat_all,_,Y_emb_all = self.decoder(dec_input, state_neg, logits_new)
            # Y_hat = Y_hat_all[:actions.shape[0]]
            # Y_emb = Y_emb_all[:actions.shape[0]]
            # neg_seq_emb = Y_emb_all[actions.shape[0]:]

            #
            logits_new_pos=logits_new[:actions.shape[0]]
            Y_hat,_,Y_emb = self.decoder(dec_input, state_allstep[i], logits_new_pos)

            dec_input_neg = torch.repeat_interleave(dec_input,8,0)
            state_neg = torch.repeat_interleave(state_allstep[i],8,1)
            logits_new_neg=logits_new[actions.shape[0]:]
            Y_hat_all,_,Y_emb_all = self.decoder(dec_input_neg, state_neg, logits_new_neg)
            # neg_seq_emb = Y_emb_all[actions.shape[0]:]
            neg_seq_emb = Y_emb_all






            y_len_step=y_len[:,i]
            # dec_X = Y_hat.argmax(dim=2)
            # print('dec_X',dec_X[0])
            # print('target',targets_seq[0])
            # y_=torch.ones_like(y_len)
          
            loss_step1 = loss_func(Y_hat, targets_seq, y_len_step)
            loss_step1=loss_step1.mean()
            # y_pred=Y_hat.argmax(dim=2)
            
            pos_seq_emb=self.action_encoder.embedding(pos_seq)
            # neg_seq_emb=self.action_encoder.embedding(neg_seq)
            pos_score = bleu_emb_pos(Y_emb, pos_seq_emb, y_len_step)
            return_step_one=return_step[:,i]

            neg_score = bleu_emb(Y_emb, neg_seq_emb, y_len_step, return_step_one)
            # loss_contrast = InfoNCE(Y_emb, pos_seq_emb, neg_seq_emb, y_len_step,return_step_one)

            # loss_step2 = -loss_func(Y_hat, neg_seq, y_len_step)
            # loss_step2=loss_step1 + neg_score
            loss_step=loss_step1 

            # loss_step=loss_step1 + neg_score
            # print('1',loss_step1.shape)
            # de
            # print('pos',pos_score)
            # print('neg',neg_score)
            # loss_step = loss_step1 + 0.1 * loss_contrast
            # loss_step=loss_step1- 0.1 * (pos_score - neg_score)
            # loss_step=-pos_score + neg_score
            loss.append(loss_step)
            # print('loss',loss)

        loss_mean=sum(loss)/len(loss)
        # print(loss_mean) 
        # de

        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits[:actions.shape[0]], loss_mean
    #@save

    def predict_seq2seq(self, states, actions, actions_len, targets, rtgs,r_step, timesteps, num_steps,
                        device, save_attention_weights=False):
        """序列到序列模型的预测"""
        # 在预测时将net设置为评估模式
        device=rtgs.device
        state_embeddings=torch.zeros([states.shape[0], states.shape[1], 128])
        for i in range(states.shape[1]):
            states_seq=states[:,i,:].type(torch.long).squeeze(1)
            # bos = torch.tensor([5011] * Y.shape[0]).reshape(-1, 1)
            output, state = self.state_encoder(states_seq)
            # output=output.permute(1, 0, 2) 
            # state_embeddings[:,i,:]=output[:,-1,:]
            context=state.permute(1, 0, 2) 
            state_embeddings[:,i,:]=context[:,-1,:]

        action_embeddings=torch.zeros([actions.shape[0], actions.shape[1], 128])
        state_allstep=[]
        for i in range(actions.shape[1]):
            action_seq=actions[:,i,:].type(torch.long).squeeze(1)
            # bos = torch.tensor([5011] * Y.shape[0]).reshape(-1, 1)
            output, state = self.action_encoder(action_seq)
            context=state.permute(1, 0, 2) 
            action_embeddings[:,i,:]=context[:,-1,:]
            state_allstep.append(state)
            

        
        # #my state embedding
        # state_embeddings = self.state_embeddings(states.type(torch.long).squeeze(-1))

        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        token_embeddings=token_embeddings.to(device)
        x = self.drop(token_embeddings + position_embeddings)
        logits = self.blocks(x)
        # logits = token_embeddings



        # x = self.ln_f(x)
        # logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss_func = MaskedSoftmaxCELoss()
        y_pred=torch.zeros_like(actions)
        # time1=time.time()
        for j in range(actions.shape[1]):
            logits_new=logits[:,j,:].squeeze(1)
            targets_seq=targets[:,j,:].type(torch.long).squeeze(1)
            # bos = torch.tensor([5011] * targets_seq.shape[0]).reshape(-1, 1)
            # dec_input = torch.cat([bos, targets_seq[:, :-1]], 1)

            score=[]
            # dec_state=state[:,i,:]
            seq_len=actions_len[:,j]

            # # step
            # for i in range(rtgs.shape[0]):
            #     dec_state=state_allstep[j][:,i,:].unsqueeze(1)
            #     output_seq, attention_weight_seq = [], []
            #     dec_X = torch.unsqueeze(torch.tensor([5011], dtype=torch.long, device=device), dim=0)
                
            #     for _ in range(num_steps):
            #         Y, dec_state = self.decoder(dec_X, dec_state, logits_new[i])
            #         # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
            #         dec_X = Y.argmax(dim=2)
            #         pred = dec_X.squeeze(dim=0).type(torch.int32).item()
            #         # 保存注意力权重（稍后讨论）
            #         if save_attention_weights:
            #             attention_weight_seq.append(self.decoder.attention_weights)
            #         # 一旦序列结束词元被预测，输出序列的生成就完成了
            #         output_seq.append(pred)
            #         if pred == 5012:
            #             break
            #     y_pred[i,j,:]=torch.tensor(self._padding_sequence(output_seq, actions.shape[2]), dtype=torch.long, device=device)

                    # output_seq.append(pred)
                # tar=targets_seq[i][:seq_len[i]]
                # score_batch = bleu(output_seq,tar,1)
                # score.append(score_batch)
            # score=sum(score)/len(score)
            # return score
        # time2=time.time()
        # print(time2-time1)
        # debug

            #batch
            dec_state=state_allstep[j]
            output_seq, attention_weight_seq = [], []
            dec_X = torch.tensor([5011] * targets_seq.shape[0]).reshape(-1, 1).to(device)
            # print(dec_X)
            for k in range(num_steps):
                Y, dec_state, Y_emb = self.decoder(dec_X, dec_state, logits_new)
                # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
                dec_X = Y.argmax(dim=2)
                # print(dec_X)
                # debug
                pred = dec_X.type(torch.int32)
                # 保存注意力权重（稍后讨论）
                if save_attention_weights:
                    attention_weight_seq.append(self.decoder.attention_weights)
                # 一旦序列结束词元被预测，输出序列的生成就完成了
                # output_seq.append(pred)
                
                # if pred == 5012:
                #     break
               
                y_pred[:,j,k]=pred.squeeze(1)
        # # time2=time.time()
        # # print(time2-time1)
        # y_shape=y_pred.shape
        # sim_dict={'0':[0,0],'1':[0,0],'2':[0,0],'3':[0,0],'4':[0,0],'5':[0,0],'6':[0,0],'7':[0,0]}
        # for i in range(y_shape[0]):
        #     for j in range(y_shape[1]):
        #         for k in range(y_shape[2]):
        #             # print(y_pred[i,j,k].item())
        #             if y_pred[i,j,k].item()==5012:
        #                 y_ten=y_pred[i,j,:k+1]

        #                 y_ten=y_ten.cpu().numpy().tolist()

        #                 y_seq=torch.tensor(self._padding_sequence(y_ten, actions.shape[2]), dtype=torch.long, device=device)                                      
        #                 y_pred[i,j,:]=y_seq
        #                 break
             
        #         similar=bleu_seq(y_pred[i,j,:],targets[i,j,:])
        #         sim_dict[str(int(r_step[i,j].item()))][0]+=similar
        #         sim_dict[str(int(r_step[i,j].item()))][1]+=1
        # # for i in range(7):
        # #     sim_dict[str(i)][0]/=sim_dict[str(i)][1]

        score = []
        for i in range(y_pred.shape[0]):
            score_i = bleu_seq(y_pred[i,-1,:],targets[i,-1,:])
            score.append(score_i)
        score = sum(score)/len(score)
        
        rouge_score = []
        for i in range(y_pred.shape[0]):
            rouge_score_i = ROUGE(y_pred[i,-1,:],targets[i,-1,:], actions_len[i,-1])
            rouge_score.append(rouge_score_i)
        rouge_score = sum(rouge_score)/len(rouge_score)
        return score, rouge_score





        # return y_pred, sim_dict
def ROUGE(y_pred,y,y_len):
    score_sum=0
    for i in range(y_len):
        for j in range(y_pred.shape[0]):
            if y[i]==y_pred[j]:
                score_sum+=1
                break
    score=score_sum/y_len.item()
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

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = 2
        self.n_heads = config.n_head
        self.hidden_size = config.n_embd  # same as embedding_size
        self.inner_size = 2*config.n_embd  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = 0.2
        self.attn_dropout_prob = config.attn_pdrop
        self.hidden_act = config['gelu']
        self.layer_norm_eps = 1e-12

        self.initializer_range = 0.02
        # define layers and loss
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output 


