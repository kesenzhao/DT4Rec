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
from torch import nn

import collections
from torch import nn
from d2l import torch as d2l
from gpu_mem_track import MemTracker

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

    # def forward(self, x, paddle_mask):
    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # print('x',x.shape)
        # print('mask',self.mask.shape)
        # print('att',att.shape)
        # print(self.mask)
        # debug
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # att = att.masked_fill(paddle_mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

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

    # def forward(self, x, paddle_mask):
    #     x = x + self.attn(self.ln1(x),paddle_mask)
    #     x = x + self.mlp(self.ln2(x))
    #     return x
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
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state

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

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    retA=[]
    for i in pred_seq:
        if i in label_seq:
            retA.append(i)
    score=len(retA)/len(pred_seq)
    return score

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
        # self.blocks=nn.Sequential(Block(config),Block(config),Block(config),Block(config))
        # self.blocks = Block(config)
        # self.blocks2 = Block(config)
        # self.blocks3 = Block(config)
        # self.blocks4 = Block(config)

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 1, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        # self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
        #                          nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #                          nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #                          nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
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
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # gpu_tracker.track()
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        # state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        #my state embedding
        # state_embeddings = self.state_embeddings(states.type(torch.long).squeeze(-1))
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
        for i in range(actions.shape[1]):
            action_seq=actions[:,i,:].type(torch.long).squeeze(1)
            output, state = self.action_encoder(action_seq)
            # output=output.permute(1, 0, 2) 
            # action_embeddings[:,i,:]=output[:,-1,:]
            context=state.permute(1, 0, 2) 
            action_embeddings[:,i,:]=context[:,-1,:]
        
        
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = state_embeddings
            token_embeddings[:,1::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
            token_embeddings[:,2::3,:] = rtg_embeddings
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:] # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
           
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()
        # gpu_tracker.track()
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        token_embeddings = token_embeddings.to(device)
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
    
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
        loss = None
        # logits=logits.long()
        # targets=targets.long()

        if targets is not None:
            # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss_func=nn.MSELoss(reduction='mean')

            # loss_mean=(logits.reshape(-1)-targets.reshape(-1))**2
            
            # print('max',logits.reshape(-1).max())
            # print(loss_mean[:30])
            # ind=torch.argmax(loss_mean)
            # print(torch.max(loss_mean))
            # print(loss_mean[ind])
            # print(ind)
            # print('ind',logits.reshape(-1)[ind])
            # print(targets.reshape(-1)[ind])
            # loss_mean=torch.mean(loss_mean)
            # print(loss_mean)
            # debug
            # print(logits[0])
            loss = loss_func(logits.reshape(-1), targets.reshape(-1))
        # gpu_tracker.track()


        # del targets
        # del rtgs
        # del state_embeddings
        # del states_seq
        # del states
        # del output
        # del context
        # del token_embeddings
        # del position_embeddings
        # del all_global_pos_emb
        # del action_embeddings
        # del rtg_embeddings





        # torch.cuda.empty_cache()
        return logits, loss



    # def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
    #     # gpu_tracker.track()
    #     # states: (batch, block_size, 4*84*84)
    #     # actions: (batch, block_size, 1)
    #     # targets: (batch, block_size, 1)
    #     # rtgs: (batch, block_size, 1)
    #     # timesteps: (batch, 1, 1)

    #     # state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
    #     # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
    #     #my state embedding
    #     # state_embeddings = self.state_embeddings(states.type(torch.long).squeeze(-1))
    #     device=rtgs.device
    #     state_embeddings=torch.zeros([states.shape[0], states.shape[1], 128])
    #     for i in range(states.shape[1]):
    #         states_seq=states[:,i,:].type(torch.long).squeeze(1)
    #         # bos = torch.tensor([5011] * Y.shape[0]).reshape(-1, 1)
    #         output, state = self.state_encoder(states_seq)
    #         # output=output.permute(1, 0, 2) 
    #         # state_embeddings[:,i,:]=output[:,-1,:]
    #         context=state.permute(1, 0, 2) 
    #         state_embeddings[:,i,:]=context[:,-1,:]
        
    #     action_embeddings=torch.zeros([actions.shape[0], actions.shape[1], 128])

    #     token_embeddings = torch.zeros((states.shape[0], states.shape[1]*22 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
    #     token_embeddings[:,20::22,:] = state_embeddings
    #     # token_embeddings[:,1::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
    #     rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
    #     token_embeddings[:,21::22,:] = rtg_embeddings
    #     paddle_mask_flag=0
    #     for i in range(actions.shape[1]):
    #         action_seq=actions[:,i,:].type(torch.long).squeeze(1)

            
    #         paddle=torch.cat((action_seq,torch.ones((action_seq.shape[0],1),device=device),torch.ones((action_seq.shape[0],1),device=device)),1)
    #         pad=torch.zeros_like(paddle)
    #         paddle_mask_step=(paddle==pad)
    #         if paddle_mask_flag==0:
    #             paddle_mask=paddle_mask_step
    #             paddle_mask_flag+=1
    #         else:
    #             paddle_mask=torch.cat((paddle_mask,paddle_mask_step),1)


    #         # output, state = self.action_encoder(action_seq)
    #         # # output=output.permute(1, 0, 2) 
    #         # # action_embeddings[:,i,:]=output[:,-1,:]
    #         # context=state.permute(1, 0, 2) 
    #         # action_embeddings[:,i,:]=context[:,-1,:]
    #         action_embeddings = self.action_embeddings(action_seq.type(torch.long).squeeze(-1))
    #         token_embeddings[:,(0+22*i):(20+22*i),:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]

    #     token_embeddings=token_embeddings[:,:22,:]
    #     # print(paddle_mask.shape)
    #     paddle_mask=paddle_mask[:,:22].unsqueeze(1).expand(token_embeddings.shape[0],22,22).unsqueeze(1)

        
        
    #     targets=targets[:,0]
        
    #     # if actions is not None and self.model_type == 'reward_conditioned': 
    #     #     rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            
    #     #     # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

    #     #     token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
    #     #     token_embeddings[:,::3,:] = state_embeddings
    #     #     token_embeddings[:,1::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
    #     #     token_embeddings[:,2::3,:] = rtg_embeddings
    #     # elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
    #     #     rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
    #     #     # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

    #     #     token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
    #     #     token_embeddings[:,::2,:] = state_embeddings # really just [:,0,:]
    #     #     token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:] # really just [:,1,:]
    #     # elif actions is not None and self.model_type == 'naive':
    #     #     # action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
           
    #     #     token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
    #     #     token_embeddings[:,::2,:] = state_embeddings
    #     #     token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
    #     # elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
    #     #     token_embeddings = state_embeddings
    #     # else:
    #     #     raise NotImplementedError()

        
    #     # gpu_tracker.track()
    #     batch_size = states.shape[0]
    #     # all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
    #     # position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
    #     token_embeddings = token_embeddings.to(device)
    #     x = self.drop(token_embeddings)
    #     # print(self.blocks)

    #     x = self.blocks(x,paddle_mask)
    #     x = self.blocks2(x,paddle_mask)
    #     x = self.blocks3(x,paddle_mask)
    #     x = self.blocks4(x,paddle_mask)
    #     x = self.ln_f(x)
    #     logits = self.head(x)
    
    #     # if actions is not None and self.model_type == 'reward_conditioned':
    #     #     logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
    #     # elif actions is None and self.model_type == 'reward_conditioned':
    #     #     logits = logits[:, 1:, :]
    #     # elif actions is not None and self.model_type == 'naive':
    #     #     logits = logits[:, ::2, :] # only keep predictions from state_embeddings
    #     # elif actions is None and self.model_type == 'naive':
    #     #     logits = logits # for completeness
    #     # else:
    #     #     raise NotImplementedError()
    #     logits = logits[:, 20::22, :]

    #     # if we are given some desired targets also calculate the loss
    #     loss = None
    #     # logits=logits.long()
    #     # targets=targets.long()

    #     if targets is not None:
    #         # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    #         loss_func=nn.MSELoss(reduction='mean')

    #         # loss_mean=(logits.reshape(-1)-targets.reshape(-1))**2
            
    #         # print('max',logits.reshape(-1).max())
    #         # print(loss_mean[:30])
    #         # ind=torch.argmax(loss_mean)
    #         # print(torch.max(loss_mean))
    #         # print(loss_mean[ind])
    #         # print(ind)
    #         # print('ind',logits.reshape(-1)[ind])
    #         # print(targets.reshape(-1)[ind])
    #         # loss_mean=torch.mean(loss_mean)
    #         # print(loss_mean)
    #         # debug
    #         loss = loss_func(logits.reshape(-1), targets.reshape(-1))
    #     # gpu_tracker.track()


    #     # del targets
    #     # del rtgs
    #     # del state_embeddings
    #     # del states_seq
    #     # del states
    #     # del output
    #     # del context
    #     # del token_embeddings
    #     # del position_embeddings
    #     # del all_global_pos_emb
    #     # del action_embeddings
    #     # del rtg_embeddings





    #     # torch.cuda.empty_cache()
    #     return logits, loss
