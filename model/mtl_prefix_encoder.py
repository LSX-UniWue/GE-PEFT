""" Utility classes and functions related to MoCL (NAACL 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import copy
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from utils.utilities import mahalanobis

class PrefixEncoder(nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.residual_prefix_projection = config.residual_prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            # train self.embedding equals to train prompts in the input
            self.embedding = nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(config.hidden_size, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)


    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            
            if self.residual_prefix_projection:
                # Use residual reparameterization
                past_key_values = self.trans(prefix_tokens) + prefix_tokens
            else:
                past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
            
        return past_key_values


class PrefixContinualMTLEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seed = config.seed
        self.task_list = config.cl_language_list.split('_')
        self.num_tasks = len(self.task_list)
        self.task_count = 0
        self.prompt_layers = list(range(config.num_hidden_layers)) # TODO: specify layers
        
        self.n_head = config.num_attention_heads
        if self.config.model_type == 't5':
            self.n_embd = config.d_kv
        else:
            self.n_embd = config.hidden_size // config.num_attention_heads
        
        # uniform
        self.prompt_init_func = nn.init.orthogonal_ if config.prompt_init_func == 'orthogonal' else nn.init.uniform_
        
        self.steps = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_val = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_final = {t_id: 0 for t_id in range(self.num_tasks)}
        
        tb_log_dir = os.path.join(self.config.output_dir, f"tensorboard_seed{self.seed}")
        if not os.path.exists(tb_log_dir):
            os.mkdir(tb_log_dir)
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        
        if self.config.concat_prompts or self.config.task_identify_epi: # progressive prompts or EPI
            if torch.cuda.is_available():
                prompts = nn.Parameter(torch.randn((self.num_tasks, config.num_hidden_layers, config.pre_seq_len*2, self.n_head * self.n_embd), requires_grad=True, device='cuda'))
            else:
                prompts = nn.Parameter(torch.randn((self.num_tasks, config.num_hidden_layers, config.pre_seq_len*2, self.n_head * self.n_embd), requires_grad=True, device='cpu'))
            self.prompt_init_func(prompts)
            self.prompts = prompts
            if self.config.concat_prompts:
                self.prefix_len_exceeded = False
        else: # vanilla continual learning
            if torch.cuda.is_available():
                prompts = nn.Parameter(torch.randn((config.num_hidden_layers, config.pre_seq_len*2, self.n_head * self.n_embd), requires_grad=True, device='cuda'))
            else:
                prompts = nn.Parameter(torch.randn((config.num_hidden_layers, config.pre_seq_len*2, self.n_head * self.n_embd), requires_grad=True, device='cpu'))
            self.prompt_init_func(prompts)
            self.prompts = prompts
                
            
    def forward(self, batch_size, task_id, train=True, final=False, prompt_select_mode='local_compose', task_id_list=None, id_pred_on=False): # here local_compose means local concatenation; global_compose means global concatenation (for Progressive Prompts)
        if train and not final: # training
            self.steps[task_id] += 1
        elif not train and not final: # validation
            self.steps_val[task_id] += 1
        else: # final evaluation
            self.steps_final[task_id] += 1
            
        if self.config.concat_prompts:
            pre_seq_len = self.config.pre_seq_len * self.num_tasks if prompt_select_mode=='global_compose' else self.config.pre_seq_len * (task_id+1)
            # Prun the concatenated prefix if it exceeds the model's maximum supporting sequence length
            if self.config.model_type != 't5':
                if pre_seq_len > (self.config.max_position_embeddings - self.config.max_seq_length) or self.prefix_len_exceeded:
                    if not self.prefix_len_exceeded:
                        self.prefix_len_exceeded = True
                        self.max_pre_seq_len = self.config.pre_seq_len * ((self.config.max_position_embeddings -  self.config.max_seq_length) // self.config.pre_seq_len)
                        print(f"Concatenating prefixes of the last {pre_seq_len//self.config.pre_seq_len} tasks only (max. sequence length exceeded)...")
                    pre_seq_len = self.max_pre_seq_len
        else:
            pre_seq_len = self.config.pre_seq_len

        if torch.cuda.is_available():
            past_key_values = torch.zeros((
                self.config.num_hidden_layers * 2,
                batch_size,
                self.n_head,
                pre_seq_len,
                self.n_embd
            ), device='cuda')
        else:
            past_key_values = torch.zeros((
                self.config.num_hidden_layers * 2,
                batch_size,
                self.n_head,
                pre_seq_len,
                self.n_embd
            ), device='cpu')

        if self.config.concat_prompts: # progressive prompts
            if prompt_select_mode=='global_compose':
                P = self.prompts.detach().clone()
                P_k = torch.cat([P[i][:, :self.config.pre_seq_len] for i in range(P.shape[0])], dim=1)
                P_v = torch.cat([P[i][:, self.config.pre_seq_len:] for i in range(P.shape[0])], dim=1)
                
            elif task_id>0: 
                P_prev = self.prompts[:task_id].detach().clone()
                if task_id > 1:
                    P_k_prev = torch.cat([P_prev[i][:, :self.config.pre_seq_len] for i in range(P_prev.shape[0])], dim=1)
                    P_v_prev = torch.cat([P_prev[i][:, self.config.pre_seq_len:] for i in range(P_prev.shape[0])], dim=1)
                else:
                    P_k_prev = P_prev[0][:, :self.config.pre_seq_len]
                    P_v_prev = P_prev[0][:, self.config.pre_seq_len:]
                
                P_k = torch.cat((P_k_prev, self.prompts[task_id][:, :self.config.pre_seq_len]), dim=1) # n*lxd
                P_v = torch.cat((P_v_prev, self.prompts[task_id][:, self.config.pre_seq_len:]), dim=1) # n*lxd
            else:
                P = self.prompts[task_id] #lxd
                
                P_k = P[:, :self.config.pre_seq_len, :] #n*lxd
                P_v = P[:, self.config.pre_seq_len:, :] #n*lxd
                
            # Prun the concatenated prefix if it exceeds the model's maximum supporting sequence length
            if P_k.shape[1] != pre_seq_len:
                P_k = P_k[:, -pre_seq_len:]
                P_v = P_v[:, -pre_seq_len:]
                
        elif self.config.task_identify_epi:
            if not final:
                P = self.prompts[task_id] #lxd
                P_k = P[:, :self.config.pre_seq_len, :] #n*lxd
                P_v = P[:, self.config.pre_seq_len:, :] #n*lxd
                
            elif id_pred_on and task_id_list is not None:
                P_list = []
                for i in task_id_list:
                    P_list.append(self.prompts[i])
                P = torch.stack(P_list)
                P_k = P[:, :, :self.config.pre_seq_len, :] #n*lxd
                P_v = P[:, :, self.config.pre_seq_len:, :] #n*lxd
            
            else:
                raise ValueError("For the EPI final inference, the predicted task IDs are required!")

            
        else: # vanilla continual learning
            P = self.prompts #lxd
            
            P_k = P[:, :self.config.pre_seq_len, :] #n*lxd
            P_v = P[:, self.config.pre_seq_len:, :] #n*lxd

        if P_k.dim() < 4:
            P_k = P_k.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1, 1) 
            P_v = P_v.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1, 1) 
            
        P_k_ = P_k.view(
            batch_size,
            self.config.num_hidden_layers,
            pre_seq_len, 
            self.n_head, 
            self.n_embd
            ).permute([1,0,3,2,4])
        P_v_ = P_v.view(
            batch_size,
            self.config.num_hidden_layers,
            pre_seq_len, 
            self.n_head, 
            self.n_embd
            ).permute([1,0,3,2,4])
    
        for layer in self.prompt_layers:
            past_key_values[layer*2] = P_k_[layer]
            past_key_values[layer*2+1] = P_v_[layer]
        
        past_key_values = past_key_values.split(2)
        
        return past_key_values
    

class PrefixQKVEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seed = config.seed
        
        self.task_list = config.cl_language_list.split('_')
        self.num_tasks = len(self.task_list)
        
        self.prompt_layers = list(range(config.num_hidden_layers)) # TODO: specify layers
        self.ortho_loss_key = config.ortho_loss_key if config.key_init_func == 'orthogonal' else False
        self.ortho_loss_prompt = config.ortho_loss_prompt if config.prompt_init_func == 'orthogonal' else False
        self.ortho_loss_coeff = config.ortho_loss_coeff
        
        self.task_count = 0
        self.steps = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_val = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_final = {t_id: 0 for t_id in range(self.num_tasks)}
        
        self.log_weights = config.log_weights
        tb_log_dir = os.path.join(self.config.output_dir, f"tensorboard_seed{self.seed}")
        if not os.path.exists(tb_log_dir):
            os.mkdir(tb_log_dir)
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        
        self.n_head = config.num_attention_heads
        if self.config.model_type == 't5':
            self.n_embd = config.d_kv
        else:
            self.n_embd = config.hidden_size // config.num_attention_heads
        
        self._initialize_keys_prompts()

            
    def forward(self, x_query, task_id, train=True, final=False, prompt_select_mode='local_compose', task_id_list=None, id_pred_on=False):
        if train and not final: # training
            self.steps[task_id] += 1
        elif not train and not final: # validation
            self.steps_val[task_id] += 1
        else: # final evaluation
            self.steps_final[task_id] += 1
            
        batch_size = x_query.shape[0]
        if torch.cuda.is_available():
            loss = torch.tensor(0., requires_grad=True).cuda()
        else:
            loss = torch.tensor(0., requires_grad=True).cpu()

        prefix_len = self.config.pre_seq_len + self.config.pre_seq_len_g if self.config.add_general_prompt else self.config.pre_seq_len
        if torch.cuda.is_available():
            past_key_values = torch.zeros((
                self.config.num_hidden_layers * 2,
                batch_size, # batch_size
                self.n_head,
                prefix_len,
                self.n_embd
            ), device='cuda')
        else:
            past_key_values = torch.zeros((
                self.config.num_hidden_layers * 2,
                batch_size, # batch_size
                self.n_head,
                prefix_len,
                self.n_embd
            ), device='cpu')
        
        K = self.keys   # number of task x dim
        P = self.prompts
        A = self.attns if self.config.add_attention_filter else None

        if prompt_select_mode == 'global_compose':
            cos_sim = self._query_key_match(x_query, K, A)
        else:
            K, P, cos_sim = self._compose_prompts(x_query, K, P, task_id, train, prompt_select_mode, A, task_id_list, id_pred_on)
        
        if self.config.epi_for_composition or self.config.embed_prototypes_for_composition or self.config.direct_compose:
            w = cos_sim
        else:
            w = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1)

        if prompt_select_mode == 'task_id':
            P_ = P.repeat(batch_size, 1, 1, 1) if P.shape[0] == 1 else P
        else:
            if self.config.detach_w_from_task_loss:
                compose_w = w.clone().detach()
                P_ = torch.einsum('bn,nkld->bkld', compose_w, P[:compose_w.shape[-1]]) #bxNlx2lxd
            else:
                P_ = torch.einsum('bn,nkld->bkld', w, P[:w.shape[-1]]) #bxNlx2lxd    weighted prompts sums
                
            
        P_k = P_[:, :, :self.config.pre_seq_len, :] #bxNlxlxd
        P_v = P_[:, :, self.config.pre_seq_len:, :] #bxNlxlxd
        
        if self.config.add_general_prompt:
            GP = self.g_prompt
            GP = GP[None].repeat(batch_size, 1, 1, 1)
            
            GP_k = GP[:, :, :self.config.pre_seq_len_g, :]
            GP_v = GP[:, :, self.config.pre_seq_len_g:, :]
            
            P_k = torch.cat((P_k, GP_k), dim=2)
            P_v = torch.cat((P_v, GP_v), dim=2) # does the order matter? now consistent with DualPrompt

        P_k_ = self._refactor_prompts(P_k, batch_size)
        P_v_ = self._refactor_prompts(P_v, batch_size)
        
        for layer in self.prompt_layers:    
            past_key_values[layer*2] = P_k_[layer]
            past_key_values[layer*2+1] = P_v_[layer]

        w_global = None
        if train:
            loss, ortho_loss_key, ortho_loss_prompt, matching_loss, w_global = self._add_kp_losses(K, P, w, cos_sim, loss, batch_size, task_id, x_query)

        if not train and not final: # during inference along with training
            w_avg = torch.mean(w.detach().clone(), dim=0)
            if self.attn_weights[task_id] is None:
                self.attn_weights[task_id] = w_avg
            else:
                self.attn_weights[task_id] = ((self.attn_weights[task_id] * self.steps_val[task_id]) + w_avg) / (self.steps_val[task_id]+1)
        
        self._log_weights(w, task_id, train, final, prompt_select_mode, w_global)
                    
        
        if train:
            self._log_losses(task_id, ortho_loss_key, ortho_loss_prompt, matching_loss)
            
        return past_key_values.split(2), loss
    
                
    def _initialize_keys_prompts(self):
        self.attn_weights = [None for _ in range(self.num_tasks)]
        if torch.cuda.is_available():
            keys = nn.Parameter(torch.randn((self.num_tasks, self.config.key_dim), requires_grad=True, device='cuda'))
            prompts = nn.Parameter(torch.randn(
                (self.num_tasks, self.config.num_hidden_layers, self.config.pre_seq_len * 2, self.n_head * self.n_embd),
                requires_grad=True, device='cuda'))
        else:
            keys = nn.Parameter(torch.randn((self.num_tasks, self.config.key_dim), requires_grad=True, device='cpu'))
            prompts = nn.Parameter(torch.randn((self.num_tasks, self.config.num_hidden_layers, self.config.pre_seq_len*2, self.n_head * self.n_embd), requires_grad=True, device='cpu'))

        self.keys_mask = [False for _ in range(self.num_tasks)]
        
        if self.config.key_init_func == 'uniform':
            nn.init.uniform_(keys)
            
        if self.config.prompt_init_func == 'uniform':
            nn.init.uniform_(prompts)
        
        if self.config.key_init_func == 'normal':
            nn.init.normal_(keys, p=2, dim=-1)
            
        if self.config.prompt_init_func == 'normal':
            nn.init.normal_(prompts, p=2, dim=-1)
        
        if self.config.key_init_func == 'orthogonal':
            keys = self.gram_schmidt(keys)
            
        if self.config.prompt_init_func == 'orthogonal':
            prompts = self.gram_schmidt(prompts)
        
        self.keys = keys
        self.prompts = prompts
        
        
        if self.config.add_general_prompt:
            g_prompt = nn.Parameter(torch.randn((self.config.num_hidden_layers, self.config.pre_seq_len_g*2, self.n_head * self.n_embd), requires_grad=True, device='cuda'))
            
            if self.config.general_prompt_init_func == 'uniform':
                nn.init.uniform_(g_prompt)
                
            self.g_prompt = g_prompt
            self.g_prompt_log = []
        
        if self.config.add_attention_filter:
            attns = nn.Parameter(torch.randn((self.num_tasks, self.config.key_dim), requires_grad=True, device='cuda'))
            nn.init.uniform_(attns)
            if self.config.attn_init_func == 'orthogonal':
                attns = self.gram_schmidt(attns)
            self.attns = attns
    
    
    def _compose_prompts(self, query, K, P, task_id, train, prompt_select_mode, A=None, task_id_list=None, id_pred_on=False):
        if task_id_list is None and not id_pred_on:
            if train and self.task_count > 0:
                # freeze keys of previous tasks
                K = torch.cat((K[:task_id].detach().clone(), K[task_id][None]), dim=0)
                if A is not None:
                    A = torch.cat((A[:task_id].detach().clone(), A[task_id][None]), dim=0)
            else:
                K = K[:task_id+1] # nxd
                if A is not None:
                    A = A[:task_id+1] # nxd  
            
        if prompt_select_mode == 'local_compose':
            if train and self.task_count > 0:
                # freeze keys & prompts of previous tasks
                # K = torch.cat((K[:task_id].detach().clone(), K[task_id][None]), dim=0)
                P = torch.cat((P[:task_id].detach().clone(), P[task_id][None]), dim=0)
                # if A is not None:
                #     A = torch.cat((A[:task_id].detach().clone(), A[task_id][None]), dim=0)
            else:
                if id_pred_on and task_id_list is not None:
                    # if per-instance task_id given, now just use the mode as task_id
                    task_id_max = max(task_id_list.cpu()).item()
                    P = P[:task_id_max+1] # nxNlx2lxd
                    K = K[:task_id_max+1] # nxd
                    if A is not None:
                        A = A[:task_id_max+1] # nxd  
                else:
                    P = P[:task_id+1] # nxNlx2lxd
                
        elif prompt_select_mode == 'task_id':
                # K = K[task_id][None,:] # 1xd
                # if A is not None:
                #     A = [task_id][None,:] # 1xd  
                    
                if id_pred_on and task_id_list is not None:
                    # if per-instance task_id given
                    P_list = []
                    for i in task_id_list:
                        P_list.append(P[i])
                    P = torch.stack(P_list)
                else:
                    P = P[task_id][None,:] # 1xNlx2lxd
        
        if (self.config.epi_for_composition or self.config.embed_prototypes_for_composition) and self.config.compose_prompts:
            if self.config.epi_for_composition:
                cos_sim = self._get_epi_ids(query)
                if not train:
                    cos_sim = cos_sim[:, :P.shape[0]]
                    cos_sim = nn.functional.softmax(cos_sim, dim=-1)
            else:
                if not train and id_pred_on and task_id_list is not None:
                    task_id_max = max(task_id_list.cpu()).item()
                    cos_sim = self._get_task_embed_pred(query, task_id_max, train)
                else:   
                    cos_sim = self._get_task_embed_pred(query, task_id, train)

        else:         
            cos_sim = self._query_key_match(query, K, A)
        
        return K, P, cos_sim
    
    
    def _query_key_match(self, Q, K, A=None):
        n_k = nn.functional.normalize(K, dim=1)
        
        if A is not None:
            a_query = torch.einsum('bj,kj->bkj', Q, A)
            n_q = nn.functional.normalize(a_query, dim=1)
            cos_sim = torch.einsum('bkj,kj->bk', n_q, n_k)
        else:
            n_q = nn.functional.normalize(Q, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', n_q, n_k)
            
        for i in range(cos_sim.shape[-1]-1):
            if self.keys_mask[i]:
                cos_sim = cos_sim.clone()
                cos_sim[:, i] = -1e8
        return cos_sim

    
    def _get_epi_ids(self, query):
        scores_over_tasks = []
        for mean_over_classes in self.task_means_over_classes:
            num_labels = mean_over_classes.shape[0]
            score_over_classes = []
            for l in range(num_labels):
                score = mahalanobis(query, mean_over_classes[l], self.cov_inv, norm=2)
                score_over_classes.append(score)
            
            # [num_labels, bs]
            score_over_classes = torch.stack(score_over_classes)
            score, _ = score_over_classes.min(dim=0)
            
            scores_over_tasks.append(score)
        
        # [num_tasks, bs]
        scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
        # _, ids = torch.min(scores_over_tasks, dim=0)
        
        prob_over_tasks = (scores_over_tasks*(-1)).T.softmax(dim=-1)
        
        return prob_over_tasks
    
    def _get_task_embed_pred(self, query, task_id, train):
        embed_prototypes = torch.stack(self.embed_prototypes[:task_id+1])
        cos_sim = torch.einsum('bj,kj->bk', query, embed_prototypes.cuda())
        return cos_sim.softmax(dim=1)
    
    
    def _refactor_prompts(self, prompt, batch_size):
        prefix_len = self.config.pre_seq_len + self.config.pre_seq_len_g if self.config.add_general_prompt else self.config.pre_seq_len
        return prompt.view(
            batch_size,
            self.config.num_hidden_layers,
            prefix_len,
            self.n_head,
            self.n_embd
        ).permute([1,0,3,2,4])
    
    
    def _add_kp_losses(self, K, P, w, cos_sim, loss, batch_size, task_id, x_query):
        ortho_loss_key, ortho_loss_prompt = 0, 0
        if self.ortho_loss_key or self.ortho_loss_prompt:
            ortho_loss_key, ortho_loss_prompt = 0, 0
            if self.ortho_loss_key:
                ortho_loss_key = self.ortho_loss_coeff * self.ortho_penalty(K)
                loss += ortho_loss_key
            if self.ortho_loss_prompt:
                ortho_loss_prompt = self.ortho_loss_coeff * self.ortho_penalty(P.view(P.shape[0], -1))
                loss += ortho_loss_prompt
            
        matching_loss = 0
        if self.config.matching_loss:
            # expect the (correct) weight digit = 1
            matching_loss = (1.0-w[:, -1]).mean() * self.config.matching_loss_coeff
            loss += matching_loss 
            
        if self.config.matching_loss_v2:
            # expect the (correct) query-key similarity = 1
            matching_loss = (1.0-cos_sim[:, -1]).mean() * self.config.matching_loss_coeff
            loss = loss + matching_loss # my loss += matching_loss
            
        if self.config.matching_loss_cls:
            # consider it as a classification task; gt_w is the ground truth one-hot vector
            gt_w = torch.zeros_like(w)
            gt_w[:, task_id] = 1.
            gt_w = gt_w[:, :task_id+1]
            
            matching_loss = torch.abs(gt_w - w).mean() * self.config.matching_loss_coeff
            loss = loss + matching_loss #my originally was loss+= caused error
                
        if self.config.matching_loss_cls_all:
            # consider it as a classification task, consider all keys (also include keys of future tasks)
            
            K_all = torch.cat((self.keys[:task_id].detach().clone(), self.keys[task_id][None]), dim=0) if task_id !=0 else self.keys[0][None]
            if task_id != self.num_tasks-1:
                K_late = self.keys[task_id+1:].detach().clone() if task_id != self.num_tasks-2 else self.keys[task_id+1].detach().clone()[None]
                K_all = torch.cat((K_all, K_late), dim=0)
            
            A_all = None
            if self.config.add_attention_filter:
                A_all = torch.cat((self.attns[:task_id].detach().clone(), self.attns[task_id][None]), dim=0) if task_id !=0 else self.attns[0][None]
                if task_id != self.num_tasks-1:
                    A_late = self.attns[task_id+1:].detach().clone() if task_id != self.num_tasks-2 else self.attns[task_id+1].detach().clone()[None]
                    A_all = torch.cat((A_all, A_late), dim=0)
                    
            cos_sim = self._query_key_match(x_query, K_all, A_all)
            
            w = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) if not self.config.direct_compose else cos_sim
            
            # gt_w is the ground truth one-hot vector
            gt_w = torch.zeros_like(w)
            gt_w[:, task_id] = 1.
            
            matching_loss = torch.abs(gt_w - w).mean() * self.config.matching_loss_coeff
            loss += matching_loss 
            
            return loss, ortho_loss_key, ortho_loss_prompt, matching_loss, w
            
        return loss, ortho_loss_key, ortho_loss_prompt, matching_loss, None
    
    
    def _add_ortho_loss(self, keys, prompts, loss):
        ortho_loss_key, ortho_loss_prompt = 0, 0
        if self.ortho_loss_key:
            ortho_loss_key = self.ortho_loss_coeff * self.ortho_penalty(keys)
            loss += ortho_loss_key
        if self.ortho_loss_prompt:
            ortho_loss_prompt = self.ortho_loss_coeff * self.ortho_penalty(prompts.view(prompts.shape[0], -1))
            loss += ortho_loss_prompt
        return loss, ortho_loss_key, ortho_loss_prompt
    
    
    def _log_weights(self, w, task_id, train, final, prompt_select_mode, w_global=None):
        if self.log_weights and prompt_select_mode != 'task_id':
            # In total 4 modes: 1-train, 2-inference, 3-final-local, 4-final-global
            w_avg = torch.mean(w.detach().clone(), dim=0)
            
            for t in range(w_avg.shape[-1]):
                if train and not final:
                    step = self.steps[task_id]
                    log_name = f'training_weight_{task_id}/{t}'
                elif not train and not final:
                    step = self.steps_val[task_id]
                    log_name = f'validation_weight_{task_id}/{t}'
                elif final:
                    step = self.steps_final[task_id]
                    log_name = f'final_{prompt_select_mode}_weight_{task_id}/{t}'

                self.writer.add_scalar(log_name, w_avg[t], step)
                
            if w_global is not None:
                w_global_avg = torch.mean(w_global.detach().clone(), dim=0)
                for t in range(self.num_tasks):
                    if train and not final:
                        step = self.steps[task_id]
                        log_name = f'training_weight_{task_id}/{t}'
                    elif not train and not final:
                        step = self.steps_val[task_id]
                        log_name = f'validation_weight_{task_id}/{t}'
                    elif final:
                        step = self.steps_final[task_id]
                        log_name = f'final_{prompt_select_mode}_weight_{task_id}/{t}'
                        
                    self.writer.add_scalar(log_name.replace('weight_', 'global_weight_'), w_global_avg[t], step)
                
    
    def _log_losses(self, task_id, ortho_loss_key, ortho_loss_prompt, matching_loss):
        if self.ortho_loss_key:
            self.writer.add_scalar(f'ortho_loss/task_{task_id}', ortho_loss_key.cpu(), self.steps[task_id])
        if self.ortho_loss_prompt:
            self.writer.add_scalar(f'ortho_loss/task_{task_id}', ortho_loss_prompt.cpu(), self.steps[task_id])
        if self.config.matching_loss or self.config.matching_loss_v2 or self.config.matching_loss_cls or self.config.matching_loss_cls_all:
            self.writer.add_scalar(f'matching_loss/task_{task_id}', matching_loss.cpu(), self.steps[task_id])

    
    # code for this function is from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u
        
        # check if the tensor > 2D and flatten the last dimensions if necessary
        is_nd = len(vv.shape) > 2
        if is_nd:
            shape_nd = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)
            
        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        
        s = self.task_count
        if s>0:
            uu[:, 0:s] = vv[:, 0:s].clone()
            
        for k in range(s, nk):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                # vk = vv[:, k].clone()
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
                    
        for k in range(s, nk):
            uk = uu[:, k].clone()
            uu[:, k] = uk / uk.norm()
            
        # undo swapping of rows and columns
        uu = uu.T 
        
        # return from 2D
        if is_nd:
            uu = uu.view(shape_nd)
        
        return nn.Parameter(uu) 
    
    
    def ortho_penalty(self, t):
        # return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()
        return ((t @t.T - torch.eye(t.shape[0]).cuda()).abs()).mean()

    
    def process_task_count(self):
        self.task_count += 1
        
        # project untrained keys to ensure orthogonality
        if self.config.key_init_func == 'orthogonal':
            self.keys = self.gram_schmidt(self.keys)
        
        task_id = self.task_count - 1
        # update task-specific prompt: use composed prompts for further usage (Only during training)
        if self.task_count > 1 and self.config.composed_prompt_for_usage:
            P = self.prompts[:self.task_count].detach().clone() # nxN_layerx2lxd
            w = self.attn_weights[task_id].detach().clone()
            P_i = torch.einsum('n,nkld->kld', w, P)
            with torch.no_grad():
                self.prompts[task_id] = nn.Parameter(P_i)
        
        # log general prompts 
        if self.config.add_general_prompt:
            self.g_prompt_log.append(self.g_prompt.detach().clone())



class PrefixMaskEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pre_seq_len = config.pre_seq_len
        self.seed = config.seed
        self.task_list = config.cl_language_list.split('_')
        self.num_tasks = len(self.task_list)
        self.task_count = 0
        self.prompt_layers = list(range(config.num_hidden_layers))  # TODO: specify layers
        self.smax = config.smax
        self.task_pre_seq_len = {}

        self.n_head = config.num_attention_heads
        if self.config.model_type == 't5':
            self.n_embd = config.d_kv
        else:
            self.n_embd = config.hidden_size // config.num_attention_heads

        self.prompt_init_func = nn.init.orthogonal_ if config.prompt_init_func == 'orthogonal' else nn.init.uniform_

        self.steps = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_val = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_final = {t_id: 0 for t_id in range(self.num_tasks)}

        tb_log_dir = os.path.join(self.config.output_dir, f"tensorboard_seed{self.seed}")
        if not os.path.exists(tb_log_dir):
            os.mkdir(tb_log_dir)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # if self.config.concat_prompts or self.config.task_identify_epi:  # progressive prompts or EPI
        #     if torch.cuda.is_available():
        #         prompts = nn.Parameter(torch.randn(
        #             (self.num_tasks, config.num_hidden_layers, config.pre_seq_len * 2, self.n_head * self.n_embd),
        #             requires_grad=True, device='cuda'))
        #     else:
        #         prompts = nn.Parameter(torch.randn(
        #             (self.num_tasks, config.num_hidden_layers, config.pre_seq_len * 2, self.n_head * self.n_embd),
        #             requires_grad=True, device='cpu'))
        #     self.prompt_init_func(prompts)
        #     self.prompts = prompts
        #     if self.config.concat_prompts:
        #         self.prefix_len_exceeded = False
        # else:  # vanilla continual learning
        # ToDo use a single set of parameters for all tasks
        if torch.cuda.is_available():
            prompts = nn.Parameter(
                torch.randn((config.num_hidden_layers, self.pre_seq_len * 2, self.n_head * self.n_embd),
                            requires_grad=True, device='cuda'))
        else:
            prompts = nn.Parameter(
                torch.randn((config.num_hidden_layers, self.pre_seq_len * 2, self.n_head * self.n_embd),
                            requires_grad=True, device='cpu'))

        self.prompt_init_func(prompts)
        self.prompts = prompts

        # self.masks = torch.nn.Embedding(self.num_tasks, config.num_hidden_layers, config.pre_seq_len * 2, self.n_head * self.n_embd)
        self.masks = torch.nn.Embedding(self.num_tasks, config.num_hidden_layers * self.pre_seq_len * 2 * self.n_head * self.n_embd)
        self.gate = torch.nn.Sigmoid()



    def get_task_grow_mask(self, task_id, mask):
        task_peft_len = self.task_pre_seq_len[task_id]
        grow_fact = self.pre_seq_len - task_peft_len
        mask_reshaped = mask.view(self.config.num_hidden_layers, self.pre_seq_len * 2,
                                            self.n_head * self.n_embd)
        first_part_mask = mask_reshaped[:, :task_peft_len, :]  # First two vectors for all embeddings
        middle_part_mask = mask_reshaped[:, task_peft_len+grow_fact:(task_peft_len * 2)+grow_fact, :]  # Next two vectors for all embeddings
        reshaped_mask = torch.cat((first_part_mask, middle_part_mask), dim=1)
        reshaped_mask = reshaped_mask.view(1, self.config.num_hidden_layers * task_peft_len * 2 *
                                            self.n_head * self.n_embd)
        return reshaped_mask

    # after grow step zero prefix is inserted for new tasks, should be removed by old ones
    def get_reshaped_prefix(self, task_id, prefix):
        task_peft_len = self.task_pre_seq_len[task_id]
        grow_fact = self.pre_seq_len - task_peft_len

        first_part = prefix[:, :task_peft_len, :]
        middle_part = prefix[:, task_peft_len+grow_fact:(task_peft_len * 2)+grow_fact, :]

        reshaped_prefix = torch.cat((first_part, middle_part), dim=1)
        return reshaped_prefix



    def get_mask(self, task, s=1):
        if torch.cuda.is_available():  # local cpu run if else
            mask = self.masks(torch.LongTensor([task]).cuda())
        else:
            mask = self.masks(torch.LongTensor([task]).cpu())

        # if growing PEFT
        if task in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task]:
            mask = self.get_task_grow_mask(task, mask)


        g_mask = self.gate(s*mask)

        if s == self.smax:  # you want to use it for evluation
            g_mask = (g_mask > 0.5).float()

        return g_mask



    def expand_prompts(self, task_id, expansion_size=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        expanded_prefix = torch.randn((self.config.num_hidden_layers, expansion_size*2, self.n_head * self.n_embd)).to(device)
        self.prompt_init_func(expanded_prefix)
        first_prefix_part = expanded_prefix[:, :expansion_size, :]
        second_prefix_part = expanded_prefix[:, expansion_size:, :]

        # Step 3: Split the original tensor into parts
        first_part = self.prompts[:, :self.pre_seq_len, :]  # First half vectors
        middle_part = self.prompts[:, self.pre_seq_len:self.pre_seq_len * 2, :]  # Second half

        expanded_tensor = torch.cat((first_part, first_prefix_part, middle_part, second_prefix_part), dim=1)
        if torch.cuda.is_available():
            expanded_tensor.to('cuda')
            expanded_prompts = nn.Parameter(expanded_tensor, requires_grad=True)
        else:
            expanded_prompts = nn.Parameter(expanded_tensor, requires_grad=True)


        self.prompts = expanded_prompts

        # adjust gating embeddings
        task_idx = torch.arange(self.num_tasks).to(device)
        task_mask_embeddings = self.masks(task_idx)
        reshaped_mask = task_mask_embeddings.view(self.num_tasks, self.config.num_hidden_layers,
                                                  self.pre_seq_len * 2, self.n_head * self.n_embd)



        # create new gating embeddings
        embedding_for_expansion = torch.nn.Embedding(self.num_tasks,
                                    self.config.num_hidden_layers * expansion_size * 2 * self.n_head * self.n_embd)
        new_mask_weights = embedding_for_expansion.weight.detach().clone()
        new_mask_weights[:task_id, :] = 0
        new_mask = new_mask_weights.view(self.num_tasks, self.config.num_hidden_layers, expansion_size * 2, self.n_head * self.n_embd)
        new_mask = new_mask.to(device)
        first_new_mask_part = new_mask[:, :, :expansion_size, :]
        second_new_mask_part = new_mask[:, :, expansion_size:, :]

        first_part_mask = reshaped_mask[:, :, :self.pre_seq_len, :]  # First half vectors for all embeddings
        middle_part_mask = reshaped_mask[:, :, self.pre_seq_len:self.pre_seq_len * 2, :]  # Next half vectors for all embeddings

        expanded_mask = torch.cat((first_part_mask, first_new_mask_part, middle_part_mask, second_new_mask_part), dim=2)
        updated_dimensionality = self.config.num_hidden_layers * (self.pre_seq_len * 2 + 2 * expansion_size) * self.n_head * self.n_embd
        flattened_embeddings = expanded_mask.view(self.num_tasks, updated_dimensionality)

        expanded_masks_embedding = torch.nn.Embedding(self.num_tasks, updated_dimensionality)
        expanded_masks_embedding.weight.data = flattened_embeddings





        self.masks = expanded_masks_embedding
        self.pre_seq_len = self.pre_seq_len + expansion_size
        # self.task_pre_seq_len[task_id] = self.pre_seq_len

        print(f"Exdended prefix with {expansion_size} tokens. New prefix lentgh is {self.pre_seq_len} .")


    def forward(self, batch_size, task_id, train=True, final=False, prompt_select_mode='local_compose',
                task_id_list=None,
                id_pred_on=False, s=None):  # here local_compose means local concatenation; global_compose means global concatenation (for Progressive Prompts)


        if train and not final:  # training
            self.steps[task_id] += 1
        elif not train and not final:  # validation
            self.steps_val[task_id] += 1
        else:  # final evaluation
            self.steps_final[task_id] += 1

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            if torch.cuda.is_available():
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.task_pre_seq_len[task_id],
                    self.n_embd
                ), device='cuda')
            else:
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.task_pre_seq_len[task_id],
                    self.n_embd
                ), device='cpu')
        else:
            if torch.cuda.is_available():
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.pre_seq_len,
                    self.n_embd
                ), device='cuda')
            else:
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.pre_seq_len,
                    self.n_embd
                ), device='cpu')




        P = self.prompts  # lxd

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            P = self.get_reshaped_prefix(task_id, P)

        task_masks = self.get_mask(task_id, s=s)

        # Reshape task_masks to match the prompts shape
        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            task_masks = task_masks.view(self.config.num_hidden_layers, self.task_pre_seq_len[task_id] * 2,
                                            self.n_head * self.n_embd)
        else:
            task_masks = task_masks.view(self.config.num_hidden_layers, self.pre_seq_len * 2,
                                            self.n_head * self.n_embd)


        P = P * task_masks

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            P_k = P[:, :self.task_pre_seq_len[task_id], :]  # n*lxd
            P_v = P[:, self.task_pre_seq_len[task_id]:, :]  # n*lxd
        else:
            P_k = P[:, :self.pre_seq_len, :]  # n*lxd
            P_v = P[:, self.pre_seq_len:, :]  # n*lxd

        if P_k.dim() < 4:
            P_k = P_k.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1, 1)
            P_v = P_v.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1, 1)

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            P_k_ = P_k.view(
                batch_size,
                self.config.num_hidden_layers,
                self.task_pre_seq_len[task_id],
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])
            P_v_ = P_v.view(
                batch_size,
                self.config.num_hidden_layers,
                self.task_pre_seq_len[task_id],
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])

        else:
            P_k_ = P_k.view(
                batch_size,
                self.config.num_hidden_layers,
                self.pre_seq_len,
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])
            P_v_ = P_v.view(
                batch_size,
                self.config.num_hidden_layers,
                self.pre_seq_len,
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])

        for layer in self.prompt_layers:
            past_key_values[layer * 2] = P_k_[layer]
            past_key_values[layer * 2 + 1] = P_v_[layer]

        past_key_values = past_key_values.split(2)

        return past_key_values




class PrefixExpandOnlyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pre_seq_len = config.pre_seq_len
        self.seed = config.seed
        self.task_list = config.cl_language_list.split('_')
        self.num_tasks = len(self.task_list)
        self.task_count = 0
        self.prompt_layers = list(range(config.num_hidden_layers))  # TODO: specify layers
        self.smax = config.smax
        self.task_pre_seq_len = {}

        self.n_head = config.num_attention_heads
        if self.config.model_type == 't5':
            self.n_embd = config.d_kv
        else:
            self.n_embd = config.hidden_size // config.num_attention_heads

        self.prompt_init_func = nn.init.orthogonal_ if config.prompt_init_func == 'orthogonal' else nn.init.uniform_

        self.steps = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_val = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_final = {t_id: 0 for t_id in range(self.num_tasks)}

        tb_log_dir = os.path.join(self.config.output_dir, f"tensorboard_seed{self.seed}")
        if not os.path.exists(tb_log_dir):
            os.mkdir(tb_log_dir)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # if self.config.concat_prompts or self.config.task_identify_epi:  # progressive prompts or EPI
        #     if torch.cuda.is_available():
        #         prompts = nn.Parameter(torch.randn(
        #             (self.num_tasks, config.num_hidden_layers, config.pre_seq_len * 2, self.n_head * self.n_embd),
        #             requires_grad=True, device='cuda'))
        #     else:
        #         prompts = nn.Parameter(torch.randn(
        #             (self.num_tasks, config.num_hidden_layers, config.pre_seq_len * 2, self.n_head * self.n_embd),
        #             requires_grad=True, device='cpu'))
        #     self.prompt_init_func(prompts)
        #     self.prompts = prompts
        #     if self.config.concat_prompts:
        #         self.prefix_len_exceeded = False
        # else:  # vanilla continual learning
        # ToDo use a single set of parameters for all tasks
        if torch.cuda.is_available():
            prompts = nn.Parameter(
                torch.randn((config.num_hidden_layers, self.pre_seq_len * 2, self.n_head * self.n_embd),
                            requires_grad=True, device='cuda'))
        else:
            prompts = nn.Parameter(
                torch.randn((config.num_hidden_layers, self.pre_seq_len * 2, self.n_head * self.n_embd),
                            requires_grad=True, device='cpu'))

        self.prompt_init_func(prompts)
        self.prompts = prompts

        # self.masks = torch.nn.Embedding(self.num_tasks, config.num_hidden_layers, config.pre_seq_len * 2, self.n_head * self.n_embd)
        # self.masks = torch.nn.Embedding(self.num_tasks, config.num_hidden_layers * self.pre_seq_len * 2 * self.n_head * self.n_embd)
        # self.gate = torch.nn.Sigmoid()



    # def get_task_grow_mask(self, task_id, mask):
    #     task_peft_len = self.task_pre_seq_len[task_id]
    #     grow_fact = self.pre_seq_len - task_peft_len
    #     mask_reshaped = mask.view(self.config.num_hidden_layers, self.pre_seq_len * 2,
    #                                         self.n_head * self.n_embd)
    #     first_part_mask = mask_reshaped[:, :task_peft_len, :]  # First two vectors for all embeddings
    #     middle_part_mask = mask_reshaped[:, task_peft_len+grow_fact:(task_peft_len * 2)+grow_fact, :]  # Next two vectors for all embeddings
    #     reshaped_mask = torch.cat((first_part_mask, middle_part_mask), dim=1)
    #     reshaped_mask = reshaped_mask.view(1, self.config.num_hidden_layers * task_peft_len * 2 *
    #                                         self.n_head * self.n_embd)
    #     return reshaped_mask

    # after grow step zero prefix is inserted for new tasks, should be removed by old ones
    def get_reshaped_prefix(self, task_id, prefix):
        task_peft_len = self.task_pre_seq_len[task_id]
        grow_fact = self.pre_seq_len - task_peft_len

        first_part = prefix[:, :task_peft_len, :]
        middle_part = prefix[:, task_peft_len+grow_fact:(task_peft_len * 2)+grow_fact, :]

        reshaped_prefix = torch.cat((first_part, middle_part), dim=1)
        return reshaped_prefix







    def expand_prompts(self, task_id, expansion_size=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        expanded_prefix = torch.randn((self.config.num_hidden_layers, expansion_size*2, self.n_head * self.n_embd)).to(device)
        self.prompt_init_func(expanded_prefix)
        first_prefix_part = expanded_prefix[:, :expansion_size, :]
        second_prefix_part = expanded_prefix[:, expansion_size:, :]

        # Step 3: Split the original tensor into parts
        first_part = self.prompts[:, :self.pre_seq_len, :]  # First half vectors
        middle_part = self.prompts[:, self.pre_seq_len:self.pre_seq_len * 2, :]  # Second half

        expanded_tensor = torch.cat((first_part, first_prefix_part, middle_part, second_prefix_part), dim=1)
        if torch.cuda.is_available():
            expanded_tensor.to('cuda')
            expanded_prompts = nn.Parameter(expanded_tensor, requires_grad=True)
        else:
            expanded_prompts = nn.Parameter(expanded_tensor, requires_grad=True)


        self.prompts = expanded_prompts




        # self.masks = expanded_masks_embedding
        self.pre_seq_len = self.pre_seq_len + expansion_size
        # self.task_pre_seq_len[task_id] = self.pre_seq_len

        print(f"Exdended prefix with {expansion_size} tokens. New prefix length is {self.pre_seq_len} .")


    def forward(self, batch_size, task_id, train=True, final=False, prompt_select_mode='local_compose',
                task_id_list=None,
                id_pred_on=False, s=None):  # here local_compose means local concatenation; global_compose means global concatenation (for Progressive Prompts)


        if train and not final:  # training
            self.steps[task_id] += 1
        elif not train and not final:  # validation
            self.steps_val[task_id] += 1
        else:  # final evaluation
            self.steps_final[task_id] += 1

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            if torch.cuda.is_available():
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.task_pre_seq_len[task_id],
                    self.n_embd
                ), device='cuda')
            else:
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.task_pre_seq_len[task_id],
                    self.n_embd
                ), device='cpu')
        else:
            if torch.cuda.is_available():
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.pre_seq_len,
                    self.n_embd
                ), device='cuda')
            else:
                past_key_values = torch.zeros((
                    self.config.num_hidden_layers * 2,
                    batch_size,
                    self.n_head,
                    self.pre_seq_len,
                    self.n_embd
                ), device='cpu')




        P = self.prompts  # lxd

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            P = self.get_reshaped_prefix(task_id, P)

        # task_masks = self.get_mask(task_id, s=s)

        # Reshape task_masks to match the prompts shape
        # if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
        #     task_masks = task_masks.view(self.config.num_hidden_layers, self.task_pre_seq_len[task_id] * 2,
        #                                     self.n_head * self.n_embd)
        # else:
        #     task_masks = task_masks.view(self.config.num_hidden_layers, self.pre_seq_len * 2,
        #                                     self.n_head * self.n_embd)


        # P = P * task_masks

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            P_k = P[:, :self.task_pre_seq_len[task_id], :]  # n*lxd
            P_v = P[:, self.task_pre_seq_len[task_id]:, :]  # n*lxd
        else:
            P_k = P[:, :self.pre_seq_len, :]  # n*lxd
            P_v = P[:, self.pre_seq_len:, :]  # n*lxd

        if P_k.dim() < 4:
            P_k = P_k.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1, 1)
            P_v = P_v.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1, 1)

        if task_id in self.task_pre_seq_len.keys() and self.pre_seq_len != self.task_pre_seq_len[task_id]:
            P_k_ = P_k.view(
                batch_size,
                self.config.num_hidden_layers,
                self.task_pre_seq_len[task_id],
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])
            P_v_ = P_v.view(
                batch_size,
                self.config.num_hidden_layers,
                self.task_pre_seq_len[task_id],
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])

        else:
            P_k_ = P_k.view(
                batch_size,
                self.config.num_hidden_layers,
                self.pre_seq_len,
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])
            P_v_ = P_v.view(
                batch_size,
                self.config.num_hidden_layers,
                self.pre_seq_len,
                self.n_head,
                self.n_embd
            ).permute([1, 0, 3, 2, 4])

        for layer in self.prompt_layers:
            past_key_values[layer * 2] = P_k_[layer]
            past_key_values[layer * 2 + 1] = P_v_[layer]

        past_key_values = past_key_values.split(2)

        return past_key_values


