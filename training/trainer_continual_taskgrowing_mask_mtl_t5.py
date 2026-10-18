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
import json
import math
import shutil
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers.trainer_pt_utils import get_parameter_names
from training.early_stopping import EarlyStopping


class ContinualTrainerMTL(nn.Module):
    def __init__(self,
                 args,
                 model,
                 logger,
                 task_list,
                 early_stopping_patience=-1,
                 tokenizer=None,
                 learning_rate_list=None,
                 ):
        super(ContinualTrainerMTL, self).__init__()
        
        self.args = args
        self.seed = args.seed
        self.model = model.to(self.args.device)
        self.logger = logger
        self.task_list = task_list
        self.num_tasks = len(task_list)
        self.prompts_saved = []
        self.g_prompts_saved = []


        self.smax = args.smax
        self.thres_cosh = 50
        self.thres_emb = 6
        self.lamb = args.lamb
        self.mask_pre = None
        self.mask_back = None

        # save to restore in case no improvement with additional params
        self.mask_pre_dump = None
        self.mask_back_dump = None
        self.prefix_dump = None
        self.prefix_mask_dump = None

        self.task_grow = args.task_grow

        self.expansion_size = args.expansion_size

        self.logger.info(f"***********************************lamb: {self.lamb}***********************************")
        print(f"***********************************lamb:", str(self.lamb), "***********************************")
        self.logger.info(f"***********************************expansion_size: {args.expansion_size}***********************************")
        print(f"***********************************expansion_size:", str(args.expansion_size), "***********************************")
        self.logger.info(f"***********************************lamb: Taskgrowing MASK Prompt T5 Model ***********************************")
        print(f"*********************************** Taskgroweing MASK Prompt T5 Model***********************************")

        
        self.num_train_epochs = math.ceil(args.num_train_epochs)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = EarlyStopping(
            save_path=os.path.join(args.output_dir, 'best_checkpoint'),
            logger=self.logger,
            patience = early_stopping_patience
        )
        self.tokenizer = tokenizer
        
        try:
            self.learning_rate_list = [float(x) for x in learning_rate_list.split('_')]
        except:
            self.learning_rate_list = [self.args.learning_rate for _ in range(self.num_tasks)]
        
        self.logger.info(f"***********************************seed: {self.seed}***********************************")
        self.logger.info(f"***********************************lr: {self.learning_rate_list}***********************************")

        with open(f'{self.args.output_dir}/task_list.json', 'w') as f:
            json.dump(self.task_list, f)
            
        if self.model.config.save_initial_prompts:
            self._save_prompts(initial=True)
    
        
    def _prepare_optimizer(self, task_id=None):
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        
        # get optimizer class and kwargs
        learning_rate = self.learning_rate_list[task_id] if self.learning_rate_list is not None else self.args.learning_rate
        optimizer_kwargs = {"lr": learning_rate}
        
        adam_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        
        from torch.optim import AdamW

        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)

        # Initilaize optimizer
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


    def _process_data(self, loader, task_id=0, train=True):
        try:
            data_batch = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data_batch = next(loader[1])
        
        inputs = {k: data_batch[k].to(self.args.device) for k in data_batch}
        
        # label = data_batch["target_ids"].to(self.args.device)
        label = data_batch["target_ids"].clone().to(self.args.device)
        if train:
            label[label[:, :] == self.tokenizer.pad_token_id] = -100
        
        return data_batch, label

    
    def compute_loss(self, model, inputs, task_id, mode, final=False, prompt_select_mode='local_compose', id_pred_on=False, s=1):
        train = True if mode == 'train' else False
        loss, outputs = model(inputs, task_id, train, final, prompt_select_mode, id_pred_on, s=s)
        
        return loss, outputs
    
    
    def _prepare_dataloaders(self, dataloaders):
        loader = {}
        batch_num = []
        for task in dataloaders.keys():
            loader[task] = [dataloaders[task], iter(dataloaders[task])]
            batch_num.append(len(dataloaders[task]))
        return loader, batch_num
    
    
    def get_mask_regularisation(self, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            free_dims = 1 - self.mask_pre
            reg += (masks * free_dims).sum()
            count += free_dims.sum()

        else:
            reg += masks.sum()
            count += np.prod(masks.size()).item()
        reg /= count

        return self.lamb * reg





    def restore_prompt_from_dump(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.prefix_encoder.prompts = nn.Parameter(self.prefix_dump.to(device))
        self.model.prefix_encoder.masks = self.prefix_mask_dump.to(device)
        self.model.prefix_encoder.pre_seq_len = int(self.model.prefix_encoder.prompts.shape[1]/2)


    def peft_grow_step(self, task_id):
        # dump used in restore case
        self.prefix_dump = self.model.prefix_encoder.prompts.clone().cpu()
        mask_copy = nn.Embedding(self.model.prefix_encoder.masks.num_embeddings, self.model.prefix_encoder.masks.embedding_dim)
        mask_copy.load_state_dict(self.model.prefix_encoder.masks.state_dict())
        mask_copy = mask_copy.cpu()
        self.prefix_mask_dump = mask_copy

        pre_seq_len = self.model.prefix_encoder.pre_seq_len
        self.model.prefix_encoder.expand_prompts(task_id=task_id, expansion_size=self.expansion_size)
        # adjust mask_pre and mask_back
        if task_id > 0:
            # dump to restore if no imporovement
            self.mask_pre_dump = self.mask_pre.clone().cpu()
            self.mask_back_dump = self.mask_back.clone().cpu()
            mask_pre = self.mask_pre.view(self.model.prefix_encoder.config.num_hidden_layers, pre_seq_len * 2,
                                                self.model.prefix_encoder.n_head * self.model.prefix_encoder.n_embd)

            # device_to_use = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            extend_zero_mask = torch.zeros((self.model.prefix_encoder.config.num_hidden_layers, self.expansion_size, self.model.prefix_encoder.n_head * self.model.prefix_encoder.n_embd))#.to(device_to_use)

            first_part = mask_pre[:, :pre_seq_len, :]  # First two vectors
            middle_part = mask_pre[:, pre_seq_len:pre_seq_len * 2, :]  # 3rd and 4th vectors

            # concatenate parts to form the final tensor
            expanded_mask_pre = torch.cat((first_part, extend_zero_mask, middle_part, extend_zero_mask), dim=1)

            mask_pre = expanded_mask_pre.view(1, self.model.prefix_encoder.config.num_hidden_layers * (pre_seq_len * 2 + 2 * self.expansion_size) * self.model.prefix_encoder.n_head * self.model.prefix_encoder.n_embd)
            self.mask_pre = mask_pre

            self.mask_back = 1 - self.mask_pre
            self.mask_back = self.mask_back.view(self.model.prefix_encoder.config.num_hidden_layers,
                                                 self.model.prefix_encoder.pre_seq_len * 2,
                                                 self.model.prefix_encoder.n_head * self.model.prefix_encoder.n_embd)



    def get_free_peft(self):
        free_params = 0
        all_params = 0
        if self.mask_pre is not None:
            free_dims = 1 - self.mask_pre
            free_params += free_dims.sum()
            all_params += np.prod(self.mask_pre.size()).item()
            print('All parameters number: ', str(all_params))
            print('Free parameters number: ', str(free_params))
            print('Free parameters proportion: ',str(free_params/all_params))
            return free_params/all_params

        else:
            print('Fraction of free parameters: 1')
            return 1


    def get_free_peft(self):
        free_params = 0
        all_params = 0
        if self.mask_pre is not None:
            free_dims = 1 - self.mask_pre
            free_params += free_dims.sum()
            all_params += np.prod(self.mask_pre.size()).item()
            print('All parameters number: ', str(all_params))
            print('Free parameters number: ', str(free_params))
            print('Free parameters proportion: ',str(free_params/all_params))
            return free_params/all_params

        else:
            print('Fraction of free parameters: 1')
            return 1


    def train_task(self,
                   task=None,
                   task_id=None,
                   num_train_batch=None,
                   num_training_steps=None,
                   len_dataloader=None,
                   train_loader=None,
                   val_loader=None,
                   val_batch=None,
                   test_loader=None,
                   test_batch=None,
                   fwd_pass_results=None,
                   eval_f1=None,
                   test_f1=None,
                   eval_acc=None,
                   test_acc=None,
                   task_grow_step=0
                   ):

        free_peft_params = self.get_free_peft()
        print('There are ', free_peft_params, ' free parameters before starting training task ', task_id, ' ',
              task, 'grow step', str(task_grow_step))
        self.logger.info(f" There are  = {free_peft_params} before starting training task {task_id} {task} grow step {task_grow_step}")

        # check wether already added additional parameters. If so needs to track if improved.
        if task_grow_step > 0: # added additional parameters:
            improved_acc = False

        for epoch in range(self.num_train_epochs):
            self.model.epoch = epoch
            self.model.train()
            train_step = 0
            for batch_index in tqdm(range(num_train_batch), total=num_training_steps):

                s = (self.smax - 1 / self.smax) * train_step / len_dataloader + 1 / self.smax

                train_input, _ = self._process_data(train_loader[task], task_id)

                prompt_select_mode = 'local_compose'
                train_loss, _ = self.compute_loss(self.model, train_input, task_id, mode='train',
                                                  prompt_select_mode=prompt_select_mode, s=s)
                mask = self.model.prefix_encoder.get_mask(task_id, s=s).to('cpu')
                reg = self.get_mask_regularisation(mask)
                train_loss = train_loss + reg



                self.model.zero_grad()
                train_loss.backward()

                # Restrict layer gradients in prefix backprop
                if task_id > 0:
                    # Reshape task_masks to match the prompts shape
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # Reshape task_masks to match the prompts shape
                    self.mask_back = self.mask_back.to(device)
                    self.model.prefix_encoder.prompts.grad.data *= self.mask_back

                # Compensate embedding gradients
                num = torch.cosh(
                    torch.clamp(s * self.model.prefix_encoder.masks.weight.data, -self.thres_cosh,
                                self.thres_cosh)) + 1
                den = torch.cosh(self.model.prefix_encoder.masks.weight.data) + 1
                self.model.prefix_encoder.masks.weight.grad.data *= self.smax / s * num / den

                self.optimizer.step()
                train_step += 1

                # Constrain embeddings
                self.model.prefix_encoder.masks.weight.data = torch.clamp(self.model.prefix_encoder.masks.weight.data,
                                                                          -self.thres_emb, self.thres_emb)

                # recover the frozen previous prompts. Although the gradient of previous prompts are 0, the value will still be changes as we are using ADAM optimizer (due to the momentum)
                if self.model.config.multi_peft_modules and (
                        self.model.config.compose_prompts or self.model.config.concat_prompts):
                    try:
                        for prev_id in range(task_id):
                            with torch.no_grad():
                                self.model.prefix_encoder.prompts[prev_id] = self.prompts_saved[prev_id]
                    except:
                        print('Implementation wrong, prompts are not frozen during training')

            eval_acc_ = self.eval(val_loader[task], val_batch[task_id], task_id, mode='val')
            eval_acc.append(eval_acc_)

            if epoch == 0:
                if self.model.config.task_identify_epi:
                    self._statistics_task_mean_cov(task_id)

                if task == self.task_list[-1]:
                    if self.model.config.multi_peft_modules and self.model.config.compose_prompts and (
                            self.model.config.classifier_match_embed or self.model.config.task_identify_epi):
                        self._save_task_specific_embeds()

            best_epoch = epoch
            if (epoch > 0 or task_grow_step > 0) and self.early_stopping_patience > 0 and (
                    self.early_stopping(eval_acc_, self.model, task) or epoch == self.num_train_epochs - 1):
                best_acc_eval = self.early_stopping.best_score
                best_epoch = eval_acc.index(best_acc_eval)

                self.logger.info(f"Early stop!!! Best epoch for [{task}]: {best_epoch}")
                self.logger.info(f"Best evaluation accuracy [{task}]: {round(best_acc_eval, 4)}")

                if task_grow_step > 0 and self.early_stopping.counter == 0:
                    improved_acc = True
                # load weights of the best model
                model_path = os.path.join(self.early_stopping.save_path, f'best_checkpoint_{task}.pth')
                self.logger.info(f"Loading weights from the best model (best epoch {best_epoch})...")
                if task_grow_step > 0 and not improved_acc: # best model with less params
                    self.logger.info(f"Restore smaller model (best epoch {best_epoch})...")
                    self.restore_prompt_from_dump()
                    if task_id > 0:
                        if torch.cuda.is_available():
                            self.mask_pre = self.mask_pre_dump
                            self.mask_pre.to('cpu')
                            self.mask_back = self.mask_back_dump
                            self.mask_back.to('cuda')
                        else:
                            self.mask_pre = self.mask_pre_dump
                            self.mask_pre.to('cpu')
                            self.mask_back = self.mask_back_dump
                            self.mask_back.to('cpu')

                self.model.load_state_dict(torch.load(model_path))

                # evaluate on the test set using the best model; Foward pass test is TIL-1 with prompt_select_mode='local_compose' by default
                best_acc_test = self.eval(test_loader[task], test_batch[task_id], task_id, mode='test')

                self.logger.info(f"Best test accuracy [{task}]: {round(best_acc_test, 4)}")
                fwd_pass_results[task_id] = round(best_acc_test, 4)

                if self.model.config.multi_peft_modules and self.model.config.compose_prompts:
                    self.logger.info(
                        f"Weight distribution up to the {task_id}-th task: {self.model.prefix_encoder.attn_weights[task_id]}")

                break

            if task_grow_step > 0 and self.early_stopping.counter == 0:
                improved_acc = True

        if task_grow_step == 0 or (
                task_grow_step > 0 and improved_acc):  # first train loop, or improved after growing
            self.peft_grow_step(task_id)
            self.early_stopping.counter = 0
            self.early_stopping.early_stop = False
            task_grow_step += 1
            self.train_task(task=task, task_id=task_id,
                            num_train_batch=num_train_batch, num_training_steps=num_training_steps,
                            len_dataloader=len_dataloader, train_loader=train_loader,
                            val_loader=val_loader, val_batch=val_batch,
                            test_loader=test_loader, test_batch=test_batch,
                            fwd_pass_results=fwd_pass_results,
                            eval_f1=eval_f1,
                            test_f1=test_f1,
                            eval_acc=eval_acc,
                            test_acc=test_acc,
                            task_grow_step=task_grow_step
                            )

        else:
            if task_id != len(self.task_list) - 1:
                os.remove(model_path)



    def train(self,
              train_dataloaders, 
              val_dataloaders,
              test_dataloaders,
              ):

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        val_loader, val_batch = self._prepare_dataloaders(val_dataloaders)
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        
        fwd_pass_results = [0. for _ in range(self.num_tasks)]
        if self.args.do_train:
            for task_id, task in enumerate(self.task_list):
                num_train_batch = train_batch[task_id]
                num_training_steps = self.num_train_epochs * num_train_batch
                num_examples_dataset = train_batch[task_id] * self.args.per_device_train_batch_size
                num_examples_training = num_training_steps * self.args.per_device_train_batch_size

                len_dataloader = len(train_dataloaders[task])

                self._prepare_optimizer(task_id)

                self.model.train_loss_buffer = np.zeros([self.num_tasks, self.num_train_epochs]) # needed?
                self.model.epochs = self.num_train_epochs
                self.model.task_id = task_id
                self.early_stopping.reinit()
                eval_acc, test_acc = [], []

                # Train!
                self.logger.info(f"***** Running training - task {task_id}: {task}*****")
                self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
                self.logger.info(f"  Num examples datasets = {num_examples_dataset}")
                self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
                self.logger.info(f"  Total optimization steps = {num_training_steps}")
                self.logger.info(f"  Num examples training = {num_examples_training}")

                

                self.train_task(task=task, task_id=task_id,
                                num_train_batch=num_train_batch, num_training_steps=num_training_steps,
                                len_dataloader=len_dataloader, train_loader=train_loader,
                                val_loader=val_loader, val_batch=val_batch,
                                test_loader=test_loader, test_batch=test_batch,
                                fwd_pass_results=fwd_pass_results,
                                eval_acc=eval_acc,
                                test_acc=test_acc,
                                task_grow_step=0
                                )


                # save task prefix length
                self.model.prefix_encoder.task_pre_seq_len[task_id] = self.model.prefix_encoder.pre_seq_len

                self.logger.info(f" Prefix length is  {self.model.prefix_encoder.pre_seq_len} of  task {task_id} {task}")
                print(" Prefix length is ", str(self.model.prefix_encoder.pre_seq_len), "of task ", str(task_id))

                free_peft_params = self.get_free_peft()
                print('There are ', free_peft_params, ' free parameters After finishing training task ', task_id, ' ',
                      task)
                self.logger.info(f" There are  = {free_peft_params} After Finishing training task {task_id} {task}")


                #  update the  masks of reserved prefix dim
                mask = self.model.prefix_encoder.get_mask(task_id, s=self.smax)
                mask = torch.autograd.Variable(mask.data.clone(), requires_grad=False)
                mask = mask.to('cpu')
                if task_id == 0:
                    self.mask_pre = mask
                else:
                    self.mask_pre = torch.max(self.mask_pre, mask)

                self.mask_back = 1 - self.mask_pre
                self.mask_back = self.mask_back.view(self.model.prefix_encoder.config.num_hidden_layers,
                                    self.model.prefix_encoder.pre_seq_len * 2,
                                    self.model.prefix_encoder.n_head * self.model.prefix_encoder.n_embd)


                free_peft_params = self.get_free_peft()
                print('There are ', free_peft_params, ' free parameters After finishing training task ', task_id, ' ',
                      task)
                self.logger.info(f" There are  = {free_peft_params} After Finishing training task {task_id} {task}")




                if self.model.config.multi_peft_modules and (self.model.config.compose_prompts or self.model.config.concat_prompts):
                    if self.model.config.compose_prompts:
                        self.model.prefix_encoder.process_task_count()

                    # store task-specific prompt
                    self.prompts_saved.append(self.model.prefix_encoder.prompts[task_id].detach().clone())
                    
                    if self.model.config.compose_prompts and self.model.config.add_general_prompt:
                        self.g_prompts_saved.append(self.model.prefix_encoder.g_prompt.detach().clone())
                    
            if self.model.config.multi_peft_modules and self.model.config.compose_prompts:      
                self._save_weight_dict()
                if self.model.config.classifier_match_embed or self.model.config.task_identify_epi:
                    self._save_task_specific_embeds()
            
            
            if self.model.config.compose_prompts and self.model.config.save_prompts:
                self._save_prompts()
            
            try:
                # In the 'ComposePrompts' mode, log the compositional weights
                self.model.prefix_encoder.log_weights = True
                tb_log_dir = os.path.join(self.args.output_dir, f"tensorboard_final_eval_seed{self.seed}")
                if not os.path.exists(tb_log_dir):
                    os.mkdir(tb_log_dir)
                self.model.prefix_encoder.writer = SummaryWriter(log_dir=tb_log_dir)
                self.model.prefix_encoder.steps_val = {t_id: 0 for t_id in range(self.num_tasks)}
            except:
                pass
        
        if not self.model.config.multi_peft_modules:
            test_results_acc = []
            for t_id, t in enumerate(test_loader.keys()):
                
                self.logger.info(f"------- Final evaluation (seed-{self.seed}) on task [{t}] -------")
                test_acc = self.eval(test_loader[t], test_batch[t_id], t_id, mode='test', final=True)
                test_results_acc.append(round(test_acc, 4))
                
            self._log_final_results(test_loader, test_results_acc, fwd_pass_results, key='acc')

        else:
            test_results = {
                'forward_pass (til-1)': fwd_pass_results, # forward == til-1 when using composed prompt, otherwise final til-1 doesn't make sense
                'til': [],
            }
            self._log_final_results(test_loader, test_results, key='acc')
            
            task_identify_results = {k: [] for k in test_results.keys()}
            for t_id, t in enumerate(test_loader.keys()):
            
                if self.model.config.compose_prompts:
                    # Task-Incremental Learning (TIL), identical to forward pass results
                    prompt_select_mode = 'local_compose' if not self.model.config.composed_prompt_for_usage else 'task_id'
                    self.logger.info(f"------- Final TIL evaluation (seed-{self.seed}) on task [{t}] -------")
                    self.logger.info(f"prompt_select_mode: [{prompt_select_mode}]")
                    test_acc = self.eval(test_loader[t], test_batch[t_id], t_id, mode='test', final=True, prompt_select_mode=prompt_select_mode)
                    test_results['til'].append(round(test_acc, 4))
                    
                    # Class-Incremental Learning evaluation (CIL) based on key matching                        # CIL-top1-select-p
                    prompt_select_mode = 'local_compose' if not self.model.config.composed_prompt_for_usage else 'task_id'
                    self.logger.info(f"------- Final CIL evaluation (seed-{self.seed}) on task [{t}] -------")
                    self.logger.info(f"prompt_select_mode: [{prompt_select_mode}]")
                    test_acc = self.eval(test_loader[t], test_batch[t_id], t_id, mode='test', final=True, prompt_select_mode=prompt_select_mode, id_pred_on=True)
                    test_results['cil'].append(round(test_acc, 4))
                    
                    identify_acc = self._print_task_infer_acc()
                    task_identify_results['cil'].append(identify_acc)
            
            self._log_final_results(test_loader, test_results, key='acc', task_identify_results=task_identify_results)

        # remove checkpoints
        shutil.rmtree(self.early_stopping.save_path)
        # log key similarities between tasks, the lower(less similair), the better, means keys are orthogonal
        try:
            # In the 'ComposePrompts' mode, log the keys similarity
            self.log_keys_sim()
        except:
            pass
        
        
    def eval(self, loader, batch, task_id, mode='test', final=False, prompt_select_mode='local_compose', id_pred_on=False):

        self.model.eval()
        corr, total = 0, 0
        with torch.no_grad():
            for batch_index in tqdm(range(batch)):
                s = self.smax
                input, gt = self._process_data(loader, train=False)
                _, outputs = self.compute_loss(self.model, input, task_id, mode=mode, final=final, prompt_select_mode=prompt_select_mode, id_pred_on=id_pred_on, s=s)
                dec = [self.tokenizer.decode(ids,skip_special_tokens=True) for ids in outputs]
                targets = [self.tokenizer.decode(ids,skip_special_tokens=True) for ids in gt]

                corr += np.sum([self.normalize_text(x)==self.normalize_text(y) for x,y in zip(dec, targets)])
                total += len(targets)
        
        accuracy = corr / total
        
        if final:
            self.logger.info(f"***** Task {self.task_list[task_id]} - Final evaluation: {mode} *****")
        else:
            self.logger.info(f"***** Task {self.task_list[task_id]} - Epoch {self.model.epoch}: {mode} (seed {self.seed}) *****")
        
        self.logger.info(f"Accuracy: {round(accuracy, 4)}")

        return accuracy


    # Process string for validation (remove pad and end tokens)
    def normalize_text(self, s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the|)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            text2 = text.replace('<pad>', '').replace('</s>', '')
            exclude = set(string.punctuation)
            return "".join(ch for ch in text2 if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    
    def _log_final_results(self, test_loader, test_results, fwd_pass_results=None, key='acc', task_identify_results=None):
        self.logger.info(f"***********************************seed: {self.seed} - {key}***********************************")
        
        self.logger.info(f"Test_tasks: {test_loader.keys()}".replace('dict_keys', ''))
        if self.model.config.multi_peft_modules:
            for test_setting in test_results.keys():
                if test_results[test_setting] != [] and test_results[test_setting] != [0. for _ in range(self.num_tasks)]:
                    self.logger.info(f"Test setting [{test_setting}]: {test_results[test_setting]}")
                    self.logger.info(f"Average test_tasks_{key}: {round(sum(test_results[test_setting])/len(test_results[test_setting]), 4)}")
                
                if task_identify_results and task_identify_results[test_setting] != []:
                    self.logger.info(f"Task identify acc: {task_identify_results[test_setting]}")
                    try:
                        self.logger.info(f"Average task identify acc: {round(sum(task_identify_results[test_setting])/len(task_identify_results[test_setting]), 4)}")
                    except:
                        pass
                    
        else:
            if fwd_pass_results:
                self.logger.info(f"test_tasks_{key} (fwd_pass): {fwd_pass_results}")
                self.logger.info(f"Average test_tasks_{key} (fwd_pass): {round(sum(fwd_pass_results)/len(fwd_pass_results), 4)}")
        
            self.logger.info(f"test_tasks_{key}: {test_results}")
            self.logger.info(f"Average test_tasks_{key}: {round(sum(test_results)/len(test_results), 4)}")
            
            
    def _statistics_task_mean_cov(self, task_id):
        task_embeds = torch.tensor(self.model.task_embeds[task_id])
        task_labels = torch.tensor(self.model.task_labels[task_id])
        labels_space = torch.unique(task_labels[:, 0])
        
        task_mean = task_embeds.mean(dim=0)
        task_cov = torch.cov((task_embeds - task_mean).T)
        
        mean_over_classes = []
        cov_over_classes = []
        for l in labels_space:
            embeds_l = task_embeds[task_labels[:, 0]==l]
            if embeds_l.numel() > 0 and embeds_l.shape[0] > 1:
                mean_l = embeds_l.mean(dim=0) 
                cov_l = torch.cov((embeds_l - mean_l).T)
            else:
                mean_l = task_mean
                cov_l = task_cov
                
            mean_over_classes.append(mean_l)
            cov_over_classes.append(cov_l)

        mean_over_classes = torch.stack(mean_over_classes) # N_class x dim
        shared_cov = torch.stack(cov_over_classes).mean(dim=0) # dim x dim (averaged over N_class)
        
        self.model.task_means_over_classes.append(nn.Parameter(mean_over_classes.cuda(), requires_grad=False))
        self.model.accumulate_shared_covs.data = self.model.accumulate_shared_covs.data.cpu()
        self.model.accumulate_shared_covs += shared_cov
        
        self.model.cov_inv = nn.Parameter(torch.linalg.pinv(
            self.model.accumulate_shared_covs / (task_id+1), hermitian=True), requires_grad=False) # E: Computes the pseudoinverse (Moore-Penrose inverse) 
        
        
    def _print_task_infer_acc(self, keyword="key_match"):
        avg_infer_acc = round(sum(self.model.task_infer_acc)/len(self.model.task_infer_acc), 4)
        self.logger.info(f"Task inference acc ({keyword}): {avg_infer_acc}")
        self.model.task_infer_acc = []
        self.model.prefix_encoder.steps_final = {t_id: 0 for t_id in range(self.num_tasks)}
        return avg_infer_acc


    def log_keys_sim(self):
        writer = self.model.prefix_encoder.writer
        K = self.model.prefix_encoder.keys
        K_sim = torch.empty((self.num_tasks, self.num_tasks))
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                K_sim[i][j] = torch.cosine_similarity(K[i][None], K[j][None])
                writer.add_scalar(f'keys_similarity/task_{i}', K_sim[i][j], j)
    
    
    def _save_prompts(self, initial=False):
        prompts_save_path = os.path.join(self.args.output_dir, f"initial_prompts_seed{self.seed}") if initial \
            else os.path.join(self.args.output_dir, f"prompts_seed{self.seed}")
        if not os.path.exists(prompts_save_path):
                os.mkdir(prompts_save_path)
                
        for ti, task in enumerate(self.task_list):
            torch.save(self.model.prefix_encoder.keys[ti].detach().clone(), f'{prompts_save_path}/{task}_key.pt')
            torch.save(self.model.prefix_encoder.prompts[ti].detach().clone(), f'{prompts_save_path}/{task}_prompt.pt')
        print(f'Task specific keys and prompts saved.')
        
        if self.model.config.add_general_prompt:
            # save general prompt
            if initial:
                torch.save(self.model.prefix_encoder.g_prompt.detach().clone(), f'{prompts_save_path}/general_prompt.pt')
            else:
                torch.save(torch.stack(self.g_prompts_saved), f'{prompts_save_path}/general_prompt.pt')
            print(f'General g_prompts saved.')
        
    
    def _save_weight_dict(self, weight_matrix=None, valid_only=False):
        """
        Convert the list of weight tensors into a dictionary with a nested structure and save it as a JSON file.
        """
        def tensor_to_value(tensor):
            """Convert a tensor to its real value."""
            real_value = tensor.item() if isinstance(tensor, torch.Tensor) else tensor
            return round(real_value, 4)
        
        weight_list = weight_matrix if weight_matrix is not None else self.model.prefix_encoder.attn_weights
        # Ensure the lists are of the same length
        assert len(weight_list) == len(self.task_list)
        
        # Create the nested dictionary with tensors converted to lists
        weight_dict = {}
        for i, task in enumerate(self.task_list):
            weight_dict[task] = {self.task_list[j]: tensor_to_value(weight_list[i][j]) for j in range(len(weight_list[i])) if j<=i}
        
        output_path = os.path.join(self.args.output_dir, "weight_dict_")
        output_path += "all.json" if not valid_only else "valid_only.json"
        
        with open(output_path, 'w') as f:
            json.dump(weight_dict, f)
            
        return weight_dict
    
    
    def _save_task_specific_embeds(self):
        save_dir = os.path.join(self.args.output_dir, "task_embeds")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        if self.model.config.task_identify_epi and self.model.config.save_embed_statistics:
            for ti, task in enumerate(self.task_list):
                task_embeds = torch.tensor(self.model.task_embeds[ti])
                task_mean = task_embeds.mean(dim=0).detach().clone()
                task_cov = torch.cov((task_embeds - task_mean).T)

                torch.save(task_mean, os.path.join(save_dir, f"{task}_task_mean.pt"))
                torch.save(task_cov, os.path.join(save_dir, f"{task}_task_cov.pt"))
        
        if self.model.config.classifier_match_embed and self.model.config.save_embed_prototypes:
            for ti, task in enumerate(self.task_list):
                embed = self.model.embed_prototypes[ti].detach().clone()
                
                torch.save(embed, os.path.join(save_dir, f"{task}_task_embed.pt"))
            
                
                
                