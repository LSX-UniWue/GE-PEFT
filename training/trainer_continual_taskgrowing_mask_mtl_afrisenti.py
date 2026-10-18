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


from datasets.load import load_metric
from transformers import EvalPrediction
from transformers.trainer_pt_utils import get_parameter_names
from training.early_stopping import EarlyStopping


class ContinualTrainerMTL(nn.Module):
    def __init__(self,
                 args,
                 model,
                 logger,
                 task_list,
                 label_list, 
                 early_stopping_patience=-1,
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

        self.num_train_epochs = math.ceil(args.num_train_epochs)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = EarlyStopping(
            save_path=os.path.join(args.output_dir, 'best_checkpoint'),
            logger=self.logger,
            patience = early_stopping_patience
        )
        
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

        print('MASKED BERT AfriSenti TRAINER')
    
        
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
        
        # if self.args.optim == OptimizerNames.ADAMW_HF:
        from torch.optim import AdamW

        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)

        # Initilaize optimizer
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


    def _process_data(self, loader, task_id=0):
        try:
            data_batch = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data_batch = next(loader[1])
        try:
            inputs = {"input_ids": data_batch["input_ids"].to(self.args.device), 
                        "attention_mask": data_batch["attention_mask"].to(self.args.device), 
                        "token_type_ids": data_batch["token_type_ids"].to(self.args.device),
                        "labels": data_batch["labels"].to(self.args.device)}
        except:
            inputs = {"input_ids": data_batch["input_ids"].to(self.args.device), 
                        "attention_mask": data_batch["attention_mask"].to(self.args.device), 
                        "labels": data_batch["labels"].to(self.args.device)}
        
        label = data_batch["labels"].to(self.args.device)
        
        return inputs, label

    
    def compute_loss(self, model, inputs, task_id, mode, final=False, prompt_select_mode=None, classifier_select_mode='task_id', id_pred_on=False, s=1):
        if self.model.config.multi_peft_modules:
            prompt_select_mode = 'local_compose' if prompt_select_mode is None else prompt_select_mode
            
            inputs['task_id'] = task_id
            inputs['train'] = True if mode == 'train' else False
            inputs['final'] = final
            inputs['prompt_select_mode'] = prompt_select_mode
            inputs['classifier_select_mode'] = classifier_select_mode
            inputs['id_pred_on'] = id_pred_on
            inputs['s'] = s
            outputs, labels = model(**inputs)
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return loss, outputs, labels
        else:
            inputs['return_dict'] = True
            outputs = model(**inputs)
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
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



    def get_free_peft(self):
        free_params = 0
        all_params = 0
        if self.mask_pre is not None:
            free_dims = 1 - self.mask_pre
            free_params += free_dims.sum()
            all_params += np.prod(self.mask_pre.size()).item()
            return free_params/all_params

        else:
            return 1


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
                   fwd_pass_results_f1=None,
                   fwd_pass_results_acc=None,
                  eval_f1=None,
                  test_f1=None,
                  eval_acc=None,
                  test_acc=None,
                  task_grow_step=0
    ):

        # check wether already added additional parameters. If so needs to track if improved.
        if task_grow_step > 0: # added additional parameters:
            improved_acc = False


        for epoch in range(self.num_train_epochs):
            self.model.epoch = epoch
            self.model.train()
            train_preds, train_gts = [], []
            train_step = 0


            for batch_index in tqdm(range(num_train_batch), total=num_training_steps):


                s = (self.smax - 1 / self.smax) * train_step / len_dataloader + 1 / self.smax

                train_input, train_gt = self._process_data(train_loader[task], task_id)

                train_loss, train_pred = self.compute_loss(self.model, train_input, task_id, mode='train', s=s)[:2]
                mask = self.model.prefix_encoder.get_mask(task_id, s=s).to('cpu')
                reg = self.get_mask_regularisation(mask)
                train_loss = train_loss + reg

                train_preds.append(train_pred)
                train_gts.append(train_gt)

                self.model.zero_grad()
                train_loss.backward()

                # Restrict layer gradients in prefix backprop
                if task_id > 0:
                    # Reshape task_masks to match the prompts shape
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.mask_back = self.mask_back.to(device)
                    self.model.prefix_encoder.prompts.grad.data *= self.mask_back

                # Compensate embedding gradients
                num = torch.cosh(
                    torch.clamp(s * self.model.prefix_encoder.masks.weight.data, -self.thres_cosh, self.thres_cosh)) + 1
                den = torch.cosh(self.model.prefix_encoder.masks.weight.data) + 1
                self.model.prefix_encoder.masks.weight.grad.data *= self.smax / s * num / den

                self.optimizer.step()
                train_step += 1

                # Constrain embeddings
                self.model.prefix_encoder.masks.weight.data = torch.clamp(
                    self.model.prefix_encoder.masks.weight.data, -self.thres_emb, self.thres_emb)

                # recover the frozen previous prompts. Although the gradient of previous prompts are 0, the value will still be changes as we are using ADAM optimizer (due to the momentum)
                if self.model.config.multi_peft_modules and (
                        self.model.config.compose_prompts or self.model.config.concat_prompts):
                    try:
                        for prev_id in range(task_id):
                            with torch.no_grad():
                                self.model.prefix_encoder.prompts[prev_id] = self.prompts_saved[prev_id]
                    except:
                        print('Implementation wrong, prompts are not frozen during training')


            preds = np.concatenate(np.array(
                [pred.logits.view(-1, pred.logits.shape[-1]).detach().to("cpu").numpy() for pred in train_preds],
                dtype=object), axis=0)
            label_ids = np.concatenate(
                np.array([label_id.view(-1).detach().to("cpu").numpy() for label_id in train_gts], dtype=object),
                axis=0)
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            # self._display_results(metrics, task_id, mode='train')

            eval_acc_, eval_f1_ = self.eval(val_loader[task], val_batch[task_id], task_id, mode='val')
            eval_acc.append(eval_acc_)
            eval_f1.append(eval_f1_)

            if epoch == 0:
                if self.model.config.task_identify_epi and not self.model.config.epi_for_composition:  # did the statistics already!
                    self._statistics_task_mean_cov(task_id)

                if task == self.task_list[-1]:
                    if self.model.config.multi_peft_modules and self.model.config.compose_prompts and (
                            self.model.config.classifier_match_embed or self.model.config.task_identify_epi):
                        self._save_task_specific_embeds()

            best_epoch = epoch
            if self.early_stopping_patience > 0 and (
                    self.early_stopping(eval_f1_, self.model, task) or epoch == self.num_train_epochs - 1):
                best_f1_eval = self.early_stopping.best_score
                best_epoch = eval_f1.index(best_f1_eval)
                best_acc_eval = eval_acc[best_epoch]

                self.logger.info(f"Early stop!!! Best epoch for [{task}]: {best_epoch}")
                self.logger.info(f"Best evaluation accuracy [{task}]: {round(best_acc_eval, 4)}")
                self.logger.info(f"Best evaluation F1 [{task}]: {round(best_f1_eval, 4)}")

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

                # evaluate on the test set using the best model; Foward pass test is TIL-1 with prompt_select_mode='local_compose' and classifier_select_mode='task_id' by default (except during warmup epochs)
                best_acc_test, best_f1_test = self.eval(test_loader[task], test_batch[task_id], task_id, mode='test')

                self.logger.info(f"Best test accuracy [{task}]: {round(best_acc_test, 4)}")
                self.logger.info(f"Best test F1 [{task}]: {round(best_f1_test, 4)}")
                fwd_pass_results_acc[task_id] = round(best_acc_test, 4)
                fwd_pass_results_f1[task_id] = round(best_f1_test, 4)

                break
            if task_grow_step > 0 and self.early_stopping.counter == 0:
                improved_acc = True

        if task_grow_step == 0 or (task_grow_step > 0 and improved_acc): # first train loop, or improved after growing
            self.peft_grow_step(task_id)
            self.early_stopping.counter = 0
            self.early_stopping.early_stop = False
            task_grow_step+=1
            self.train_task(task=task, task_id=task_id,
                            num_train_batch=num_train_batch, num_training_steps=num_training_steps,
                            len_dataloader=len_dataloader, train_loader=train_loader,
                            val_loader=val_loader, val_batch=val_batch,
                            test_loader=test_loader, test_batch=test_batch,
                            fwd_pass_results_f1=fwd_pass_results_f1,
                            fwd_pass_results_acc=fwd_pass_results_acc,
                            eval_f1=eval_f1,
                            test_f1=test_f1,
                            eval_acc=eval_acc,
                            test_acc=test_acc,
                            task_grow_step=task_grow_step
                            )
        else:
            # no improvement
            if task_id != len(self.task_list) - 1: # remove only when all finished
                os.remove(model_path)







    def train(self,
              train_dataloaders, 
              val_dataloaders,
              test_dataloaders,
              ):

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        val_loader, val_batch = self._prepare_dataloaders(val_dataloaders)
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        
        fwd_pass_results_f1 = [0. for _ in range(self.num_tasks)]
        fwd_pass_results_acc = [0. for _ in range(self.num_tasks)]
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
                eval_f1, test_f1 = [], []
                eval_acc, test_acc = [], []

                # Train!
                self.logger.info(f"***** Running training - task {task_id}: {task}*****")
                self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
                self.logger.info(f"  Num examples datasets = {num_examples_dataset}")
                self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
                self.logger.info(f"  Total optimization steps = {num_training_steps}")
                self.logger.info(f"  Num examples training = {num_examples_training}")


                if self.model.config.epi_for_composition or self.model.config.embed_prototypes_for_composition and self.model.config.compose_prompts:
                    # get epi task statistics / task embedding prototypes
                    for batch_index in range(num_train_batch):
                        train_input, train_gt = self._process_data(train_loader[task], task_id)
                        input_embed = self.model.query_encoder(train_input["input_ids"], attention_mask=train_input["attention_mask"])[0]
                        self.model.process_query(input_embed, train_input["attention_mask"], True, task_id, train_gt)

                    if self.model.config.epi_for_composition:
                        assert self.model.config.task_identify_epi
                        self._statistics_task_mean_cov(task_id)

                        self.model.prefix_encoder.task_means_over_classes = self.model.task_means_over_classes
                        self.model.prefix_encoder.cov_inv = self.model.cov_inv

                    elif self.model.config.embed_prototypes_for_composition:
                        assert self.model.config.classifier_match_embed
                        self.model.embed_prototypes[task_id] = torch.tensor(self.model.embed_prototypes[task_id]).mean(dim=0)
                        self.model.prefix_encoder.embed_prototypes = self.model.embed_prototypes




                self.train_task(task=task, task_id=task_id,
                                num_train_batch=num_train_batch, num_training_steps=num_training_steps,
                                len_dataloader=len_dataloader, train_loader=train_loader,
                                val_loader=val_loader, val_batch=val_batch,
                                test_loader=test_loader, test_batch=test_batch,
                                fwd_pass_results_f1=fwd_pass_results_f1,
                                fwd_pass_results_acc=fwd_pass_results_acc,
                                eval_f1=eval_f1,
                                test_f1=test_f1,
                                eval_acc=eval_acc,
                                test_acc=test_acc,
                                task_grow_step=0
                                )




                # save task prefix length
                self.model.prefix_encoder.task_pre_seq_len[task_id] = self.model.prefix_encoder.pre_seq_len
                    

                # - update the  masks of reserved prefix dim
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
                
            self._log_final_results(test_loader, test_results_acc, fwd_pass_results_acc, key='acc')

        else:
            test_results = {
                'forward_pass (til-1)': fwd_pass_results_f1, # forward == til when using composed prompt, otherwise final til-1 doesn't make sense
                'til': [], 
            }
            for t_id, t in enumerate(test_loader.keys()):
                if self.model.config.compose_prompts:
                    # Task-Incremental Learning (TIL), identical to forward pass results
                    prompt_select_mode = 'local_compose' if not self.model.config.composed_prompt_for_usage else 'task_id'
                    classifier_select_mode = 'task_id'
                    self.logger.info(f"------- Final TIL evaluation (seed-{self.seed}) on task [{t}] -------")
                    self.logger.info(f"prompt_select_mode: [{prompt_select_mode}] & classifier_select_mode [{classifier_select_mode}]")
                    _, test_f1 = self.eval(test_loader[t], test_batch[t_id], t_id, mode='test', final=True, prompt_select_mode=prompt_select_mode, classifier_select_mode=classifier_select_mode)
                    test_results['til'].append(round(test_f1, 4))

            self._log_final_results(test_loader, test_results, key='f1')

        # remove checkpoints
        shutil.rmtree(self.early_stopping.save_path)
        # log key similarities between tasks, the lower(less similair), the better, means keys are orthogonal
        try:
            # In the 'ComposePrompts' mode, log the keys similarity
            self.log_keys_sim()
        except:
            pass
        
        
    def eval(self, loader, batch, task_id, mode='test', final=False, prompt_select_mode='local_compose', classifier_select_mode='task_id', id_pred_on=False):

        self.model.eval()
        preds, gts = [], []
        # cc = 0
        with torch.no_grad():
            for batch_index in tqdm(range(batch)):

                s = self.smax
                input, gt = self._process_data(loader)
                
                if self.model.config.multi_peft_modules:
                    loss, pred, label = self.compute_loss(self.model, input, task_id, mode=mode, final=final, prompt_select_mode=prompt_select_mode, classifier_select_mode=classifier_select_mode, id_pred_on=id_pred_on,s=s)
                    gts.append(label)
                else:
                    loss, pred = self.compute_loss(self.model, input, task_id, mode=mode, final=final, s=s)
                    gts.append(gt)
                
                preds.append(pred)
                
                
        preds = np.concatenate(np.array([pred.logits.view(-1, pred.logits.shape[-1]).detach().to("cpu").numpy() for pred in preds], dtype=object), axis=0)
        label_ids = np.concatenate(np.array([label_id.view(-1).detach().to("cpu").numpy() for label_id in gts], dtype=object), axis=0)    
        metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        self._display_results(metrics, task_id, mode, final)
        
        return metrics["accuracy"], metrics['f1_weighted']


    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        f1_metric = load_metric("f1", experiment_id=f"{self.args.output_dir.split('/')[-1]}")

        accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
        f1_weighted = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"]
        f1_macro = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]

        return {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            
        }
        
        
    def _display_results(self, metrics, task_id, mode, final=False):
        if final:
            self.logger.info(f"***** Task {self.task_list[task_id]} - Final evaluation: {mode} *****")
        else:
            self.logger.info(f"***** Task {self.task_list[task_id]} - Epoch {self.model.epoch}: {mode} (seed {self.seed}) *****")
        
        accuracy = metrics["accuracy"]
        f1_weighted = metrics["f1_weighted"]
        f1_macro = metrics["f1_macro"]
        
        self.logger.info(f"Accuracy: {round(accuracy, 4)}")
        self.logger.info(f"F1 weighted: {round(f1_weighted, 4)}")
        self.logger.info(f"F1 macro: {round(f1_macro, 4)}")
    
    
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
                    
        else:
            if fwd_pass_results:
                self.logger.info(f"test_tasks_{key} (fwd_pass): {fwd_pass_results}")
                self.logger.info(f"Average test_tasks_{key} (fwd_pass): {round(sum(fwd_pass_results)/len(fwd_pass_results), 4)}")
        
            self.logger.info(f"test_tasks_{key}: {test_results}")
            self.logger.info(f"Average test_tasks_{key}: {round(sum(test_results)/len(test_results), 4)}")
            
            
    def _statistics_task_mean_cov(self, task_id):
        task_embeds = torch.tensor(self.model.task_embeds[task_id])
        task_labels = torch.tensor(self.model.task_labels[task_id])
        labels_space = torch.unique(task_labels)
        
        task_mean = task_embeds.mean(dim=0)
        task_cov = torch.cov((task_embeds - task_mean).T)
        
        mean_over_classes = []
        cov_over_classes = []
        for l in labels_space:
            embeds_l = task_embeds[task_labels==l]
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
            
                
                
                