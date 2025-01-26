import os
import json
import pdb
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict
import math

import numpy as np
import torch
from torch import nn
from torch import device, Tensor
from tqdm.autonotebook import trange
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import distributed as dist
import matplotlib.pyplot as plt
import transformers

WEIGHTS_NAME = "pytorch_model.bin"

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        #print("Trainer is used")
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        pass


    def train(self,
              model,
              train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
              eval_dataloader=None,
              evaluator=None,
              epochs: int = 1,
              steps_per_epoch=None,
              scheduler: str = 'WarmupCosine',
              warmup_steps: int = 10000,
              warmup_ratio: float = 0.01,
              optimizer_class: Type[Optimizer] = torch.optim.AdamW,
              optimizer_params: Dict[str, object] = {'lr': 2e-5},
              weight_decay: float = 0.01,
              evaluation_steps: int = 100,
              save_steps: int = 100,
              output_path: str = None,
              save_best_model: bool = True,
              max_grad_norm: float = 1,
              use_amp: bool = False,
              accumulation_steps: int = 1,
              callback: Callable[[float, int, int], None] = None,
              show_progress_bar: bool = True,
              checkpoint_path: str = None,
              checkpoint_save_total_limit: int = 0,
              load_best_model_at_last: bool = True):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.best_score = -9999999
        self.accumulation_steps = accumulation_steps
        self.evaluator = evaluator
        self.eval_dataloader = eval_dataloader

        dataloaders = [dataloader for dataloader, _, _ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio)

        loss_models = [loss for _, loss, _ in train_objectives]
        train_weights = [weight for _, _, weight in train_objectives]

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        # Use CPU
        device = torch.device("cpu")
        model.to(device)
        for loss_model in loss_models:
            loss_model.to(device)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        train_loss_dict = defaultdict(list)
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            #print(f"Epoch {epoch + 1}, Classifier bias: {loss_model.model.fc.bias}")

            epoch_train_losses = []
            epoch_train_correct = 0
            epoch_train_total = 0

            for train_iter in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(len(train_objectives)):
                    loss_model = loss_models[train_idx]
                    loss_model.zero_grad()
                    loss_model.train()

                    loss_weight = train_weights[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    if data is None:
                        print("Skipping empty batch")
                        continue

                    for key in data:
                        data[key] = data[key].to(device)

                    # Forward pass and loss computation
                    outputs = loss_model(pixel_values=data['pixel_values'], labels=data['labels'])
                    loss_value = loss_weight * outputs['loss_value'] / self.accumulation_steps
                    loss_value.backward()

                    # Gradient clipping and optimizer step
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_train_losses.append(loss_value.item())
                    predictions = (torch.sigmoid(outputs['logits']) > 0.5).float()
                    if predictions.shape != data['labels'].shape:
                        predictions = predictions.view_as(data['labels'])

                    epoch_train_correct += (predictions == data['labels']).sum().item()
                    epoch_train_total += data['labels'].numel()


                    #print(f"Logits: {outputs['logits'][:5]}")
                    #print(f"Predictions: {predictions[:5]}")
                    #print(f"Labels: {data['labels'][:5]}")

                scheduler.step()
                global_step += 1

                # Evaluate at specified steps
                if evaluation_steps > 0 and global_step % evaluation_steps == 0 and self.evaluator is not None:
                    scores = self.evaluator.evaluate()
                    print(f"\n######### Eval {global_step} #########")
                    print(f"TP: {scores['tp']}, TN: {scores['tn']}, FP: {scores['fp']}, FN: {scores['fn']}")
                    print(f"Accuracy: {scores['accuracy']:.4f}, Recall: {scores['recall']:.4f}")

            # Log training metrics
            epoch_train_loss = np.mean(epoch_train_losses)
            #print("epoch_train_correct: ", epoch_train_correct)
            #print("epoch_train_total: ", epoch_train_total)
            epoch_train_accuracy = epoch_train_correct / epoch_train_total
            #print("epoch_train_accuracy: ", epoch_train_accuracy)

            self.train_losses.append(epoch_train_loss)
            self.train_accuracies.append(epoch_train_accuracy)

            # Evaluate at the end of the epoch
            if evaluator is not None:
                eval_scores = evaluator.evaluate()
                self.val_losses.append(eval_scores['val_loss'])
                self.val_accuracies.append(eval_scores['accuracy'])

            #print(f"Epoch {epoch + 1} - Training Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_accuracy:.4f}")
            #if evaluator:
            #    print(f"Validation Loss: {eval_scores['val_loss']:.4f}, Accuracy: {eval_scores['accuracy']:.4f}")

            # Save checkpoint after each epoch
            epoch_save_dir = os.path.join(output_path, f'epoch_{epoch + 1}')
            os.makedirs(epoch_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(epoch_save_dir, 'pytorch_model.bin'))
            print(f"Epoch {epoch + 1} checkpoint saved to {epoch_save_dir}\n")

        # Save the final model
        if output_path is not None:
            final_model_path = os.path.join(output_path, 'final_model.bin')
            torch.save(model.state_dict(), final_model_path)
            print(f"\nFinal model saved to {final_model_path}")

        # Save training and validation plots
        self.save_plots(output_path)

    def save_plots(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        # Loss plot
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_path, 'loss_plot.png'))
        plt.close()

        # Accuracy plot
        plt.figure()
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_path, 'accuracy_plot.png'))
        plt.close()

        print("Training and validation plots saved successfully.")

    """
    #trainer before saving each epoch and final model
    def train(self,
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        eval_dataloader = None,
        evaluator=None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 100,
        save_steps : int = 100,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_total_limit: int = 0,
        load_best_model_at_last: bool = True,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.best_score = -9999999
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.score_logs = defaultdict(list)
        self.evaluator = evaluator
        self.eval_dataloader = eval_dataloader

        dataloaders = [dataloader for dataloader,_,_ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        loss_models = [loss for _, loss,_ in train_objectives]
        train_weights = [weight for _,_,weight in train_objectives]

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        # map models to devices
        #model = model.cuda()
        device = torch.device("cpu") #changed to CPU
        #print("Using device:", device) 
        model.to(device)
        for loss_model in loss_models:
            loss_model.to(device)

        # execute training on multiple GPUs
        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)
        num_train_objectives = 1 #added

        skip_scheduler = False
        train_loss_dict = defaultdict(list)
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            print(f"Epoch {epoch + 1}, Classifier bias: {model.fc.bias}")
            training_steps = 0
            for train_iter in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):

                # check if model parameters keep same
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    loss_model.zero_grad()
                    loss_model.train()

                    loss_weight = train_weights[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        # for train_idx in range(num_train_objectives):
                        if '_build_prompt_sentence' in dir(dataloaders[train_idx].dataset):
                            dataloaders[train_idx].dataset._build_prompt_sentence()
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    for key in data:
                        data[key] = data[key].to(device)

                    if use_amp:
                        with autocast():
                            loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return['loss_value']
                        loss_value = loss_value
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        #print("Loss model signature:", loss_model.forward.__annotations__)
                        #loss_model_return = loss_model(**data) #sandra auskommentiert

                        #print("Passing data to loss_model...")
                        ##new
                        #outputs = loss_model(
                        #    pixel_values=data['pixel_values'], labels=data['labels']
                        #)
                        #print("Inside SuperviseClassifier")
                        #print("Data passed to loss_model, pixel_values shape:", data['pixel_values'].shape if data['pixel_values'] is not None else "None")

                        outputs = loss_model.forward(
                            pixel_values=data['pixel_values'], labels=data['labels']
                        )
                        #print("Input pixel_values shape:", pixel_values.shape)

                        # Extract logits and loss
                        logits = outputs['logits']  # Shape: [batch_size, 1]
                        loss_value = outputs['loss_value'] if 'loss_value' in outputs else None

                        # Extract logits and loss (if available)
                        logits = outputs['logits']
                        #print("Output logits shape:", logits.shape)

                        loss_value = outputs['loss_value'] if 'loss_value' in outputs else None

                        # Compute the loss using the logits and labels
                        if len(data['labels'].shape) == 1:  # If labels are [batch_size], reshape to [batch_size, 1]
                            data['labels'] = data['labels'].view(-1, 1).float()

                        #print(f"Logits shape: {logits.shape}, Labels shape: {data['labels'].shape}")

                        # Compute loss
                        loss_model_return = {'loss_value': loss_model.loss_fn(logits, data['labels'])}
                        #loss_model_return = loss_model(input=data['pixel_values'], target=data['labels']) #sandra neew
                        
                        loss_value = loss_weight * loss_model_return['loss_value'] / self.accumulation_steps
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    train_loss_dict[train_idx].append(loss_value.item())
                    optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps>0 and global_step % evaluation_steps == 0:
                    print('\n######### Train Loss #########')
                    for key in train_loss_dict.keys():
                        print('{} {:.4f} \n'.format(key, np.mean(train_loss_dict[key])))
                    train_loss_dict = defaultdict(list)

                    #TODO: update prompt sentences
                    # for train_idx in range(num_train_objectives):
                    #     if '_build_prompt_sentence' in dir(dataloaders[train_idx].dataset):
                    #         dataloaders[train_idx].dataset._build_prompt_sentence()

                if evaluation_steps > 0 and global_step % evaluation_steps == 0 and self.evaluator is not None:
                    scores = self.evaluator.evaluate()
                    print(f'\n######### Eval {global_step} #########')
                    print(f"TP: {scores['tp']}, TN: {scores['tn']}, FP: {scores['fp']}, FN: {scores['fn']}")
                    print(f"Accuracy: {scores['accuracy']:.4f}, Recall: {scores['recall']:.4f}")

                
                #commented out from original version
                if evaluation_steps > 0 and global_step % evaluation_steps == 0 and self.evaluator is not None:
                    scores = self.evaluator.evaluate()
                    print(f'\n######### Eval {global_step} #########')
                    for key in scores.keys():
                        if key in ['acc', 'auc']:
                            if scores[key] is not None:
                                print('{}: {:.4f}'.format(key, scores[key]))
                            else:
                                print(f'{key}: Not computed')
                    save_dir =  os.path.join(output_path, f'{global_step}/')
                    self._save_ckpt(model, save_dir)

                    # score logs save the list of scores
                    self.score_logs['global_step'].append(global_step)
                    for key in scores.keys():
                        if key in ['acc','auc']:
                            if scores[key] is not None:
                                self.score_logs[key].append(scores[key])
                            else:
                                print(f'{key}: Not saved in logs')
                #---

                if self.evaluator is None and global_step % save_steps == 0:
                    state_dict = model.state_dict()
                    save_dir =  os.path.join(output_path, f'{global_step}/')
                    self._save_ckpt(model, save_dir)
                    print('model saved to', os.path.join(output_path, WEIGHTS_NAME))

        if save_best_model:
            import pandas as pd
            from distutils.dir_util import copy_tree
            res = pd.DataFrame(self.score_logs)

            # Check if the DataFrame is empty
            if res.empty:
                print("Skipping model save. The DataFrame is empty.")
                save_best_model = None
            else:
                res = res.set_index('global_step')
                # Take the average column best
                best_iter = res.mean(1).idxmax()
                best_save_path = os.path.join(output_path, './best')
                if not os.path.exists(best_save_path): 
                    os.makedirs(best_save_path)
                best_origin_path = os.path.join(output_path, f'./{best_iter}')
                print(f'save best checkpoint at iter {best_iter} to', best_save_path)
                copy_tree(best_origin_path, best_save_path)

        if eval_dataloader is None and output_path is not None:   #No evaluator, but output path: save final model version
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_path, WEIGHTS_NAME))
            print('model saved to', os.path.join(output_path, WEIGHTS_NAME))

        if eval_dataloader is not None and load_best_model_at_last and save_best_model and evaluator is not None:
            state_dict = torch.load(os.path.join(best_save_path, WEIGHTS_NAME), map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f'load best checkpoint at last from {best_save_path}')
        """

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, WEIGHTS_NAME))
