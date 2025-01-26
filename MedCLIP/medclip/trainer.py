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
