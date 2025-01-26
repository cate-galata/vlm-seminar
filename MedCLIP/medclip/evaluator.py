import pdb

import pandas as pd
import numpy as np
from sklearn import multiclass
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

from tqdm import tqdm
from sklearn.exceptions import UndefinedMetricWarning

from . import constants

class Evaluator:
    '''do evaluation on chexpert5x200 zero-shot classification
    '''
    def __init__(self,
        medclip_clf,
        eval_dataloader=None,
        mode=None,
        ) -> None:
        '''specify class_names if doing zero-shot classification.
        mode: `binary`, 'multiclass`, or `multilabel`,
        if set None, the method will automatically decide from data.
        recommend to set explicitly to avoid errors.
        '''
        self.clf = medclip_clf
        self.mode = mode
        self.eval_dataloader = eval_dataloader

    def evaluate(self, eval_dataloader=None):
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
        else:
            eval_dataloader = self.eval_dataloader

        pred_list = []
        label_list = []
        total_loss = 0.0
        total_batches = 0
        total_ones, total_zeros = 0, 0

        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)  # Forward pass
                pred = outputs['logits']  # Predictions
                loss = outputs.get('loss_value', None)  # Loss if available

            # Sum up the loss for all batches
            if loss is not None:
                total_loss += loss.item()
                total_batches += 1

            labels = data['labels'].cpu().numpy()
            pred_list.append(pred)
            label_list.append(data['labels'])

            # Count the number of 1's and 0's in the batch labels
            total_ones += (labels == 1).sum()
            total_zeros += (labels == 0).sum()

        # Print the total count after the loop
        print(f"Total labels in evaluation/testing dataset: {total_ones} 1's, {total_zeros} 0's")
        #print(f"Unique labels: {np.unique(labels, return_counts=True)}")

        # Calculate metrics
        pred_list = torch.cat(pred_list, 0)
        labels = torch.cat(label_list, 0).cpu().detach().numpy()
        pred_scores = torch.sigmoid(pred_list).cpu().detach().numpy()
        pred_labels = (pred_scores > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate average validation loss
        avg_val_loss = total_loss / total_batches if total_batches > 0 else None

        print("\nConfusion Matrix:")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        if avg_val_loss is not None:
            print(f"Validation Loss: {avg_val_loss:.4f}")

        # Return metrics
        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "recall": recall,
            "val_loss": avg_val_loss,  # Include validation loss
        }
    """
    #original functoin from original medclip repo
    def evaluate(self, eval_dataloader=None):
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                pred = outputs['logits']
            pred_list.append(pred)
            label_list.append(data['labels'])
        
        pred_list = torch.cat(pred_list, 0)
        labels = torch.cat(label_list, 0).cpu().detach().numpy()

        pred = pred_list.cpu().detach().numpy()        
        outputs = {'pred':pred, 'labels':labels}

        if self.mode is None:
            if len(labels.shape) == 1:
                if len(np.unique(labels)) == 2:
                    self.mode = 'binary'
                else:
                    self.mode = 'multiclass'
            else:
                self.mode = 'multilabel'
            print(f'no mode specified, will pick mode `{self.mode}` by data.')

        if self.mode == 'binary':
            if pred.shape[1] == 1:
                pred_score = torch.tensor(pred).sigmoid().numpy().flatten()
                #auc = roc_auc_score(labels, pred_score)
                print(f"Labels shape: {labels.shape}, Unique labels: {np.unique(labels)}")
                print(f"Predicted scores shape: {pred_score.shape}")
                try:
                    auc = roc_auc_score(labels, pred_score)
                except ValueError as e:
                    print(f"ROC AUC computation skipped: {e}")
                    auc = None
                outputs['auc'] = auc
                pred_label = np.ones(len(pred))
                pred_label[pred_score<0.5] = 0
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc
                

            else: # have 2 outputs
                pred_score = torch.tensor(pred).sigmoid().numpy()
                pred_label = np.argmax(pred_score, 1)
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc

                # cnf_matrix = confusion_matrix(labels, pred_label)
                # res = self.process_confusion_matrix(cnf_matrix)
                # outputs.update(res)

            res = classification_report(labels, pred_label, output_dict=True)
            res = res['macro avg']
            res.pop('support')
            outputs.update(res)

        if self.mode == 'multiclass':
            pred_label = pred.argmax(1)
            acc = (pred_label == labels).mean()
            outputs['acc'] = acc
            res = classification_report(labels, pred_label, output_dict=True)
            res = res['macro avg']
            res.pop('support')
            outputs.update(res)

            # cnf_matrix = confusion_matrix(labels, pred_label)
            # res = self.process_confusion_matrix(cnf_matrix)
            # outputs.update(res)
        
        if self.mode == 'multilabel':
            pred_score = torch.tensor(pred).sigmoid().numpy()
            auroc_list, auprc_list = [], []
            for i in range(pred_score.shape[1]):
                y_cls = labels[:, i]
                pred_cls = pred_score[:, i]
                auprc_list.append(average_precision_score(y_cls, pred_cls))
                auroc_list.append(roc_auc_score(y_cls, pred_cls))
            outputs['auc'] = np.mean(auroc_list)
            outputs['auprc'] = np.mean(auprc_list)
        return outputs
    """
    
    def process_confusion_matrix(self, cnf_matrix):
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        outputs = {}
        # Sensitivity, hit rate, recall, or true positive rate
        outputs['tpr'] = TP/(TP+FN)
        # Specificity or true negative rate
        outputs['tnr'] = TN/(TN+FP) 
        # Precision or positive predictive value
        outputs['ppv'] = TP/(TP+FP)
        # Negative predictive value
        outputs['npv'] = TN/(TN+FN)
        # Fall out or false positive rate
        outputs['fpr'] = FP/(FP+TN)
        # False negative rate
        outputs['fnr'] = FN/(TP+FN)
        # False discovery rate
        outputs['fdr'] = FP/(TP+FP)

        # Overall accuracy for each class
        # outputs['acc'] = (TP+TN)/(TP+FP+FN+TN)
        if cnf_matrix.shape[0] > 2: # multiclass
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = np.mean(v)
        else:
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = v[1]
        return outputs
