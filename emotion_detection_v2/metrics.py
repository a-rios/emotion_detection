import torch
import numpy as np
from typing import List
from torchmetrics.functional import confusion_matrix, stat_scores, f1_score, accuracy

def calculate_metrics(logits: torch.tensor,
                      labels: torch.tensor,
                      emotions: dict):

    # outputs a tensor (len(emotions), 5), where 2nd dim for each label: [tp fp tn fn tp+fn]
    stats =  stat_scores(preds=logits, target=labels.int(), reduce='macro', mdmc_reduce='global', top_k=1, num_classes=len(emotions), multiclass=None).int()
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1])
    y_pred = torch.argmax(logits, dim=1)
    preds_per_class = torch.nn.functional.one_hot(y_pred, num_classes=logits.shape[1])

    return {'vloss': None,
            'tp': stats[:,0],
            'fp': stats[:,1],
            'fn': stats[:,3],
            'labels':  labels_one_hot.sum(dim=0), # (batch, classes) -> (classes)
            'predicted_classes' : preds_per_class.sum(0),
            'y_pred': y_pred,
            'y_true': labels.int()
            }

def round_metric(value: float,
                 decimals: int=4):
    return round(value.item()*100, decimals)

def get_log_scores(outputs: List[torch.tensor],
                   emotions: dict):

    tqdm_dict = {}
    tqdm_dict['y_pred'] = torch.cat([outputs[i]['y_pred'] for i in range(len(outputs))])
    tqdm_dict['y_true'] = torch.cat([outputs[i]['y_true'] for i in range(len(outputs))])
    tqdm_dict['predicted_classes'] = torch.stack([outputs[i]['predicted_classes'] for i in range(len(outputs))], dim=0).sum(dim=0)
    tqdm_dict['labels'] = torch.stack([outputs[i]['labels'] for i in range(len(outputs))], dim=0).sum(dim=0)
    tqdm_dict['tp'] = torch.stack([outputs[i]['tp'] for i in range(len(outputs))], dim=0).sum(dim=0)
    tqdm_dict['fp'] = torch.stack([outputs[i]['fp'] for i in range(len(outputs))], dim=0).sum(dim=0)
    tqdm_dict['fn'] = torch.stack([outputs[i]['fn'] for i in range(len(outputs))], dim=0).sum(dim=0)
    loss = torch.stack([outputs[i]['vloss'] for i in range(len(outputs))], dim=0).mean(dim=0)
    tqdm_dict['vloss'] = round(loss.item(), 2)

    tqdm_dict['microF1'] = round_metric(
                                    f1_score(preds=tqdm_dict['y_pred'], target=tqdm_dict['y_true'], average='micro', num_classes=7)
                            )
    tqdm_dict['macroF1'] = round_metric(
                                    f1_score(preds=tqdm_dict['y_pred'], target=tqdm_dict['y_true'], average='macro', num_classes=7)
                            )
    F1_per_class = f1_score(preds=tqdm_dict['y_pred'], target=tqdm_dict['y_true'], average='none', num_classes=7)
    tqdm_dict['F1_per_class'] = [round_metric(v) for v in F1_per_class ]

    tqdm_dict['acc_unweighted'] = round_metric(
                                    accuracy(preds=tqdm_dict['y_pred'], target=tqdm_dict['y_true'], average='macro', num_classes=7)
                                    )
    tqdm_dict['acc_weighted'] =  round_metric(
                                    accuracy(preds=tqdm_dict['y_pred'], target=tqdm_dict['y_true'], average='weighted', num_classes=7)
                                    )
    acc_per_class = accuracy(preds=tqdm_dict['y_pred'], target=tqdm_dict['y_true'], average='none', num_classes=7)
    tqdm_dict['acc_per_class'] = [round_metric(v) for v in acc_per_class ]

    y_true = tqdm_dict.pop("y_true", None)
    y_pred = tqdm_dict.pop("y_pred", None)

    return tqdm_dict
