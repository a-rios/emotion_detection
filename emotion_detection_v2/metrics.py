import torch
import numpy as np
from typing import List
#from torchmetrics import F1, ConfusionMatrix
from torchmetrics.functional import confusion_matrix, stat_scores


def calculate_metrics(logits: torch.tensor,
                      labels: torch.tensor,
                      emotions: dict):

    # outputs a tensor (len(emotions), 5), where 2nd dim for each label: [tp fp tn fn tp+fn]
    stats =  stat_scores(preds=logits, target=labels.int(), reduce='macro', mdmc_reduce='global', top_k=1, num_classes=len(emotions), multiclass=None).int().cpu().numpy()
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1])
    y_pred = torch.argmax(logits, dim=1)
    preds_per_class = torch.nn.functional.one_hot(y_pred, num_classes=logits.shape[1])

    return {'vloss': None,
            'tp': stats[:,0],
            'fp': stats[:,1],
            'fn': stats[:,3],
            'labels':  labels_one_hot.sum(dim=0).cpu().numpy(), # (batch, classes) -> (classes)
            'predicted_classes' : preds_per_class.sum(0).cpu().numpy(),
            'y_pred': y_pred.cpu().numpy(),
            'y_true': labels.int().cpu().numpy()
            }

    ## torch code, TODO check torchmetrics.F1 to calculare scores

## code based on https://github.com/neeraj310/Master_Thesis_EA_In_ERC.git

def calculate_f1_score(tp, fp, fn):
    prec_rec_f1 = {}
    tp_fn = tp+fn
    tp_fp = tp+fp


    Recall = [np.round(tp/tp_fn*100, 2) if tp_fn>0 else 0.0 for tp,tp_fn in zip(tp,tp_fn)]
    prec_rec_f1['microRecall'] = np.round((sum(tp)/ sum(tp_fn))*100, 2)
    prec_rec_f1['macroRecall'] = np.round (sum(Recall) / len(Recall),2)

    Precision = [np.round(tp/tp_fp*100, 2) if tp_fp>0 else 0.0 for tp,tp_fp in zip(tp,tp_fp)]
    prec_rec_f1['microPrecision'] = np.round((sum(tp)/ sum(tp_fp))*100, 2)
    prec_rec_f1['macroPrecision'] = np.round(sum(Precision) / len(Precision),2)

    per_class_f1score = []
    f1_numenator = [2*x*y for (x, y) in zip(Recall, Precision)]
    f1_denominator = [x + y for (x, y) in zip(Recall, Precision)]

    for num1, num2 in zip(f1_numenator,f1_denominator):
        if  num2:
            per_class_f1score.append(np.round(num1 / num2, 2))
        else:
            per_class_f1score.append(0.0)

    #import IPython; IPython.embed();  exit(1)
    macroPrecision = prec_rec_f1['macroPrecision']
    macroRecall = prec_rec_f1['macroRecall']
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0

    prec_rec_f1['microF1_score'] = 2 * (prec_rec_f1['microPrecision'] * prec_rec_f1['microRecall']) / (prec_rec_f1['microPrecision'] + prec_rec_f1['microRecall'])
    prec_rec_f1['macroF1_score'] = np.round(macroF1, 2)
    prec_rec_f1['macroF1_per_class'] = per_class_f1score

    return prec_rec_f1


def get_log_scores(outputs: List[torch.tensor],
                   emotions: dict,
                   class_weights: torch.tensor):

    tqdm_dict = {}
    tqdm_dict['predicted_classes'] = [0 for w in emotions]
    tqdm_dict['labels'] = [0 for w in emotions]
    tqdm_dict['tp'] = [0 for w in emotions]
    tqdm_dict['fp'] = [0 for w in emotions]
    tqdm_dict['fn'] = [0 for w in emotions]
    tqdm_dict['acc_per_class'] = [0 for w in emotions]
    tqdm_dict['y_pred'] = []
    tqdm_dict['y_true'] = []
    tqdm_dict['vloss'] = 0

    for metric_name in outputs[0].keys():
        for output in outputs:
            metric_value = output[metric_name]
            if metric_name in ['y_pred', 'y_true']:
                tqdm_dict[metric_name].extend(metric_value)
            else:
                tqdm_dict[metric_name] += metric_value

        if metric_name in ['vloss']:
            tqdm_dict[metric_name] =  tqdm_dict[metric_name] / len(outputs)

    for i in range(len(emotions)):
        if  tqdm_dict['labels'][i]:
            tqdm_dict['acc_per_class'][i] = np.round((tqdm_dict['tp'][i] / tqdm_dict['labels'][i])*100, 2)
        else:
            tqdm_dict['acc_per_class'][i] = 0.0

    tqdm_dict['acc_unweighted'] = np.round(sum(tqdm_dict['acc_per_class']) / len(tqdm_dict['acc_per_class']),2)
    tqdm_dict['acc_weighted'] = sum(weight * value for weight, value in zip(class_weights, tqdm_dict['acc_per_class']))
    tqdm_dict['acc_weighted'] = (tqdm_dict['acc_weighted'] * 10**2).round() / (10**2)

    prec_rec_f1 = calculate_f1_score(tqdm_dict['tp'], tqdm_dict['fp'], tqdm_dict['fn'])
    tqdm_dict.update(prec_rec_f1)

    y_true = tqdm_dict.pop("y_true", None)
    y_pred = tqdm_dict.pop("y_pred", None)

    return tqdm_dict
