import torch
import numpy as np
from typing import List

## code based on https://github.com/neeraj310/Master_Thesis_EA_In_ERC.git

def calculate_metrics(logits: torch.tensor,
                      labels: torch.tensor,
                      class_weights: torch.tensor,
                      emotions: dict):

    labelslist = labels.cpu().tolist()
    preds = torch.argmax(logits, dim=1)
    predslist = preds.cpu().tolist()

    cm = np.zeros((len(emotions.keys()),len(emotions.keys())), dtype=np.int64) # recall

    for label, pred in zip(labels.view(-1), preds.view(-1)):
        cm[label.long(), pred.long()] += 1
        cm = cm[0:len(emotions), 0:len(emotions)]
        gt_labels_per_class =  cm.sum(axis = 1)
        preds_per_class =  cm.sum(axis = 0)
        tp = cm.diagonal()[0:]
        fp = preds_per_class-tp
        fn = gt_labels_per_class- tp

    return {'vloss': None,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'predicted_classes':preds_per_class,
            'labels': gt_labels_per_class,
            'y_pred':predslist,
            'y_true':labelslist
            }

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
