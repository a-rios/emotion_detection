import torch
import numpy as np

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
            'preds':preds_per_class,
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
