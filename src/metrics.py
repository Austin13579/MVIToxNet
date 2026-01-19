from sklearn.metrics import recall_score, f1_score,precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score,average_precision_score, confusion_matrix


def evaluate(label, prob):
    pred_label = (prob > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(label, pred_label).ravel()
    specificity = tn / (tn + fp)

    res = {
        'Accuracy': balanced_accuracy_score(label, pred_label),
        'AUROC': roc_auc_score(label, prob),
        'AUPRC': average_precision_score(label, prob),
        'Recall': recall_score(label, pred_label),
        'F1': f1_score(label, pred_label),
        'MCC': matthews_corrcoef(label, pred_label)
    }

    return res