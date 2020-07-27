# encoding: utf-8
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluation1(true_label, predict_score, input_classes, predict_label, model_name=None, global_step=None,
                sample_dir=None):
    count_tp, count_fn, count_fp, count_tn = 0, 0, 0, 0
    for i in range(len(input_classes)):
        if input_classes[i] == predict_label[i] == '0':
            count_tp += 1
        elif input_classes[i] == predict_label[i] == '1':
            count_tn += 1
        elif input_classes[i] == '0' and predict_label[i] == '1':
            count_fn += 1
        elif input_classes[i] == '1' and predict_label[i] == '0':
            count_fp += 1
        # else:
        #     print(input_classes[i], predict_label[i])
    tpr = float(count_tp) / (count_tp + count_fn) if (count_tp + count_fn) > 0 else -1
    fpr = float(count_fp) / (count_fp + count_tn) if (count_fp + count_tn) > 0 else -1
    tnr = float(count_tn) / (count_fp + count_tn) if (count_fp + count_tn) > 0 else -1
    acc = (float(count_tp) + float(count_tn)) / (len(input_classes))
    precision = float(count_tp) / (count_tp + count_fp) if (count_tp + count_fp) > 0 else -1
    recall = float(count_tp) / (count_tp + count_fn) if (count_tp + count_fn) > 0 else -1
    f1 = (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1

    fpr_roc1, tpr_roc1, threshold1 = roc_curve(true_label, predict_score)  #
    roc_auc1 = auc(fpr_roc1, tpr_roc1)  #
    if sample_dir:
        # Compute ROC curve and ROC area for each class
        fpr_roc, tpr_roc, threshold = roc_curve(true_label, predict_score)  #
        roc_auc = auc(fpr_roc, tpr_roc)  #
        plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr_roc, tpr_roc, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.6f)' % roc_auc)  #
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate(FPR)')
        plt.ylabel('True Positive Rate(TPR)')
        plt.title('DeepSVFilter(' + model_name + ') ROC Curve {iter_num = ' + str(global_step) + '}')
        plt.legend(loc="lower right")
        plt.savefig(sample_dir + model_name + "_test_ROC.png")
        # plt.show()
    print("count_tp=", count_tp, "count_tn=", count_tn, "count_fn=", count_fn, "count_fp=", count_fp)
    return count_tp, count_tn, count_fn, count_fp, tpr, fpr, tnr, acc, precision, recall, f1, roc_auc1
