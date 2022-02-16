import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def compute_metrics_multiclass(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate f1_score using sklearn's function
    f1_s = f1_score(labels, preds, average='micro')
    return {'f1_score_micro': f1_s}


def multiclass_report(y_true, y_pred, labels):

    report_classes = classification_report(y_true=y_true,
                                           y_pred=y_pred,
                                           labels=labels,
                                           output_dict=True)

    confusion_matrix_array = confusion_matrix(y_true=y_true,
                                              y_pred=y_pred,
                                              labels=labels)

    confusion_matrix_dataframe = pd.DataFrame(confusion_matrix_array, index=labels, columns=labels)

    return report_classes, confusion_matrix_dataframe
