import numpy as np
from sklearn.cross_validation import KFold

def get_results(y_true, y_pred, labels=None):
    conf_mat = my_confusion_matrix(y_true,y_pred,labels)
    p, r, f, a = my_precision_recall_f1_acc(conf_mat)
    return conf_mat, p, r, f, a

def my_confusion_matrix(y_true, y_pred, labels = None):
    """
    Calculates a labels * labels confusion matrix, showing
    correct classifications along the diagonal and incorrect
    classification everywhere else.
    """
    try:
        conf_mat = np.zeros([len(labels), len(labels)])
    except:
        raise TypeError("labels needs to be a list-like")
        return

    for t, p in zip(y_true, y_pred):
        """
        finds which cell to add a point to.
        when i == j the classifier
        correctly identified the class
        """
        for i in range(len(labels)): 
            if t == labels[i]:
                break

        for j in range(len(labels)):
            if p == labels[j]:
                break

        conf_mat[i][j] += 1
    return conf_mat

def my_precision_recall_f1_acc(conf_mat):
    # count true positives and false positives and negatives for each class
    tps = np.zeros(conf_mat.shape[0])
    fps = np.zeros(conf_mat.shape[0])
    fns = np.zeros(conf_mat.shape[0])

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            if i == j:
                tps[j] += conf_mat[i][j]
            else:
                fns[i] += conf_mat[i][j]
                fps[j] += conf_mat[i][j]

    precisions = np.zeros(conf_mat.shape[0])
    recalls = np.zeros(conf_mat.shape[0])
    accuracy = 0
    for i in range(conf_mat.shape[0]):
        # precision for each class
        tot = tps[i] + fps[i]
        if tot != 0:
            precisions[i] = tps[i] / tot
        else:
            precisions[i] = 1.0

        # recall for each class
        tot = tps[i] + fns[i]
        if tot != 0:
            recalls[i] = tps[i] / tot
        else:
            recalls[i] = 1.0

        tot = tps[i] + fns[i] + fps[i]
    

    """
    average the measures for each class
    """
    tot_prec = sum(precisions) / conf_mat.shape[0]
    tot_reca = sum(recalls) / conf_mat.shape[0]
    tot_acc = sum(tps) / conf_mat.sum()
    f1 = 2*tot_prec*tot_reca / (tot_prec + tot_reca)
    return tot_prec, tot_reca, f1, tot_acc
