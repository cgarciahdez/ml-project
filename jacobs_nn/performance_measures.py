import numpy as np

def my_confusion_matrix(y_true, y_pred, labels = None):
    try:
        conf_mat = np.zeros([len(labels), len(labels)])
    except:
        raise TypeError("labels needs to be a list-like")
        return
#    print("confusion matrix")
#    print(y_true)
#    print(y_pred)
    for t, p in zip(y_true, y_pred):
        for i in range(len(labels)): 
#            print(t)
#            print(labels[i])
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
    accuracies = np.zeros(conf_mat.shape[0])
    for i in range(conf_mat.shape[0]):
        tot = tps[i] + fps[i]
        if tot != 0:
            precisions[i] = tps[i] / tot
        else:
            precisions[i] = 1.0

        tot = tps[i] + fns[i]
        if tot != 0:
            recalls[i] = tps[i] / tot
        else:
            recalls[i] = 1.0

        tot = tps[i] + fns[i] + fps[i]
        if tot != 0:
            accuracies[i] = tps[i] / tot
        else:
            accuracies[i] = 1.0

    tot_prec = sum(precisions) / conf_mat.shape[0]
    tot_reca = sum(recalls) / conf_mat.shape[0]
    tot_acc = sum(accuracies) / conf_mat.shape[0]
    f1 = 2*tot_prec*tot_reca / (tot_prec + tot_reca)
    return tot_prec, tot_reca, f1, tot_acc
