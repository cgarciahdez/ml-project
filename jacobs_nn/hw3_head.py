import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_validation import KFold

from data_loader import load_hw3data
from performance_measures import my_confusion_matrix, my_precision_recall_f1_acc
from my_classifiers import MyLogisticRegression, MyFeedForwardMLP
from collections import defaultdict
from functools import reduce

POS_INF = float("inf")
NEG_INF = float("-inf")

def main():
    
    np.set_printoptions(precision = 6, suppress = True)

    # path to datasets
    data_paths = [  './data/optdigits.tes', \
                    './data/iris.data']

    data_X = []
    data_y = []
    for dat in data_paths:
        temp_X, temp_y = load_hw3data(dat)
        data_X.append(temp_X)
        data_y.append(temp_y)


    t_folds = 10
    print("number of folds: "+str(t_folds))

    # Logistic Regression
    clf_names = ["Logistic Regression"]
    classifiers = [MyLogisticRegression()]
    classes = []

    logreg, classes = produce_output(data_X, data_y, \
                                    clf_names, classifiers, t_folds)
    logreg_scores = compute_performance(logreg, classifiers, classes)
    print_scores(logreg_scores, data_paths, clf_names, t_folds, classes)

    # Feed Forward MLP
    clf_names = ["Feed Forward"]
    classifiers = [MyFeedForwardMLP()]

    ff_mlp, classes = produce_output(data_X, data_y, \
                            clf_names, classifiers, t_folds)
    ff_mlp_scores = compute_performance(ff_mlp, classifiers, classes)
    print_scores(ff_mlp_scores, data_paths, clf_names, t_folds, classes)

def produce_output(data_x, data_y, clf_names, classifiers, t_folds):
    # classifier and fold get flipped partway through
    # the final structure is:
    # output[dataset][classifier][fold][0 = true, 1 = pred]
    output = []
    # needs to keep track of all classes
    classes = []
#    d_num = 0 #keeps track of the set number for file naming
    num_dat = 0
    for d_x, d_y in zip(data_x, data_y): #dataset
        num_points = len(d_x)
        kf = KFold(n=num_points, n_folds=t_folds, shuffle=True)
        curr_dataset = [[] for i in range(len(classifiers))]
#        t_num = 0 #keeps track of the test number for file naming graphs
        for train, test in kf: #fold
            X_train = [d_x[i] for i in train]
            X_test = [d_x[i] for i in test]
            y_train = [d_y[i] for i in train]
            y_test = [d_y[i] for i in test]
            for i in range(len(classifiers)): #classifier
                classifiers[i].fit(X_train, y_train)
                y_pred = [classifiers[i].predict(ex) for ex in X_test]
#                plt.plot(X_train, y_train, 'bo', X_test, y_pred, 'ro')
#                sav_name = "./graphs/svar/"+clf_names[i]+"_set"+str(d_num)+"_test"+str(t_num)+".png"
#                plt.savefig(sav_name)
#                plt.close()
#                for t,p in zip(y_test, y_pred):
#                    print((t,p))
                curr_dataset[i].append([y_test, y_pred])
#            t_num += 1
        if num_dat < len(data_y):
            classes.append(classifiers[0].classes)
            num_dat += 1
        output.append(curr_dataset)
#        d_num += 1
    return output, classes

# computes the confusion matrix for each classifier
def compute_performance(clf_output, classifiers, classes):
    # output[dataset][classifier][fold][0 = true, 1 = pred]
    dataset_measures = []
    j = 0 # counter to keep track of current classifier
    da = 0 # dataset
    for d in clf_output: # dataset
        clf_measures = []
        for c in d: # classifier
            fold_measures = []
            for f in c: # fold
                fold_mat = my_confusion_matrix(y_true = np.array(f[0]), \
                                            y_pred = np.array(f[1]), \
                                            labels = classes[da])
#                for honk, desu in zip(f[0], f[1]):
#                    print(honk, desu)
                prec, rec, f1, acc = my_precision_recall_f1_acc(fold_mat)
                fold_measures.append([fold_mat, prec, rec, f1, acc])
            j += 1
            clf_measures.append(fold_measures)
        j = 0
        da += 1
        dataset_measures.append(clf_measures)
    return dataset_measures

def print_scores(clf_scores, data_names, clf_names, t_folds, classes):
    for i in range(len(data_names)): # dataset
        print(data_names[i])
        for j in range(len(clf_names)): # classifier
            print(clf_names[j], end=": \n")
            tot_conf_mat = None
            # precision, recall, f1, acc
            avg_measures = np.array([0.0, 0.0, 0.0, 0.0])

            for k in range(t_folds):
                if k==0:
                    tot_conf_mat = clf_scores[i][j][k][0]
                else:
                    tot_conf_mat += clf_scores[i][j][k][0]
                avg_measures[0] += clf_scores[i][j][k][1]
                avg_measures[1] += clf_scores[i][j][k][2]
                avg_measures[2] += clf_scores[i][j][k][3]
                avg_measures[3] += clf_scores[i][j][k][4]

            avg_measures = np.array(list(map(lambda a: a/t_folds, avg_measures)))
            print("confusion matrix: ")
            print(classes[i])
            print(tot_conf_mat)
            print("mean precision: " + str(avg_measures[0]))
            print("mean recall: " + str(avg_measures[1]))
            print("mean f1: " + str(avg_measures[2]))
            print("mean accuracy: " + str(avg_measures[2]))
            print("********************")
 
if __name__ == '__main__':
    main()
