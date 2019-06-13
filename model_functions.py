from sklearn.metrics import (
    recall_score,
    hamming_loss,
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np
import itertools


def get_scores(y_train, y_train_hat, y_test, y_test_hat):
    """Prints the recall and hamming-loss scores for
       the training and testing data"""
    rec_train = recall_score(y_train, y_train_hat)
    rec_test = recall_score(y_test, y_test_hat)
    hamming_train = hamming_loss(y_train, y_train_hat)
    hamming_test = hamming_loss(y_test, y_test_hat)

    print(f"""
            Training Recall: {rec_train} \n
            Testing Recall: {rec_test} \n
            Training Hamming-Loss: {hamming_train} \n
            Testing Hamming-Loss: {hamming_test}""")


def get_auc_scores(clf, X_train_full, X_test_full, y_train, y_test):
    """Prints the AUC scores for training and testing data
       and returns testing score"""
    y_train_score = clf.predict_proba(X_train_full)[:, 1]
    y_test_score = clf.predict_proba(X_test_full)[:, 1]

    auc_train = roc_auc_score(y_train, y_train_score)
    auc_test = roc_auc_score(y_test, y_test_score)

    print(f"""
            Training AUC: {auc_train}
            Testing AUC: {auc_test}""")
    return y_test_score


def plot_roc_curve(y_test, y_test_score):
    """Plot ROC curve for testing data"""
    fpr, tpr, _ = roc_curve(y_test, y_test_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


def show_cm(y_true, y_pred, class_names=None, model_name=None):
    """Show confusion matrix"""
    cf = confusion_matrix(y_true, y_pred)
    plt.imshow(cf, cmap=plt.cm.Blues)

    if model_name:
        plt.title("Confusion Matrix: {}".format(model_name))
    else:
        plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
    else:
        class_names = set(y_true)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

    thresh = cf.max() / 2.0

    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(
            j,
            i,
            cf[i, j],
            horizontalalignment="center",
            color="white" if cf[i, j] > thresh else "black",
        )

    plt.colorbar()
