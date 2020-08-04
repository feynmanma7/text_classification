from jiangziya.utils.config import get_train_data_dir, get_model_dir, get_data_dir
import os
from sklearn import svm
from sklearn.datasets import load_svmlight_file
import time, joblib
from sklearn.metrics import classification_report
import numpy as np


def test_sk_svm_on_text_data():
    train_path = os.path.join(get_train_data_dir(), "train_tfidf.txt")
    model_path = os.path.join(get_model_dir(), "sk_libsvm.pkl")

    # clf = svm.SVC(C=100, kernel='linear')
    clf = svm.SVC()
    print(clf)

    start = time.time()
    X, y = load_svmlight_file(train_path)
    end = time.time()
    last = end - start
    print("Load lasts %.2fs" % last)

    start = time.time()
    clf.fit(X, y)
    end = time.time()
    last = end - start
    print("Train lasts %.2fs" % last)

    joblib.dump(clf, model_path)
    print("Save model to %s" % model_path)


def load_label_data(data_path=None):
    data = np.loadtxt(data_path, dtype=np.float, delimiter=' ')
    y = data[:, 0]
    X = data[:, 1:]
    return X, y

def test_sk_svm_on_sample_data():
    # data from mllib
    train_data_path = os.path.join(get_data_dir(), "mllib", "sample_svm_train")
    test_data_path = os.path.join(get_data_dir(), "mllib", "sample_svm_test")

    start = time.time()
    #clf = svm.SVC(kernel='linear', C=1)#acc=0.50, k<x, x'> = x^T x'
    #clf = svm.SVC(kernel='poly', C=1, gamma='scale') #acc=0.61, k<x, x'> = (\gamma x^Tx' + r)^d
    clf = svm.SVC(kernel='rbf', C=1, gamma='scale') #acc=0.62, k<x, x'> = exp(-gamma ||x - x'||^2)
    #clf = svm.SVC(kernel='sigmoid', C=10) # acc=0.54, k<x, x'> = tanh (gamma x^Tx' + r)
    print(clf)
    #X, y = load_svmlight_file(train_data_path)
    X, y = load_label_data(data_path=train_data_path)
    clf.fit(X, y)
    end = time.time()
    last = end - start
    print("Train lasts %.2fs" % last)

    X, y_true = load_label_data(data_path=test_data_path)
    y_pred = clf.predict(X)
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    test_sk_svm_on_sample_data()

