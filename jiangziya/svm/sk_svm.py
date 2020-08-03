from jiangziya.utils.config import get_train_data_dir, get_model_dir
import os
from sklearn import svm
from sklearn.datasets import load_svmlight_file
import time, joblib


if __name__ == '__main__':

    train_path = os.path.join(get_train_data_dir(), "train_tfidf.txt")
    model_path = os.path.join(get_model_dir(), "sk_libsvm.pkl")

    #clf = svm.SVC(C=100, kernel='linear')
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
