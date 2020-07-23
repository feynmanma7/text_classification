from sklearn.naive_bayes import MultinomialNB
from jiangziya.utils.config import get_train_data_dir, get_model_dir
import os
from sklearn.datasets import load_svmlight_file
import time, joblib

if __name__ == '__main__':
    train_path = os.path.join(get_train_data_dir(), "train_libsvm.txt")
    model_path = os.path.join(get_model_dir(), "sk_naive_bayes.pkl")

    clf = MultinomialNB()
    print(clf)

    start = time.time()
    X, y = load_svmlight_file(train_path)
    end = time.time()
    last = end - start
    print("Load lasts %.2f" % last)

    start = time.time()
    clf.fit(X, y)
    end = time.time()
    last = end - start
    print("Train lasts %.2f" % last)

    joblib.dump(clf, model_path)
    print("Save model to %s" % model_path)