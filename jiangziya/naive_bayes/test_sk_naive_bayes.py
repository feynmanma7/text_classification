from jiangziya.utils.config import get_train_data_dir, get_model_dir
import os
from sklearn.datasets import load_svmlight_file
import time, joblib

if __name__ == '__main__':
    test_path = os.path.join(get_train_data_dir(), "test_libsvm.txt")
    test_result_path = os.path.join(get_train_data_dir(), "thucnews_test_sk_nb.txt")
    model_path = os.path.join(get_model_dir(), "sk_naive_bayes.pkl")

    clf = joblib.load(model_path)
    print(clf)
    print("Load model done!")

    start = time.time()
    X, y = load_svmlight_file(test_path)
    end = time.time()
    last = end - start
    print("Load data lasts %.2fs" % last)

    start = time.time()
    y_pred_list = clf.predict(X)
    end = time.time()
    last = end - start
    print("Test lasts %.2fs" % last)

    with open(test_result_path, 'w', encoding='utf-8') as fw:
        for y_true, y_pred in zip(y, y_pred_list):
            fw.write(str(y_true) + '\t' + str(y_pred) + '\n')
        print("Write test result to %s" % test_result_path)

