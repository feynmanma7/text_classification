from jiangziya.utils.config import get_train_data_dir, get_model_dir
import os
from sklearn.datasets import load_svmlight_file
import time, joblib

if __name__ == '__main__':
    test_path = os.path.join(get_train_data_dir(), "test_libsvm.txt")
    test_result_path = os.path.join(get_train_data_dir(), "thucnews_test_sk_libsvm.txt")
    model_path = os.path.join(get_model_dir(), "sk_libsvm.pkl")

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
        line_cnt = 0
        for y_true, y_pred in zip(y, y_pred_list):
            fw.write(str(y_true) + '\t' + str(y_pred) + '\n')
            line_cnt += 0
            if line_cnt % 1000 == 0:
                print(line_cnt)
        print(line_cnt)
        print("Write test result to %s" % test_result_path)

