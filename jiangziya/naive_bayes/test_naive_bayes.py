from jiangziya.utils.config import get_train_data_dir, get_model_dir
import os, time
from jiangziya.naive_bayes.naive_bayes import NaiveBayes

if __name__ == '__main__':
    #test_path = os.path.join(get_train_data_dir(), "thucnews_test_seg.txt")
    test_path = os.path.join(get_train_data_dir(), "tmp_seg.txt")
    test_result_path = os.path.join(get_train_data_dir(), "thucnews_test_nb.txt")
    model_path = os.path.join(get_model_dir(), "naive_bayes.pkl")

    start = time.time()
    nb = NaiveBayes()
    nb.load_model(model_path=model_path)
    nb.predict_on_file(test_path=test_path, test_result_path=test_result_path)
    end = time.time()
    last = end - start
    print("Predict done! Lasts %.2fs" % last)


