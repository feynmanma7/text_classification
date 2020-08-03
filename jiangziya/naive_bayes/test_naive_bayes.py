from jiangziya.utils.config import get_train_data_dir, get_model_dir
from jiangziya.utils.dictionary import load_dict
import os, time
from jiangziya.naive_bayes.naive_bayes import NaiveBayes


if __name__ == '__main__':
    test_path = os.path.join(get_train_data_dir(), "thucnews_test_seg.txt")
    #test_path = os.path.join(get_train_data_dir(), "tmp_seg.txt")
    test_result_path = os.path.join(get_train_data_dir(), "thucnews_test_nb.txt")
    model_path = os.path.join(get_model_dir(), "naive_bayes.pkl")

    chosen_word_dict_path = os.path.join(get_train_data_dir(), "chosen_word_dict.pkl")

    chosen_word_dict = load_dict(chosen_word_dict_path)
    print("#chosen_word_dict = %d" % len(chosen_word_dict))

    start = time.time()
    nb = NaiveBayes()
    nb.load_model(model_path=model_path)
    nb.predict_on_file(test_path=test_path, test_result_path=test_result_path,
                       chosen_word_dict=chosen_word_dict)
    end = time.time()
    last = end - start
    print("Predict done! Lasts %.2fs" % last)


