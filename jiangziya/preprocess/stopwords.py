from jiangziya.utils.config import get_data_dir
import os, pickle
import string
from zhon import hanzi


def load_stopwords_dict(stopwords_dict_path=None):
    with open(stopwords_dict_path, 'rb') as fr:
        stopwords_dict = pickle.load(fr)
        return stopwords_dict
    return None


def generate_stopwords_dict(stopwords_path_list=None,
                            stopwords_dict_path=None):

    stopwords_dict = {}

    for stopwords_path in stopwords_path_list:
        with open(stopwords_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                stopword = line[:-1]
                stopwords_dict[stopword] = True

    # English punctuation
    for char in string.punctuation:
        stopwords_dict[char] = True

    # Chinese punctuation
    for char in hanzi.punctuation:
        stopwords_dict[char] = True

    with open(stopwords_dict_path, 'wb') as fw:
        pickle.dump(stopwords_dict, fw)
        print("Write done! %s" % stopwords_dict_path)


if __name__ == '__main__':
    baidu_stopwords_path = os.path.join(get_data_dir(), "nlp", "baidu_stopwords.txt")
    stopwords_dict_path = os.path.join(get_data_dir(), "nlp", "stopwords_dict.pkl")

    generate_stopwords_dict(stopwords_path_list=[baidu_stopwords_path],
                            stopwords_dict_path=stopwords_dict_path)
