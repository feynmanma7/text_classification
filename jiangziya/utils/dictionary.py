import pickle


def load_pickle_dict(pickle_dict_path=None):
    with open(pickle_dict_path, 'rb') as fr:
        pickle_dict = pickle.load(fr)
        return pickle_dict

    return None


def get_stopwords_dict(stopwords_dict_path_list=None):
    stopwords_dict = {}
    for stopwords_dict_path in stopwords_dict_path_list:
        with open(stopwords_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopword = line[:-1]
                stopwords_dict[stopword] = True
        return stopwords_dict
    return None