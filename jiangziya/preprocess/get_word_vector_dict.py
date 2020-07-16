from jiangziya.utils.config import get_model_dir
import os
import numpy as np
import pickle


def load_word_vector_dict(word_vector_dict_path=None):
    with open(word_vector_dict_path, 'rb') as fr:
        word_vector_dict = pickle.load(fr)
        return word_vector_dict

    return {}


def get_word_vector_dict(word_vector_path=None):
    # word_vector_path:
    # header line: 365076 300; num_words, embedding_dim
    # remain lines: word \t vec_1 \s vec_2 ... \s vec_300

    # Output: {word: vec in numpy.ndarray}
    word_vector_dict = {}
    with open(word_vector_path, 'r', encoding='utf-8') as f:
        header = False
        for line in f:
            if not header:
                header = True
                continue

            buf = line[:-1].strip().split(' ')
            word = buf[0]
            # [300, ]
            vec = np.array(list(map(lambda x: float(x), buf[1:])),
                                dtype=np.float32)
            word_vector_dict[word] = vec

    return word_vector_dict


if __name__ == '__main__':
    word_vector_path = os.path.join(get_model_dir(), "sgns.sogou.char")
    word_vector_dict_path = os.path.join(get_model_dir(), "sogou_vectors.pkl")

    word_vector_dict = get_word_vector_dict(
        word_vector_path=word_vector_path)

    with open(word_vector_dict_path, 'wb') as fw:
        pickle.dump(word_vector_dict, fw)
        print("Write done! %s" % word_vector_dict_path)