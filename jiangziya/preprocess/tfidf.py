from jiangziya.utils.config import get_train_data_dir, get_label_dict
from jiangziya.utils.dictionary import load_dict
import os, pickle, numpy as np


def compute_tfidf(data_path=None,
                  idf_dict=None,
                  word2id_dict=None,
                  tfidf_path=None):
    # data: label \t title_words \t text_words, split by '\s'
    # idf_dict: {word: idf}
    # word2id_dict: {word: index}
    # tfidf: label_index \s word_index:tfidf \s word_index:tfidf, word_index sorted ASC.

    fw = open(tfidf_path, 'w', encoding='utf-8')
    label_dict = get_label_dict()

    with open(data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf = line[:-1].split('\t')
            if len(buf) != 3:
                continue

            label_name = buf[0]
            label_index = label_dict[label_name]

            title = buf[1]
            text = buf[2]

            # === Count tf
            tf_dict = {}
            for word in (title + ' ' + text).split(' '):
                if word not in word2id_dict:
                    continue

                if word not in tf_dict:
                    tf_dict[word] = 1
                else:
                    tf_dict[word] += 1

            # === Compute tfidf
            tfidf_dict = {}
            for word, tf in tf_dict.items():
                if word not in idf_dict:
                    continue

                idf = idf_dict[word]
                tfidf = tf * idf

                word_index = word2id_dict[word]
                tfidf_dict[word_index] = tfidf

            # === Store in the format of libsvm
            # LIBSVM: sort key ASC {key: value}.
            tfidf_list = []
            for word_index, tfidf in sorted(tfidf_dict.items(), key=lambda x: x[0]):
                tfidf_list.append(str(word_index) + ':' + "{:.4f}".format(tfidf))

            if len(tfidf_list) > 0:
                fw.write(str(label_index) + ' ' + ' '.join(tfidf_list) + '\n')
        fw.close()
        print("Write done! %s " % tfidf_path)


if __name__ == '__main__':
    #data_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
    #tfidf_path = os.path.join(get_train_data_dir(), "train_tfidf.txt")

    data_path = os.path.join(get_train_data_dir(), "thucnews_test_seg.txt")
    tfidf_path = os.path.join(get_train_data_dir(), "test_tfidf.txt")

    idf_path = os.path.join(get_train_data_dir(), "train_idf.pkl")
    chosen_word_dict_path = os.path.join(get_train_data_dir(), "chosen_word_dict.pkl")

    # === Load chosen_word_dict
    chosen_word_dict = load_dict(dict_path=chosen_word_dict_path)
    print("#chosen_word_dict = %d" % len(chosen_word_dict))

    # === Get word_to_index_mapping_dict, of word ASC
    word2id_dict = {}
    index = 0
    for word in sorted(chosen_word_dict.keys(), key=lambda x:x[0]):
        word2id_dict[word] = index
        index += 1

    # === Load idf
    idf_dict = load_dict(dict_path=idf_path)

    # === Compute tf-idf
    compute_tfidf(data_path=data_path,
                  idf_dict=idf_dict,
                  word2id_dict=word2id_dict,
                  tfidf_path=tfidf_path)

