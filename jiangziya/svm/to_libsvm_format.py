from jiangziya.utils.config import get_train_data_dir, get_label_dict
import os, pickle
import numpy as np


def get_train_word2id_dict(tfidf_path=None, word2id_dict_path=None):
    # tfidf: label_name \t word:tfidf \s word:tfidf
    # word2id_dict: {word: index}, start from 0
    word2id_dict = {}
    index = 0
    with open(tfidf_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf = line[:-1].split('\t')
            if len(buf) != 2:
                continue
            for pair in buf[1].split(' '):
                word = pair.split(':')[0]
                if word not in word2id_dict:
                    word2id_dict[word] = index
                    index += 1

    with open(word2id_dict_path, 'wb') as fw:
        pickle.dump(word2id_dict, fw)
        print("Write done! #word2id_dict=%d" % len(word2id_dict))


def tfidf_to_libsvm_format(tfidf_path=None,
                           libsvm_path=None,
                           word2id_dict_path=None):
    # tfidf: label_name \t word:tfidf \s word:tfidf
    # libsvm: label_index \s word_index:tfidf, in sorted word_index
    # word2id_dict: {word: index}, start from 0
    with open(word2id_dict_path, 'rb') as fr:
        word2id_dict = pickle.load(fr)
        print("#word2id_dict=%d" % len(word2id_dict))

    with open(libsvm_path, 'w', encoding='utf-8') as fw:
        with open(tfidf_path, 'r', encoding='utf-8') as fr:

            label_dict = get_label_dict()
            line_cnt = 0
            for line in fr:
                buf = line[:-1].split('\t')
                if len(buf) != 2:
                    continue
                label_name = buf[0]
                label_index = label_dict[label_name]

                libsvm_dict = {}

                for pair in buf[1].split(' '):
                    if len(pair.split(':')) != 2:
                        continue
                    word = pair.split(':')[0]
                    if word not in word2id_dict:
                        continue
                    word_index = word2id_dict[word]
                    tfidf = pair.split(':')[1]

                    libsvm_dict[word_index] = tfidf

                libsvm_list = []
                # === sort in word_index ASC
                for word_index, tfidf in sorted(libsvm_dict.items(), key=lambda x: x[0]):
                    libsvm_list.append(str(word_index) + ':' + tfidf)

                fw.write(str(label_index) + ' ' + ' '.join(libsvm_list) + '\n')
                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)
            print("Total line %d" % line_cnt)


def word_to_libsvm_format(text_path=None,
                           libsvm_path=None,
                           word2id_dict_path=None,
                           df_info_dict_path=None):
    # text: label_name \t title_words \t text_words; words split by '\s'
    # libsvm: label_index \s word_index:tfidf, in sorted word_index
    # word2id_dict: {word: index}, start from 0
    # df_info: {'df_dict': df_dict, 'total_num_doc': total_num_doc}
    # df_dict: {df: count}

    with open(word2id_dict_path, 'rb') as fr:
        word2id_dict = pickle.load(fr)
        print("#word2id_dict=%d" % len(word2id_dict))

    with open(df_info_dict_path, 'rb') as fr:
        df_info_dict = pickle.load(fr)
        total_num_doc = df_info_dict['total_num_doc']
        df_dict = df_info_dict['df_dict']
        print('total_num_doc %d' % total_num_doc)
        print("#df_dict=%d" % len(df_dict))

    with open(libsvm_path, 'w', encoding='utf-8') as fw:
        with open(text_path, 'r', encoding='utf-8') as fr:

            label_dict = get_label_dict()
            line_cnt = 0
            for line in fr:
                buf = line[:-1].split('\t')
                if len(buf) != 3:
                    continue
                label_name = buf[0]
                label_index = label_dict[label_name]

                title = buf[1]
                text = buf[2]

                # === count tf in current doc
                tf_dict = {} # {word: tf_count}
                for word in (title + ' ' + text).split(' '):
                    if len(word) == 0 or word not in word2id_dict or word not in df_dict:
                        continue

                    if word not in tf_dict:
                        tf_dict[word] = 1
                    else:
                        tf_dict[word] += 1

                # === compute tf-idf
                tfidf_dict = {} #{word_index: tfidf}
                for word, tf in tf_dict.items():
                    word_index = word2id_dict[word]
                    df = int(df_dict[word])
                    tfidf = tf * np.log(total_num_doc / (df + 1))
                    tfidf_dict[word_index] = tfidf

                # === sort in word_index ASC
                libsvm_list = []
                for word_index, tfidf in sorted(tfidf_dict.items(), key=lambda x:x[0]):
                    libsvm_list.append(str(word_index) + ':' + "{:.4f}".format(tfidf))

                fw.write(str(label_index) + ' ' + ' '.join(libsvm_list) + '\n')
                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)
            print("Total line %d" % line_cnt)


if __name__ == '__main__':
    word2id_dict_path = os.path.join(get_train_data_dir(), "train_word2id.pkl")
    df_info_dict_path = os.path.join(get_train_data_dir(), "train_df.pkl")

    """
    tfidf_path = os.path.join(get_train_data_dir(), "train_tfidf.txt")
    libsvm_path = os.path.join(get_train_data_dir(), "train_libsvm.txt")
    tfidf_to_libsvm_format(tfidf_path=tfidf_path,
                           libsvm_path=libsvm_path,
                           word2id_dict_path=word2id_dict_path)
    """

    text_path = os.path.join(get_train_data_dir(), "thucnews_test_seg.txt")
    libsvm_path = os.path.join(get_train_data_dir(), "test_libsvm.txt")
    word_to_libsvm_format(text_path=text_path,
                          libsvm_path=libsvm_path,
                          df_info_dict_path=df_info_dict_path,
                          word2id_dict_path=word2id_dict_path)




