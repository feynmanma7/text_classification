from jiangziya.utils.config import get_train_data_dir
import os, pickle, numpy as np


def compute_idf(train_path=None, idf_path=None):
    # train_data: label \t title_words \t text_words, words split by '\s'
    # idf_dict: {word: train_idf}

    df_dict = {}
    total_num_doc = 0

    with open(train_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf = line[:-1].split('\t')
            if len(buf) != 3:
                continue

            total_num_doc += 1

            title = buf[1]
            text = buf[2]

            word_in_cur_doc = {}

            for word in (title + ' ' + text).split(' '):
                if word in word_in_cur_doc:
                    continue

                word_in_cur_doc[word] = True

                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1

    print("Total_num_doc = %d" % total_num_doc)
    print("#word = %d" % len(df_dict))
    idf_dict = {}
    # idf = log N/(df+1)
    for word, df in df_dict.items():
        idf = np.log(total_num_doc/(df+1))
        idf_dict[word] = idf

    with open(idf_path, 'wb') as fw:
        pickle.dump(idf_dict, fw)
        print("Write done! %s" % idf_path)


if __name__ == '__main__':
    train_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
    idf_path = os.path.join(get_train_data_dir(), "train_idf.pkl")

    compute_idf(train_path=train_path, idf_path=idf_path)

