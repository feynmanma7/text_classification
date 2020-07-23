from jiangziya.utils.config import get_train_data_dir
import os, pickle, numpy as np


def get_tfidf(data_path=None, df_path=None, tfidf_path=None):
    # data: label \t title_words \t text_words, split by '\s'
    # df: pickle of df_dict: {word: df_count}
    # tfidf: label \t word:tfidf \s word:tfidf

    # df: {word: df_count}
    df_dict = {}
    total_num_doc = 0
    with open(data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf = line[:-1].split('\t')
            if len(buf) != 3:
                continue

            total_num_doc += 1

            # label = buf[0]
            title = buf[1]
            text = buf[2]

            word_in_cur_doc = {}

            for word in (title + ' ' + text).split(' '):
                if len(word) == 0 or word in word_in_cur_doc:
                    continue

                word_in_cur_doc[word] = True

                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1

    with open(tfidf_path, 'w', encoding='utf-8') as fw:
        with open(data_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                buf = line[:-1].split('\t')
                if len(buf) != 3:
                    continue

                label = buf[0]
                title = buf[1]
                text = buf[2]

                tf_dict = {}
                for word in (title + ' ' + text).split(' '):
                    if len(word) == 0:
                        continue

                    if word not in tf_dict:
                        tf_dict[word] = 1
                    else:
                        tf_dict[word] += 1

                tfidf_list = []
                for word, tf in tf_dict.items():
                    if word not in df_dict:
                        continue
                    df = df_dict[word]
                    tfidf = tf * np.log(total_num_doc/(df+1))

                    tfidf_list.append(word + ':' + "{:.4f}".format(tfidf))

                fw.write(label + '\t' + ' '.join(tfidf_list) + '\n')
            print("Write tfidf done! %s" % tfidf_path)

    with open(df_path, 'wb') as fw:
        df_info = {'df_dict': df_dict, 'total_num_doc': total_num_doc}
        pickle.dump(df_info, fw)
        print("Write df_info done! %s" % df_path)


if __name__ == '__main__':
    train_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
    df_path = os.path.join(get_train_data_dir(), "train_df.pkl")
    tfidf_path = os.path.join(get_train_data_dir(), "train_tfidf.txt")

    get_tfidf(data_path=train_path,
              df_path=df_path,
              tfidf_path=tfidf_path)

