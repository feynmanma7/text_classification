from jiangziya.utils.config import get_train_data_dir, get_label_list
import os, time
import pickle
import numpy as np


# information gain = mutual information, only for compute on all of the classes (t, C).
def compute_information_gain_on_file(train_data_path=None,
                                       information_gain_dict_path=None):
    # train_data: label_name \t title_words \t text_words, words split by '\s'
    # information_gain_dict: {label_name: {word: information_gain}}

    label_count_dict = {} # {label_name: count}

    # {label_name: word_doc_count_dict}
    #   word_doc_count_dict: {word: word_doc_count}
    label_word_count_dict = {}

    df_dict = {} # {word, num_doc}

    with open(train_data_path, 'r', encoding='utf-8') as fr:
        line_cnt = 0
        for line in fr:
            buf = line[:-1].split('\t')
            if len(buf) != 3:
                continue

            label_name = buf[0]
            title = buf[1]
            text = buf[2]

            # === Count y=C_k
            if label_name not in label_count_dict:
                label_count_dict[label_name] = 1
            else:
                label_count_dict[label_name] += 1

            if label_name not in label_word_count_dict:
                label_word_count_dict[label_name] = {}

            word_in_cur_doc_dict = {} # {word: True/False}, True in current doc, for df_dict

            # === Count X_i=1|y=C_k
            for word in (title + ' ' + text).split(' '):

                if word not in word_in_cur_doc_dict:
                    word_in_cur_doc_dict[word] = True

                    # === Count word in df_dict only once in the doc.
                    if word not in df_dict:
                        df_dict[word] = 1
                    else:
                        df_dict[word] += 1

                if word not in label_word_count_dict[label_name]:
                    label_word_count_dict[label_name][word] = 1
                else:
                    label_word_count_dict[label_name][word] += 1

            line_cnt += 1
            if line_cnt % 1000 == 0:
                print(line_cnt)
        print("Total line_cnt = %d" % line_cnt)

    print('Total #word=%d' % len(df_dict))
    print('label_count_dict', label_count_dict)
    # === N, total_num_doc
    N = sum(label_count_dict.values())
    print("N = %d" % N)
    # assert total_num_doc == line_cnt

    information_gain_dict = {}

    label_prob_dict = {}
    for label, label_count in label_count_dict.items():
        label_prob_dict[label] = label_count / N

    for word in df_dict:
        mutual_info = compute_information_gain(label_word_count_dict=label_word_count_dict,
                                                 word_doc_count_dict=df_dict,
                                                 label_prob_dict=label_prob_dict,
                                                 word=word,
                                                 N=N)

        information_gain_dict[word] = mutual_info

    with open(information_gain_dict_path, 'wb') as fw:
        pickle.dump(information_gain_dict, fw)
        print("Write done!")


def compute_information_gain(label_word_count_dict=None,
                               word_doc_count_dict=None,
                               label_prob_dict=None,
                               word=None,
                               label=None,
                               N=None):
    """
    t: term, word,
    C: Category, class
    mu(t, C) = \sum_t \sum_c p(t, c) \log \frac{p(t, c)}{p(t)p(c)}

    For each term, t=1

    mu(t, C_i) = p(t=1, c) \log \frac{p(t=1, c)}{p(t=1)p(c)}

    To count:
    p(t) = N(t) / N, df of term
    p(c) = N(c) / N
    p(t, c) = N(t, c) / N(t, C) = N(t, c) / N(C) = N(t, c) / N, count p[c][t]
    """

    info_gain = 0

    p_t = word_doc_count_dict[word] / N

    for label, word_count_dict in label_word_count_dict.items():
        p_c = label_prob_dict[label]
        if word not in word_count_dict:
            continue
        N_t_c = word_count_dict[word]
        p_t_c = N_t_c / N

        info_gain += p_t_c * np.log(p_t_c/(p_t * p_c))

    return info_gain


def test_compute_information_gain():
    """
    doc_0: 0: I love you
    doc_1: 0: I love
    doc_2: 0: love you
    doc_3: 1: you
    """

    label_word_count_dict = {0: {'I': 2, 'love': 3, 'you': 2}, 1: {'you': 1}}
    word_doc_count_dict = {'I': 2, 'love': 3, 'you': 3}
    label_prob_dict = {0: 0.75, 1: 0.25}

    words = ['I', 'love', 'you']
    N = 4

    for word in words:
        mutual_info = compute_information_gain(label_word_count_dict=label_word_count_dict,
                                             word_doc_count_dict=word_doc_count_dict,
                                             label_prob_dict=label_prob_dict,
                                             word=word,
                                             N=N)
        print('word: %s \tmi=%.4f' % (word, mutual_info))


if __name__ == '__main__':
    test_compute_information_gain()

    train_data_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
    #train_data_path = os.path.join(get_train_data_dir(), "thucnews_val_seg.txt")
    information_gain_dict_path = os.path.join(get_train_data_dir(), "information_gain.pkl")
    
    start = time.time()
    compute_information_gain_on_file(train_data_path=train_data_path,
                                       information_gain_dict_path=information_gain_dict_path)
    end = time.time()
    last = end - start
    print("Compute information_gain done! Lasts %.2fs" % last)

