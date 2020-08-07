from jiangziya.utils.config import get_train_data_dir, get_label_list
import os, time
import pickle
import numpy as np


# information gain = mutual information, only for compute on all of the classes (t, C).
def compute_mutual_information_on_file(train_data_path=None,
                                       mutual_information_dict_path=None):
    # train_data: label_name \t title_words \t text_words, words split by '\s'
    # label_mutual_information_dict: {label_name: {word: mutual_information}}

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

    label_mutual_information_dict = {} # {label: {word: mi}}

    label_prob_dict = {}
    for label, label_count in label_count_dict.items():
        label_prob_dict[label] = label_count / N
        label_mutual_information_dict[label] = {}

    for word in df_dict:
        # mutual_info_dict: {label: mi} for current word
        mutual_info_dict = compute_mutual_information(label_word_count_dict=label_word_count_dict,
                                                 word_doc_count_dict=df_dict,
                                                 label_count_dict=label_count_dict,
                                                 word=word,
                                                 N=N)

        for label, mi in mutual_info_dict.items():
            label_mutual_information_dict[label][word] = mi

    with open(mutual_information_dict_path, 'wb') as fw:
        pickle.dump(label_mutual_information_dict, fw)
        print("Write done!")


def compute_mutual_information(label_word_count_dict=None,
                               word_doc_count_dict=None,
                               label_count_dict=None,
                               word=None,
                               N=None):
    """
    t: term, word,
    C: Category, class
    mu(t, C) = \sum_t \sum_c p(t, c) \log \frac{p(t, c)}{p(t)p(c)}

             c=1    c=0  total
    t=1     N(t, c)       N(t)
    t=0
    total   N(c1)          N

    For class c_i,

    mu(t, c_i) = \sum_t p(t, c_i) \log \frac{p(t, c_i)}{p(t) p(c_i)}

    """

    mutual_info_dict = {}

    N_t = word_doc_count_dict[word]

    for label, word_count_dict in label_word_count_dict.items():
        N_c = label_count_dict[label]
        if word not in word_count_dict:
            continue
        N_t_c = word_count_dict[word]

        mutual_info_dict[label] = _compute_mutual_information(N_t, N_c, N_t_c, N)

    return mutual_info_dict


def _compute_mutual_information(N_t, N_c, N_t_c, N):
    """
             c=1    c=0   total
    t=1     N(t,c)         N(t)
    t=0
    total   N(c)            N
    """
    mutual_info = 0

    p_t_c = N_t_c / N
    p_t = N_t / N
    p_c = N_c / N

    p_not_t = 1 - p_t
    p_not_c = 1 - p_c

    p_not_t_c = p_c - p_t_c
    p_t_not_c = p_t - p_t_c
    p_not_t_not_c = 1 - p_t_c - p_not_t_c - p_t_not_c

    mutual_info += _compute_one_mutual_information(p_t_c, p_t, p_c)
    mutual_info += _compute_one_mutual_information(p_not_t_c, p_not_t, p_c)
    mutual_info += _compute_one_mutual_information(p_t_not_c, p_t, p_not_c)
    mutual_info += _compute_one_mutual_information(p_not_t_not_c, p_not_t, p_not_c)

    return mutual_info


def _compute_one_mutual_information(p_t_c, p_t, p_c):
    if p_t < 1e-4 or p_c < 1e-4 or p_t_c < 1e-4:
        return 0
    return p_t_c * np.log(p_t_c / (p_t * p_c))


def test_compute_mutual_information():
    """
    doc_0: 0: I love you
    doc_1: 0: I love
    doc_2: 0: love you
    doc_3: 1: you
    """

    label_word_count_dict = {0: {'I': 2, 'love': 3, 'you': 2}, 1: {'you': 1}}
    word_doc_count_dict = {'I': 2, 'love': 3, 'you': 3}
    label_count_dict = {0: 3, 1: 1}

    words = ['I', 'love', 'you']
    N = 4

    for word in words:
        mutual_info_dict = compute_mutual_information(label_word_count_dict=label_word_count_dict,
                                             word_doc_count_dict=word_doc_count_dict,
                                             label_count_dict=label_count_dict,
                                             word=word,
                                             N=N)
        print("word: %s mutual_info_dict %s" % (word, mutual_info_dict))


if __name__ == '__main__':
    test_compute_mutual_information()


    train_data_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
    #train_data_path = os.path.join(get_train_data_dir(), "thucnews_val_seg.txt")
    mutual_information_dict_path = os.path.join(get_train_data_dir(), "mutual_information.pkl")
    
    start = time.time()
    compute_mutual_information_on_file(train_data_path=train_data_path,
                                       mutual_information_dict_path=mutual_information_dict_path)
    end = time.time()
    last = end - start
    print("Compute mutual_information done! Lasts %.2fs" % last)



