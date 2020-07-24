from jiangziya.utils.config import get_train_data_dir, get_label_list
import os, time
import pickle


def compute_chi_square_on_file(train_data_path=None, chi_square_dict_path=None):
    # train_data: label_name \t title_words \t text_words, words split by '\s'
    # chi_square_dict: {label_name: {word: chi_square}}

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

    print('label_count_dict', label_count_dict)
    # === N, total_num_doc
    N = sum(label_count_dict.values())
    print("N = %d" % N)
    # assert total_num_doc == line_cnt

    # ===
    # \chi^2 = N * (\sum_k O_{1k}^2 / (R1 * C_k) + \sum_k O_{0k}^2 / (R0 * C_k) - 1)
    # R1 = df(word), R0 = N - R1

    chi_square_dict = {}

    # O_11 O_12 | R_1=df   O_11 = word_count_in_cur_label
    # O_21 O_22 | R_2
    # C_1  C_2  | N

    # === For each label
    for label_name, label_count in label_count_dict.items():

        C_1 = label_count
        C_2 = N - label_count

        col_list = [C_1, C_2]

        word_chi_square_dict = {} # {word: chi_square}

        # === For each word, compute chi-square.
        for word, word_count in label_word_count_dict[label_name].items():
            R_1 = df_dict[word]
            R_2 = N - R_1
            row_list = [R_1, R_2]

            O_11 = word_count # word_count in this label
            O_12 = R_1 - O_11

            O_21 = C_1 - O_11
            O_22 = C_2 - O_12

            observed_array = [[O_11, O_12], [O_21, O_22]]
            chi_square = compute_chi_square(row_list=row_list,
                                            col_list=col_list,
                                            observed_array=observed_array,
                                            N=N)

            word_chi_square_dict[word] = chi_square

        chi_square_dict[label_name] = word_chi_square_dict

    with open(chi_square_dict_path, 'wb') as fw:
        pickle.dump(chi_square_dict, fw)
        print("Write done!")


def compute_chi_square(row_list=None, col_list=None, observed_array=None, N=None):
    # observed_array: [n_row, n_col]

    chi_square = 0

    for i in range(len(row_list)):
        row = row_list[i]
        for k in range(len(col_list)):
            col = col_list[k]
            chi_square += observed_array[i][k] ** 2 / (row * col)

    chi_square = (chi_square - 1) * N

    return chi_square


def test_compute_chi_square():
    row_list = [40, 44]
    col_list = [54, 30]
    N = 84
    observed_array = [[34, 6], [20, 24]]

    chi_square = compute_chi_square(row_list=row_list,
                                    col_list=col_list,
                                    observed_array=observed_array,
                                    N=N)
    # 14.2715
    print(chi_square)


if __name__ == '__main__':
    #train_data_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
    train_data_path = os.path.join(get_train_data_dir(), "thucnews_val_seg.txt")
    chi_square_dict_path = os.path.join(get_train_data_dir(), "chi_square_dict.pkl")

    #test_compute_chi_square()

    start = time.time()
    compute_chi_square_on_file(train_data_path=train_data_path,
                               chi_square_dict_path=chi_square_dict_path)
    end = time.time()
    last = end - start
    print("Compute word_chi_square done! Lasts %.2fs" % last)
