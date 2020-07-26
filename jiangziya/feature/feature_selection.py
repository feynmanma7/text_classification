from jiangziya.utils.config import get_train_data_dir, get_label_list
import os, time
import pickle


def select_by_chi_square(chi_square_dict_path=None,
                         chosen_word_dict_path=None,
                         min_chi_square=3.84,
                         max_word_of_label=10000):
    # chi_square_dict: {label_name: {word: chi_square}}
    # chosen_word_dict: {word: True}

    chosen_word_dict = {}
    with open(chi_square_dict_path, 'rb') as fr:
        chi_square_dict = pickle.load(fr)

        for label_name, word_chi_square_dict in chi_square_dict.items():
            for word, chi_square in sorted(word_chi_square_dict.items(), key=lambda x:-x[1])[:max_word_of_label]:
                if chi_square < min_chi_square or word in chosen_word_dict:
                    continue
                chosen_word_dict[word] = True

    print("#chosen_word_dict = %d" % len(chosen_word_dict))
    with open(chosen_word_dict_path, 'wb') as fw:
        pickle.dump(chosen_word_dict, fw)


if __name__ == '__main__':
    chi_square_dict_path = os.path.join(get_train_data_dir(), "chi_square_dict.pkl")
    chosen_word_dict_path = os.path.join(get_train_data_dir(), "chosen_word_dict.pkl")
    min_chi_square = 3.84
    max_word_of_label = 50

    select_by_chi_square(chi_square_dict_path=chi_square_dict_path,
                         chosen_word_dict_path=chosen_word_dict_path,
                         min_chi_square=min_chi_square,
                         max_word_of_label=max_word_of_label)