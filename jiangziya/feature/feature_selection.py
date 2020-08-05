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


def select_by_mutual_information(mutual_information_dict_path=None,
                         chosen_word_dict_path=None,
                         num_chosen_word=1000):
    # mutual_information_dict_path: {word: mutual_information}
    # chosen_word_dict: {word: True}

    with open(mutual_information_dict_path, 'rb') as fr:
        mutual_information_dict = pickle.load(fr)

    chosen_word_dict = {}
    i = 0
    for word, mutual_information in sorted(mutual_information_dict.items(),
                                           key=lambda x:-x[1])[:num_chosen_word]:
        if i < 10:
            print(word, mutual_information)
            i += 1
        chosen_word_dict[word] = True

    print("#chosen_word_dict = %d" % len(chosen_word_dict))
    with open(chosen_word_dict_path, 'wb') as fw:
        pickle.dump(chosen_word_dict, fw)


if __name__ == '__main__':
    chosen_word_dict_path = os.path.join(get_train_data_dir(), "chosen_word_dict.pkl")

    """
    chi_square_dict_path = os.path.join(get_train_data_dir(), "chi_square_dict.pkl")
    min_chi_square = 3.84
    max_word_of_label = 50

    select_by_chi_square(chi_square_dict_path=chi_square_dict_path,
                         chosen_word_dict_path=chosen_word_dict_path,
                         min_chi_square=min_chi_square,
                         max_word_of_label=max_word_of_label)
    """

    mutual_information_dict_path = os.path.join(get_train_data_dir(), "mutual_information.pkl")
    num_chosen_word = 300
    select_by_mutual_information(mutual_information_dict_path=mutual_information_dict_path,
                                 chosen_word_dict_path=chosen_word_dict_path,
                                 num_chosen_word=num_chosen_word)

    print("Write done!", chosen_word_dict_path)