from jiangziya.utils.config import get_train_data_dir, get_label_list
import os, time
import pickle

if __name__ == '__main__':
    chi_square_dict_path = os.path.join(get_train_data_dir(), "chi_square_dict.pkl")

    with open(chi_square_dict_path, 'rb') as fr:
        chi_square_dict = pickle.load(fr)

        word_chi_square_dict = chi_square_dict['体育']
        print(len(word_chi_square_dict))

        cnt = 0
        for i, (word, chi_square) in enumerate(sorted(word_chi_square_dict.items(), key=lambda x: -x[1])):
            if chi_square > 3.84:
                cnt += 1
        print(cnt)