from jiangziya.utils.config import get_train_data_dir
import os, pickle


def word_to_index(word_cnt_path=None, word2id_path=None, min_count=5):
	index = 2 # 0 for `pad`, 1 for `unk`



if __name__ == '__main__':
	word_cnt_path = os.path.join(get_train_data_dir(), "word_cnt_dict.pkl")
	word2id_path = os.path.join(get_train_data_dir(), "word2id_dict.pkl")
	min_count = 5

	word_to_index(word_cnt_path=word_cnt_path,
				  word2id_path=word2id_path,
				  min_count=min_count)


