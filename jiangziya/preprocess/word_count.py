from jiangziya.utils.config import get_train_data_dir
import os, pickle


def word_count(data_path=None, word_cnt_path=None):
	# data: \label \t word \s word \s
	# word_cnt_dict: {word: count}

	with open(data_path, 'r', encoding='utf-8') as fr:
		word_cnt_dict = {}
		line_cnt = 0
		for line in fr:
			buf = line[:-1].split('\t')
			if len(buf) == 1:
				continue
			#label = buf[0]
			words = buf[1].split(' ')
			if len(words) == 1:
				continue

			for word in words:
				if word not in word_cnt_dict:
					word_cnt_dict[word] = 1
				else:
					word_cnt_dict[word] += 1

			line_cnt += 1
			if line_cnt % 10000 == 0:
				print(line_cnt)

		print(line_cnt)

		print("#word_cnt_dict = %d" % len(word_cnt_dict))
		with open(word_cnt_path, 'wb') as fr:
			pickle.dump(word_cnt_dict, fr)
			print("Write done!", word_cnt_path)


if __name__ == '__main__':
	data_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
	word_cnt_path = os.path.join(get_train_data_dir(), "word_cnt_dict.pkl")

	word_count(data_path=data_path, word_cnt_path=word_cnt_path)