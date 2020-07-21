from jiangziya.utils.config import get_train_data_dir
import os

class TfIdf:
	def __init__(self):
		pass

	def fit(self):
		pass


if __name__ == '__main__':
	train_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
	tfidf_path = os.path.join(get_train_data_dir(), "thucnews_train_tfidf.txt")
