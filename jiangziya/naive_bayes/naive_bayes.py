from jiangziya.utils.config import get_train_data_dir, get_model_dir, get_label_dict
from jiangziya.utils.dictionary import load_pickle_dict
import os, pickle, time
import numpy as np


class NaiveBayes:
	def __init__(self):
		super(NaiveBayes, self).__init__()

		self.label_count = {} # {label_name: label_count}

		# {label_name: prob} <== {label_name: label_doc_count/#total_doc}
		self.label_prob = {}

		# {label_name: {word: count}}
		self.label_word_count = {}

		# {label_name: {word: prob}}, {word: prob} <== {word: word_in_num_doc_label / num_doc_label}
		self.label_word_prob = {}

	def fit(self, train_path=None, chosen_word_dict=None):
		# train: label_name \t title_words \t text_words

		with open(train_path, 'r', encoding='utf-8') as fr:
			line_cnt = 0
			for line in fr:
				buf = line[:-1].split('\t')
				if len(buf) != 3:
					continue

				label_name = buf[0]

				# === p(y = C_k)
				if label_name not in self.label_count: # self.label_label ==> self.label_count, bug
					self.label_count[label_name] = 1
				else:
					self.label_count[label_name] += 1

				if label_name not in self.label_word_count:
					self.label_word_count[label_name] = {}
					self.label_word_prob[label_name] = {}

				title = buf[1]
				text = buf[2]

				# === p(X_i = x_i | y = C_k)
				# VSM, 0-1 for a word
				word_dict = {} # {word: True}
				for word in (title + ' ' + text).split(' '):
					if word not in chosen_word_dict: # add feature selection
						continue
					if word in word_dict:
						continue
					word_dict[word] = True

					# word_in_num_doc_cur_label + 1
					if word not in self.label_word_count[label_name]:
						self.label_word_count[label_name][word] = 1
					else:
						self.label_word_count[label_name][word] += 1

				line_cnt += 1
				if line_cnt % 1000 == 0:
					print(line_cnt)
			print(line_cnt)

			# === Normalize p(y = C_k), Laplace Smoothing
			N = sum(self.label_count.values())
			print("N = %d" % N)
			for label_name, label_count in self.label_count.items():
				self.label_prob[label_name] = (label_count + 1) / (N + len(self.label_count))

			# === Normalize p(X_i = x_i | y = C_k)
			V = len(chosen_word_dict)
			for label_name, word_doc_count_dict in self.label_word_count.items():
				#N_k = sum(word_doc_count_dict.values())
				N_k = self.label_count[label_name]
				for word, doc_count in word_doc_count_dict.items():
					self.label_word_prob[label_name][word] = (doc_count + 1) / (N_k + V)

	def save_model(self, model_path=None):
		with open(model_path, 'wb') as fw:
			model_dict = {'label_count': self.label_count,
						  'label_prob': self.label_prob,
						  'label_word_count': self.label_word_count,
						  'label_word_prob': self.label_word_prob}

			pickle.dump(model_dict, fw)
			print("Save model done! %s" % model_path)

	def load_model(self, model_path=None):
		with open(model_path, 'rb') as fr:
			model_dict = pickle.load(fr)
			self.label_count = model_dict['label_count']
			self.label_prob = model_dict['label_prob']
			self.label_word_count = model_dict['label_word_count']
			self.label_word_prob = model_dict['label_word_prob']
			print("Load model done! %s" % model_path)

	def predict_on_file(self, test_path=None, test_result_path=None, chosen_word_dict=None):
		# test_data: label_name \t title \t text
		fw = open(test_result_path, 'w', encoding='utf-8')
		line_cnt = 0

		with open(test_path, 'r', encoding='utf-8') as fr:
			label_dict = get_label_dict() # {label_name: label_index}
			for line in fr:
				buf = line[:-1].split('\t')
				if len(buf) != 3:
					continue

				y_true_label_name = buf[0].strip()
				y_true_label_index = str(label_dict[y_true_label_name])
				title = buf[1]
				text = buf[2]

				word_dict = {} # {word: True}
				for word in (title + ' ' + text).split(' '):
					word = word.strip()
					if word not in chosen_word_dict: # feature selection
						continue
					if word in word_dict:
						continue
					word_dict[word] = True

				probs = {} # {label_name: prob}
				V = len(chosen_word_dict)
				for label_name, label_prob in self.label_prob.items():
					prob = np.log(label_prob)
					N_k = self.label_count[label_name]
					for word in word_dict.keys():
						if word not in self.label_word_prob[label_name]:
							#continue
							word_pos_prob = 1. / (N_k + V)
							#print(label_name, word, word_pos_prob)
						else:
							word_pos_prob = self.label_word_prob[label_name][word]
						prob += np.log(word_pos_prob)

					probs[label_name] = prob

				# === Sort by prob, DESC
				# sorted[top_0][first_item=label_name]
				y_pred_label_name = sorted(probs.items(), key=lambda x:-x[1])[0][0]
				y_pred_label_index = str(label_dict[y_pred_label_name])

				fw.write(str(y_true_label_index) + '\t' + str(y_pred_label_index) + '\n')
				line_cnt += 1
				if line_cnt % 1000 == 0:
					print(line_cnt)
			print(line_cnt)
			print("Predict done! %s" % test_result_path)


if __name__ == "__main__":
	train_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
	#train_path = os.path.join(get_train_data_dir(), "thucnews_val_seg.txt")
	model_path = os.path.join(get_model_dir(), "naive_bayes.pkl")
	chosen_word_dict_path = os.path.join(get_train_data_dir(), "chosen_word_dict.pkl")

	chosen_word_dict = load_pickle_dict(chosen_word_dict_path)
	print("#chosen_word_dict = %d" % len(chosen_word_dict))

	start = time.time()
	nb = NaiveBayes()
	nb.fit(train_path=train_path, chosen_word_dict=chosen_word_dict)
	end = time.time()
	last = end - start
	print("Train done! Lasts %.2fs" % last)

	nb.save_model(model_path=model_path)


