from jiangziya.utils.config import get_train_data_dir, get_label_dict, get_model_dir
from jiangziya.preprocess.get_word_vector_dict import load_word_vector_dict
import tensorflow as tf
import numpy as np
import os


def dataset_generator(data_path=None,
                      epochs=10,
                      shuffle_buffer_size=1024,
                      batch_size=16,
                      max_seq_len = 100,
                      word_vec_dict=None):
    # input_data: label \t title_words \t text_words, words split by ','
    # Output: label [word_index_1, word_index_2, ..., word_index_max_seq_len], 0 for pad, 1 for unk
    #   `pad` for less sequence, while `truncate` for longer one.

    # {label_name: label_index}
    label_dict = get_label_dict()

    def generator():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                buf = line[:-1].split('\t')
                if len(buf) != 3:
                    continue
                label_name = buf[0]
                label = int(label_dict[label_name])

                title = buf[1]
                text = buf[2]

                words = (title + ' ' + text).split(' ')

                inputs = []
                for i, word in zip(range(max_seq_len), words):
                    if word not in word_vec_dict:
                        continue
                    word_vec = word_vec_dict[word]
                    inputs.append(word_vec)

                for i in range(len(inputs), max_seq_len):
                    # embedding_dim: 300, hard code here
                    inputs.append([0] * 300)

                inputs = np.array(inputs)
                yield inputs, [label]

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=((max_seq_len, 300, ), (1, )),
                                             output_types=(tf.float32, tf.int32))

    return dataset.repeat(epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
        .batch(batch_size=batch_size)


def get_dataset(data_path=None,
                epochs=10,
                shuffle_buffer_size=4,
                batch_size=2,
                word_vec_dict=None,
                max_seq_len=10):
    return dataset_generator(data_path=data_path,
                             epochs=epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             batch_size=batch_size,
                             word_vec_dict=word_vec_dict,
                             max_seq_len=max_seq_len)


if __name__ == "__main__":
    train_path = os.path.join(get_train_data_dir(), "thucnews_train_seg.txt")
    word2id_dict_path = os.path.join(get_train_data_dir(), "word2id_dict.pkl")
    word_vector_dict_path = os.path.join(get_model_dir(), "sogou_vectors.pkl")

    # === Load word_vec_dict
    word_vec_dict = load_word_vector_dict(word_vector_dict_path=word_vector_dict_path)
    print("#word_vec_dict = %d" % len(word_vec_dict))

    train_dataset = get_dataset(data_path=train_path,
                                batch_size=2,
                                word_vec_dict=word_vec_dict,
                                max_seq_len=5)

    # inputs: [None, 300]
    # label: [None, ]
    print("%d\tinputs\tlabels")
    for i, (inputs, labels) in zip(range(2), train_dataset):
        print(i, inputs.shape, labels.shape)