from jiangziya.utils.config import get_data_dir, get_label_dict
import tensorflow as tf
import numpy as np
import os


def dataset_generator(data_path=None,
                      epochs=10,
                      shuffle_buffer_size=1024,
                      batch_size=16,
                      max_seq_len = 100,
                      word2id_dict=None):
    # input_data: label \t word_1, word_2, ..., split by ' '
    # Output: label [word_index_1, word_index_2, ..., word_index_max_seq_len], 0 for pad, 1 for unk
    #   `pad` for less sequence, while `truncate` for longer one.

    # {label_name: label_index}
    label_dict = get_label_dict()

    def generator():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                buf = line[:-1].split('\t')
                label_name = buf[0]
                label = int(label_dict[label_name])

                inputs = np.array(list(map(lambda x: float(x), buf[1].split(','))),
                                  dtype=np.float32)

                yield inputs, [label]

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=((300, ), (1, )),
                                             output_types=(tf.float32, tf.int32))

    return dataset.repeat(epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
        .batch(batch_size=batch_size)


def get_dataset(data_path=None,
                epochs=10,
                shuffle_buffer_size=1024,
                batch_size=16,
                word2id_dict=None):
    return dataset_generator(data_path=data_path,
                             epochs=epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             batch_size=batch_size,
                             word2id_dict=word2id_dict)


if __name__ == "__main__":
    data_dir = os.path.join(get_data_dir(), "text_classification")
    train_path = os.path.join(data_dir, "thucnews_train_vec.txt")

    # TODO
    word2id_dict =

    train_dataset = get_dataset(data_path=train_path,
                                batch_size=4,
                                word2id_dict=word2id_dict)

    # inputs: [None, 300]
    # label: [None, ]
    for i, (inputs, labels) in zip(range(2), train_dataset):
        print(i, inputs.shape, labels.shape)