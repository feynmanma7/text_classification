from jiangziya.utils.config import get_data_dir, get_label_dict
import tensorflow as tf
import numpy as np
import os


def pretrained_dataset_generator(data_path=None,
                      epochs=10,
                      shuffle_buffer_size=1024,
                      batch_size=16):
    # input_data: label \t vec_1, vec_2, ..., vec_300, split by ','
    # Output: inputs, label; [None, 300], [None, 1]

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


def get_pretrained_dataset(data_path=None,
                epochs=10,
                shuffle_buffer_size=1024,
                batch_size=16):
    return pretrained_dataset_generator(data_path=data_path,
                             epochs=epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             batch_size=batch_size)


if __name__ == "__main__":
    data_dir = os.path.join(get_data_dir(), "text_classification")
    train_path = os.path.join(data_dir, "thucnews_train_vec.txt")

    train_dataset = get_pretrained_dataset(data_path=train_path,
                                batch_size=4)

    # inputs: [None, 300]
    # label: [None, ]
    for i, (inputs, labels) in zip(range(2), train_dataset):
        print(i, inputs.shape, labels.shape)