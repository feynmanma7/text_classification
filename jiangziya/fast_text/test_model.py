from jiangziya.utils.config import get_data_dir, get_model_dir, get_label_dict
from jiangziya.fast_text.fast_text import PretrainedFastText, test_pretrained_fast_text_once
import tensorflow as tf
tf.random.set_seed(7)
import os
import numpy as np
np.random.seed(7)


def test_model(model=None, vec_path=None, result_path=None):
    # data: label_name \t vec_1, vec_2, ..., vec_300; split by ','
    # result: true_label_index \t pred_label_index

    with open(vec_path, 'r', encoding='utf-8') as fr:
        with open(result_path, 'w', encoding='utf-8') as fw:

            # {label_name: label_index}
            label_dict = get_label_dict()

            line_cnt = 0
            for line in fr:
                buf = line[:-1].split('\t')
                label_name = buf[0]
                true_label = label_dict[label_name]

                # [1, 300]
                inputs = np.array(list(map(lambda x: float(x), buf[1].split(','))),
                                  dtype=np.float32).reshape((-1, 300))
                # [1, num_classes=14]
                softmax = model(inputs)

                pred_label = np.argmax(softmax, axis=1)
                fw.write(str(true_label) + '\t' + str(pred_label[0]) + '\t' + '\n')
                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)
            print("Total line %d" % line_cnt)


if __name__ == '__main__':
    num_classes = 14
    checkpoint_dir = os.path.join(get_model_dir(), "fast_text")

    data_dir = os.path.join(get_data_dir(), "text_classification")
    #val_path = os.path.join(data_dir, "thucnews_val_vec.txt")
    test_path = os.path.join(data_dir, "thucnews_test_vec.txt")
    test_result_path = os.path.join(data_dir, "thucnews_test_fast_text.txt")

    # === Build and compile model.
    model = PretrainedFastText(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=['acc'])

    # === Load weights.
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    model.load_weights(checkpoint)

    # === Run once, to load weights of checkpoint.
    test_pretrained_fast_text_once(model=model)

    # === Test
    test_model(model=model, vec_path=test_path, result_path=test_result_path)
    print("Write done! %s" % test_result_path)