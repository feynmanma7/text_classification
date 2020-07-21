from jiangziya.utils.config import get_data_dir, get_model_dir, get_label_dict
from jiangziya.text_cnn.pretrained_text_cnn import PretrainedTextCNN, test_model_once
from jiangziya.preprocess.get_word_vector_dict import load_word_vector_dict
import tensorflow as tf
tf.random.set_seed(7)
import os
import numpy as np
np.random.seed(7)


def test_model(model=None,
               test_path=None,
               result_path=None,
               word_vec_dict=None,
               max_seq_len=100):
    # test_data: \label \t title_words \t text_words
    # result: true_label_index \t pred_label_index

    with open(test_path, 'r', encoding='utf-8') as fr:
        with open(result_path, 'w', encoding='utf-8') as fw:

            # {label_name: label_index}
            label_dict = get_label_dict()

            line_cnt = 0
            for line in fr:
                buf = line[:-1].split('\t')

                if len(buf) != 3: # label \t title \t text
                    continue

                label_name = buf[0]
                true_label = label_dict[label_name]

                title = buf[1]
                text = buf[2]
                words = (title + ' ' + text).split(' ')

                inputs = []
                i = 0
                for word in words:
                    if word not in word_vec_dict:
                        continue
                    word_vec = word_vec_dict[word]
                    inputs.append(word_vec)
                    i += 1
                    if i >= max_seq_len:
                        break

                for _ in range(len(inputs), max_seq_len):
                    inputs.append([0] * 300) # 300: embedding_dim

                inputs = np.expand_dims(np.array(inputs), axis=0)
                #print(inputs.shape)

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
    max_seq_len = 350
    model_name = "pretrained_text_cnn"

    checkpoint_dir = os.path.join(get_model_dir(), model_name)

    data_dir = os.path.join(get_data_dir(), "text_classification")
    #val_path = os.path.join(data_dir, "thucnews_val_vec.txt")
    test_path = os.path.join(data_dir, "thucnews_test_seg.txt")
    test_result_path = os.path.join(data_dir, "thucnews_test_" + model_name + ".txt")

    word_vector_dict_path = os.path.join(get_model_dir(), "sogou_vectors.pkl")

    # === Load word_vec_dict
    word_vec_dict = load_word_vector_dict(word_vector_dict_path=word_vector_dict_path)
    print("#word_vec_dict = %d" % len(word_vec_dict))

    # === Build and compile model.
    model = PretrainedTextCNN(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=['acc'])

    # === Load weights.
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    model.load_weights(checkpoint)

    # === Run once, to load weights of checkpoint.
    test_model_once(model=model, seq_len=max_seq_len)

    # === Test
    test_model(model=model,
               test_path=test_path,
               result_path=test_result_path,
               word_vec_dict=word_vec_dict,
               max_seq_len=max_seq_len)
    print("Write done! %s" % test_result_path)