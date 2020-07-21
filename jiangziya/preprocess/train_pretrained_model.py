from jiangziya.text_cnn.pretrained_text_cnn import PretrainedTextCNN
from jiangziya.utils.config import get_model_dir, get_data_dir, get_log_dir
from jiangziya.text_cnn.pretrained_dataset import get_dataset
from jiangziya.preprocess.get_word_vector_dict import load_word_vector_dict
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import time
import os
import pickle


def train_model():
    total_num_train = 669589 # num_lines of thucnews_train_seg.txt
    total_num_val = 83316 # num_lines of thucnews_test_seg.txt

    max_seq_len = 350 # avg #words in sequence = 380, remove stop-words will go to ~350.

    num_classes = 14
    epochs = 100
    #epochs = 3
    shuffle_buffer_size = 1024 * 2
    batch_size = 32
    patience = 10 # for early stopping

    model_name = "pretrained_text_cnn"

    data_dir = os.path.join(get_data_dir(), "text_classification")
    train_path = os.path.join(data_dir, "thucnews_train_seg.txt")
    val_path = os.path.join(data_dir, "thucnews_test_seg.txt")

    log_dir = os.path.join(get_log_dir(), model_name)
    checkpoint_path = os.path.join(get_model_dir(), model_name, "ckpt")
    history_path = os.path.join(get_log_dir(), "history", model_name + ".pkl")

    word_vector_dict_path = os.path.join(get_model_dir(), "sogou_vectors.pkl")

    # === Load word_vec_dict
    word_vec_dict = load_word_vector_dict(word_vector_dict_path=word_vector_dict_path)
    print("#word_vec_dict = %d" % len(word_vec_dict))

    num_train_batch = total_num_train // batch_size + 1
    num_val_batch = total_num_val // batch_size + 1

    # === tf.data.Dataset
    train_dataset = get_dataset(data_path=train_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size,
                                max_seq_len=max_seq_len,
                                word_vec_dict=word_vec_dict)

    val_dataset = get_dataset(data_path=val_path,
                              epochs=epochs,
                              shuffle_buffer_size=shuffle_buffer_size,
                              batch_size=batch_size,
                              max_seq_len=max_seq_len,
                              word_vec_dict=word_vec_dict)

    # === model
    model = PretrainedTextCNN(num_classes=num_classes)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)

    # loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=['acc'])

    # callbacks
    callbacks = []

    early_stopping_cb = EarlyStopping(monitor='val_loss',
                                      patience=patience,
                                      restore_best_weights=True)
    callbacks.append(early_stopping_cb)

    tensorboard_cb = TensorBoard(log_dir=log_dir)
    callbacks.append(tensorboard_cb)

    checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True,
                                    save_best_only=True)
    callbacks.append(checkpoint_cb)

    # === Train
    history = model.fit(train_dataset,
               epochs=epochs,
               steps_per_epoch=num_train_batch,
               validation_data=val_dataset,
               validation_steps=num_val_batch,
               callbacks=callbacks)

    print(model.summary())

    return history


if __name__ == "__main__":
    start = time.time()
    train_model()
    end = time.time()
    last = end - start
    print("\nTrain done! Lasts: %.2fs" % last)