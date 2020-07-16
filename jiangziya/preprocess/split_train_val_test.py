from jiangziya.utils.config import get_data_dir
import os
import numpy as np


def split_train_val_test(text_path, train_path, val_path, test_path,
                         train_ratio=0.8, val_ratio=0.1):

    fw_train = open(train_path, 'w', encoding='utf-8')
    fw_val = open(val_path, 'w', encoding='utf-8')
    fw_test = open(test_path, 'w', encoding='utf-8')

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = np.random.random()
            if r < train_ratio:
                fw_train.write(line)
            elif r >= train_ratio and r < train_ratio + val_ratio:
                fw_val.write(line)
            else:
                fw_test.write(line)

    fw_train.close()
    fw_val.close()
    fw_test.close()


if __name__ == '__main__':
    text_path = os.path.join(get_data_dir(), "thucnews.txt")
    train_path = os.path.join(get_data_dir(), "thucnews_train.txt")
    val_path = os.path.join(get_data_dir(), "thucnews_val.txt")
    test_path = os.path.join(get_data_dir(), "thucnews_test.txt")

    train_ratio = 0.8
    val_ratio = 0.1
    split_train_val_test(text_path, train_path, val_path, test_path,
                         train_ratio=train_ratio,
                         val_ratio=val_ratio)

    print("Write done!", train_path)