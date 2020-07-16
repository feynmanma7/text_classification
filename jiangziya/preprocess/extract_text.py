from jiangziya.utils.config import get_data_dir
from glob import glob
import os, time


def extract_text(data_path=None):
    # data_path: **/体育/*.txt
    label_name = data_path.split('/')[-2]

    with open(data_path, 'r', encoding='utf-8') as f:
        is_header = False
        title = ''
        text = []
        for line in f:
            line = line[:-1]
            if not is_header:
                title = line
                is_header = True

            if len(line) == 1:
                continue

            text.append(line)

        return label_name + '\t' + title + '\t' + ''.join(text)


def extract_dir(text_data_dir=None, text_path=None):
    # text_data_dir: THUNews/体育/*.txt
    # return: label_name \t title \t text

    text_data_path = os.path.join(text_data_dir, "**/*.txt")
    fw = open(text_path, 'w', encoding='utf-8')

    start = time.time()
    line_cnt = 0
    for data_path in glob(text_data_path, recursive=True):
        text = extract_text(data_path=data_path)
        fw.write(text + '\n')
        line_cnt += 1
        if line_cnt % 1000 == 0:
            last = time.time() - start
            print("%d lasts %.2fs" % (line_cnt, last))
            fw.flush()
    fw.close()

    last = time.time() - start
    print("%d lasts %.2fs" % (line_cnt, last))
    print("Write done! %s" % text_path)


if __name__ == '__main__':
    text_data_dir = os.path.join(get_data_dir(), "text_classification", "THUCNews")
    text_path = os.path.join(get_data_dir(), "text_classification", "thucnews.txt")

    extract_dir(text_data_dir=text_data_dir, text_path=text_path)

