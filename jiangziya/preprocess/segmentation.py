from jiangziya.utils.config import get_data_dir
from jiangziya.preprocess.stopwords import load_stopwords_dict
from jiangziya.utils.print_util import print_time
import jieba
import os


def segmentation_text(text=None, stopwords_dict=None):
    words = []
    for word in jieba.cut(text, ):
        word = word.strip()
        if len(word) == 0:
            continue
        if word in stopwords_dict:
            continue
        words.append(word)
    return ' '.join(words)


@print_time
def segmentation_file(file_path=None, seg_path=None, stopwords_dict=None):
    # file: label \t title \t text
    # seg_file: \label \t seg_title \t seg_text  split by '\s'

    jieba.enable_paddle()

    stopwords_dict[' '] = True
    stopwords_dict['\t'] = True
    stopwords_dict['\n'] = True

    with open(seg_path, 'w', encoding='utf-8') as fw:
        with open(file_path, 'r', encoding='utf-8') as fr:
            line_cnt = 0
            for line in fr:
                buf = line[:-1].split('\t')
                label = buf[0].strip()
                title = buf[1].strip()
                text = buf[2].strip()

                seg_title = segmentation_text(text=title,
                                              stopwords_dict=stopwords_dict)
                seg_text = segmentation_text(text=text,
                                             stopwords_dict=stopwords_dict)

                fw.write(label + '\t' + seg_title + '\t' + seg_text + '\n')
                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)

            print(line_cnt)


if __name__ == '__main__':
    #file_type = "thucnews_train"
    #file_type = "thucnews_val"
    file_type = "thucnews_test"

    file_name = "shuf_" + file_type + ".txt" # use shuffled file to seg
    seg_file_name = file_type + "_seg.txt"
    file_path = os.path.join(get_data_dir(), "text_classification", file_name)
    seg_path = os.path.join(get_data_dir(), "text_classification", seg_file_name)

    #stopwords_path = os.path.join(get_data_dir(), "nlp", "baidu_stopwords.txt")
    stopwords_dict_path = os.path.join(get_data_dir(), "nlp", "stopwords_dict.pkl")

    # === Load stopwords_dict
    stopwords_dict = load_stopwords_dict(stopwords_dict_path=stopwords_dict_path)
    print("#stopwords_dict = %d" % len(stopwords_dict))

    # === Segmentation
    segmentation_file(file_path=file_path,
                 seg_path=seg_path,
                 stopwords_dict=stopwords_dict)
    print("Write done! %s" % seg_path)
