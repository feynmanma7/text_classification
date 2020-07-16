from jiangziya.utils.config import get_data_dir, get_model_dir
from jiangziya.preprocess.get_word_vector_dict import load_word_vector_dict
from jiangziya.utils.time_util import print_time
import os, copy


@print_time
def compute_vectors(text_path=None, vec_path=None, word_vec_dict=None):
    # text: \label \t seg_title \t seg_text; word split by '\s'
    # vec: \label \t vec_1, vec_2, ..., vec_300, split by ','; merge title and text

    with open(vec_path, 'w', encoding='utf-8') as fw:
        with open(text_path, 'r', encoding='utf-8') as fr:
            line_cnt = 0
            for line in fr:
                buf = line[:-1].strip().split('\t')
                label = buf[0]
                title = buf[1].split(' ')
                text = buf[2].split(' ')

                vecs = None
                num_word = 0
                for word in title + text:
                    if word not in word_vec_dict:
                        continue
                    word_vec = word_vec_dict[word] # np.ndarray, [300, ]
                    num_word += 1
                    if vecs is None:
                        vecs = copy.deepcopy(word_vec)
                    else:
                        vecs += copy.deepcopy(word_vec)

                if vecs is None or num_word == 0:
                    continue
                vecs /= num_word

                vecs_str = ','.join(list(map(lambda x: str(x), vecs)))
                fw.write(label + '\t' + vecs_str + '\n')
                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)
            print("Total line %d" % line_cnt)


if __name__ == '__main__':
    file_type = "thucnews_train"
    text_path = os.path.join(get_data_dir(), "text_classification", file_type + "_seg.txt")
    vec_path = os.path.join(get_data_dir(), "text_classification", file_type + "_vec.txt")

    word_vector_dict_path = os.path.join(get_model_dir(), "sogou_vectors.pkl")

    # === Load word_vec_dict
    word_vec_dict = load_word_vector_dict(word_vector_dict_path=word_vector_dict_path)
    print("#word_vec_dict = %d" % len(word_vec_dict))

    # === Compute vectors for file, merge title and text as ONE file.
    compute_vectors(text_path=text_path, vec_path=vec_path, word_vec_dict=word_vec_dict)
    print("Write done! %s" % vec_path)
