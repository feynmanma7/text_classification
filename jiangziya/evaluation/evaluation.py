from jiangziya.utils.config import get_data_dir, get_model_dir, get_label_dict
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score
import os


def format_score(score):
    return "{:.4f}".format(score)


def get_markdown_table(result_dict=None):
    print('|%s|' % '|'.join(result_dict.keys()))
    print('|%s|' % '|'.join(['---'] * len(result_dict)))
    print('|%s|' % '|'.join(list(map(lambda x: "{:.4f}".format(x), result_dict.values()))))


def evaluation(test_result_path):
    # test_result: y_true_label \t y_pred_label
    y_true = []
    y_pred = []
    with open(test_result_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf = line[:-1].split('\t')
            y_true.append(int(buf[0]))
            y_pred.append(int(buf[1]))

        print("#y_true = %d #y_pred = %d" % (len(y_true), len(y_pred)))
        print(classification_report(y_true=y_true, y_pred=y_pred))
        print('\n')

        accuracy = accuracy_score(y_true, y_pred)
        #recall = recall_score(y_true, y_pred, average='micro')
        #precision_recall_curve(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        result_dict = {'accuracy': accuracy,
                  'f1_micro': f1_micro,
                  'f1_marco': f1_macro,
                  'f1_weighted': f1_weighted}

        get_markdown_table(result_dict)


if __name__ == '__main__':

    #method = "fast_text"
    #method = "pretrained_text_cnn"
    method = "nb"

    data_dir = os.path.join(get_data_dir(), "text_classification")
    test_result_path = os.path.join(data_dir, "thucnews_test_" + method + ".txt")
    #test_result_path = '/Users/flyingman/Data/text_classification/ft_result.txt'

    evaluation(test_result_path=test_result_path)