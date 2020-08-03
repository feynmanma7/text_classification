# Data

+ Download THUCNews.zip.

+ Unzip and let one line per-article, thucnews.txt. 

> label \t title \t text 

+ Segmentation, thucnews_train_seg.txt

> label \t title_words \t text_words, words split by '\s'.

# Feature Selection

+ Chi-square

chi_square_dict.pkl

> {label: {word: chi_square}} 

+ Df, document frequency of train data

train_df.pkl

> {word: document_count}

> {word: True}

# Sparse Features

In LibSVM format.

# Dense Features

To use LR, SVM, GBDT easily.


+ Tf-idf, tf * train_df, using feature selection (such as Chi-square)

chosen_word_dict.pkl, word chosen by chi-square.