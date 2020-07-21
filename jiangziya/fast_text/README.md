# FastText

# 1. Preprocess

+ Segmentation, remove stop-words.

# 2. Pre-trained WordVectors FastText

+ Remove stop-words for representation. 

+ Compute average-embedding of the text.

+ Add dense layers.

+ Add softmax layer in the end. 

# 3. End-to-End FastText

+ Add 2-gram representation.

+ For skip-gram model, given segmentation sequence, 

> A B C D E

Assume `C` is the center word, window size is 2, the context sequence will be

> A, AB, B, BC, CD, D, DE, E   

# 4. Facebook FastText

+ Download and compile facebook fasttext source code.

> https://github.com/facebookresearch/fastText

+ Add `__label__` prefix to **segmentation** data of label.

> awk -F '\t' '{print "__label__"$1" "$2" "$3}' thucnews_train_seg.txt > ft_train.txt

Sample of train data:

> __label__科技 中秋 国庆 黄金 双周 出游 三成 机票 网友 抢走 中秋 国庆 黄金 双周 出游 三成 机票 网友
抢走 中秋 国庆 放假 安排 密度 大 、 间隔 短 ， 称为 史上 最 零碎 长 假期 。 很多 人 避开 国庆 旅游
黄金周 ， 休假 出行 计划 提前 中秋 。

+ Train 

> ./fasttext supervised -input ft_train.txt -output model
 
+ Test model

> ./fasttext test model.bin ft_val.txt 1 

Test result of validate data.

> P@1: 0.95
>
> R@1: 0.95

+ Predict and evaluate 

> ./fasttext predict model.bin ft_test.txt 1 > ft_pred.txt

Data of predict.


# 5. Performance

|method|accuracy|f1_micro|f1_marco|f1_weighted|
|---|---|---|---|---|
|Word2vec FastText, 3 epochs|0.9326|0.9326|0.9149|0.9324|
|Word2vec FastText, 11 epochs|0.9377|0.9377|0.9216|0.9375|
|Word2vec FastText, Stop|0.9373|0.9373|0.9216|0.9371|
|Facebook FastText|0.9496|0.9496|0.9380|0.9496|

