# Text Classification

# 1. Dataset

THUCTC, [http://thuctc.thunlp.org/](http://thuctc.thunlp.org/)

Total files: 836075

num_classes: 14

classes:

> 体育   娱乐   家居   彩票   房产   教育   时尚   
> 时政   星座   游戏   社会   科技   股票   财经

num_files of each class: 

>3578 星座

>37098 财经

>92632 娱乐

>24373 游戏

>50849 社会

>20050 房产

>131604 体育

>41936 教育

>7588 彩票

>154398 股票

>32586 家居

>162929 科技

>13368 时尚

>63086 时政

# 2. Preprocess

+ Get a big file of `label \t title \t text`, one line per text.

> preprocess/extract_text.py 

+ Split into train, validate and test data.

> preprocess/split_train_val_test.py
   
+ Shuffle the whole data with shell.

> shuf a.txt > b.txt (for Unix/Linux)
> gshuf a.txt > b.txt (for MacOS)



# 3. Evaluation Metrics

# 4. Performance

|method|accuracy|f1_micro|f1_marco|f1_weighted|
|---|---|---|---|---|
|FastText|0.9326|0.9326|0.9149|0.9324|

# Reference

1. 知乎: 中文文本分类 pytorch实现, [https://zhuanlan.zhihu.com/p/73176084](https://zhuanlan.zhihu.com/p/73176084)