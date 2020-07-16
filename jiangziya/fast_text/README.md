# FastText

# 1. Preprocess

1. Segmentation, remove stop-words.

# 2. Pre-trained Model

1. Remove stop-words for representation. 

2. Compute average-embedding of the text.

3. Add dense layers.

4. Add softmax layer in the end. 

# 3. End-to-End Fast Text

1. Add 2-gram representation.

2. For skip-gram model, given segmentation sequence, 

> A B C D E

Assume `C` is the center word, window size is 2, the context sequence will be

> A, AB, B, BC, CD, D, DE, E   

