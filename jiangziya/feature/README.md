# Feature Selection

# 1. Chi-Square 

Pronounced `Kappa` in Greece.

> $\chi^2 = \frac{(Observed - Expected)^2}{Expected}$

## 1.1 Example

|Feature\Label|Y=1|Y=0|Total|
|-----|-------|-------|-----|
|X=1|O11=34|O12=6|R1=40|
|X=0|O21=20|O22=24|R2=44|
|Total|C1=54|C2=30|N=84|

The expected value of (row1, col1) is, 
> $E11 = \frac{R1}{N} * \frac{C1}{N} * N = \frac{R1*C1}{N}$ 

The expected value of (rowp, colq) is, 
> $Epq = \frac{Rp*Cq}{N}$

> $\chi^2 = \sum_p\sum_q \frac{(Opq-Epq)^2}{Epq}=N(\sum_p\sum_q \frac{Opq^2}{Rp * Cq} - 1)$

In this example, 

> $\chi^2 = 84 * (\frac{34^2}{40*54} + \frac{20^2}{44*54} + \frac{6^2}{40*30} + \frac{24^2}{44*30} - 1) = 14.2715$

Freedom is, 

> $(#R-1) * (#C-1) = (2-1) * (2-1) = 1 * 1 = 1$

Check the Chi-squaure distribution table, the `1`-freedom, `95%`-confidence-level value(`0.05`-p-value) is 3.84.

For $\chi^2 = 14.2715 > 3.84$, thus the original hyper-thesis

> the two random varariables X and Y are not correlated.

is `WRONG` under the 0.95 confidence level.

Which is to say, 

> X and Y are correlated.

For classification problem, 

> The `Bigger` the Chi-square value, the `Importance` the feature.

Preference:[https://www.cnblogs.com/dacc123/p/8746247.html](https://www.cnblogs.com/dacc123/p/8746247.html)

## 1.2 Text Classification

Total number of classes is 14. Treat each class separately.

|Feature\Label|Y=(class_i=True)|Y=(class_i=False)|Total|
|-----|-------|-------|---|
|X_i=(word_i=True)|count_1|count_2|R1|
|X_i=(word_i=False)|count_1|count_2|R2|
|Total|C1|C2|N

> $\chi^2 = \sum_{p=1}^2\sum_{q=1}^{2} \frac{(Opq-Epq)^2}{Epq}=N(\sum_p\sum_q \frac{Opq^2}{Rp * Cq} - 1)$

Freedom is, 

> $(#R-1) * (#C-1) = (2-1) * (2-1) = 1$

The 95%-confidence-level of freedom 1 is `3.84`。

For programming, 

|Word|Y=class_1|Y!=class_1|Total|
|--|--|--|--|
|X_i=True|O11|O12|R1=df
|X_i=False|C1-O11|C2-O12|R2
|Total|C1|C2|N

1. Count $C_k = \sum_{i=1}^M \mathbb{I}(y_i=C_k)$ and $N$.

2. For each word, count $O_{1k}$, then $O_{2k} = C_k - O_{1k}$.

3. For each word, $R1 = \sum_k O_{1k} = df$, document frequenct

4. For each label, compute $\chi^2$ for each word.

5. For each label, sort word by $\chi^2$, do feature selection.

# 