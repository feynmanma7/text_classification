# Naive Bayes

> $y^* = \arg\max_{C_k} p(y=C_k|X)$

> $= \arg\max_{C_k} p(X|y=C_k) p(y=C_k) / p(X)$

> $= \arg\max_{C_k} p(X|y=C_k) p(y=C_k)$

> $= \arg\max_{C_k} \prod_i p(X_i | y=C_k) p(y=C_k)$

> $= \arg\max_{C_k} \sum_i log p(X_i | y=C_k) + log p(y=C_k)$

# Parameters Estimation
 
> $p(y=C_k) = N_{C_k} / N$

> $p(X_i | y=C_k) = N_{X_i=x_i} / N_{C_k}$

# Laplace Smoothing, $\lambda = 1$, 

> $p(y=C_k) = (N_{C_k} + 1) / (N + K)$

Prior of $p(y=C_k)$ is $1/K$.

 $p(X_i=x_i | y=C_k) = (N_{X_i=x_i} + 1) / (N_{C_k} + S_i)$

$S_i$ is the total number of value of feature $i$, for word is 2 (exists or not).

Prior of $p(X_i=x_i | y=C_k)$ is 1/2.

# Train

Count and compute $p(y=C_k)$ and $p(X_i|y=C_k)$

# Predict

For given inputs $X={x_{a_1}, x_{a_2}, ..., x_{a_n}}$ with $n$ words, but is far less than the total number of words $N$.

> $y^* = \arg\max_{C_k} \sum_{i=1}^N log p(X_i | y=C_k) + log p(y=C_k)$ 

The left part is 

> $\sum \log p(X_{+}|y=C_k) + \sum \log p(X_{-}|y=C_k)$

$X_{+}$ are words exist in test text, while $X_{-}$ are missing words which are too many.

## Example

Denote $+_1 = \log p(X_{+_1}|y=C_k)$.

To compute 

> $\arg\max_{C_k} \sum +_1, +_2, -_3, -_4, -_5$

> $= \arg\max_{C_k} \sum (+_1, +_2)+ \sum (-_3, -_4, -_5)$

> $= \arg\max_{C_k} \sum (+_1, +_2) + (\sum_{i=1}^5 -_i) - \sum (-_1, -_2)$

# Evaluation 

For text classification, 

Naive Bayes + Chi-square (50 words per label)

|method|accuracy|f1_micro|f1_marco|f1_weighted|Note|
|---|---|---|---|---|---|
|Naive Bayes|0.8211|0.8211|0.8138|0.8273|Chi-Square, 50 words per label|
|Naive Bayes|0.1526|0.1526|0.1703|0.1496|Mutual information, total words=300|
|Naive Bayes|0.6126|0.6126|0.5921|0.6055|MI, 5 words per label
|Naive Bayes|0.6489|0.6489|0.6396|0.6613|MI, 10 words per label|
|Naive Bayes|0.5742|0.5742|0.5815|0.6069|MI, 25 words per label|
|Naive Bayes|0.3940|0.3940|0.4493|0.4329|MI, 50 words per label|
|Naive Bayes|0.1474|0.1474|0.2428|0.1463|MI, 100 words per label|
|Naive Bayes|0.0459|0.0459|0.1019|0.0424|MI, 500 words per label|