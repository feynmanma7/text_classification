# Naive Bayes

> $y^* = \arg\max_{C_k} p(y=C_k|X) = \arg\max_{C_k} p(X|y=C_k) p(y=C_k) / p(X)$
> $= \arg\max_{C_k} p(X|y=C_k) p(y=C_k)$
> $= \arg\max_{C_k} \prod_i p(X_i | y=C_k) p(y=C_k)$
> $= \arg\max_{C_k} \sum_i log p(X_i | y=C_k) + log p(y=C_k)

Parameters Estimation:
 
1. > $p(y=C_k) = N_{C_k} / N$

2. > $p(X_i | y=C_k) = N_{X_i=x_i} / N_{C_k}$

Laplace Smoothing, $lambda = 1$, 

1. > $p(y=C_k) = (N_{C_k} + 1) / (N + K）$

Prior of $p(y=C_k)$ is 

> $frac{1}{K}$
 
2. $p(X_i=x_i | y=C_k) = (N_{X_i=x_i} + 1) / (N_{C_k} + S_i)$

$S_i$ is the total number of value of feature $i$, for word is 2 (exists or not).

Prior of $p(X_i=x_i | y=C_k)$ is 

> $\frac{1}{2}$
