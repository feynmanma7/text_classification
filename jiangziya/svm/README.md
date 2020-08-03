# 0. Perceptron

+ Assume the data are linear classifiable.

## 0.0 Loss function

> $L(W, b)=max(0, -y_i(W^TX_i + b))$

## 0.1 Decision function

> $y=sgn(W^TX + b)$

# 1. SVM

## 1.0 Decision function

> $y=sgn(W^Tx + b)$

## 1.1 Point-to-line distance

Distance from point $(x_0, y_0)$ to line $Ax + By + C = 0$ is, 

> $d = \frac{Ax_0 + By_0 + C}{\sqrt{A^2 + B^2}}$

## 1.2 Geometry distance

For a correctly classified point $(x_i, y_i)$,

> $y_i (W^T_i + b) = c >= 0$ 

Distance between the correctly classified point and the decision plane ($W^X+b = 0$).

> $gd(W, b) = \frac{y_i(W^TX_i + b)}{||W||_2}$

## 1.3 Maximum-minimum geometry distance

Among all of the classifier separate hyper-planes, $y=(W^TX+ b)$, choose the hyper-plane with maximum value among minimum geometry distance.

> $W^* = \max \gamma$

> $\gamma = \min \frac{y_i(W^TX_i + b)}{||W||_2}$

## 1.4 Object function

> $\max_{W, b} \gamma$

> $s.t. \frac{y_i(W^TX_i + b)}{||W||_2} \ge \gamma$

Thus, 
> $y_i(\frac{W}{||W||_2}\centerdot X_i + \frac{b}{||W||_2}) \ge \gamma$

> $y_i(\frac{W}{||W||_2 \gamma} \centerdot X_i + \frac{b}{||W||_2 \gamma}) \ge 1$

Let $W = \frac{W}{||W||_2 \gamma}$, $b = \frac{b}{||W||_2 \gamma}$, then $\gamma = \frac{W}{||W||_2 W} = \frac{1}{||W||_2}$.

The object function becomes, 

> $\max_{W, b} \frac{1}{||W||_2}$

> $s.t.$  $y_i(W^TX_i + b) \ge 1$

Eqivalent to, 

> $\min_{W, b} \frac{1}{2} ||W||^2$

> $s.t.$  $y_i(W^TX_i + b) \ge 1$

## 1.5 Constraint optimization

Add Lagrange mulipliers, 

> $L(W, b, \alpha) = \frac{1}{2}||W||^2 + \sum_{i=1}^N \alpha_i(1 - y_i(W^TX_i + b))$

> $s.t.$ $\alpha_i \ge 0$

### 1.5.0 Primal problem

Let 

> $\theta(W, b) = \max_{\alpha} L(W, b, \alpha)$

If the constraint of Lagrange function fails, 

> $y_i(W^TX_i + b) < 1$, 

$\theta(W, b) = + \infty$ when $\alpha = + \infty$.

Thus the primal problem is, 

> $\min_{W, b} \theta(W, b)$

> $=\min_{W, b} \max_{\alpha} L(W, b, \alpha)$

> $=p^*$

### 1.5.1 Dual problem

The dual problem is, 

> $d^* = \max_{\alpha}\min_{W, b} L(W, b, \alpha) \le \min_{W, b}\max_{\alpha}L(W, b, \alpha) = p^*$

### 1.5.2 KKT conditions

One of the necessary and sufficient way the dual problem ans primal problem has the equivalent optimal solution are KKT conditions.

For primal problem, 

> $\min f(x)$

> $s.t.$ $\sum g(x_i) = 0$

> $s.t.$ $\sum h(x_i) \le 0$

Add Lagrange multipliers, 

> $L(W, b) = f(x) + \sum^M \alpha_i g(x_i) + \sum^N \beta_i h(x_i)$

The KKT conditions are, 

> $\nabla_{x}L(W, b) = 0$

> $g(x_i) = 0, i = 1, 2, ..., M$

> $h(x_i) \le 0$

> $\beta_i \ge 0$

> $\beta_i h(x_i) = 0$

### 1.5.3 Solution of SVM 

Solve the dual problem, 

> $\max_{\alpha} \min_{W, b} L(W, b, \alpha)$

$L(W, b, \alpha) = \frac{1}{2} ||W||^2 + \sum_{i=1}^N \alpha_i (1 - y_i (W^TX_i + b))$

Solve the minimum problem first, 

> $\min_{W, b} L(W, b, \alpha)$

KKT conditions of the problems are, 

> $\nabla_{W, b}L(W, b, \alpha) = 0$

> $\alpha_i \ge 0$

> $(1 - y_i (W^TX_i + b)) \le 0$

> $\alpha_i (1 - y_i(W^TX_i + b)) = 0$

Thus, 

> $W = \sum_{i=1}^N \alpha_i y_i X_i$

> $\sum_{i=1}^N \alpha_i y_i = 0$

Insert the result above to the Lagrange function, 

> $L(W, b, \alpha)=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i \alpha_j y_i y_j(X_i X_j)$ $- \sum_{i=1}^N \alpha_i y_i (\sum_{j=1}^N \alpha_j y_j X_j)x_i$ $+ \sum_{i=1}^N \alpha_i$

> $=-\frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (X_i X_j)$ $+ \sum_{i=1}^N \alpha_i$

Compute the maximum problem $\max_{\alpha} L(W, b, \alpha)$

> $\max_{\alpha} L(W, b, \alpha)$

> $s.t.$ $\sum_{i=1}^N \alpha_i y_i = 0$

> $\alpha_i \ge 0, i = 1, 2, ..., N$

Got another optimization problem, 

> $min_{\alpha} - L(W, b, \alpha)$

> $s.t.$ $\sum_{i=1}^N \alpha_i y_i = 0$

> $\alpha_i \ge 0, i = 1, 2, ..., N$

## Loss function




