---
title: Basic MLP
date: 2022-09-04 09:00:58
tags: 
- MLP
categories:
- Deep Learning
---
- [Neuron Layer](#neuron-layer)
  - [Weight matrix of a layer](#weight-matrix-of-a-layer)
  - [Synaptic input at a layer for single input](#synaptic-input-at-a-layer-for-single-input)
  - [Synaptic input to a layer for batch input](#synaptic-input-to-a-layer-for-batch-input)
  - [Activation at a layer for batch input](#activation-at-a-layer-for-batch-input)
  - [SGD for single layer](#sgd-for-single-layer)
  - [GD for single layer](#gd-for-single-layer)
- [Perceptron Layer](#perceptron-layer)
  - [SGD for perceptron layer](#sgd-for-perceptron-layer)
  - [GD for perceptron layer](#gd-for-perceptron-layer)

# Neuron Layer
A Layer of perceptrons performs
## Weight matrix of a layer
Consider a layer of K neurons:
![K neurons](../figures/K%20neurons.png)

Let $\mathbf{W_k}$ and $b_k$ denote the weight vector and bias of $k$ th neuron. Weights connected to a neuron layer is represented by a weight matrix $\mathbf{W}$ where the columns are given by weight vectors of individual neurons:
$$\mathbf{W} = (\mathbf{w_1}\quad\mathbf{w_2}\quad\dots\quad\mathbf{w_k})$$
and a bias vector $\mathbf{b}$ where each element corresponds to a bias of a neuron:
$$\mathbf{b} = (b_1, b_2, \dots, b_K)^T$$
## Synaptic input at a layer for single input
Given an input pattern $x\in \mathbb{R}^n$ to a layer of $K$ neurons. The synaptic input $u_k$ to $k$ th neuron is:
$$u_k = \mathbf{w}_k^T\mathbf{x}+ b_k$$
where $\mathbf{w}_k$ and $b_k$ denote the weight vector and bias of $k$th neuron. Synaptic input vector $u$ to the layer is:
$$\mathbf{u} = \left(\begin{array}{cc}
    u_1\\u_2\\\vdots\\u_k
\end{array}\right) = \left(\begin{array}{cc}
    \mathbf{w}_1^T\mathbf{x} + b_1\\
    \mathbf{w}_2^T\mathbf{x} + b_2\\
    \vdots\\
    \mathbf{w}_k^T\mathbf{x} + b_k
\end{array}\right) = \left(\begin{array}{cc}
    \mathbf{w}_1^T\\
    \mathbf{w}_2^T\\
    \vdots\\
    \mathbf{w}_k^T\\
\end{array}\right)\mathbf{x} + \left(\begin{array}{cc}
    b_1\\b_2\\\vdots\\b_k
\end{array}\right) = \mathbf{W}^T\mathbf{x} + \mathbf{b}$$
where $\mathbf{W}$ is the weight matrix and $\mathbf{b}$ is the bias vector of the layer.

## Synaptic input to a layer for batch input

Given a set $\{\mathbf{x}_p \}_{p=1}^P$ input patterns to a layer of $K$ neurons where $\mathbf{x}_p\in \mathbb{R}^n$.

Synaptic input $\mathbf{u}_p$ to the layer for an input pattern $\mathbf{x}_p$ is:
$$\mathbf{u}_p = \mathbf{W}^T\mathbf{x}_p + \mathbf{b}$$
The synaptic input matrix $\mathbf{U}$ to the layer for P patterns:
$$\mathbf{U} = \left(\begin{array}{cc}
    \mathbf{x}_1^T\mathbf{W} + \mathbf{b}^T\\
    \mathbf{x}_2^T\mathbf{W} + \mathbf{b}^T\\
    \vdots\\
    \mathbf{x}_P^T\mathbf{W} + \mathbf{b}^T
\end{array}\right) = \left(\begin{array}{cc}
    \mathbf{x}_1^T\\
    \mathbf{x}_2^T\\
    \vdots\\
    \mathbf{x}_P^T
\end{array}\right)\mathbf{W} + \left(\begin{array}{cc}
    \mathbf{b}^T\\\mathbf{b}^T\\\vdots\\\mathbf{b}^T
\end{array}\right) = \mathbf{XW} + \mathbf{B}$$
where rows $\mathbf{U}$ are synaptic inputs corresponding to individual input patterns.

The matrix 
$$\mathbf{B} = \left(\begin{array}{cc}
    \mathbf{b}^T\\\mathbf{b}^T\\\vdots\\\mathbf{b}^T
\end{array}\right)$$ 
has bias vector propagated as rows.

Using batch in deep learning could accerate the speed of training.

## Activation at a layer for batch input

Activation of the layer of synaptic input to the layer due to a batch of patterns:

$$f(\mathbf{U}) = \left(\begin{array}{cc}
    f(\mathbf{u}_1^T) \\ f(\mathbf{u}_2^T) \\\vdots \\f(\mathbf{u}_P^T)
\end{array}\right) = \left(\begin{array}{cc}
    f(\mathbf{u}_1)^T\\
    f(\mathbf{u}_2)^T\\
    \vdots\\
    f(\mathbf{u}_P)^T
\end{array}\right)$$
where activation of each pattern is written as rows.

## SGD for single layer
Computational graph for processing input $(\mathbf{x}, \mathbf{d})$:
![SGD for single layer](../figures/SGD-for-single-layer.png)

$J$ denotes the cost function. Now, we need to compute gradients $\nabla_\mathbf{W}J$ and $\nabla_\mathbf{b}J$ to learn the weight matrix $\mathbf{W}$ and the bias vector $\mathbf{b}$.

Consider $k$th neuron at the layer:
$$u_k = \mathbf{w}_k^T\mathbf{x} + b_k = \sum_{i=0}^nw_{ki}x_i + b_k$$
where n is the rank of $\mathbf{x}$. So that
$$\begin{aligned}
    &\frac{\partial u_k}{\partial\mathbf{W}_k}\\
    =& \left(\begin{array}{cc}
        \frac{\partial (\sum_{i=0}^nw_{ki}x_i + b_k)}{\partial w_{k1}}\\
        \frac{\partial (\sum_{i=0}^nw_{ki}x_i + b_k)}{\partial w_{k2}}\\
        \vdots\\
        \frac{\partial (\sum_{i=0}^nw_{ki}x_i + b_k)}{\partial w_{kn}}
    \end{array}\right)\\
    =&\left(\begin{array}{cc}
        x_1\\x_2\\\vdots\\x_d
    \end{array}\right)\\
    =&\mathbf{x}
\end{aligned}$$
The gradient of the cost with respect to the weight connected to $k$th neuron is:

$$\nabla_{\mathbf{w}_k}J = \frac{\partial J}{\partial u_k}\frac{\partial u_k}{\partial \mathbf{w}_k} = \mathbf{x}\frac{\partial J}{\partial u_k}$$
$$\nabla_{b_k}J = \frac{\partial J}{\partial u_k}\frac{\partial u_k}{\partial b_k} = \frac{\partial J}{\partial u_k}$$
The gradient of $J$ with respect to $\mathbf{W} = (\mathbf{w_1}\quad \mathbf{w_2}\quad\dots\quad\mathbf{w}_k)$:
$$\begin{aligned}
    \nabla_wJ &= (\nabla_{\mathbf{w}1}J\quad\nabla_{\mathbf{w}2}J\quad\dots\quad\nabla_{\mathbf{w}K}J)\\
    &=\left(\mathbf{x}\frac{\partial J}{\partial u_1}\quad\mathbf{x}\frac{\partial J}{\partial u_2}\quad\dots\quad\mathbf{x}\frac{\partial J}{\partial u_K}\right)\\
    &=\mathbf{x}\left(\frac{\partial J}{\partial u_1}\quad\frac{\partial J}{\partial u_2}\quad\dots\quad\frac{\partial J}{\partial u_K}\right)\\
    &=\mathbf{x}(\nabla_\mathbf{u}J)^T
\end{aligned}$$
where
$$\nabla_\mathbf{u}J = \frac{\partial J}{\partial \mathbf{u}} = \left(\begin{array}{cc}
    \frac{\partial J}{\partial u_1}\\\frac{\partial J}{\partial u_2}\\\vdots\\\frac{\partial J}{\partial u_K}
\end{array}\right)$$
Similarly, by substituting $\frac{\partial J}{\partial b_k} = \frac{\partial J}{\partial u_k}$:
$$\nabla_\mathbf{b}J = \left(\begin{array}{cc}
        \frac{\partial J}{\partial b_1}\\
        \frac{\partial J}{\partial b_2}\\\vdots\\
        \frac{\partial J}{\partial b_K}
    \end{array}\right) = \left(\begin{array}{cc}
        \frac{\partial J}{\partial u_1}\\
        \frac{\partial J}{\partial u_2}\\\vdots\\
        \frac{\partial J}{\partial u_K}
    \end{array}\right) = \nabla_\mathbf{u}J$$
Thus, for any income data $\mathbf{x}$, the gradients of $\mathbf{w}$ and $b$ are:
$$\nabla_\mathbf{w}J = \mathbf{x}(\nabla_\mathbf{u}J)^T\\
\nabla_\mathbf{b}J = \nabla_\mathbf{u}J$$
That is, by computing gradient $\nabla_\mathbf{u}J$ with respect to synaptic input $\mathbf{u}$, the gradient of cost $J$ with respect to the weights and biases is obtained.

## GD for single layer
Given a set of patterns $\{(\mathbf{x}_p, \mathbf{d}_p)\}_{p=1}^P$ where $\mathbf{x}_p\in \mathbb{R}^n$ and $\mathbf{d}_p\in \mathbb{R}^K$ for regression and $d_p\in \{1, 2, \dots, K\}$ for classification.

The cost $J$ is given by the sum of cost due to individual patterns:
$$J = \sum_{p=1}^PJ_p$$
where then:
$$\nabla_\mathbf{W}J = \sum_{p=1}^P\nabla_\mathbf{w}J_p$$
Substituting $\nabla_\mathbf{W}J_p = \mathbf{x}_p(\nabla_{\mathbf{u}_p}J_p)^T$:
$$\begin{aligned}
    \nabla_\mathbf{W}J &= \sum_{p=1}^P\mathbf{x}_p(\nabla_{\mathbf{u}_p}J_p)^T\\
    &=\sum_{p=1}^P\mathbf{x}_p(\nabla_{\mathbf{u}_p}J)^T\qquad\text{since }\nabla_{\mathbf{u}_p}J = \nabla_{\mathbf{u}_p}J_p\\
    &=\mathbf{x}_1(\nabla_{\mathbf{u}_1}J)^T + \mathbf{x}_2(\nabla_{\mathbf{u}_2}J)^T + \dots + \mathbf{x}_P(\nabla_{\mathbf{u}_P}J)^T\\
    &= (\mathbf{x}_1\quad\mathbf{x}_2\quad\dots\quad\mathbf{x}_P)\left(\begin{array}{cc}
        (\nabla_{\mathbf{u}_1}J)^T\\
        (\nabla_{\mathbf{u}_2}J)^T\\
        \vdots\\
        (\nabla_{\mathbf{u}_P}J)^T
    \end{array}\right)\\
    &= \mathbf{X}^T\nabla_\mathbf{U}J
\end{aligned}$$
Note that $\mathbf{X} = \left(\begin{array}{cc}
    \mathbf{x}_1^T\\\mathbf{x}_2^T\\\vdots\\\mathbf{x}_P^T
\end{array}\right)$ and $\mathbf{U} = \left(\begin{array}{cc}
    \mathbf{u}_1^T\\\mathbf{u}_2^T\\\vdots\\\mathbf{u}_P^T
\end{array}\right)$
For the biases, substitute with $\nabla_{\mathbf{u}_p}J = \nabla_{\mathbf{u}_p}J_p$
$$\begin{aligned}
    \nabla_\mathbf{b}J &= \sum_{p=1}^P\nabla_\mathbf{b}J_p\\
    &=\sum_{p=1}^P\nabla_{\mathbf{u}_p}J_p\\
    &=\sum_{p=1}^P\nabla_{\mathbf{u}_p}J\\
    &=\nabla_{\mathbf{u}_1}J + \nabla_{\mathbf{u}_2}J + \dots + \nabla_{\mathbf{u}_P}J\\
    &= (\nabla_{\mathbf{u}_1}J\quad\nabla_{\mathbf{u}_2}J\quad\dots\quad\nabla_{\mathbf{u}_P}J)\left(\begin{array}{cc}
        1\\1\\\vdots\\1
    \end{array}\right)\\
    &= (\nabla_\mathbf{U})^T\mathbf{1}_P
\end{aligned}$$
where $\mathbf{1}_P = (1, 1, \dots, 1)^T$ is a vector of $P$ ones.

That is, by computing gradient $\nabla_\mathbf{U}J$ with respect to synaptic input, the weights and biases can be updated.

# Perceptron Layer
A layer of perceptrons performs **multidimensional non-linear regression** and learns a multidimensional non-linear mapping:
$$\phi = \mathbb{R}^n \rightarrow \mathbb{R}^K$$
## SGD for perceptron layer
Given a training pattern $(\mathbf{x}, \mathbf{d})$, note $\mathbf{x} = (x_1, x_2, \dots, x_n)^T \in \mathbb{R}^n$ and $\mathbf{d} = (d_1, d_2, \dots, d_K)^T\in \mathbb{R}^K$. The square error cost function is:
$$J = \frac{1}{2}\sum_{k=1}^K(d_k - y_k)^2$$
where $y_k = f(u_k) = \frac{1}{1+e^{-u_k}}$ and $u_k = \mathbf{x}^T\mathbf{w}_k + b_k$. $\mathbf{u_k}$ is the synaptic input of the $k$th neuron. $J$ is the sum of square errors of all the nueron outputs. Gradient of $J$ with respect to $u_k$ is:
$$\frac{\partial J}{\partial u_k} = \frac{\partial J}{\partial y_k}\frac{\partial y_k}{\partial u_k} = -(d_k - y_k)\frac{\partial y_k}{\partial u_k} = -(d_k - y_k)f'(u_k)$$
Substituting $\nabla_{u_k}J = \frac{\partial J}{\partial u_k} = -(d_k - y_k)f'(u_k)$:
$$\nabla_\mathbf{u}J = -\left(\begin{array}{cc}
    (d_1 - y_1)f'(u_1)\\(d_2 - y_2)f'(u_2)\\\vdots\\(d_1 - y_1)f'(u_1)
\end{array}\right) = -(\mathbf{d} - \mathbf{y})\cdot f'(\mathbf{u})$$
where "$\cdot$" means element-wise multiplication.

The algorithm is:
$$\begin{aligned}
    &\text{Given a training dataset} \{(\mathbf{x}, \mathbf{d})\}\\
    &\text{Set learning parameter }\alpha\\
    &\text{Initialize $\mathbf{W}$ and $\mathbf{b}$}\\
    &\text{Repeat until convergence:}\\
    &\qquad\text{For every pattern }(\mathbf{x}, \mathbf{d}):\\
    &\qquad\qquad \mathbf{u} = \mathbf{W}^T\mathbf{x} + \mathbf{b}\\
    &\qquad\qquad \mathbf{y} = f(\mathbf{u}) = \frac{1}{1+e^{-\mathbf{u}}}\\
    &\qquad\qquad\nabla_\mathbf{u}J = -(\mathbf{d} - \mathbf{y})\cdot f'(\mathbf{u})\\
    &\qquad\qquad\nabla_\mathbf{W}J = \mathbf{x}(\nabla_\mathbf{u}J)^T\\
    &\qquad\qquad\nabla_\mathbf{b}J = \nabla_\mathbf{u}J\\
    &\qquad\qquad\mathbf{W}\leftarrow\mathbf{W} - \alpha\nabla_\mathbf{W}J\\
    &\qquad\qquad\mathbf{b}\leftarrow\mathbf{b} - \alpha\nabla_\mathbf{b}J
\end{aligned}$$
## GD for perceptron layer
Given a training dataset $\{\mathbf{x}_p, \mathbf{d}_p\}_{p=1}^P$. Note $\mathbf{x}_p = (x_{p1}, x_{p2}, \dots, x_{pn})^T \in \mathbb{R}^n$ and $\mathbf{d}_p = (d_{p1}, d_{p2}, \dots, d_{pK})^T\in \mathbb{R}^K$.

The cost function $J$ is given by the sum of square errors:
$$J = \frac{1}{2}\sum_{p=1}^P\sum_{k=1}^K(d_{pk} - y_{pk})^2$$