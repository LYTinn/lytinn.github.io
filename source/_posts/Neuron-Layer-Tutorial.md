---
title: Neuron Layer Tutorial
date: 2022-09-05 21:04:55
tags:
- tutorial
- Neuron Layer
categories:
- Deep Learning
---
- [Example 1](#example-1)
- [Example 2](#example-2)
- [Example 3](#example-3)
- [Example 4](#example-4)
- [Tutorial Q1](#tutorial-q1)
  - [a](#a)
  - [b](#b)
  - [c](#c)
  - [d](#d)
- [Tutorial Q2](#tutorial-q2)
- [Tutorial Q3](#tutorial-q3)

Import corresponding packages first:
```python
import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    os.makedirs('figures')
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```
# Example 1
A perceptron layer of 3 neurons shown in the figure receives 2-dimensional inputs $(x_1, x_2)^T$, and has a weight matrix $\mathbf{W}$ and a bias vector $\mathbf{b}$ given by
$$
\mathbf{W} = \left(\begin{matrix}
  0.133&0.072&-0.155\\-0.001&0.062&-0.072
\end{matrix}\right)
$$

$$
\mathbf{b} = \left(\begin{matrix}
  0.017\\0.009\\0.069
\end{matrix}\right)
$$
Using batch processing, find the output for input patterns:
$$
\left(\begin{matrix}
  0.5\\-1.66
\end{matrix}\right),
\left(\begin{matrix}
  -1.0\\-0.51
\end{matrix}\right),
\left(\begin{matrix}
  0.78\\-0.65
\end{matrix}\right),
\left(\begin{matrix}
  0.04\\-0.2
\end{matrix}\right)
$$
```python
# input data
X = np.array([[0.5,-1.66],[-1.0,-0.51],[0.78,-0.65],[0.04,-0.2]])
print('x:{}'.format(X))
# define weight and bias
W = np.array([[0.133, 0.072, -0.155],
             [-0.001, 0.062, -0.072]])
b = np.array([0.017, 0.009, 0.069])
print('W:{}'.format(W))
print('b:{}'.format(b))
```
```
x:[[ 0.5  -1.66]
 [-1.   -0.51]
 [ 0.78 -0.65]
 [ 0.04 -0.2 ]]
W:[[ 0.133  0.072 -0.155]
 [-0.001  0.062 -0.072]]
b:[0.017 0.009 0.069]
```
```python
# define a class for a perceptron layer
class Layer:
    def __init__(self):
        self.w = tf.Variable(W, dtype=tf.float64)
        self.b = tf.Variable(b, dtype=tf.float64)     
    def __call__(self, x):
        u = tf.matmul(x, self.w) + self.b
        y = tf.sigmoid(u)
        return u, y
```
```python
model = Layer()
print('w: {}, \nb: {}'.format(model.w.numpy(), model.b.numpy()))
```
```
w: [[ 0.133  0.072 -0.155]
 [-0.001  0.062 -0.072]], 
b: [0.017 0.009 0.069]
```
```python
u, y = model(X)
print('u: {}'.format(u))
print('y: {}'.format(y))
```
```
u: [[ 0.08516 -0.05792  0.11102]
 [-0.11549 -0.09462  0.26072]
 [ 0.12139  0.02486 -0.0051 ]
 [ 0.02252 -0.00052  0.0772 ]]
y: [[0.52127714 0.48552405 0.52772653]
 [0.47115955 0.47636263 0.56481328]
 [0.53031029 0.50621468 0.498725  ]
 [0.50562976 0.49987    0.51929042]]

```
# Example 2
# Example 3
# Example 4
# Tutorial Q1
## a
## b
## c
## d
# Tutorial Q2
# Tutorial Q3

