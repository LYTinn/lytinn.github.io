---
title: Neuron Layer Tutorial
date: 2022-09-05 21:04:55
tags:
- tutorial
- Neuron Layer
categories:
- Deep Learning
---
- [Tutorial Q1](#tutorial-q1)
  - [a](#a)
    - [Solution](#solution)
  - [b](#b)
    - [Solution](#solution-1)
  - [c](#c)
    - [Solution](#solution-2)
  - [d](#d)
    - [Solution](#solution-3)
- [Tutorial Q2](#tutorial-q2)
    - [Solution](#solution-4)
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

# Tutorial Q1
Train a softmax layer of neurons to perform the following classification, given the inputs $\mathbf{x} = (x_1, x_2)$ and target class labels $d$:

![Question 1](/figures/dl_tutorial3/Q1.png)
## a
Show on iteration of gradient descent learning at a learning factor 0.05. Initialize the weight to $\left(\begin{matrix}
    0.88&0.08&-0.34\\0.68&-0.39&-0.19
\end{matrix}\right)$ and biases to zero.

### Solution

First import the libraries.
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
Set the parameters and prepare the data
```python
# set parameters
no_epochs = 2000
lr = 0.05
no_inputs = 18
no_classes = 3

# prepare the data
X = np.array([[0,4],[-1,3],[2,3],[-2,2],[0,2],[1,2],[-1,2],[-3,1],[-1,1],
             [2,1],[4,1],[-2,0],[1,0],[3,0],[-3,-1],[-2,-1],[2,-1],[4,-1]]).astype(float)

Y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2]).astype(int)
K = np.zeros((no_inputs, no_classes)).astype(float)
for i in range(len(Y)):
    K[i, Y[i]] = 1
```
Plot the data to see the distribution
```python
plt.figure(1)
plot_pred = plt.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class A')
plot_original = plt.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class B')
plot_original = plt.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class C')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('data points')
plt.legend()
plt.savefig('./figures/t3q1_1.png')
```
The figure is
![t3q1-1](/figures/dl_tutorial3/t3q1_1.png)

Define softmax class, loss function, and training step:
```python
class Softmax:
    def __init__(self):
        self.w = tf.Variable([[0.88, 0.08, -0.34],
                             [0.68, -0.39, -0.19]], dtype=tf.float64)
        self.b = tf.Variable(np.zeros(3), dtype=tf.float64)
        
    def __call__(self, X):
        u = tf.matmul(X, self.w) + self.b
        # reduce_sum will sum the values on dimension of K(neurons)
        p = tf.exp(u)/tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)
        y = tf.argmax(p, axis=1) # on dimension of K(neurons)
        # p.shape = (batch_size, k)
        return u, p, y

def loss(p, k, y):
    loss = -tf.reduce_sum(tf.math.log(p)*k) # sum all elements
    err = tf.reduce_sum(tf.cast(tf.not_equal(
        tf.argmax(k,1), y), tf.int32)) # argmax from the second dimension
    return loss, err

def train(model, inputs, k, learning_rate):
    _, p, y = model(inputs)
    grad_u = -(k - p)
    grad_w = tf.matmul(tf.transpose(inputs), grad_u)
    grad_b = tf.reduce_sum(grad_u, axis = 0)
    
    model.w.assign_sub(learning_rate * grad_w)
    model.b.assign_sub(learning_rate * grad_b)
    
    return grad_u, grad_w, grad_b
```
The initial parameters are:
```python
model = Softmax()

print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
```
> w: [[ 0.88  0.08 -0.34]\
 [ 0.68 -0.39 -0.19]], b: [0. 0. 0.]

The training process:
```python
loss_, err_ = [], []
for epoch in range(no_epochs):
    u_, p_, y_ = model(X)
    l_, e_ = loss(p_, K, y_)
    grad_u_, grad_w_, grad_b_ = train(model, X, K, lr)
    
    if (epoch == 0):
        print('iter: {}'.format(epoch+1))
        print('u: {}'.format(u_))
        print('p: {}'.format(p_))
        print('y: {}'.format(y_))
        print('entropy: {}'.format(l_))
        print('error: {}'.format(e_))
        print('grad_u: {}'.format(grad_u_))
        print('grad_w: {}'.format(grad_w_))
        print('grad_b: {}'.format(grad_b_))

        print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
  
    loss_.append(l_), err_.append(e_)

    if not epoch%100:
        print('epoch:{}, loss:{}, error:{}'.format(epoch,loss_[epoch], err_[epoch]))
```
> iter: 1
u: [[ 2.72 -1.56 -0.76]\
 [ 1.16 -1.25 -0.23]\
 [ 3.8  -1.01 -1.25]\
 [-0.4  -0.94  0.3 ]\
 [ 1.36 -0.78 -0.38]\
 [ 2.24 -0.7  -0.72]\
 [ 0.48 -0.86 -0.04]\
 [-1.96 -0.63  0.83]\
 [-0.2  -0.47  0.15]\
 [ 2.44 -0.23 -0.87]\
 [ 4.2  -0.07 -1.55]\
 [-1.76 -0.16  0.68]\
 [ 0.88  0.08 -0.34]\
 [ 2.64  0.24 -1.02]\
 [-3.32  0.15  1.21]\
 [-2.44  0.23  0.87]\
 [ 1.08  0.55 -0.49]\
 [ 2.84  0.71 -1.17]]\
p: [[0.95725834 0.013251   0.02949065]\
 [0.74688701 0.06708188 0.18603111]\
 [0.98565168 0.00803095 0.00631737]\
 [0.27804803 0.162032   0.55991997]\
 [0.7732904  0.09098136 0.13572824]\
 [0.90523572 0.04785595 0.04690833]\
 [0.53868681 0.14105281 0.32026038]\
 [0.04747873 0.17951914 0.77300213]\
 [0.31422361 0.23987186 0.44590453]\
 [0.90434851 0.06262815 0.03302335]\
 [0.98312509 0.01374584 0.00312907]\
 [0.05738527 0.28423113 0.6583836 ]\
 [0.57321072 0.25756018 0.1692291 ]\
 [0.89569581 0.08125569 0.0230485 ]\
 [0.00794311 0.25526562 0.73679127]\
 [0.02335079 0.33718476 0.63946445]\
 [0.55659139 0.32761246 0.11579616]\
 [0.87953015 0.10452098 0.01594887]]\
y: [0 0 0 2 0 0 0 2 2 0 0 2 0 0 2 2 0 0]\
entropy: 34.49939181290479\
error: 14\
grad_u: [[-0.04274166  0.013251    0.02949065]\
 [-0.25311299  0.06708188  0.18603111]\
 [-0.01434832  0.00803095  0.00631737]\
 [ 0.27804803 -0.837968    0.55991997]\
 [ 0.7732904  -0.90901864  0.13572824]\
 [-0.09476428  0.04785595  0.04690833]\
 [ 0.53868681 -0.85894719  0.32026038]\
 [ 0.04747873 -0.82048086  0.77300213]\
 [ 0.31422361 -0.76012814  0.44590453]\
 [ 0.90434851  0.06262815 -0.96697665]\
 [ 0.98312509  0.01374584 -0.99687093]\
 [ 0.05738527 -0.71576887  0.6583836 ]\
 [ 0.57321072  0.25756018 -0.8307709 ]\
 [ 0.89569581  0.08125569 -0.9769515 ]\
 [ 0.00794311 -0.74473438  0.73679127]\
 [ 0.02335079 -0.66281524  0.63946445]\
 [ 0.55659139  0.32761246 -0.88420384]\
 [ 0.87953015  0.10452098 -0.98405113]]\
grad_w: [[ 12.0257068   12.49953702 -24.52524382]\
 [  2.79893186  -5.36663209   2.56770023]]\
grad_b: [ 6.42794117 -5.32631825 -1.10162292]\
w: [[ 0.27871466 -0.54497685  0.88626219]\
 [ 0.54005341 -0.1216684  -0.31838501]], b: [-0.32139706  0.26631591  0.05508115]\
epoch:0, loss:34.49939181290479, error:14\
epoch:100, loss:1.9333247697935543, error:1\
epoch:200, loss:1.5358837613051912, error:0\
epoch:300, loss:1.3178982377920088, error:0\
epoch:400, loss:1.1695641281651357, error:0\
epoch:500, loss:1.0584597889286107, error:0\
epoch:600, loss:0.9705543681923241, error:0\
epoch:700, loss:0.8984667101829489, error:0\
epoch:800, loss:0.8378243245467658, error:0\
epoch:900, loss:0.7858224171419618, error:0\
epoch:1000, loss:0.7405559495288825, error:0\
epoch:1100, loss:0.70067378582978, error:0\
epoch:1200, loss:0.6651844165017479, error:0\
epoch:1300, loss:0.6333398149163699, error:0\
epoch:1400, loss:0.60456250515345, error:0\
epoch:1500, loss:0.5783979014158144, error:0\
epoch:1600, loss:0.5544821144294328, error:0\
epoch:1700, loss:0.5325195879400361, error:0\
epoch:1800, loss:0.5122671850752352, error:0\
epoch:1900, loss:0.49352262405229586, error:0



## b
Find the weights and bias at the convergence of learning

### Solution
```python
print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
print('p: {}'.format(p_))
print('y: {}'.format(y_))
print('entropy: {}'.format(l_))
print('error: {}'.format(e_))
print('loss:{}, error:{}'.format(loss_[-1], err_[-1]))
```
> w: [[-0.12579118 -3.75032318  4.49611436]\
 [ 5.85809031 -1.23774231 -4.520348  ]], b: [-8.88901039  6.560547    2.32846339]\
p: [[9.99997579e-01 2.42123350e-06 6.98879273e-14]\
 [9.01412271e-01 9.85877291e-02 1.98989511e-11]\
 [9.99974747e-01 2.07701744e-06 2.31758057e-05]\
 [2.02372285e-04 9.99797628e-01 1.41111127e-12]\
 [2.21399607e-01 7.78584445e-01 1.59470295e-05]\
 [9.08146113e-01 8.52056726e-02 6.64821481e-03]\
 [7.52961175e-03 9.92470383e-01 5.33616405e-09]\
 [4.48102390e-09 9.99999996e-01 9.86940548e-15]\
 [6.29515832e-06 9.99993562e-01 1.43223115e-07]\
 [4.18582845e-05 1.26276834e-04 9.99831865e-01]\
 [4.05286113e-09 8.70305629e-12 9.99999996e-01]\
 [1.39363095e-10 9.99999999e-01 1.00151567e-09]\
 [1.30189769e-07 1.77411099e-02 9.82258760e-01]\
 [1.28309324e-11 1.24460046e-09 9.99999999e-01]\
 [3.08521981e-15 1.00000000e+00 7.00324926e-12]\
 [1.15638432e-13 9.99999973e-01 2.66785383e-08]\
 [4.06213763e-14 1.77986902e-07 9.99999822e-01]\
 [3.93243891e-18 1.22648772e-14 1.00000000e+00]]\
y: [0 0 0 1 1 0 1 1 1 2 2 1 2 2 1 1 2 2]\
entropy: 0.4762838541742147\
error: 0\
loss:0.4762838541742147, error:0
```python
# plot learning curves
plt.figure(2)
plt.plot(range(no_epochs), loss_)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/t3q1_2.png')

plt.figure(3)
plt.plot(range(no_epochs), err_)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/t3q1_3.png')
```
![t3q1_2](/figures/dl_tutorial3/t3q1_2.png)
![t3q1_3](/figures/dl_tutorial3/t3q1_3.png)

## c
Indicate the probabilities that the network predicts the classes of trained patterns
### Solution
The probabilities are displayed in the previous result, the $p$ matrix.
## d
Plot the decision boundaries separating the three classes
### Solution
Find out the function of boundaries.
```python
w_, b_ = model.w.numpy(), model.b.numpy()
ww, bb = np.zeros((3, 2)), np.zeros(3)
for i in range(3):
    ww[i, :] = w_[:, i] - w_[:, (i+1)%3]
    bb[i] = b_[i] - b_[(i+1)%3]
    
print('ww: {}'.format(ww))
print('bb: {}'.format(bb))

m = -ww[:, 0]/ww[:, 1]
c = -bb/ww[:, 1]

print('m: {}'.format(m))
print('c: {}'.format(c))

def compute_line(x):
    y = np.zeros((3, x.shape[0]))
    for i in range(3):
        y[i] = m[i]*x + c[i]
        
    return y

xx = np.arange(-4.5, 4.5, 0.01)
yy = compute_line(xx)
```
Plot each boundary individually:
```python
plt.figure(4)
l1 = plt.subplot(1, 1, 1)
l1.plot(xx, yy[0], color = 'black', linestyle = '-')
plot_pred = l1.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class A')
plot_original = l1.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class B')
plot_original = l1.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class C')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-4.5, 4.5)
plt.xlim(xx.min(), xx.max())
plt.text(4.3, 0.1, r'$x_1$')
plt.text(0.1, 4.3, r'$x_2$')
plt.text(-4.5, 2.5, r'$3.3x_1+6.3x_2-13.6=0$')
plt.legend()
plt.savefig('./figures/t3q1_4.png')
```
![t3q1_4](/figures/dl_tutorial3/t3q1_4.png)
```python
plt.figure(5)
l2 = plt.subplot(1, 1, 1)
l2.plot(xx, yy[1], color = 'black', linestyle = '-')
plot_pred = l2.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class A')
plot_original = l2.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class B')
plot_original = l2.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class C')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-4.5, 4.5)
plt.xlim(xx.min(), xx.max())
plt.text(4.3, 0.1, r'$x_1$')
plt.text(0.1, 4.3, r'$x_2$')
plt.text(-4.0, -3.0, r'$-7.6x_1+3.1x_2+3.8=0$')
plt.legend()
plt.savefig('./figures/t3q1_5.png')
```
![t3q1_5](/figures/dl_tutorial3/t3q1_5.png)
```python
plt.figure(6)
l3 = plt.subplot(1, 1, 1)
l3.plot(xx, yy[2], color = 'black', linestyle = '-')
plot_pred = l3.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class A')
plot_original = l3.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class B')
plot_original = l3.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class C')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-4.5, 4.5)
plt.xlim(xx.min(), xx.max())
plt.text(4.3, 0.1, r'$x_1$')
plt.text(0.1, 4.3, r'$x_2$')
plt.text(2.5, 2.0, r'$4.3x_1-9.4x_2+9.8=0$')
plt.legend()
plt.savefig('./figures/t3q1_6.png')
```
![t3q1_6](/figures/dl_tutorial3/t3q1_6.png)

Plot all the boundaries together:
```python
plt.figure(7)
line = plt.subplot(1, 1, 1)
line.plot(xx, yy[0], color = 'black', linestyle = '-')
line.plot(xx, yy[1], color = 'black', linestyle = '-')
line.plot(xx, yy[2], color = 'black', linestyle = '-')
plot_pred = line.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class A')
plot_original = line.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class B')
plot_original = line.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class C')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-4.5, 4.5)
plt.xlim(xx.min(), xx.max())
plt.text(4.3, 0.1, r'$x_1$')
plt.text(0.1, 4.3, r'$x_2$')
plt.legend()
plt.savefig('./figures/t3q1_7.png')
```
![t3q1_7](/figures/dl_tutorial3/t3q1_7.png)

Display only the useful segments:
```python
plt.figure(8)
line = plt.subplot(1, 1, 1)
line.plot(xx[yy[0] > yy[2]], yy[0, yy[0] > yy[2]], color = 'black', linestyle = '-')
line.plot(xx[yy[1] < yy[2]], yy[1, yy[1] < yy[2]], color = 'black', linestyle = '-')
line.plot(xx[yy[2] > yy[0]], yy[2, yy[2] > yy[0]], color = 'black', linestyle = '-')
plot_pred = line.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class A')
plot_original = line.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class B')
plot_original = line.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class C')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-4.5, 4.5)
plt.xlim(xx.min(), xx.max())
plt.text(4.3, 0.1, r'$x_1$')
plt.text(0.1, 4.3, r'$x_2$')
plt.legend()
plt.savefig('./figures/t3q1_8.png')
```
![t3q1_8](/figures/dl_tutorial3/t3q1_8.png)
# Tutorial Q2
This tutorial question is exactly the same as [example 3](#example-3-softmax-classification-of-iris-data), but we will use the build-in function of tensorflow.

### Solution
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import pylab as plt

# set parameters
no_epochs = 1000
batch_size = 16
lr = 0.01

tf.random.set_seed(10)
np.random.seed(1000)

# read the data
iris = datasets.load_iris()
iris.data -= np.mean(iris.data, axis=0)

x_train, x_test, y_train, y_test = train_test_split(iris.data, 
                                                   iris.target,
                                                   test_size=0.2,
                                                   random_state=2)

# build the model
model = Sequential([
    Dense(3, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=lr),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(x_train, y_train,
    epochs=no_epochs,
    batch_size = batch_size,
    verbose=2,
    use_multiprocessing=False,
    validation_data=(x_test, y_test))
```
```python
print('weights: {}'.format(model.layers[0].get_weights()[0]))
print('bias: {}'.format(model.layers[0].get_weights()[1]))
print('entropy = %2.3f, accuracy = %2.3f'%(history.history['val_loss'][-1], history.history['val_accuracy'][-1]))
```
> weights: [[ 0.08131782  1.0666023   0.42606768]\
 [ 1.3533359  -0.31473705  0.36418575]\
 [-3.1573186  -0.43541595  2.0120568 ]\
 [-0.81039417 -1.3970811   1.7261641 ]]\
bias: [-0.78853196  2.226858   -1.4383247 ]\
entropy = 0.112, accuracy = 0.967

```python
plt.figure(1)
plt.plot(range(no_epochs), history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/t3q2a_1.png')

plt.figure(2)
plt.plot(range(no_epochs), history.history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('classification accuracy')
plt.savefig('./figures/t3q2a_2.png')
```
![t3q2a_1](/figures/dl_tutorial3/t3q2a_1.png)
![t2q2a_2](/figures/dl_tutorial3/t3q2a_2.png)
# Tutorial Q3

