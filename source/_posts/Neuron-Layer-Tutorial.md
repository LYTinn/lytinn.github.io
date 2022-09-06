---
title: Neuron Layer Tutorial
date: 2022-09-05 21:04:55
tags:
- tutorial
- Neuron Layer
categories:
- Deep Learning
---
- [Example 1 Output of perceptron layer](#example-1-output-of-perceptron-layer)
- [Example 2 GD of a softmax classification](#example-2-gd-of-a-softmax-classification)
- [Example 3 Softmax classification of iris data](#example-3-softmax-classification-of-iris-data)
- [Example 4 GD of a perceptron layer](#example-4-gd-of-a-perceptron-layer)
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
# Example 1 Output of perceptron layer
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
>x:[[ 0.5  -1.66]\
 [-1.   -0.51]\
 [ 0.78 -0.65]\
 [ 0.04 -0.2 ]]\
W:[[ 0.133  0.072 -0.155]\
 [-0.001  0.062 -0.072]]\
b:[0.017 0.009 0.069]

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
>w: [[ 0.133  0.072 -0.155]\
 [-0.001  0.062 -0.072]], \
b: [0.017 0.009 0.069]

```python
u, y = model(X)
print('u: {}'.format(u))
print('y: {}'.format(y))
```
>u: [[ 0.08516 -0.05792  0.11102]\
 [-0.11549 -0.09462  0.26072]\
 [ 0.12139  0.02486 -0.0051 ]\
 [ 0.02252 -0.00052  0.0772 ]]\
y: [[0.52127714 0.48552405 0.52772653]\
 [0.47115955 0.47636263 0.56481328]\
 [0.53031029 0.50621468 0.498725  ]\
 [0.50562976 0.49987    0.51929042]]


# Example 2 GD of a softmax classification
Train a softmax regression layer of neurons to perform the following classification:
$$
(0.94\quad0.18)\rightarrow \text{class A}\\
(-0.58\quad-0.53)\rightarrow \text{class B}\\
(-0.23\quad-0.31)\rightarrow \text{class B}\\
(0.42\quad-0.44)\rightarrow \text{class A}\\
(0.5\quad-1.66)\rightarrow \text{class C}\\
(-1.0\quad-0.51)\rightarrow \text{class B}\\
(0.78\quad-0.65)\rightarrow \text{class A}\\
(0.04\quad-0.20)\rightarrow \text{class C}
$$
use a learning factor $\alpha = 0.05$.
```python
# set parameters of the layer and for learning
num_epochs = 3000
num_inputs = 2
num_classes = 3
lr = 0.05

SEED = 10
np.random.seed(SEED)
```
```python
# prepare inputs and outputs
X = np.array([[0.94, 0.18],[-0.58, -0.53],[-0.23, -0.31],[0.42, -0.44],
              [0.5, -1.66],[-1.0, -0.51],[0.78, -0.65],[0.04, -0.20]])
Y = np.array([0, 1, 1, 0, 2, 1, 0, 2])
K = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]]).astype(float)

print(X)
print(Y)
print(lr)
```
>[[ 0.94  0.18]\
 [-0.58 -0.53]\
 [-0.23 -0.31]\
 [ 0.42 -0.44]\
 [ 0.5  -1.66]\
 [-1.   -0.51]\
 [ 0.78 -0.65]\
 [ 0.04 -0.2 ]]\
[0 1 1 0 2 1 0 2]\
0.05

```python
# define the class for the softmax layer
class Softmax_Layer:
    def __init__(self, no_inputs, no_classes):
        self.w = tf.Variable(np.random.rand(no_inputs, no_classes), dtype=tf.float64)
        self.b = tf.Variable(tf.zeros([no_classes], dtype=tf.float64))
        
    def __call__(self, x):
        u = tf.matmul(x, self.w) + self.b
        p = tf.exp(u)/tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)
        y = tf.argmax(p, axis=1)
        return u, p, y
    
def loss(p, k, y):
    entropy = -tf.reduce_sum(tf.math.log(p)*k)
    error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(k,1),y),tf.int32))
    return entropy, error

def train(model, inputs, k, learning_rate):
    _, p, y = model(inputs)
    grad_u = -(k-p)
    grad_w = tf.matmul(tf.transpose(inputs), grad_u)
    grad_b = tf.reduce_sum(grad_u, axis=0) # axis refers to the dimension of tensor
    
    model.w.assign_sub(learning_rate * grad_w)
    model.b.assign_sub(learning_rate * grad_b)
    
    return grad_u, grad_w, grad_b
```
```python
# Initialize the layer
model = Softmax_Layer(num_inputs, num_classes)

print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
```
>
w: [[0.44183317 0.43401399 0.61776698]\
 [0.51313824 0.65039718 0.60103895]], b: [0. 0. 0.]

```python
loss_, err_ = [], []
for epoch in range(num_epochs):
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

>iter: 1\
u: [[ 0.50768807  0.52504465  0.68888797]\
 [-0.52822651 -0.59643862 -0.67685549]\
 [-0.26069449 -0.30144634 -0.32840848]\
 [-0.04021089 -0.10388888 -0.00499501]\
 [-0.6308929  -0.86265233 -0.68884117]\
 [-0.70353368 -0.76571656 -0.92429684]\
 [ 0.01109002 -0.08422725  0.09118292]\
 [-0.08495432 -0.11271888 -0.09549711]]\
p: [[0.31092953 0.31637331 0.37269716]\
 [0.35766004 0.33407677 0.30826319]\
 [0.34547147 0.33167587 0.32285266]\
 [0.33623047 0.31548744 0.34828209]\
 [0.36538548 0.28980071 0.34481382]\
 [0.36474817 0.34275787 0.29249396]\
 [0.33417186 0.30379045 0.36203769]\
 [0.33759491 0.32835067 0.33405442]]\
y: [2 0 0 2 0 0 2 0]\
entropy: 8.78616183666182\
error: 8\
grad_u: [[-0.68907047  0.31637331  0.37269716]\
 [ 0.35766004 -0.66592323  0.30826319]\
 [ 0.34547147 -0.66832413  0.32285266]\
 [-0.66376953  0.31548744  0.34828209]\
 [ 0.36538548  0.28980071 -0.65518618]\
 [ 0.36474817 -0.65724213  0.29249396]\
 [-0.66582814  0.30379045  0.36203769]\
 [ 0.33759491  0.32835067 -0.66594558]]\
grad_w: [[-1.90130829  2.02207871 -0.12077043]\
 [-0.55592222  0.0692429   0.48667931]]\
grad_b: [-0.24780807 -0.43768692  0.68549499]\
w: [[0.53689859 0.33291006 0.6238055 ]\
 [0.54093435 0.64693504 0.57670499]], b: [ 0.0123904   0.02188435 -0.03427475]\
epoch:0, loss:8.78616183666182, error:8\
epoch:100, loss:2.8684515778635955, error:1\
epoch:200, loss:2.2883120189878925, error:1\
epoch:300, loss:1.9730210412198046, error:0\
epoch:400, loss:1.7592913803333123, error:0\
epoch:500, loss:1.5998765664851753, error:0\
epoch:600, loss:1.4740705847133975, error:0\
epoch:700, loss:1.370955701326433, error:0\
epoch:800, loss:1.2841033606723224, error:0\
epoch:900, loss:1.2094309742118017, error:0\
epoch:1000, loss:1.144196610636459, error:0\
epoch:1100, loss:1.0864768286088007, error:0\
epoch:1200, loss:1.034874189355273, error:0\
epoch:1300, loss:0.9883435309164786, error:0\
epoch:1400, loss:0.9460838064030327, error:0\
epoch:1500, loss:0.9074680758355147, error:0\
epoch:1600, loss:0.8719966867182616, error:0\
epoch:1700, loss:0.8392650728596022, error:0\
epoch:1800, loss:0.8089410612158957, error:0\
epoch:1900, loss:0.780748532875915, error:0\
epoch:2000, loss:0.7544554326694685, error:0\
epoch:2100, loss:0.7298648182889294, error:0\
epoch:2200, loss:0.7068080743670997, error:0\
epoch:2300, loss:0.6851396950546537, error:0\
epoch:2400, loss:0.6647332206708897, error:0\
epoch:2500, loss:0.6454780355864276, error:0\
epoch:2600, loss:0.6272768172078264, error:0\
epoch:2700, loss:0.6100434831452572, error:0\
epoch:2800, loss:0.5937015238244387, error:0\
epoch:2900, loss:0.5781826364208404, error:0

```python
print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
print('loss: {}'.format(loss_[-1]))
```
>w: [[ 14.24025851 -13.00690374   0.26025937]\
 [  4.56356541  -1.94667656  -0.85231447]], b: [-0.52891834 -0.47039985  0.99931818]\
loss: 0.5635695814410673

```python
plt.figure(1)
plot_pred = plt.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class 1')
plot_original = plt.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class 2')
plot_original = plt.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class 3')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('data points')
plt.legend()
plt.savefig('./figures/3.2_1.png')
```
![3.21](/public/figures/dl_tutorial3/3.2_1.png)
```python
plt.figure(2)
plt.plot(range(num_epochs), loss_)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/3.2_2.png')
```
![3.22](/public/figures/dl_tutorial3/3.2_2.png)
```python
plt.figure(3)
plt.plot(range(num_epochs), err_)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/3.2_3.png')
```
![3.23](/public/figures/dl_tutorial3/3.2_3.png)
# Example 3 Softmax classification of iris data
Iris data has three classes of iris flower. There are four features:
- Sepal length
- Sepal width
- Petal Length
- Petal width

Using softmax classification to classify the iris data.
```python
# set the parameters
no_epochs = 2500
lr = 0.5

SEED = 100
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
```python
# prepare iris dataset
from sklearn import datasets

no_inputs = 4
no_outputs = 3

# input data
iris = datasets.load_iris()
iris.data -= np.mean(iris.data, axis=0)

no_data = len(iris.data)

X = iris.data

# convert the targets into one-hot matrix
Y = np.zeros((no_data, no_outputs))
for i in range(no_data):
    Y[i, iris.target[i]] = 1
    
print(np.shape(X))
print(np.shape(Y))
```
>(150, 4)\
(150, 3)

```python
# define a class of softmax layer
class SoftmaxLayer():
    def __init__(self, no_inputs, no_outputs):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(w_init(shape=(no_inputs,no_outputs), dtype=tf.float64))
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(b_init(shape=(no_outputs), dtype=tf.float64))

    def __call__(self, x):
        u = tf.matmul(x, self.w) + self.b
        return tf.exp(u)/tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)

def loss(targets, logits):
    t_float = tf.cast(targets, tf.float64)
    losses = -tf.reduce_mean(tf.reduce_sum(tf.math.log(logits)*targets, axis=1))
    class_err = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(logits, axis=1), tf.argmax(targets, axis=1)), dtype=tf.int32))
    return losses, class_err

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss, current_err = loss(outputs, model(inputs))
    dw, db = t.gradient(current_loss, [model.w, model.b])
    model.w.assign(model.w - learning_rate * dw)
    model.b.assign(model.b - learning_rate * db)
```
```python
# initialize the softmax layer
model = SoftmaxLayer(no_inputs, no_outputs)

print(model.w.numpy(), model.b.numpy())
```
>[[ 0.00470517  0.00424244 -0.0228833 ]\
 [-0.0086293  -0.03198624  0.05250187]\
 [ 0.10071415 -0.00131456 -0.00903195]\
 [-0.01193019 -0.04326576 -0.04804788]] [0. 0. 0.]


```python
# train the model
entropy, err = [], []
for epoch in range(no_epochs):
    entropy_, err_ = loss(Y, model(X))
    entropy.append(entropy_), err.append(err_)

    train(model, X, Y, learning_rate=lr)

    if epoch%100 == 0:
        print('Epoch %2d:  loss=%2.5f:  error=%3d'%(epoch, entropy[-1], err[-1]))

entropy_, err_ = loss(Y, model(X))
print('test error=%3d'%err_)
```

>Epoch  0:  loss=1.20477:  error=140\
Epoch 100:  loss=0.16400:  error=  5\
Epoch 200:  loss=0.11847:  error=  4\
Epoch 300:  loss=0.09852:  error=  3\
Epoch 400:  loss=0.08708:  error=  3\
Epoch 500:  loss=0.07958:  error=  2\
Epoch 600:  loss=0.07424:  error=  2\
Epoch 700:  loss=0.07023:  error=  2\
Epoch 800:  loss=0.06710:  error=  2\
Epoch 900:  loss=0.06459:  error=  2\
Epoch 1000:  loss=0.06252:  error=  2\
Epoch 1100:  loss=0.06078:  error=  3\
Epoch 1200:  loss=0.05930:  error=  3\
Epoch 1300:  loss=0.05802:  error=  3\
Epoch 1400:  loss=0.05690:  error=  3\
Epoch 1500:  loss=0.05592:  error=  3\
Epoch 1600:  loss=0.05504:  error=  3\
Epoch 1700:  loss=0.05426:  error=  3\
Epoch 1800:  loss=0.05355:  error=  3\
Epoch 1900:  loss=0.05291:  error=  3\
Epoch 2000:  loss=0.05233:  error=  3\
Epoch 2100:  loss=0.05179:  error=  3\
Epoch 2200:  loss=0.05130:  error=  3\
Epoch 2300:  loss=0.05085:  error=  3\
Epoch 2400:  loss=0.05043:  error=  3\
test error=  3

```python
# print learned weights
print('w: %s, b: %s'%(model.w.numpy(), model.b.numpy()))
```

>w: [[-5.64468184e-01  1.29034580e+00 -7.39813302e-01]\
 [ 2.43509283e+00  1.08464962e-03 -2.42429116e+00]\
 [-5.19366825e+00 -2.52485178e-01  5.53652107e+00]\
 [-2.38732752e+00 -2.88848158e+00  5.17256527e+00]], b: [-0.98272709  5.52615415 -4.54342706]
 
 ```python
 # plot learning curves
plt.figure(2)
plt.plot(range(no_epochs), entropy)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/3.3_1.png')

plt.figure(3)
plt.plot(range(no_epochs), np.array(err))
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/3.3_2.png')
```
![3.21](/public/figures/dl_tutorial3/3.2_1.png)
![3.22](/public/figures/dl_tutorial3/3.2_2.png)
# Example 4 GD of a perceptron layer
Design a perceptron layer to perform the following mapping using GD learning and learning rate $\alpha=0.1$:
![GD of a perceptron layer](/public/figures/dl_tutorial3/GD-perceptron-layer.png)
```python
# set the parameter for the layer and training
no_features = 3
no_labels = 2
no_data = 8

lr = 0.1
no_iters = 10000

SEED = 10
np.random.seed(SEED)
```
```python
# generate training data
X = np.random.rand(no_data, no_features)
Y = np.zeros((no_data, no_labels))
Y[:,0] = (X[:,0] + X[:,1]**2 + X[:, 2]**3 + np.random.rand(no_data))/4
Y[:,1] = (X[:,0] + X[:,1] + X[:, 2] + X[:,0]*X[:,1]*X[:,2] + np.random.rand(no_data))/5

print('X = {}'.format(X))
print('Y = {}'.format(Y))
print('alpha = {}'.format(lr))
```
> X = [[0.77132064 0.02075195 0.63364823]\
 [0.74880388 0.49850701 0.22479665]\
 [0.19806286 0.76053071 0.16911084]\
 [0.08833981 0.68535982 0.95339335]\
 [0.00394827 0.51219226 0.81262096]\
 [0.61252607 0.72175532 0.29187607]\
 [0.91777412 0.71457578 0.54254437]\
 [0.14217005 0.37334076 0.67413362]]\
Y = [[0.36700015 0.46890243]\
 [0.36067172 0.37505132]\
 [0.34976828 0.24872748]\
 [0.48444787 0.41710316]\
 [0.36332573 0.28887784]\
 [0.43984029 0.51677508]\
 [0.59832905 0.51552032]\
 [0.27739117 0.37034263]]\
alpha = 0.1

```python
class Perceptron_Layer():
    def __init__(self, no_features, no_labels):
        self.w = tf.Variable(np.random.rand(no_features, no_labels)*0.05, dtype=tf.float64)
        self.b = tf.Variable(tf.zeros([no_labels], dtype=tf.float64))

    def __call__(self, x):
        u = tf.matmul(x, self.w) + self.b
        y = tf.sigmoid(u)
        return u, y
    
def loss(predicted_y, target_y):
    return tf.reduce_mean(tf.reduce_sum(tf.square(target_y - predicted_y), axis=1))

def train(model, inputs, outputs, learning_rate):
    _, y = model(inputs)
    dy = y*(1 - y)
    grad_u = -(outputs - y)*dy
    grad_w = tf.matmul(tf.transpose(inputs), grad_u)
    grad_b = tf.reduce_sum(grad_u, axis = 0)
    
    model.w.assign_sub(learning_rate * grad_w)
    model.b.assign_sub(learning_rate * grad_b)
    
    return dy, grad_u, grad_w, grad_b
```
```python
model = Perceptron_Layer(no_features, no_labels)

print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
```
> w: [[0.02737931 0.04096435]\
 [0.00994738 0.04284252]\
 [0.01758263 0.03773238]], b: [0. 0.]

```python
# train the perceptron layer
cost = []
for i in range(no_iters):
    u_, y_ = model(X)
    loss_ = loss(y_, Y)
    dy_, grad_u_, grad_w_, grad_b_ = train(model, X, Y, lr)
    
    if (i < 2 or i == no_iters - 1):
        print('iter: {}'.format(i+1))
        print('u: {}'.format(u_))
        print('y: {}'.format(y_))
    
        print('m.s.e: {}'.format(loss_))
          
        print('dy: {}'.format(dy_))
        print('grad_u: {}'.format(grad_u_))
        print('grad_w: {}'.format(grad_w_))
        print('grad_b: {}'.format(grad_b_))

        print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
  
    cost.append(loss_)

    if not i%200:
        print('epoch:{}, loss:{}'.format(i,cost[i]))
```
> iter: 1
u: [[0.03246586 0.05639477]\
 [0.02941309 0.06051367]\
 [0.01596152 0.04707752]\
 [0.02599938 0.06895513]\
 [0.01949109 0.05276747]\
 [0.02908206 0.06702673]\
 [0.04177553 0.08868174]\
 [0.01945932 0.04725543]]\
y: [[0.50811575 0.51409496]\
 [0.50735274 0.5151238 ]\
 [0.5039903  0.51176721]\
 [0.50649948 0.51723195]\
 [0.50487262 0.51318881]\
 [0.50727    0.51675041]\
 [0.51044236 0.52215592]\
 [0.50486468 0.51181166]]\
m.s.e: 0.040125181814308984\
dy: [[0.24993413 0.24980133]\
 [0.24994594 0.24977127]\
 [0.24998408 0.24986153]\
 [0.24995776 0.24970306]\
 [0.24997626 0.24982606]\
 [0.24994715 0.24971942]\
 [0.24989096 0.24950912]\
 [0.24997633 0.24986048]]\
grad_u: [[ 3.52696045e-02  1.12891542e-02]\
 [ 3.66623250e-02  3.49860812e-02]\
 [ 3.85530485e-02  6.57235085e-02]\
 [ 5.51197077e-03  2.50024655e-02]\
 [ 3.53833616e-02  5.60387245e-02]\
 [ 1.68538644e-02 -6.16146671e-06]\
 [-2.19620871e-02  1.65564230e-03]\
 [ 5.68629937e-02  3.53475203e-02]]\
grad_w: [[0.06117103 0.05689372]\
 [0.08792995 0.12787342]\
 [0.10245525 0.1202335 ]]\
grad_b: [0.20313508 0.23003693]\
w: [[0.0212622  0.03527498]\
 [0.00115438 0.03005517]\
 [0.00733711 0.02570903]], b: [-0.02031351 -0.02300369]\
epoch:0, loss:0.040125181814308984\
iter: 2\
u: [[ 0.00075957  0.02111881]\
 [-0.00216746  0.02417237]\
 [-0.01398353  0.01118853]\
 [-0.01064889  0.02522192]\
 [-0.01367601  0.01342131]\
 [-0.00431515  0.02779948]\
 [ 0.00400599  0.04479576]\
 [-0.01191349  0.0105635 ]]\
y: [[0.50018989 0.50527951]\
 [0.49945813 0.5060428 ]\
 [0.49650418 0.5027971 ]\
 [0.4973378  0.50630515]\
 [0.49658105 0.50335528]\
 [0.49892122 0.50694942]\
 [0.5010015  0.51119707]\
 [0.49702166 0.50264085]]\
m.s.e: 0.03653319230796048\
dy: [[0.24999996 0.24997213]\
 [0.24999971 0.24996348]\
 [0.24998778 0.24999218]\
 [0.24999291 0.24996025]\
 [0.24998831 0.24998874]\
 [0.24999884 0.24995171]\
 [0.249999   0.24987463]\
 [0.24999113 0.24999303]]\
grad_u: [[ 0.03329743  0.00909326]\
 [ 0.03469656  0.03274309]\
 [ 0.03668218  0.06351542]\
 [ 0.00322239  0.02229695]\
 [ 0.03331227  0.05361694]\
 [ 0.01477016 -0.00245594]\
 [-0.02433179 -0.00108027]\
 [ 0.05490568  0.03307363]]\
grad_w: [[0.05386745 0.04849972]\
 [0.07892824 0.11736362]\
 [0.09336808 0.10968475]]\
grad_b: [0.18655489 0.21080307]\
w: [[ 0.01587546  0.03042501]\
 [-0.00673844  0.01831881]\
 [-0.0019997   0.01474056]], b: [-0.038969 -0.044084]\
epoch:200, loss:0.01236920027196445\
epoch:400, loss:0.009547814990014315\
epoch:600, loss:0.007721123872008595\
epoch:800, loss:0.006416033024162599\
epoch:1000, loss:0.005465560905166514\
epoch:1200, loss:0.004769636527024338\
epoch:1400, loss:0.0042587309513468456\
epoch:1600, loss:0.003882875651306887\
epoch:1800, loss:0.0036058395774580592\
epoch:2000, loss:0.003401265014784655\
epoch:2200, loss:0.0032499348472099436\
epoch:2400, loss:0.0031378085266889396\
epoch:2600, loss:0.003054605115715101\
epoch:2800, loss:0.0029927797035383145\
epoch:3000, loss:0.0029467830194868506\
epoch:3200, loss:0.0029125249131228717\
epoch:3400, loss:0.00288698468227491\
epoch:3600, loss:0.002867927335918271\
epoch:3800, loss:0.002853696444605626\
epoch:4000, loss:0.0028430625060447585\
epoch:4200, loss:0.0028351116650964653\
epoch:4400, loss:0.002829163852288894\
epoch:4600, loss:0.0028247124291994052\
epoch:4800, loss:0.002821379599288177\
epoch:5000, loss:0.0028188834048013638\
epoch:5200, loss:0.002817013258288543\
epoch:5400, loss:0.002815611774449621\
epoch:5600, loss:0.002814561262001985\
epoch:5800, loss:0.0028137736683666154\
epoch:6000, loss:0.00281318308672499\
epoch:6200, loss:0.002812740167301663\
epoch:6400, loss:0.0028124079455334743\
epoch:6600, loss:0.002812158725670525\
epoch:6800, loss:0.002811971751327711\
epoch:7000, loss:0.0028118314633049208\
epoch:7200, loss:0.002811726195991678\
epoch:7400, loss:0.0028116472015323915\
epoch:7600, loss:0.002811587919073888\
epoch:7800, loss:0.0028115434273657297\
epoch:8000, loss:0.002811510034592676\
epoch:8200, loss:0.002811484970959764\
epoch:8400, loss:0.0028114661582394566\
epoch:8600, loss:0.0028114520369808592\
epoch:8800, loss:0.0028114414369319316\
epoch:9000, loss:0.002811433479853794\
epoch:9200, loss:0.00281142750662048\
epoch:9400, loss:0.002811423022529595\
epoch:9600, loss:0.0028114196562707195\
epoch:9800, loss:0.0028114171291381976\
iter: 10000\
u: [[-0.65450324 -0.1528308 ]\
 [-0.46704496 -0.33307843]\
 [-0.753545   -0.90354437]\
 [-0.08014989 -0.39632383]\
 [-0.57954836 -0.68057356]\
 [-0.21956645 -0.34271239]\
 [ 0.38789226  0.21420738]\
 [-0.78656788 -0.69533828]]\
y: [[0.34197546 0.4618665 ]\
 [0.3853159  0.41749178]\
 [0.32004935 0.28832267]\
 [0.47997325 0.4021959 ]\
 [0.35903652 0.3361333 ]\
 [0.44532785 0.41515076]\
 [0.5957752  0.55334801]\
 [0.31290609 0.3328466 ]]\
m.s.e: 0.0028114152401259125\
dy: [[0.22502824 0.24854584]\
 [0.23684756 0.24319239]\
 [0.21761777 0.20519271]\
 [0.24959893 0.24043436]\
 [0.2301293  0.22314771]\
 [0.24701096 0.24280061]\
 [0.24082711 0.24715399]\
 [0.21499587 0.22205974]]\
grad_u: [[-0.00563126 -0.00174875]\
 [ 0.00583691  0.0103212 ]\
 [-0.00646737  0.00812464]\
 [-0.00111686 -0.00358422]\
 [-0.00098707  0.01054495]\
 [ 0.00135549 -0.02467445]\
 [-0.00061504  0.00934927]\
 [ 0.00763556 -0.00832636]]\
grad_w: [[-4.95517633e-06 -3.08863809e-06]\
 [-7.26261986e-06 -4.22286188e-06]\
 [-7.40727970e-06 -4.64967168e-06]]\
grad_b: [1.03651294e-05 6.27722059e-06]\
w: [[1.08203095 1.14034734]\
 [1.42454337 0.39919455]\
 [1.14653011 0.84453119]], b: [-2.24515506 -1.57582409]

```python
print('w: {}, b: {}'.format(model.w.numpy(), model.b.numpy()))
print('loss:{}'.format(cost[i]))
```
> w: [[1.08203095 1.14034734]\
 [1.42454337 0.39919455]\
 [1.14653011 0.84453119]], b: [-2.24515506 -1.57582409]\
loss:0.0028114152401259125

```python
# plot learning curves
plt.figure(1)
plt.plot(range(no_iters), cost)
plt.xlabel('iterations')
plt.ylabel('mean square error')
plt.title('gd with alpha = {}'.format(lr))
plt.savefig('./figures/3.4_1.png')
```
![3.4.1](/public/figures/dl_tutorial3/3.4_1.png)
```python
_, pred = model(X)

plt.figure(2)
plot_targets = plt.plot(Y[:,0], Y[:,1], 'b^', label='targets')
plot_pred = plt.plot(pred[:,0], pred[:,1], 'ro', label='predicted')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('gd outputs')
plt.legend()
plt.savefig('./figures/3.4_2.png')
```
![3.4.2](/public/figures/dl_tutorial3/3.4_2.png)

# Tutorial Q1
## a
## b
## c
## d
# Tutorial Q2
# Tutorial Q3

