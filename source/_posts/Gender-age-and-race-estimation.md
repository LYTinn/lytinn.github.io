---
title: Gender, age, and race estimation
date: 2022-11-30 15:47:48
tags:
- Deep Learning practice
categories:
- Deep Learning
---
- [Introduction](#introduction)
- [Related Work](#related-work)
  - [Transfer Learning](#transfer-learning)
- [Method](#method)
  - [ResNet](#resnet)
  - [Gender, Age, Race classification](#gender-age-race-classification)
  - [Transfer Learning](#transfer-learning-1)
  - [Data Argumentation](#data-argumentation)
  - [Gender specific fusion model](#gender-specific-fusion-model)
- [Experiment](#experiment)
  - [Setup](#setup)

> Author: Hu Kairui, Hu Siyuan, Hu Tianrun
# Introduction
Gender, race classification and age estimation based on face images are three common tasks conducted by human eyes on a daily basis, and they also form an important field that has been long researched by computer vision scientists. Gender race classification and age estimation have a broad application in real world scenarios, such as security and e-commerce, to name a few. Therefore, a high performance model tackling these problems is of great significance. However, different from face detection and recognition tasks, accurate gender race classification and age estimation involve multiple challenges. Firstly, the gradual aging process results in similar facial appearances corresponding to similar ages, while large age differences lead to notably different facial appearances. Secondly, most of the datasets available are small and imbalanced, which may introduce bias to experiment results. Lastly, the facial appearance variation pattern due to aging is complex and may be caused by multiple intrinsic and extrinsic factors, such as race, gender and lifestyle. Gender age classification and age estimation can be formulated into the following problem, given a latent embedding of the face images $\mathcal{X}$, a gender as one of $\{g\}_1^c$ and an race as one of $\{r\}_1^c$ is classified corresponding to an image $x$, $g_i\in\mathbb{Z}^+$, $r_i\in\mathbb{Z}^+$, $x\in\mathcal{X}$. It is worth noting that age estimation are commonly formulated into a scalar regression problem, where age is represented as $a\in\mathbb{R}^+$ that is continual. However, in our setting age is represented as $a\in\mathbb{Z}^+$ that corresponds to discrete age groups. This is to utilize datasets with face images not labeled with specific age values but age groups. While for datasets whose images are labeled with specific age values, the interval between age groups would be set to 1, which will convert the classification problem equivalent to the regression problem. The successful use of Deep Learning-based methods in computer vision tasks has paved the path for deriving end-to-end gender race classification and age estimation models, and Convolutional Neural Networks are the most representative ones. CNN-based methods learn local features based on the age differences as a metric measure. 
![result](/figures/gender-age-race/result_2_F_East_Asian_9220.jpg)

In this work we first proposed an end-to-end gender-age-race ternary classification model as a baseline. Afterwards, we proposed a Gender-specific Fusion model to harness the gender-specific information and further improve the performance, depicted in Fig. \textcolor{red}{1}. Firstly, we looked into the model performance degradation problem caused by training on unbalanced datasets, and conducted experiments to reveal its impact. Secondly, with an appropriate dataset selection as the starting point, we employed the idea of Transfer Learning by using a pre-trained model trained on ImageNet dataset, and performed downstream fine-tuning on our face attribute classification datasets. This can help leverage the strong capacity of large-scale models and save computing costs by skipping early epochs. Thirdly, to verify our intuitive insight that age-race prediction of different genders tends to exhibit different patterns, for example, females are more effective at hiding their ages due to social commentaries, we employed a gender-specific stacked fusion model that the gender is classified first, then the image is fed into different gender-specific classifiers to do the further predictions on age and race. 
	
In particular, we proposed the following ideas:

$\bullet$ We proposed an end-to-end gender-age-race classification solution based on a ResNet pre-trained model as the foundation.

$\bullet$ We proposed a Gender-specific Fusion model with a two-stage architecture for the gender classification and the age-race estimation sub-tasks, considering the gender-specific information. 

$\bullet$ Our proposed fusion model achieves a higher accuracy than the original end-to-end models when applied to contemporary face image datasets UTKface and FairFace. 

# Related Work
The task of classifying different human characteristics from facial appearance, such as gender, race, age, emotions, expressions, or other facial aspects, is known as face attribute recognition. Multiple computer vision systems have included face attribute recognition as a minor component. For instance, Kumar et al. employed features for face verification that included facial characteristics that characterize individual traits, such as gender, race, hair style, expressions, and accessories. Additionally, attributes are frequently used to re-identify people in photos or videos by fusing characteristics of the human face and body, which is particularly successful when faces are obscured or too small. These systems can be used for security purposes such as CCTV monitoring or electronic device authentication (like unlocking cellphones). Face attribute recognition is frequently used for demographic research in social science and marketing, which aims to determine how social actions of people relate to their backgrounds in terms of demographics. Social scientists, who traditionally did not utilize photos, have started to use images of people to infer their demographic traits and evaluate their activities in many research using commercial services and off-the-shelf techniques. Examples include demographic studies of social media users that use their images.
## Transfer Learning
Deep neural network architectures such as convolutional neural networks (CNNs) and more recently transformers have achieved many successes in image classification tasks. It has been consistently shown that these models work best when big models can be trained and there is an abundance of labeled data available for the task \cite{kolesnikov2020big, mahajan2018exploring, ngiam2018domain}. However, there are numerous situations in real life where the need for a lot of training data cannot be satisfied. Among them are:
	
$\bullet$ Insufficient data due to privacy concerns or the rarity of the data. For instance, because to the rarity of the examples themselves as well as privacy considerations, training data for novel and rare disease identification tasks in the medical area is scarce. 

$\bullet$ The cost of data collection and/or labeling is prohibitive. For instance, only highly qualified subject-matter specialists can perform labeling.

We may want to learn from a limited number of training instances for a variety of additional reasons as well: 

$\bullet$ From a cognitive science perspective, it is intriguing to make an effort to emulate the human capacity to learn general concepts from a sparse sample size. 

$\bullet$ Compute resource limitations could make it difficult to train a big model from random initialization with big data. Consider environmental issues \cite{strubell2019energy}. 

Transfer learning frequently significantly improves performance in all of these instances. In this paradigm, the trained weights are utilized to initialize a model for the target task after the model has been trained on a similar dataset and task for which additional data are available. The dataset must be sufficiently connected and best practice techniques must be applied for this process to boost performance rather than degrade it.

# Method
## ResNet
For our gender, age, and race classification task, we used ResNet, a deep residual learning framework, as the pre-trained model. ResNet helps address the model degradation problem by enabling stacked layers to fit a residual mapping. Mathematically, let the targeted underlying mapping be $H(x)$, then the stacked layers fit the mapping of $F(x): H(x)-x$. The original mapping becomes $F(x)+x$. Compared with the original mapping, it is easier to  learn the residual mapping. We can explain the learning mathematically. The residual unit can be denoted as: 
$$
y_l=h(x_l)+F(x_l, W_l),
x_{l+1}=f(y_l)
$$
With $x_l$ and $x_{l+1}$ be the input and output of the $l$th residual unit. $F$ is the residual function indicating the learned residual, $h(x_l)=x_l$ denotes the identity mapping, and $f$ is the ReLU activation function. The feature we learn from layer $l$ to deeper layer $L$ is:
$$
x_L=x_l+\sum_{i=l}^{L-1}F(x_i, W_i)
$$
	
	
Based on the chain rule, we can obtain the gradient:
	
$$
\begin{aligned}
    \frac{\partial loss}{\partial x_l} & = \frac{\partial loss}{\partial x_L}*\frac{\partial x_L}{\partial x_l} \\
    & = \frac{\partial loss}{\partial x_L}*(1+\frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F(x_i, W_i))
\end{aligned}
$$
With $\frac{\partial loss}{\partial x_L}$ denoting the gradient of the loss function at L, 1 in parentheses indicates that the shortcut connection propagates the gradient losslessly, and the presence of 1 does not cause the gradient to vanish. Hence, residual learning will be easier.


## Gender, Age, Race classification
We first perform the three classification tasks together with the pre-trained ResNet model. For the labeling purpose in training, we labeled the gender, age, and race as the corresponding natural numbers starting from 0. The model takes an input RGB image of size 224 x 224. It consists of the pre-trained ResNet and 3 additional parallel dense layers as classification heads for producing the corresponding 3 output labels. We select the output label with the SoftMax activation, which is defined below:
$$
\begin{aligned}
    \sigma(\boldsymbol{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
\end{aligned}
$$
$$
\begin{aligned}
    \text{for } i=1,...K \text{ and } \boldsymbol{z}=(z_1,...,z_K)\in\mathbb{R}^K
\end{aligned}
$$


The cross-entropy loss function defined is used to optimize the classification model:
$$
\begin{aligned}
L=-\frac{1}{m}\sum_{i=1}^{m}y_i*log(\hat{y_i})
\end{aligned}
$$

The epoch for the training process is set to 30. In order to prevent overfitting, we implemented early stopping with a patience of 5. The model is optimized by the Adam optimizer, and we implemented learning rate decay as the number of epoch goes up. 

![original](/figures/gender-age-race/original.png)

## Transfer Learning
Transfer Learning refers to pre-training on large-scale general datasets and fine-tuning on small datasets according to the requirements of downstream tasks. Therefore, Transfer Learning can greatly benefit face image related tasks, as models can directly inherit the knowledge of human faces' basic representations from pre-trained models, thus time-consuming early epochs can be skipped. To leverage the advantages of pre-trained models, we employed multiple models well-trained on ImageNet as the foundation, as ImageNet contains rich image representation knowledge. Moreover, the original last layer is replaced by 3 parallel dense softmax classification heads, and a flatten layer is added as the penultimate layer to reshape the classification head input. After modifying the model architecture to adapt to the gender, age, race classification task, further fine-tuning on our face image datasets can be performed.

## Data Argumentation
As complex network architectures are employed in our model design, dataset augmentation is of great importance to prevent overfitting on small datasets. To be specific, the augmentation pipeline is composed of random cropping, flipping and light adjustment, and each will be executed according to its configured probability. Image augmentation will be executed during data loading in each epoch, therefore, the original small datasets can be enriched to a great extent. It is worth noting that the training processes of two gender-specific classification models will employ more augmentation. As the original dataset will be split in half to train the gender-specific models, while they share the architecture of same depth with the end-to-end model which is trained on the entire dataset, more image augmentation is required for the gender-specific models to ensure that they are trained to the same extent as the end-to-end model to ensure fairness. 
	
## Gender specific fusion model
There are many difference features between men and women. To harness the gender-specific information in the classification tasks, we proposed a Gender-specific Fusion model consisting of two main modules. The first module is a gender classifier that classifies the input image into one of the two gender categories. The second module consists of two gender-specific classifiers, denoted as Male-classifier and Female-classifier. The Male classifier performs age and race predictions for the input images classified as male, and the Female-classifier performs age and race predictions for the female subjects. With this two-stage architecture, given an input image for the age-gender-race classification task, we can first classify its gender, after which we put this image into the Male or Female classifier to perform further predictions on age and race. 

To effectively train the two gender-specific classifiers, we divided the training set into the male and female dataset based on the gender label. The Male classifier was solely trained on the male dataset, and the Female classifier was trained only on the female dataset. To establish a fair training setup, we performed data augmentation on the sub-datasets to ensure each of the two sub-datasets has the same number of data inputs as the original dataset, so that the two gender-specific classifiers can be trained on a dataset with the same size as the original one. We used the pre-trained ResNet model for all three classifiers. The gender classifier was added with one classification head for gender, and the two gender-specific classifiers were added with two classification head for predicting the age and race. We used cross-entropy loss function for optimization and SoftMax function for activation. The model is optimized by the Adam optimizer. The epoch in our training process was set to 30, and early stopping with a patience of 5 was implemented to avoid overfitting. 

![fusion](/figures/gender-age-race/fusion.png)
# Experiment
![utk data ana](/figures/gender-age-race/UTKface-data-ana1.png)
![utk data ana](/figures/gender-age-race/UTKface-data-ana2.png)
To solve the gender-age-race classification task, we first performed the gender-age prediction sub-task and then added the race prediction sub-task to enable the model to complete the ternary classification task. The popular IMDb-WIKI dataset was used for the gender-age prediction sub-task, and the UTKface and FairFace dataset was used for the additional race prediction sub-task, as there are no race labels in the IMDb-WIKI dataset. The sizes of different datasets vary, and there were multiple choices of pre-trained models, which guided us to determine which dataset-model combination to use. Hence we also studied the relationship between dataset size and model architecture by trying different backbones on IMDb-WIKI, and selected the top-performing network for use. As for the additional race prediction sub-task, we compared the performances of our pre-trained models including Xception and ResNet on UTKface and FairFace respectively to determine a better model and a more appropriate dataset for use. 
	
	
With the obtained comparative results, we then selected FairFace as the target dataset and ResNet as the pre-trained model to perform our age-gender-race ternary classification task. In order to optimize the original model architecture and effectively harness the gender-specific information, we implemented a gender-specific fusion model with a two-stage architecture, and compared its performance over the original model.

## Setup
**Dataset** As for the gender-age prediction sub-task, we used the IMDb-WIKI with 460,723 images from IMDb and 62,328 images from Wikipedia, which have labeled ages and genders.

As for the race prediction subtask, we evaluated our models on the UTKface\cite{zhifei2017cvpr} dataset and FairFace\cite{karkkainenfairface} dataset. The UTKface dataset contains 23706 images, with 2 genders: male and female, 117 classes of ages: 0 to 116, and 5 classes of races: White, Black, Asian, Indian, and Others. The models used 13275 examples for training, 3319 samples for evaluation, and the rest for testing.

![data](/figures/gender-age-race/screenshot001.png)
	
The FairFace contains 108k faces with 2 genders, 9 classes of ages, and 7 classes of races. It has two sets: train and test. We utilized the training set for fine-tuning and the testing set for metrics. The race distribution of FairFace is more balanced compared to that of other datasets.

![data](/figures/gender-age-race/screenshot002.png)

The pre-trained models were trained on ImageNet on the original validation labels. The last layer was dropped as the models were used for a different down-streaming task.

**Pretraining & Fine-tuning.** To train our model, we utilized the pre-trained architectures and parameters. The pre-trained models were downloaded from Keras. In this assignment, we tried Xception, VGG19, ResNet152V2, InceptionResNetV2, MobileNetV2, and DenseNet201.
	



[def]: #introduction