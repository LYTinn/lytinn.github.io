---
title: Deep Learning Topic selection
date: 2022-10-15 10:34:02
tags:
- Deep Learning
categories:
- Deep Learning
---

- [Deep Learning Model Selection](#deep-learning-model-selection)
  - [A. Speech Emotion Recognition](#a-speech-emotion-recognition)
  - [B. Text Emotion Recognition](#b-text-emotion-recognition)
  - [C. Sentiment Analysis](#c-sentiment-analysis)
  - [D. Gender Classification](#d-gender-classification)
  - [E. Material Recognition](#e-material-recognition)
  - [F. Flowers Recognition](#f-flowers-recognition)

# Deep Learning Model Selection
## A. Speech Emotion Recognition 
Speech emotion recognition (SER) is the prediction of speaker’s emotions from speech signals. SER involves extraction of audio features from speech and classification of speaker utterances to emotional
classes. Interesting projects would be:

1. To develop deep learning techniques for SER invariant to speaker characteristics such as gender, age emotion.
2. To develop unsupervised learning techniques for SER
3. To detect emotions dynamically in speech. That is, to predict emotions within subintervals of the speech utterance.


## B. Text Emotion Recognition 
Text emotion recognition (TER) involves predicting emotions expressed in text and documents. Existing algorithms find emotion by learning the relationships of words using recurrent neural networks (RNN) or convolutional neural networks (CNN). RNN and CNN capture local information (i.e., emotion of words) and ignore the global information (i.e., emotion of sentence). Interesting projects would be

1. To develop deep learning techniques for capture both local and global information. The local information refers to emotions expressed by words and global information refers to emotions expressed by the meanings of sentences.
2. To develop techniques that are invariant to speaker’s writing styles and characteristics 

## C. Sentiment Analysis
Text sentiment analysis (TSA) refers to identification of sentiments, usually positive or negative, expressed in text or document. One may want to develop deep learning techniques for TSA

1. To deal with domain adaptation, that is, how can one adapt a network train on one domain to work in another domain
2. To avoid using recurrent networks to speed up computations
3. To deal with small datasets, that is, with insufficient number of training samples

## D. Gender Classification
Automatic gender classification has been used in many applications including image analysis on social platforms. The goal of this project is to classify the gender of faces in an image. One can design a convolutional neural network to achieve this goal. Some tasks to consider:

1. Modify some previously published architectures e.g., increase the network depth, reducing their parameters, etc.
2. Consider age and gender recognition simultaneously to take advantage of the gender-specific age characteristics and age-specific gender characteristics inherent to images
3. Consider pre-training using the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## E. Material Recognition
The goal of this project is to train a convolutional neural network to classify color photographs of surfaces into one of ten common material categories: fabric, foliage, glass, leather, metal, paper, plastic, stone, water, and wood. Some tasks to consider:

1. Modify some previously published architectures e.g., increase the network depth, reducing their parameters, etc.
2. Try data augmentation to increase the number of training images
3. Try a larger dataset, Materials in Context Database (MINC)

## F. Flowers Recognition
The Oxford Flowers 102 dataset is a collection of 102 flower categories commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is divided into a training set, a validation set and a test set. The training set and validation set each consist of 10 images per class (a total of 1020 images each). The test set consists of the remaining 6149 images (minimum 20 per class). Some tasks to consider:

1. Modify some previously published architectures e.g., increase the network depth, reducing their parameters, etc.
2. Analyze the results of using fewer training images, i.e., few-shot learning
3. Try advanced techniques such as visual prompt tuning with a Transformer architecture
4. Try more advanced loss function such as triplet loss