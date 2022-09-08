---
title: Sentiment Analysis paper read
date: 2022-09-05 14:03:13
tags:
- NLP
- Sentiment Analysis
categories:
- Natural Language Processing
---
- [Introduction](#introduction)
- [SMART](#smart)
  - [Dataset](#dataset)
  - [Motivation](#motivation)
  - [Result](#result)
- [Unsupervised Data Augmentation for Consistency Training](#unsupervised-data-augmentation-for-consistency-training)
  - [Dataset](#dataset-1)
  - [Motivation](#motivation-1)
  - [Result](#result-1)
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](#xlnet-generalized-autoregressive-pretraining-for-language-understanding)
  - [Dataset](#dataset-2)
  - [Motivation](#motivation-2)
  - [Result](#result-2)
- [ERNIE-Doc: A Retrospective Long-Document Modeling Transformer](#ernie-doc-a-retrospective-long-document-modeling-transformer)
  - [Dataset](#dataset-3)
  - [Motivation](#motivation-3)
  - [Result](#result-3)
  - [Conclusion](#conclusion)
- [MA_BERT: Learning Representation by Incorporating Multi-Attribute Knowledge in Transformers.](#ma_bert-learning-representation-by-incorporating-multi-attribute-knowledge-in-transformers)
  - [Dataset](#dataset-4)
  - [Motivation](#motivation-4)
  - [Result](#result-4)


# Introduction
In this post, we will read some top papers about sentiment analysis. The papers are choosen from [paper with code](https://paperswithcode.com/task/sentiment-analysis).

# SMART
The [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/pdf/1911.03437v5.pdf) published in 2019 is the state-of-the -art for sentiment analysis on SST-2 Binary classification. It achieves 97.5% in accuracy.
## Dataset
The dataset of this paper is the **Standford Sentiment Treebank (SST)**. SST is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from **movie reviews**. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.

Each phrase is labelled as either *negative, somewhat negative, neutral, somewhat positive* or *positive*. The corpus with all 5 labels is referred to as SST-5 or SST fine-grained. Binary classification experiments on full sentences (*negative* or *somewhat negative* vs *somewhat positive* or *positive* with *neutral* sentences discarded) refer to the dataset as SST-2 or SST binary.
## Motivation
When performing NLP models on downstream tasks, because of the limit fine-tuning data resources and high complexity of pre-trained model, the fine-tuning always cause overfitting.

The team comes up with a framework to fine-tune the pre-trained model on the downstream tasks, avoiding over-fitting and getting a better performance.
## Result
Achieves a new state-of-the-art performance on many NLP benchmarks like GLUE, SNLI, SciTail, ANLI.

# Unsupervised Data Augmentation for Consistency Training
The [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848v6.pdf) published on 2019 is the state-of-the-art for sentiment analysis on Amazon Review Full. It achieves 65.83% accuracy on Amazon Review dataset.
## Dataset
This dataset contains product reviews and metadata from Amazon, including 233.1 million reviews spanning May 1996 - Oct 2018.

This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), links (also viewed/also bought graphs), and transaction metadata for each review shown on the review page.
## Motivation
Substituting simple moising operations with davanced data augmentation methods, improves the performance on consistency training framework.

**Consistency training** is used to enforce the predictions to be similar for an unlabeled example and the augmented unlabeled example

## Result
Finetuning from BERT, and yields improvements in high-data regime, such as ImageNet, whether when there is only 10% labeled data or when a full labeled set with 1.3M extra unlabeled examples is used.

# XLNet: Generalized Autoregressive Pretraining for Language Understanding
The state-of-the-art model for Sentiment Analysis on IMDb.

## Dataset
The **IMDb Movie Reviews** dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. The dataset contains an even number of positive and negative reviews. Only highly polarizing reviews are considered. A negative review has a score $\leq 4$ out of 10, and a positive review has a score $\geq 7$ out of 10. No more than 30 reviews are included per movie. The dataset contains additional unlabeled data.
## Motivation
To overcome the defects of autoencoding model (like BERT).

XLNet using autoregressive pretraining method that
- enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order
- overcomes the limitations of BERT which neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy.
## Result
Outperforms BERT on 20 tasks.

# ERNIE-Doc: A Retrospective Long-Document Modeling Transformer
An improvement to transformer architecture.
## Dataset
The **IMDb Movie Reviews** dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. The dataset contains an even number of positive and negative reviews. Only highly polarizing reviews are considered. A negative review has a score $\leq 4$ out of 10, and a positive review has a score $\geq 7$ out of 10. No more than 30 reviews are included per movie. The dataset contains additional unlabeled data.
## Motivation
Uing recurrence transformer to overcome the defect of transformer on long text.
## Result
Improved the state-of-th-art language modeling result of perplexity to 16.8 on WikiText-103. It also outperformed competitive pretraining models by a large margin on most language understanding tasks, such as text classification and question answering.
## Conclusion
Suitable to be innovations.

# MA_BERT: Learning Representation by Incorporating Multi-Attribute Knowledge in Transformers.
The state-of-the-art model for Sentiment Analysis on User and product information

## Dataset
The **IMDb Movie Reviews** dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. The dataset contains an even number of positive and negative reviews. Only highly polarizing reviews are considered. A negative review has a score $\leq 4$ out of 10, and a positive review has a score $\geq 7$ out of 10. No more than 30 reviews are included per movie. The dataset contains additional unlabeled data.

## Motivation
Incorporating attribute information from text.
![MA-BERT](/figures/NLP/MA-BERT.png)

## Result
Outperformed pre-trained BERT models and other methods incorporating external attribute knowledge.
