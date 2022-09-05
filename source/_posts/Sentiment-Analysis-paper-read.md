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


# Introduction
In this post, we will read some top papers about sentiment analysis. The papers are choosen from [paper with code](https://paperswithcode.com/task/sentiment-analysis).

# SMART
The [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/pdf/1911.03437v5.pdf) published in 2019 is the top paper for sentiment analysis on SST-2 Binary classification. It reaches 97.5% in accuracy.
## Dataset
The dataset of this paper is the **Standford Sentiment Treebank (SST)**. SST is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from **movie reviews**. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.

Each phrase is labelled as either *negative, somewhat negative, neutral, somewhat positive* or *positive*. The corpus with all 5 labels is referred to as SST-5 or SST fine-grained. Binary classification experiments on full sentences (*negative* or *somewhat negative* vs *somewhat positive* or *positive* with *neutral* sentences discarded) refer to the dataset as SST-2 or SST binary.
## Motivation
When performing NLP models on downstream tasks, because of the limit fine-tuning data resources and high complexity of pre-trained model, the fine-tuning always cause overfitting.

The team comes up with a framework to fine-tune the pre-trained model on the downstream tasks, avoiding over-fitting and getting a better performance.