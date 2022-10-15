---
title: Sentiment Analysis--model selection
date: 2022-10-03 10:47:17
tags:
- NLP
- Sentimental Analysis
categories:
- Natural Language Processing
---
- [Sentiment Analysis -- model selection](#sentiment-analysis----model-selection)
  - [Using pre-trained model to build a demo](#using-pre-trained-model-to-build-a-demo)
  - [Fine-tuning model](#fine-tuning-model)
  
# Sentiment Analysis -- model selection
![How market affected by news](/figures/Sentiment_analysis/how-market-affected.png)
## Using pre-trained model to build a demo
We can first build a demo for sentiment analysis using pre-trained model from HuggingFace. Follow this [blog](https://huggingface.co/blog/sentiment-analysis-python).
```python
from transformers import pipeline
sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
data = ["I love you", "I hate you"]
result = sentiment_pipeline(data)
print(result)
```
We can see that the result is
```
[{'label': 'POS', 'score': 0.9903131723403931}, 
{'label': 'NEG', 'score': 0.9797351360321045}]
```
## Fine-tuning model
Using fine-tuning, I trained a [model](https://huggingface.co/LYTinn/finetuning-sentiment-model-3000-samples) with data from IMDB. The model is uploaded to hugging face.