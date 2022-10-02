## Introduction

**[Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction)** is a Natural Language Processing (NLP) machine learning problem that was posted on **Kaggle**  as a Data Science Competition (2020).

In this notebook a baseline is created that demonstrates how such a problem of tweet sentiment extraction can be approached. A version of this notebook was used to participate in the NLP competition, and it secured a solo **ranking of 966 out of 2227 participants [(Top 44%)](https://www.kaggle.com/sangayb)**.
Below we provide a brief overview of what's to be expected from the notebook.

## Overview

**Description of the problem**: 
Given a tweet and its sentiment, which words or phrases best describe the sentiment. 

**Objective:**
Construct a machine learning model that will look at the labelled sentiment of the tweet and determine which words or phrases best supports the sentiment. 

**Data Description**:
The data comprises of csv files: training ([train.csv](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data)) and testing ([test.csv](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data)).  Each csv file contains the following columns: 
1. `textID` - unique ID for each piece of text
2. `text` - text of the tweet
3. `sentiment` - general sentiment of the tweet
4. `selected_text` - words or phrases in `text` that best describes the sentiment. 

**Approach:**
* Formulate the problem like a **Question-Answering** type NLP problem.
* Portion of the  `train.csv` dataset is used to train and validate the machine learning model. 

**Salient features** of this formulation are:

1. Preprocessesing the `train.csv` dataset:
   - Cleans the training dataset
   - Splits the `train.csv` data and use it for training and validation.  
   - Record the position of the `selected_text` within the `text` by denoting a starting and ending index.
2. Load the preprocessed csv-file to HuggingFace's [data.datasets](https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html). 
3. Then load it to a `tensorflow dataset` using the `.to_tf_dataset()` method.
3. Model
   - Finetuning a pretrained NLP model (RoBERTa) with a classification head on top -- RoBERTa is chosen over BERT because roBERTa tokenizer is better at handling special characters (`/`, `.`, etc) like those found in URL's. 

**Backend:** Keras Tensorflow


**References:**
1. https://huggingface.co/docs/transformers/preprocessing
2. Fine-tune a pretrained model https://huggingface.co/docs/transformers/training

3. Tensorflow Dataset: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
4. [Chris Deotte's amazing starter notebook on Tweet Sentiment Extraction:](https://www.kaggle.com/code/cdeotte/tensorflow-roberta-0-705)