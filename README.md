# Fake News Detection Using Machine Learning and Deep Learning

## Overview

This project focuses on detecting fake news articles using Natural Language Processing (NLP), traditional machine learning models, and deep learning techniques. With the rapid spread of misinformation across digital platforms, automated fake news classification has become increasingly important for preserving trust in media and reducing the societal impact of misinformation.

Using the Kaggle Fake and Real News Dataset, this project builds and compares multiple classification models including Logistic Regression, Naive Bayes, XGBoost, and LSTM to determine whether a news article is real or fake based on its textual content.

## Problem Statement

The rise of online media has significantly increased the speed at which misinformation spreads. Fake news can manipulate public opinion, influence elections, create social unrest, and damage trust in legitimate journalism.

The goal of this project is to build a robust text classification system capable of accurately distinguishing fake news articles from real news articles using supervised machine learning and deep learning models.

## Dataset

The dataset used in this project is the Fake and Real News Dataset from Kaggle.

Files:
- Fake.csv
- True.csv

Dataset Size:
- Fake News Articles: 23,502
- Real News Articles: 21,417
- Total Articles: 44,919

Features:
- title
- text
- subject
- date

## Scope

- Text preprocessing and cleaning
- Stopword removal and normalization
- Feature extraction using TF-IDF
- Training multiple supervised learning models
- Building a deep learning LSTM model using PyTorch
- Performance evaluation using classification metrics
- Comparative analysis of model effectiveness

Binary Classification:
- 0 = Fake News
- 1 = Real News

## Key Variables

Input Variable:
- text

Engineered Variables:
- cleaned_text
- TF-IDF Features

Target Variable:
- label

## Tech Stack

Programming Language:
- Python

Libraries:
- pandas
- numpy
- nltk
- scikit-learn
- xgboost
- PyTorch
- matplotlib

## Methodology

1. Load Fake.csv and True.csv
2. Merge both datasets
3. Preprocess text by removing HTML, punctuation, lowercase conversion, and stopword removal
4. Convert text to numerical vectors using TF-IDF
5. Train Logistic Regression, Naive Bayes, XGBoost, and LSTM models
6. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix

## Results

Logistic Regression:
- Accuracy: 98.89%

Naive Bayes:
- Accuracy: 93.43%

XGBoost:
- Accuracy: 99.67%

LSTM:
- Accuracy: ~99%

Best Performing Model:
- XGBoost

## How to Run

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
jupyter notebook
```

Open:
P.4.ipynb

## Project Structure

NLP-NewsClassification/
- P.4-Report.pdf
- P.4.ipynb
- README.md
- requirements.txt

---

## Author

### Aaron Tsui

Master of Science in Data Science  
Northwestern University 

Specializing in machine learning, deep learning, and applied AI systems for healthcare and large-scale predictive modeling.

- Email: aaron.tsui.careers@gmail.com  
- LinkedIn: https://www.linkedin.com/in/aaron-tsui/