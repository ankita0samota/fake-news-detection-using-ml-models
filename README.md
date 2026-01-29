# fake-news-detection-using-ml-models

## 📘 Project Overview

The rapid growth of digital news platforms has made the spread of fake news a serious global
problem. Fake or misleading information can influence public opinion, damage trust, and cause
social unrest. Manual verification of news content is slow and inefficient, making automated
solutions necessary. This project focuses on detecting fake news articles using machine learning
by analyzing textual content and classifying news as real or fake.

## 🏭 Industry Relevance

This project addresses a real-world industry problem and can be applied in:

- Social media platforms for automatic fake news flagging
- News publishing organizations for content verification
- Government and cyber security agencies for misinformation monitoring
- Online platforms aiming to reduce the spread of misleading information

## 🎯 Project Objectives

- Automatically classify news articles as fake or real
- Compare the performance of multiple machine learning models
- Improve detection accuracy using ensemble and hybrid learning techniques

## 🛠 Technology Stack

- Python
- Scikit-learn
- NLTK
- Pandas & NumPy

## 📂 Dataset

**Source:** Kaggle

This project uses a **balanced fake news dataset** consisting of two classes:

- **Fake News Dataset** (~20,000 articles)
- **True News Dataset** (~20,000 articles)

Each news article contains the following attributes:

- Title
- Text content
- Subject / category
- Publication date

The majority of the articles focus on **political and world news**, primarily from the period **2016–2017**.  
The dataset was pre-processed to remove noise, while intentionally preserving punctuation errors and grammatical inconsistencies in fake news articles to maintain realism.

🔗 **Dataset Link (Kaggle):**  
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Using balanced datasets helps reduce class bias and improves the reliability of the machine learning model during training.

Due to the large size of the dataset, it has not been uploaded to this repository. Instead, the dataset can be downloaded directly from Kaggle using the link provided above.

## ⚙ Methodology / Workflow

Data Collection → Text Preprocessing → Feature Extraction (TF-IDF) →  
Model Training → Model Evaluation

Text preprocessing includes tokenization, lowercasing, and stopword removal to improve
classification performance.

## 🤖 Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- Naive Bayes
- k-Nearest Neighbors (kNN)
- Neural Networks
- AdaBoost
- Hybrid Models

## 🔮 Future Scope

- Integration with real-time news and social media feeds
- Multilingual fake news detection
- Deployment as a web or mobile application
- Use of deep learning and transformer-based models
- Incorporation of explainable AI for better model transparency
