# NLP-Sentiment-Classification-Analysis



![Demo Video](https://datascientest.com/en/files/2023/09/nlp.jpg)

## Demo Video

[Watch the Demo Video](https://youtu.be/ChRp_Pf63jE?si=bx0eNHKJa2J4MfPl)



## Introduction
This project focuses on performing Sentiment Analysis and Text Classification on a dataset of Amazon reviews. We aim to classify the sentiment of the reviews (positive, negative, neutral) and categorize them into different classes based on the content.

## Project Description

The goal of this project is to build and deploy a robust Natural Language Processing (NLP) model that can accurately predict the sentiment and classify text data. The project includes data collection, preprocessing, feature extraction, model training, evaluation, and deployment.

## Key Features

- Scraping Amazon reviews using newspaper3k and Selenium.
- Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
- Text classification using various NLP models (Logistic Regression, SVM, LSTM).
- Deployment of the model using Flask.
- Visualization of results using Plotly.
  
## Project Structure

```
├── data
│   ├── raw_data.csv                # Raw dataset
│   ├── preprocessed_data.csv       # Preprocessed dataset
├── notebooks
│   ├── 01_data_collection.ipynb    # Data collection notebook
│   ├── 02_data_preprocessing.ipynb # Data preprocessing notebook
│   ├── 03_eda.ipynb                # Exploratory Data Analysis
│   ├── 04_modeling.ipynb           # Model training and evaluation
│   ├── 05_deployment.ipynb         # Flask app development
├── models
│   ├── sentiment_model.h5          # Sentiment analysis model
│   ├── classification_model.h5     # Text classification model
│   ├── tokenizer.pkl               # Tokenizer for text preprocessing
├── app
│   ├── app.py                      # Flask app for deployment
│   ├── requirements.txt            # Python packages required for the project
├── README.md                       # Project README file
├── LICENSE                         # License for the project
└── .gitignore                      # Git ignore file
```


## Data Collection

The dataset was collected by scraping Amazon customer reviews across various product categories including shoes, skincare products, and fitness gadgets. The scraping was performed using Selenium for dynamic content and newspaper3k for static content.

## Steps:

- Web Scraping: Collected reviews by iterating over multiple product pages.
- Data Cleaning: Removed duplicates, null values, and irrelevant information.
- Dataset Creation: Combined all reviews into a single dataset saved as raw_data.csv.
- Data Preprocessing
- Data preprocessing involved several steps to clean and prepare the text data for modeling.

## Key Steps:

- Tokenization: Split text into individual tokens using the NLTK library.
- Lowercasing: Converted all text to lowercase to maintain uniformity.
- Stopword Removal: Removed common stopwords (e.g., 'the', 'is') using the NLTK stopword list.
- Lemmatization: Reduced words to their base or root form.
- POS Tagging: Extracted Part-of-Speech tags for each token using NLTK.
- Feature Extraction: Created features like word count, sentence count, average word length, etc.
  
## Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution and characteristics of the data.

## Key Insights:

- Sentiment Distribution: Plotted the distribution of sentiment labels (positive, neutral, negative).
- Word Cloud: Generated word clouds for positive and negative reviews to visualize frequent words.
- POS Tagging: Analyzed the distribution of POS tags to understand sentence structure.
- Correlation Analysis: Examined the correlation between features like word count, sentiment score, etc.

## Modeling

Several machine learning and deep learning models were trained and evaluated for sentiment analysis and text classification.

**Models:**
- Logistic Regression: A baseline model using bag-of-words features.
- Support Vector Machine (SVM): Trained using TF-IDF features.
- LSTM (Long Short-Term Memory): A deep learning model for sequence modeling.
- GRU (Gated Recurrent Unit): An alternative to LSTM with a simpler architecture.

## Hyperparameter Tuning:

Used GridSearchCV and RandomizedSearchCV to optimize hyperparameters.

## Evaluation

The models were evaluated using various performance metrics.

## Metrics:

- Accuracy: The percentage of correctly predicted labels.
- Precision: The ratio of true positive predictions to all positive predictions.
- Recall: The ratio of true positive predictions to all actual positives.
- F1 Score: The harmonic mean of precision and recall.
- Confusion Matrix: A matrix showing the true vs. predicted labels.

## Results

**Sentiment Analysis:**

- Accuracy: 85%
- F1 Score: 0.82

**Text Classification:**

- Accuracy: 87%
- F1 Score: 0.85

**Visualizations:**

- Confusion matrix, ROC curves, and accuracy/loss plots were generated using Matplotlib and Seaborn.

## Deployment

The best-performing models were saved and deployed using Flask for real-time predictions.

**Deployment Process:**

-Flask API: Created a Flask API to handle incoming requests and return predictions.
-Model Loading: Loaded the pre-trained models (.h5 format) and tokenizer (.pkl format).
-User Interface: Designed a simple web interface using HTML and CSS for inputting text and viewing predictions.
-Hosting: The app was deployed on Heroku (or any other cloud platform).

## Installation

To run this project locally, follow these steps:

**Prerequisites:**

- Python 3.7+
- Virtual Environment (optional but recommended)

## Contact

**MA Rahman**

[LinkedIn](https://www.linkedin.com/in/rahman17309/) | [Email](mailto:rahmandatascience09@gmail.com)



