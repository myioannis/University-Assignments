# Course

This is an assigment for the 'Deep Learning for Natural Language Processing' course at the Department of Informatics and Telecommunications, University of Athens.

# Short Description

Developed a sentiment classifier with 3 classes (pro-vax, anti-vax, neutral) using **Softmax Regression** for a Twitter COVID-19 vaccine sentiment dataset.

# Tech Stack

Python, **Scikit-learn**, NLTK, Pandas, Numpy, Matplotlib

# Featured

- Data Exploration

<br>

- Data Preprocessing
  - Noise removal (URL, hashtags, emojis, tags)
  - Normalization (lowercase conversion, punctuation removal, stopword removal, inflected form handling)
  - Vectorization (BoW, TF-IDF)

<br>

- Plot Learning Curves

<br>

- Error Analysis (classification report, wrong examples, top n-grams per class)

<br>

- Fine Tuning
  - Treat data preprocessing steps as hyperparameters
  - Scikit-learn pipelines and custom transformers
  - Cross validation using GridSearchCV
  - Model parameters optimization using RandomSearchCV
