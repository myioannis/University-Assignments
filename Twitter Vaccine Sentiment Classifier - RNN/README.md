# Course

This is an assigment for the 'Deep Learning for Natural Language Processing' course at the Department of Informatics and Telecommunications, University of Athens.

# Short Description

Developed a sentiment classifier with 3 classes (pro-vax, anti-vax, neutral) using a bidirectional stacked **RNN** with **LSTM/GRU** cells for a Twitter vaccine sentiment dataset.

# Tech Stack

Python, **Pytorch**, Scikit-learn, NLTK, Pandas, Numpy, Matplotlib

# Featured

- Data Exploration

<br>

- Data Preprocessing
  - Noise removal (URL, hashtags, emojis, tags)
  - Normalization (lowercase conversion, punctuation removal, stopword removal, inflected form handling)
  - Vectorization (Pretrained word-embeddings - GloVe)

<br>

- Plot Learning Curves

<br>

- Error Analysis (classification report, wrong examples, top n-grams per class, ROC curve)

<br>

- Experimentation with the following Hyperparameters:
  - The number of stacked RNNs
  - The number of hidden layers
  - The type of cells
  - Skip connections
  - Gradient clipping
  - Dropout probability
 
<br>

- Added *Attention*
