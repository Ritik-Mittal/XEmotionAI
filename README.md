## Project Name
**XEmotionAI**

## Project Description
XEmotionAI is a comprehensive project utilizing the Sentiment140 dataset with 1.6 million tweets for sentiment analysis. Due to the large dataset size, various preprocessing steps and optimizations were implemented, including tokenization, stopword removal, and lemmatization, with intermediate results stored in pickle files (`stp_words_removed_tokenized.pkl`, `preprocessed_tweets.pkl`, and `vectorized_data.npz`). The project leverages a Sequential RNN model with LSTM layers to achieve high accuracy in sentiment classification.

## Overview

This project aims to classify the sentiment of tweets as positive or negative using the Sentiment140 dataset. Given the substantial size of the dataset, several preprocessing steps and optimizations were applied to ensure efficient handling and high performance.

### Key Features

- **Dataset**: Utilizes the Sentiment140 dataset with 1.6 million tweets.
- **Preprocessing**: Includes text normalization, contraction expansion, noisy token removal, stopword removal, tokenization, and lemmatization.
- **Vectorization**: Uses Keras' TextVectorization for transforming text into numerical format.
- **Model**: Implements an RNN model with LSTM layers for binary classification.
- **Performance**: Achieves high accuracy, precision, and recall.

### Pickle Files
- `stp_words_removed_tokenized.pkl`: Contains tokenized tweets with stopwords removed.
- `preprocessed_tweets.pkl`: Contains preprocessed and lemmatized tweets.
- `vectorized_data.npz`: Contains vectorized tweet data ready for model training.

### Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ritik-Mittal/XEmotionAI.git
   cd XEmotionAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Sentiment140 dataset**
   ```bash
   kaggle datasets download -d kazanova/sentiment140
   unzip sentiment140.zip
   ```

4. **Run the preprocessing and training script in Google Colab**
   - Upload `twitter_sentiment_classification.py` to Google Colab.
   - Ensure your dataset files are uploaded to the Colab environment.
   - Execute the script by running the cells in the Colab notebook.

### Preprocessing Steps

1. **Text Normalization**
   ```python
   def normalize_text(text):
       return text.lower()
   ```

2. **Contraction Expansion**
   ```python
   from contractions import fix
   def expand_contractions(text):
       return fix(text)
   ```

3. **Noisy Token Removal**
   ```python
   import re
   def remove_noise(text):
       text = re.sub(r'http\S+', '', text)  # Remove URLs
       text = re.sub(r'@\w+', '', text)  # Remove mentions
       text = re.sub(r'#\w+', '', text)  # Remove hashtags
       text = re.sub(r'\d+', '', text)  # Remove digits
       text = re.sub(r'\W+', ' ', text)  # Remove special characters
       return text
   ```

4. **Tokenization**
   ```python
   from nltk.tokenize import word_tokenize
   def tokenize(text):
       return word_tokenize(text)
   ```

5. **Stopword Removal**
   ```python
   from nltk.corpus import stopwords
   stop_words = set(stopwords.words('english'))
   def remove_stopwords(tokens):
       return [token for token in tokens if token not in stop_words]
   ```

6. **Lemmatization**
   ```python
   import spacy
   nlp = spacy.load('en_core_web_sm')
   def lemmatize(tokens):
       doc = nlp(' '.join(tokens))
       return [token.lemma_ for token in doc]
   ```

### Model Training

- **Architecture**
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Embedding, LSTM, Dense

   model = Sequential()
   model.add(Embedding(input_dim=vocab_size, output_dim=128))
   model.add(LSTM(units=128))
   model.add(Dense(1, activation='sigmoid'))

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   ```

- **Training**
   ```python
   history = model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.2)
   ```

### Usage

1. **Preprocess the Data**
   ```python
   preprocessed_data = preprocess(raw_data)
   ```

2. **Train the Model**
   ```python
   model.fit(preprocessed_data, labels)
   ```

3. **Evaluate the Model**
   ```python
   evaluation = model.evaluate(test_data, test_labels)
   print(f'Accuracy: {evaluation[1]*100:.2f}%')
   ```

### Conclusion

XEmotionAI offers an end-to-end solution for sentiment analysis using the Sentiment140 dataset. By employing robust preprocessing techniques and an advanced RNN model, this project ensures high accuracy in classifying tweet sentiments. The intermediate results are stored in pickle files to optimize performance and facilitate further analysis.

