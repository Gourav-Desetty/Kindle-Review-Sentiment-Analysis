# %pip install lxml
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import gensim
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('Kindle_reviews/all_kindle_review.csv')

print(df.head())
df = df[['reviewText', 'rating']].copy()
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df['rating'].unique())
print(df['rating'].value_counts())                  ### No imbalanced dataset 


## Preprocessing And Cleaning
## positive review 1 and negative review 0
df['rating'] = df['rating'].apply(lambda x:0 if x<3 else 1)
print(df['rating'].unique())
print(df['rating'].value_counts())
print(df[df['rating'] < 1])
## lowering all the cases

df['reviewText'] = df['reviewText'].str.lower()
print(df.head())


stop_words = stopwords.words('english')
lematizer = WordNetLemmatizer()

def preprocess_review(text):
    if type(text) != str:
        return ''

    text = re.sub('[^a-z A-Z 0-9]', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(text))
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = ' '.join([lematizer.lemmatize(word)for word in text.split()])
    text = ' '.join(text.split())
    return text

df['reviewText'] = df['reviewText'].apply(preprocess_review)
print(df['reviewText'].head())
print(df)



X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['rating'], test_size=0.2, random_state=42)

print(X_train.shape)


tokenize_reviews = [word_tokenize(word) for word in X_train]
print(tokenize_reviews)

model = gensim.models.Word2Vec(tokenize_reviews, vector_size=300,    
                                window=10,          
                                min_count=2,       
                                workers=4,
                                epochs=20,      
                                sg=1  )

print(model.wv.index_to_key)

print(model.corpus_count)

model.wv.similar_by_word('novel')

def avg_word2vec(text):
    words = word_tokenize(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.wv.vector_size)

X_train_vectors = [avg_word2vec(doc) for doc in X_train]
X_test_vectors = [avg_word2vec(doc) for doc in X_test]


len(X_train_vectors), X_train_vectors[0].shape

ml_model = LogisticRegression(max_iter=1000)

ml_model.fit(X_train_vectors, y_train)

y_preds = ml_model.predict(X_test_vectors)

print("Classification report:", classification_report(y_test, y_preds))
print("Accuracy:", accuracy_score(y_test, y_preds))




## Bag of words and TFIDF
bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)


tfidf=TfidfVectorizer()
X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)

nb_model_bow=MultinomialNB().fit(X_train_bow,y_train)
nb_model_tfidf=MultinomialNB().fit(X_train_tfidf,y_train)

y_pred_bow=nb_model_bow.predict(X_test_bow)
y_pred_tfidf=nb_model_bow.predict(X_test_tfidf)

print("BOW accuracy: ",accuracy_score(y_test,y_pred_bow))
print("TFIDF accuracy: ",accuracy_score(y_test,y_pred_tfidf))
print("Word2Vec accuracy", accuracy_score(y_test, y_preds))


def predict_sentiment(review_text):
    preprocessed_data = preprocess_review(review_text)
    vec = avg_word2vec(preprocessed_data)
    predict = ml_model.predict([vec])
    return 'Positive' if predict == 1 else 'Negative'
result = predict_sentiment("I didn't expect this book to be soo good")
print(result)