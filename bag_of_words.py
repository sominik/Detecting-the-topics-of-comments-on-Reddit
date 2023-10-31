import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn import svm
from sklearn.metrics import accuracy_score

stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
# nltk.download("popular")

train_dataset = pd.read_csv("train.csv")
train_comments = np.array(train_dataset['Comment'])
train_labels = np.array(train_dataset['Topic'])

test_dataset = pd.read_csv("test.csv")
test_comments = np.array(test_dataset['Comment'])
test_labels = np.array(test_dataset['Topic'])


def clean_data(comment):
    words = re.sub(r"[^a-zA-Z]", " ", comment.lower()).split(" ")
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            cleaned_words.append(word)
    cleaned_comment = " ".join(cleaned_word for cleaned_word in cleaned_words)
    return cleaned_comment


tfidf_vectorizer = TfidfVectorizer(smooth_idf=True,
                                   ngram_range=(1, 3))
train_dataset['cleaned_comment'] = train_dataset['Comment'].apply(clean_data)
tfidf_vectorizer.fit(train_dataset['cleaned_comment'])
train_vectors = tfidf_vectorizer.transform(train_dataset['cleaned_comment'])
test_dataset['cleaned_comment'] = test_dataset['Comment'].apply(clean_data)
test_vectors = tfidf_vectorizer.transform(test_dataset['cleaned_comment'])

svm = svm.SVC()

svm.fit(train_vectors, train_labels)
predicted_labels = svm.predict(test_vectors)
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"simple bag of words accuracy on test data is : {accuracy}")
