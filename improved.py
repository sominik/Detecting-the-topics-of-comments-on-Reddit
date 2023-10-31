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


def read_from_file():
    f = open("similar_words.txt", "r")
    text = f.read()
    array_of_vec = []
    text_array = text.split("\n")
    for i in range(0, len(text_array) - 1):
        vec = text_array[i].split('[')[1].split(']')[0].split(', ')
        for j in range(len(vec)):
            vec[j] = vec[j].split('\'')[1].lower()
        array_of_vec.append(vec)
    return array_of_vec


def clean_data(comment):
    words = re.sub(r"[^a-zA-Z]", " ", comment.lower()).split(" ")
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            word = porter_stemmer.stem(word)
            cleaned_words.append(word)
    cleaned_comment = " ".join(cleaned_word for cleaned_word in cleaned_words)
    return cleaned_comment


reference = read_from_file()


def bagOfWords_improvement(comment):
    words = comment.split(" ")
    for i in range(len(words)):
        for j in range(len(reference)):
            if words[i] in reference[j]:
                words[i] = reference[j][0]
    new_comment = " ".join(word for word in words)
    return new_comment
    # print(new_comment)


tfidf_vectorizer = TfidfVectorizer(smooth_idf=True,
                                   ngram_range=(1, 3))
train_dataset['cleaned_comment'] = train_dataset['Comment'].apply(clean_data)
train_dataset['improved_comment'] = train_dataset['cleaned_comment'].apply(bagOfWords_improvement)

tfidf_vectorizer.fit(train_dataset['improved_comment'])
train_vectors = tfidf_vectorizer.transform(train_dataset['improved_comment'])

test_dataset['cleaned_comment'] = test_dataset['Comment'].apply(clean_data)
test_dataset['improved_comment'] = test_dataset['cleaned_comment'].apply(bagOfWords_improvement)
test_vectors = tfidf_vectorizer.transform(test_dataset['improved_comment'])

svm = svm.SVC(kernel='linear')
svm.fit(train_vectors, train_labels)
predicted_labels = svm.predict(test_vectors)
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"improved bag of words accuracy on test data is : {accuracy}")

