from gensim.models import KeyedVectors
import re
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
print(model.word_vec)
train_dataset = pd.read_csv("train.csv")
train_comments = np.array(train_dataset['Comment'])

test_dataset = pd.read_csv("test.csv")
test_comments = np.array(test_dataset['Comment'])

stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
# nltk.download("popular")
f = open("words.txt", "a")


def extract_word_set(comments):
    word_set=[]
    for comment in comments:
        words = re.sub(r"[^a-zA-Z]", " ", comment.lower()).split(" ")
        cleaned_words = []
        for word in words:
            if word not in stop_words:
                word = porter_stemmer.stem(word)
                cleaned_words.append(word)
        word_set.extend(cleaned_words)
    words_set = np.unique(np.array(word_set))
    return words_set


def similar_word(words_set):
    similar_words = []
    i = 0
    for word in words_set:
        if word in model:
            similar_words.append([word])
            result = model.most_similar(positive=[word], topn=5)
            # print(word, " --> ", result)
            temp = [data[0] for data in result]
            for item in temp:
                if item in words_set:
                    similar_words[i] += [item]
            f.write(str(similar_words[i]) + "\n")
            i += 1


def delete_extra_words():
    f = open("words.txt", "r")
    f2 = open("similar_words.txt", "a")
    text = f.read()
    text_array = text.split("\n")
    for i in range(0, len(text_array) - 1):
        vec = text_array[i].split('[')[1].split(']')[0].split(', ')
        for j in range(len(vec)):
            vec[j] = vec[j].split('\'')[1]
        if len(vec) != 1:
            f2.write(str(vec) + "\n")


# main:
words_set = extract_word_set(train_comments)
words_set2 = extract_word_set(test_comments)
arr1 = np.array(words_set)
arr2 = np.array(words_set2)
words_set = np.concatenate((arr1, arr2))
words_set = np.unique(words_set)
similar_word(words_set)
