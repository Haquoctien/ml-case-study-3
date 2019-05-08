import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)

data.drop("article_link", axis = 1, inplace = True)

punctuations = re.compile(f'[{string.punctuation}]')

data['headline'] = data['headline'].replace(punctuations," ").str.lower()

stopWords = set(stopwords.words('english'))

data['headline'] = data['headline'].apply(lambda x: 
    ' '.join(term for term in x.split() if term not in stopWords))
    
vectorizer = TfidfVectorizer()
vectorizer.fit(data['headline'])
dictionary = vectorizer.get_feature_names()

# tách tập train và tập test
trainData = data[:20000]
trainX = np.array(trainData['headline'])
trainY = np.array(trainData['is_sarcastic'])
testData = data[20000:]
testX = np.array(testData['headline'])
testY = np.array(testData['is_sarcastic'])

# vector hóa tiêu đề
trainXvectorized = vectorizer.transform(trainX)
testXvectorized = vectorizer.transform(testX)

# mô hình
model = LogisticRegression(solver = 'saga').fit(trainXvectorized, trainY)
y_pred = model.predict(testXvectorized)
model.score(testXvectorized, testY)

