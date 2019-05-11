import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from numba import jit
from scipy.sparse import hstack, csr_matrix

data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
data.drop("article_link", axis=1, inplace=True)

punctuations = re.compile(f'[{string.punctuation}]')
data['headline'] = data['headline'].replace(punctuations, " ").str.lower()
stopWords = set(stopwords.words('english'))
data['headline'] = data['headline'].apply(
        lambda x: ' '.join(
                term for term in x.split() if term not in stopWords))

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
model = LogisticRegression(solver='saga').fit(trainXvectorized, trainY)
y_pred = model.predict(testXvectorized)
model.score(testXvectorized, testY)


# tính accuracy, precision, recall, f1-score
def score(y_true, y_pred):
    accu = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    reca = metrics.recall_score(y_true, y_pred)
    f1score = metrics.f1_score(y_true, y_pred)
    return accu, prec, reca, f1score


result = score(testY, y_pred)


# hàm tính sigmoid
@jit
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# hàm tính độ lỗi
@jit
def loss(p, y):
    return np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))


# hàm thêm cột 1 vào ma trận examples
def add1Col(x):
    intercept = np.ones((x.shape[0], 1))
    return hstack((intercept, x))


@jit
def getMomentum(gradient, vPrev, alpha, gamma=0.9):
    v = gamma*vPrev + alpha/(1-gamma)*gradient
    return v


# hàm chạy batch gradient descent
@jit
def BGD(x, y, lr, numIter=10000, epsilon=1e-10):
    x = add1Col(x)
    theta = np.zeros(x.shape[1])
    v = np.zeros(x.shape[1])
    E = -float('inf')
    for i in range(numIter):
        z = x.dot(theta)
        p = sigmoid(z)
        gradient = x.T.dot(p - y)
        v = getMomentum(gradient, v, lr)
        theta = theta - v
        Enext = loss(p, y)
        if abs(E-Enext) < epsilon:
            break
        E = Enext
        print('iteration #'+str(i)+' E:' + str(E))
    return theta


def predict_probs(X, theta):
    return sigmoid(X.dot(theta))


def predict(X, theta, threshold=0.5):
    return np.array(predict_probs(X, theta) >= threshold, dtype=int)


w = BGD(trainXvectorized, trainY, 0.001, 15000, 1e-12)
y_pred = predict(add1Col(testXvectorized), w)
print(result)
print(score(testY, y_pred))

# model KNN
    
from sklearn.neighbors import KNeighborsClassifier

KNNmodel = KNeighborsClassifier().fit(trainXvectorized, trainY)

knn_y_pred = KNNmodel.predict(testXvectorized)

print(score(testY,knn_y_pred))

# model Decision Tree

from sklearn.tree import DecisionTreeClassifier

DTmodel = DecisionTreeClassifier().fit(trainXvectorized, trainY)

dt_y_pred = DTmodel.predict(testXvectorized)

print(score(testY,dt_y_pred))

# naive bayes model
from sklearn.naive_bayes  import BernoulliNB

NBmodel = BernoulliNB().fit(trainXvectorized, trainY)

nb_y_pred = NBmodel.predict(testXvectorized)

print(score(testY, nb_y_pred))

