import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from numba import jit
from scipy.sparse import hstack

# đọc file
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

# tách tập train và tập test
trainData = data[:20000]
trainX = np.array(trainData['headline'])
trainY = np.array(trainData['is_sarcastic'])
testData = data[20000:]
testX = np.array(testData['headline'])
testY = np.array(testData['is_sarcastic'])

# vector hóa tiêu đề
vectorizer = TfidfVectorizer().fit(trainX)
trainXvectorized = vectorizer.transform(trainX)
testXvectorized = vectorizer.transform(testX)

# mô hình từ thư viện
model = LogisticRegression(solver='saga').fit(trainXvectorized, trainY)
y_pred_sklearn = model.predict(testXvectorized)
model.score(testXvectorized, testY)


# hàm tính accuracy, precision, recall, f1-score
def score(y_true, y_pred):
    accu = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    reca = metrics.recall_score(y_true, y_pred)
    f1score = metrics.f1_score(y_true, y_pred)
    return {'accuracy': accu,'precision': prec,'recall': reca,'f1': f1score}

print('SkLearn model: ')
print(score(testY, y_pred_sklearn))


# hàm tính sigmoid
@jit
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# hàm tính độ lỗi
@jit
def loss(p, y):
    return np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))

@jit
# hàm thêm cột 1 vào ma trận examples
def add1Col(x):
    intercept = np.ones((x.shape[0], 1))
    return hstack((intercept, x))

# hàm tính momentum
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
    return theta

@jit
def predict_probs(X, theta):
    return sigmoid(X.dot(theta))

@jit
def predict(X, theta, threshold=0.5):
    return np.array(predict_probs(X, theta) >= threshold, dtype=int)


w = BGD(trainXvectorized, trainY, 0.015, 200)
y_pred = predict(add1Col(testXvectorized), w)
print('BDG Logistic Regression: ')
print(score(testY, y_pred))

# model KNN
from sklearn.neighbors import KNeighborsClassifier
KNNmodel = KNeighborsClassifier(n_neighbors = 7, metric='euclidean').fit(trainXvectorized, trainY)
knn_y_pred = KNNmodel.predict(testXvectorized)
print('KNN model: ')
print(score(testY,knn_y_pred))

# model Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTmodel = DecisionTreeClassifier().fit(trainXvectorized, trainY)
dt_y_pred = DTmodel.predict(testXvectorized)
print('Decision Tree model: ')
print(score(testY,dt_y_pred))

# naive bayes model
from sklearn.naive_bayes  import BernoulliNB
NBmodel = BernoulliNB().fit(trainXvectorized, trainY)
nb_y_pred = NBmodel.predict(testXvectorized)
print('Naive Bayes model: ')
print(score(testY, nb_y_pred))
