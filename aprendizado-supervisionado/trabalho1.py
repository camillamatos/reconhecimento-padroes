from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import math
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

def print_data(name, variance, n_components, acc, cm, tpr, tnr):
  print('------------------------')
  print(f'Modelo: {name}')
  print(f'Variância acumulada: {variance}')
  print(f'Nº componentes: {n_components}')
  print(f'Acurácia: {acc}')
  print(f'TPR: {tpr}')
  print(f'TNR: {tnr}')
  print('Matriz de confusão:')
  print(cm)

def normalized(data, method="min"):
  new_data = []

  if method == "min":
    new_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

  elif method == "std":
    data_T = X.T
    for i in data_T:
      new_data.append((i - np.mean(row)) / ((np.std(i, ddof=1))**2))
    
    new_data = np.array(new_data).T

  return new_data

dataset = np.loadtxt('/content/drive/MyDrive/database.csv', delimiter=",")
X = dataset[:, 0:50]
y = dataset[:, 51]
y = [0 if i < 0 else 1 for i in y]

X = normalized(X, "min")

def train(variance, model, name):
  pca = PCA(n_components=variance, svd_solver='full')
  new_dataset = pca.fit_transform(np.copy(X), y)
  X_train, X_test, y_train, y_test = train_test_split(new_dataset, y, test_size=0.20, random_state=25)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  cm = confusion_matrix(y_test, predictions)
  tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
  acc = (tn+tp) / (tn+fp+fn+tp)
  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)
  atr = new_dataset.shape[1]
  print_data(name, variance, pca.n_components_, acc, cm, tpr, tnr)

  return atr, cm, acc, tpr, tnr

variances = [0.75, 0.9, 0.99]

for variance in variances:
  resultG = train(variance, GaussianNB(), 'Naive Bayes');
  resultSVML = train(variance, SVC(kernel='linear'), 'SVM Linear');
  resultSVMR = train(variance,  SVC(kernel='rbf'), 'SVM RBF');
  resultC = train(variance, DecisionTreeClassifier(splitter='random'), 'C4.5');
