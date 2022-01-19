from google.colab import drive
drive.mount('/content/drive')

!pip install fuzzy-c-means

import pandas as pd
import numpy as np
import math
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from fcmeans import FCM
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

dataset = np.loadtxt('/content/drive/MyDrive/trabalho.csv', delimiter=",")

X = dataset[:, 0:29]
y = dataset[:, 30]

X = normalized(X, "min")

def plotClustering(X, y_pred, centers):
  import matplotlib.pyplot as plt
  plt.figure()

  plt.scatter(centers[:, 0], centers[:, 1], c="red", marker='*', s=150)
  plt.scatter(X[:, 0], X[:, 1], c=y_pred)
  plt.title("Dados associados aos clusters")

  plt.show()

def evaluateROCCurve(y_true, scores):
  # Calcula a curva ROC
  from sklearn.metrics import roc_curve, auc

  fpr1, tpr1, thresholds = roc_curve(y_true, scores)
  auc1 = auc(fpr1, tpr1)

  return fpr1, tpr1, auc1

def plotRocCurve(fpr1, tpr1, auc1, eer=False):
  # Plota o gráfico da curva ROC
  import matplotlib.pyplot as plt
  plt.figure()
  lw = 2
  plt.plot(fpr1, tpr1, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % auc1)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  if eer is True:
    plt.plot([0, 1], [1, 0], color='red', lw=lw, linestyle='-.')

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic (ROC)')
  plt.legend(loc="lower right")
  plt.show()

def trainNoSupervised(variance, model, name):
  pca = PCA(n_components=variance, svd_solver='full')
  new_dataset = pca.fit_transform(np.copy(X), y)
  X_train, X_test, y_train, y_test = train_test_split(new_dataset, y, test_size=0.20, random_state=25)
  model.fit(X_train)
  predictions = model.predict(X_test)
  y_pred = predictions.reshape(-1)

  print("Modelo: ", name)
  print("Variância ", variance)
  print("Número de componentes: ", pca.n_components_)

  if name == 'K-Means':
    centers = model.cluster_centers_
  else:
    centers = model.centers

  silhouette_avg = silhouette_score(X_test, y_pred)
  m1 = X_test[y_pred==0].mean()
  m2 = X_test[y_pred==1].mean()
  s1 = X_test[y_pred==0].std(ddof=1)
  s2 = X_test[y_pred==1].std(ddof=1)
  fisher =  (abs(m1 - m2) ** 2) / ((s1 ** 2) + (s2 ** 2))
  print("\nCRITÉRIO INTERNO")
  print("Média silhouete: %.2f" % silhouette_avg)
  print("Critério de  fisher: %.2f" % fisher)

  cm = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  acc = (tn + tp) / (tn + fp + fn +tp)
  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)
  print("\nCRITÉRIO EXTERNO")
  print("ACC: %.2f" % acc)
  print("TPR: %.2f" % tpr)
  print("TNR: %.2f" % tnr)
  print(cm)
  plotClustering(X_test, y_pred, centers)
  print("\n------------------------------------------------------\n")

def trainSupervised(variance, model, name):
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
  print('------------------------')
  print(f'Modelo: {name}')
  print(f'Variância acumulada: {variance}')
  print(f'Nº componentes: {pca.n_components_}')
  print(f'Acurácia: {acc}')
  print(f'TPR: {tpr}')
  print(f'TNR: {tnr}')
  print('Matriz de confusão:')
  print(cm)
  fpr1, tpr1, auc1 = evaluateROCCurve(y_test, predictions)
  plotRocCurve(fpr1, tpr1, auc1)

variances = [0.50, 0.75, 0.90, 0.99]

for variance in variances:
  trainNoSupervised(variance, KMeans(n_clusters=2, random_state=0), 'K-Means');
  trainNoSupervised(variance, FCM(n_clusters=2), 'Fuzzy C-Means');

variances = [0.50, 0.75, 0.90, 0.99]

for variance in variances:
  trainSupervised(variance, SVC(kernel='linear', probability=True), 'SVM Linear');
  trainSupervised(variance,  SVC(kernel='rbf', probability=True), 'SVM RBF');
  trainSupervised(variance, DecisionTreeClassifier(splitter='random'), 'C4.5');
  trainSupervised(variance, SGDClassifier(loss='log', penalty="l2"), 'SGDC');
  trainSupervised(variance, KNeighborsClassifier(n_neighbors=2), 'KNeighbors');