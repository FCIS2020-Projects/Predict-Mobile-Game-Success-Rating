import pandas as pd
import pickle
import timeit
from sklearn.decomposition import PCA
import os
import Data_Preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import  numpy as np
import itertools
from sklearn import svm, datasets


class Classification_Testing:
    def __init__(self , X_test , Y_test):
        self.X_test = X_test
        self.Y_test = Y_test

    def OneVsOnelinear(self,PCA):
        if PCA == 0:
            loaded_model = pickle.load(open("models/OneVsOnelinear.pkl", 'rb'))
        else:
            loaded_model = pickle.load(open("modelsPCA/OneVsOnelinear.pkl", 'rb'))

        start = timeit.default_timer()
        result = loaded_model.score(self.X_test, self.Y_test)
        stop = timeit.default_timer()
        print(str(result * 100)," Time Testing : " + str(stop - start))
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))

    def OneVsOne_LinearSVC(self,PCA):
        if PCA == 0 :
            loaded_model = pickle.load(open("models/OneVsOne_LinearSVC.pkl", 'rb'))
        else:
            loaded_model = pickle.load(open("modelsPCA/OneVsOne_LinearSVC.pkl", 'rb'))

        start = timeit.default_timer()
        result = loaded_model.score(self.X_test, self.Y_test)
        stop = timeit.default_timer()
        print(str(result * 100)," Time Testing : " + str(stop - start))
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))
    def OneVsOne_rbf(self,PCA):
        if PCA == 0 :
            loaded_model = pickle.load(open("models/OneVsOne_rbf.pkl", 'rb'))
        else:
            loaded_model = pickle.load(open("modelsPCA/OneVsOne_rbf.pkl", 'rb'))

        start = timeit.default_timer()
        result = loaded_model.score(self.X_test, self.Y_test)
        stop = timeit.default_timer()
        print(str(result * 100)," Time Testing : " + str(stop - start))
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))
    def OneVsOne_ploy(self,PCA):
        if PCA == 0 :
            loaded_model = pickle.load(open("models/OneVsOne_ploy.pkl", 'rb'))
        else:
            loaded_model = pickle.load(open("modelsPCA/OneVsOne_ploy.pkl", 'rb'))

        start = timeit.default_timer()
        result = loaded_model.score(self.X_test, self.Y_test)
        stop = timeit.default_timer()
        print(str(result * 100)," Time Testing : " + str(stop - start))
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))
    def adaBoost(self,PCA):
        if PCA == 0 :
            loaded_model = pickle.load(open("models/adaBoost.pkl", 'rb'))
        else:
            loaded_model = pickle.load(open("modelsPCA/adaBoost.pkl", 'rb'))

        start = timeit.default_timer()
        result = loaded_model.score(self.X_test, self.Y_test)
        stop = timeit.default_timer()
        print(str(result * 100)," Time Testing : " + str(stop - start))
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))

    def decisionTree(self,PCA):
        if PCA == 0 :
            loaded_model = pickle.load(open("models/decisionTree.pkl", 'rb'))
        else:
            loaded_model = pickle.load(open("modelsPCA/decisionTree.pkl", 'rb'))

        start = timeit.default_timer()
        result = loaded_model.score(self.X_test, self.Y_test)
        stop = timeit.default_timer()
        print(str(result * 100)," Time Testing : " + str(stop - start))
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))

    def KNN(self,PCA):
        if PCA == 0 :
            loaded_model = pickle.load(open("models/KNN.pkl", 'rb'))
        else:
            loaded_model = pickle.load(open("modelsPCA/KNN.pkl", 'rb'))

        start = timeit.default_timer()
        result = loaded_model.score(self.X_test, self.Y_test)
        stop = timeit.default_timer()
        print(str(result * 100)," Time Testing : " + str(stop - start))
        y_pred = loaded_model.predict(self.X_test)
        print(confusion_matrix(self.Y_test, y_pred))

    def PCA_algorithm(self,xtrain,ytrain):
        # pca = PCA().fit(X_train)
        # # Plotting the Cumulative Summation of the Explained Variance
        # plt.figure()
        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # plt.xlabel('Number of Components')
        # plt.ylabel('Variance (%)')  # for each component
        # plt.title('Pulsar Dataset Explained Variance')
        # plt.show()

        pca = PCA(n_components=20)
        pca.fit(xtrain)
        x = pca.transform(xtrain)
        ratio= pca.explained_variance_ratio_
        xtest = pca.transform(self.X_test)

        print("//////////////////////// Start Training After PCA //////////////////////////////")
        self.X_test = xtest

        self.OneVsOnelinear(1)
        self.OneVsOne_LinearSVC(1)
        self.OneVsOne_ploy(1)
        self.OneVsOne_rbf(1)
        self.adaBoost(1)
        self.decisionTree(1)
        self.KNN(1)