import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing , svm
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC , LinearSVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import timeit
import time
import os
import Data_Preprocessing
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler

def Trainig():

    if (os.path.exists("PreProcessing Data/TrainingDataSet.csv")):
        data = pd.read_csv("PreProcessing Data/TrainingDataSet.csv")
        print("Training DataSet Already Existing")
        Y = data['Rate']  # Label
        X = data.drop(columns=["Rate"], inplace=False)
    else:
        print("Starting Training Data PreProcessing")
        pre_processing = Data_Preprocessing.Pre_Processing()
        X,Y = pre_processing.PreProcessing_Trainig()

    highX = X[Y == 2]
    highY = Y[Y == 2]

    intermediateX =  X[Y == 1]
    intermediateY =  Y[Y == 1]

    lowX = X[Y == 0]
    lowY = Y[Y == 0]

    HX_train, HX_test, HY_train, HY_test = train_test_split(highX, highY, test_size=0.2, shuffle=True)
    IX_train, IX_test, IY_train, IY_test = train_test_split(intermediateX, intermediateY, test_size=0.2, shuffle=True)
    LX_train, LX_test, LY_train, LY_test = train_test_split(lowX, lowY, test_size=0.2, shuffle=True)

    X_train = np.concatenate((HX_train, IX_train ,LX_train))
    y_train = np.concatenate((HY_train, IY_train ,LY_train))

    X_test = np.concatenate((HX_test, IX_test, LX_test))
    y_test = np.concatenate((HY_test, IY_test, LY_test))

    print("Start Classification Techinques")
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    np.save('modelsPCA/traindata.npy',X_train)
    np.save('modelsPCA/trainlabel.npy', y_train)

    caller(X_train, y_train,X_test, y_test,0)
    PCA_algorithm(X_train, y_train,X_test, y_test)

def PCA_algorithm(X_train, y_train,X_test, y_test):
    # pca = PCA().fit(X_train)
    # # Plotting the Cumulative Summation of the Explained Variance
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)')  # for each component
    # plt.title('Pulsar Dataset Explained Variance')
    # plt.show()

    pca = PCA(n_components=20)
    pca.fit(X_train)
    x = pca.transform(X_train)
    #ratio= pca.explained_variance_ratio_
    xtest = pca.transform(X_test)

    print("//////////////////////// Start Training After PCA //////////////////////////////")
    caller(x, y_train ,xtest,y_test,1)


    # PCA_algorithm()

def caller(X_train, y_train, X_test, y_test,PCA):
    OneVsOnelinear(X_train, y_train, X_test, y_test,PCA)
    OneVsOne_LinearSVC(X_train, y_train, X_test, y_test,PCA)
    OneVsOne_rbf(X_train, y_train, X_test, y_test,PCA)
    OneVsOne_ploy(X_train, y_train, X_test, y_test,PCA)
    adaBoost(X_train, y_train, X_test, y_test,PCA)
    decisionTree(X_train, y_train, X_test, y_test,PCA)
    KNN(X_train, y_train, X_test, y_test,PCA)

def OneVsOnelinear(X_train, y_train, X_test, y_test,PCA):
    C = 1
    start = timeit.default_timer()
    model = OneVsOneClassifier(SVC(kernel='linear', C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    start2 = timeit.default_timer()
    accuracy = model.score(X_test, y_test)
    stop2 = timeit.default_timer()
    print('One VS One SVM accuracy :> Kernel|Linear:' + str(accuracy * 100), " Time Trainig : " + str(stop - start),
          " Time Testing : " + str(stop2 - start2))

    # pred_i = model.predict(X_test)
    # error= np.mean(pred_i != y_test)
    # accuracy=np.mean(pred_i == y_test)
    # print(accuracy*100)
    filename = 'OneVsOnelinear.pkl'
    if PCA == 0:
        pickle.dump(model, open("models/" + filename, 'wb'))
    else:
        pickle.dump(model, open("modelsPCA/" + filename, 'wb'))

def OneVsRestlinear(X_train, y_train, X_test, y_test,PCA):
    C = 1
    start = timeit.default_timer()
    model = OneVsRestClassifier(SVC(kernel='linear', C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    start2 = timeit.default_timer()
    accuracy = model.score(X_test, y_test)
    stop2 = timeit.default_timer()

    print('One VS Rest SVM accuracy :> Kernel|Linear:' + str(accuracy * 100),
          " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))

    filename = 'OneVsRestlinear.pkl'
    if PCA == 0:
        pickle.dump(model, open("models/" + filename, 'wb'))
    else:
        pickle.dump(model, open("modelsPCA/" + filename, 'wb'))

def OneVsOne_LinearSVC(X_train, y_train, X_test, y_test,PCA):
    C = 1
    start = timeit.default_timer()
    model1 = OneVsOneClassifier(LinearSVC(C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    # lin_svc = svm_model_linear_ovr.predict(X_test)
    start2 = timeit.default_timer()
    accuracy1 = model1.score(X_test, y_test)
    stop2 = timeit.default_timer()
    print('One VS One SVM accuracy :> Kernel|Linear:' + str(accuracy1 * 100),
          " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))

    filename = 'OneVsOne_LinearSVC.pkl'
    if PCA == 0:
        pickle.dump(model1, open("models/" + filename, 'wb'))
    else:
        pickle.dump(model1, open("modelsPCA/" + filename, 'wb'))

def OneVsRest_LinearSVC(X_train, y_train, X_test, y_test,PCA):
    C = 1
    start = timeit.default_timer()
    model = OneVsRestClassifier(LinearSVC(C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    # lin_svc = svm_model_linear_ovr.predict(X_test)
    start2 = timeit.default_timer()
    accuracy = model.score(X_test, y_test)
    stop2 = timeit.default_timer()

    print('One VS Rest SVM accuracy :> Kernel|Linear:' + str(accuracy * 100),
          " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))

    filename = 'OneVsRest_LinearSVC.pkl'
    if PCA == 0:
        pickle.dump(model, open("models/" + filename, 'wb'))
    else:
        pickle.dump(model, open("modelsPCA/" + filename, 'wb'))

def OneVsOne_rbf(X_train, y_train, X_test, y_test,PCA):
    C = .000001
    # svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=1).fit(X_train, y_train))
    start = timeit.default_timer()
    model = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.4, C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    start2 = timeit.default_timer()
    accuracy = model.score(X_test, y_test)
    stop2 = timeit.default_timer()
    print('One VS One SVM accuracy Kernel == rbf Gaussian : ' + str(accuracy * 100),
          " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))

    filename = 'OneVsOne_rbf.pkl'
    if PCA == 0:
        pickle.dump(model, open("models/" + filename, 'wb'))
    else:
        pickle.dump(model, open("modelsPCA/" + filename, 'wb'))

def OneVsRest_rbf(X_train, y_train, X_test, y_test,PCA):
    C = 1
    # svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=1).fit(X_train, y_train))
    start = timeit.default_timer()
    svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.4, C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    start2 = timeit.default_timer()
    accuracy = svm_model_linear_ovr.score(X_test, y_test)
    stop2 = timeit.default_timer()

    print('One VS Rest SVM accuracy Kernel == rbf Gaussian : ' + str(accuracy * 100),
          " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))
    filename = 'OneVsRest_rbf.pkl'
    if PCA == 0:
        pickle.dump(svm_model_linear_ovr, open("models/" + filename, 'wb'))
    else:
        pickle.dump(svm_model_linear_ovr, open("modelsPCA/" + filename, 'wb'))

def OneVsOne_ploy(X_train, y_train, X_test, y_test,PCA):
    C = 1
    start = timeit.default_timer()
    model = OneVsOneClassifier(SVC(kernel='poly', degree=2, C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    # poly_svc = svm_model_linear_ovr.predict(X_test)
    start2 = timeit.default_timer()
    accuracy = model.score(X_test, y_test)
    stop2 = timeit.default_timer()

    print('One VS One SVM accuracy Kernel == Poly: ' + str(accuracy * 100), " Time Trainig : " + str(stop - start),
          " Time Testing : " + str(stop2 - start2))
    filename = 'OneVsOne_ploy.pkl'
    if PCA == 0:
        pickle.dump(model, open("models/" + filename, 'wb'))
    else:
        pickle.dump(model, open("modelsPCA/" + filename, 'wb'))

def OneVsRest_ploy(X_train, y_train, X_test, y_test,PCA):
    C = 1
    start = timeit.default_timer()
    model = OneVsRestClassifier(SVC(kernel='poly', degree=3, C=C)).fit(X_train, y_train)
    stop = timeit.default_timer()

    # poly_svc = svm_model_linear_ovr.predict(X_test)
    start2 = timeit.default_timer()
    accuracy = model.score(X_test, y_test)
    stop2 = timeit.default_timer()

    print('One VS Rest SVM accuracy Kernel == Poly: ' + str(accuracy * 100), " Time Trainig : " + str(stop - start),
          " Time Testing : " + str(stop2 - start2))
    filename = 'OneVsRest_ploy.pkl'
    if PCA == 0:
        pickle.dump(model, open("models/" + filename, 'wb'))
    else:
        pickle.dump(model, open("modelsPCA/" + filename, 'wb'))

def adaBoost(X_train, y_train, X_test, y_test,PCA):

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15),
                             algorithm="SAMME.R",
                             n_estimators=20)
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    #
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    start = timeit.default_timer()
    bdt.fit(X_train, y_train)
    stop = timeit.default_timer()

    start2 = timeit.default_timer()
    y_prediction = bdt.predict(X_test)
    accuracy = np.mean(y_prediction == y_test)
    stop2 = timeit.default_timer()

    print("The achieved accuracy using Adaboost is " + str(accuracy * 100),
          " Time Trainig : " + str(stop - start), " Time Testing : " + str(stop2 - start2))
    filename = 'adaBoost.pkl'
    if PCA == 0:
        pickle.dump(bdt, open("models/" + filename, 'wb'))
    else:
        pickle.dump(bdt, open("modelsPCA/" + filename, 'wb'))

def decisionTree(X_train, y_train, X_test, y_test,PCA):
    # error = []
    # accuracy = []
    # # Calculating error for K values between 1 and 40
    # for i in range(1, 100):
    #     clf = tree.DecisionTreeClassifier(max_depth=i)
    #     clf.fit(X_train, y_train)
    #     pred_i = clf.predict(X_train)
    #     error.append(np.mean(pred_i != y_train))
    #     accuracy.append(np.mean(pred_i == y_train) * 100)
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate Depth Value')
    # plt.xlabel('Max Depth Value')
    # plt.ylabel('Mean Error')
    # plt.show()

    clf = tree.DecisionTreeClassifier(max_depth=15)
    clf.fit(X_train, y_train)
    start = timeit.default_timer()
    y_prediction = clf.predict(X_test)
    stop = timeit.default_timer()

    start2 = timeit.default_timer()
    accuracy = np.mean(y_prediction == y_test)
    stop2 = timeit.default_timer()

    print("The achieved accuracy using Decision Tree is " + str(accuracy * 100),
          " Time Trainig : " + str(stop - start), " Time Trainig : " + str(stop2 - start2))
    filename = 'decisionTree.pkl'
    if PCA == 0:
        pickle.dump(clf, open("models/" + filename, 'wb'))
    else:
        pickle.dump(clf, open("modelsPCA/" + filename, 'wb'))

def KNN(X_train, y_train, X_test, y_test,PCA):
    # error = []
    # accuracy = []
    # # Calculating error for K values between 1 and 40
    # for i in range(1, 40):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     pred_i = knn.predict(X_train)
    #     error.append(np.mean(pred_i != y_train))
    #     accuracy.append(np.mean(pred_i == y_train) * 100)
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # plt.show()
    #
    start = timeit.default_timer()
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)
    stop = timeit.default_timer()

    start2 = timeit.default_timer()
    pred_i = knn.predict(X_test)
    error = (np.mean(pred_i != y_test))
    accuracy = (np.mean(pred_i == y_test))
    stop2 = timeit.default_timer()

    print("The achieved accuracy using KNN is " + str(accuracy * 100), " Time Trainig : " + str(stop - start),
          " Time Trainig : " + str(stop2 - start2))

    filename = 'KNN.pkl'
    if PCA == 0:
        pickle.dump(knn, open("models/" + filename, 'wb'))
    else:
        pickle.dump(knn, open("modelsPCA/" + filename, 'wb'))

# plt.figure(figsize=(10, 8))
# sns.heatmap(data[['Rate' ,"Price", "In-app Purchases", "Size", 'User Rating Count',
#                   'Original Release Year','Current Version Release Year',
#                   'Original Release Month','Current Version Release Month' , 'Developer'  ]].corr(), annot=True,
#             cmap="coolwarm")
# plt.show()

# scaler = StandardScaler()
    # scaler.fit(X_train)
    #
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)