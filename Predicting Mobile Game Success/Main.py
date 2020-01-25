import Classification
import Classification_Testing
import Data_Preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

print("Appstore games Classification")

input = int(input("Press 1 To Train Models Or 2 To Test Model :: "))
if input == 1 :
    Classification.Trainig()
else :
    pre_processing = Data_Preprocessing.Pre_Processing()
    X, Y = pre_processing.PreProcessing_Testing()

    print("Testing Before PCA Algorithm")
    Testing = Classification_Testing.Classification_Testing(X,Y)
    Testing.OneVsOnelinear(0)
    Testing.OneVsOne_LinearSVC(0)
    Testing.OneVsOne_ploy(0)
    Testing.OneVsOne_rbf(0)
    Testing.adaBoost(0)
    Testing.decisionTree(0)
    Testing.KNN(0)



    xtrain = np.load('modelsPCA/traindata.npy', allow_pickle=True)
    ytrain = np.load('modelsPCA/trainlabel.npy', allow_pickle=True)
    Testing.PCA_algorithm(xtrain,ytrain)
