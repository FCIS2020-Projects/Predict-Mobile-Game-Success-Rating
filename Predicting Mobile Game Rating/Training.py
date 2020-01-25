import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt

class Training:
    def __init__(self):
        self.data = pd.read_csv('sample/predicting_mobile_game_success_train_set.csv')

    def extract_column_Training(self, name):
        distinictValues = set()
        for i in self.data[name]:
            for j in i:
                distinictValues.add(j)

        for i in distinictValues:
            list_lng = []
            bool = 0
            for j in self.data[name]:
                for k in j:
                    if (i == k): bool = 1
                list_lng.append(bool)
                bool = 0

            elementum = sum(list_lng)
            if elementum > 700:
                self.data.insert(self.data.shape[1], i, list_lng, True)
        return self.data


    def Preprocessing(self):
        self.data = self.data.drop_duplicates()

        self.data.dropna(how='all', axis='columns', inplace=True)
        self.data.dropna(how='all', axis='rows', inplace=True)

        self.data.drop(columns=['URL', 'ID', 'Name', 'Icon URL', 'Description', 'Subtitle', 'Primary Genre'], inplace=True)

        self.data["Price"].fillna(self.data["Price"].sum(skipna=True) / len(self.data["Price"]), inplace=True)

        self.data["Size"].fillna(self.data["Size"].sum(skipna=True) / len(self.data["Size"]), inplace=True)
        self.data["Size"] = self.data["Size"].map(lambda x: round(x / (1024 * 1024), 2))

        avgUsercount = self.data["User Rating Count"].sum(skipna=True) / len(self.data["User Rating Count"])
        self.data['User Rating Count'].fillna(avgUsercount, inplace=True)

        self.data["Languages"].fillna('EN', inplace=True)
        self.data["Languages"] = self.data["Languages"].map(lambda x: x.lower())
        self.data["Languages"] = self.data["Languages"].map(lambda x: x.replace(" ", ""))
        self.data["Languages"] = self.data["Languages"].map(lambda x: x.split(","))

        self.data["Genres"].fillna('Games', inplace=True)
        self.data["Genres"] = self.data["Genres"].map(lambda x: x.lower())
        self.data["Genres"] = self.data["Genres"].map(lambda x: x.replace(" ", ""))
        self.data["Genres"] = self.data["Genres"].map(lambda x: x.split(","))

        self.data.dropna(subset=['Average User Rating'], inplace=True)

        self.data = self.extract_column_Training("Languages")
        self.data.drop(columns=["Languages"], inplace=True)
        self.data = self.extract_column_Training("Genres")
        self.data.drop(columns=["Genres"], inplace=True)

        self.data["Original Release Date"] = self.data["Original Release Date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y"))
        self.data["Original Release Year"] = self.data["Original Release Date"].map(lambda x: x.year)
        self.data["Original Release Month"] = self.data["Original Release Date"].map(lambda x: x.month)

        self.data["Current Version Release Date"] = self.data["Current Version Release Date"].map(
            lambda x: datetime.strptime(x, "%d/%m/%Y"))
        self.data["Current Version Release Year"] = self.data["Current Version Release Date"].map(lambda x: x.year)
        self.data["Current Version Release Month"] = self.data["Current Version Release Date"].map(lambda x: x.month)

        self.data.drop(columns=['Original Release Date', 'Current Version Release Date'], inplace=True)
        self.data.drop(columns=["Original Release Year", "Original Release Month"], inplace=True)

        avgYear = self.data["Current Version Release Year"].sum(skipna=True) / len(
            self.data["Current Version Release Year"])
        avgMonth = self.data["Current Version Release Month"].sum(skipna=True) / len(
            self.data["Current Version Release Month"])

        self.data["Age Rating"] = self.data["Age Rating"].map(lambda x: x.replace(" ", ""))
        self.data["Age Rating"] = self.data["Age Rating"].map(lambda x: float(x[:-1]))

        self.data.drop(columns=['Developer'], inplace=True)

        self.data.drop(columns=['In-app Purchases'], inplace=True)

        def Feature_Normalizer(X, cols):
            for c in cols:
                norm = preprocessing.MinMaxScaler(feature_range=(0, 1))
                norm.fit(X[c].values.reshape(-1, 1))
                X[c] = norm.transform(X[c].values.reshape(-1, 1))
            return X

        self.data = Feature_Normalizer(self.data, ['User Rating Count', 'Price',
                                         'Size', 'Age Rating',
                                         'Current Version Release Year',
                                         'Current Version Release Month'])

        X = self.data.drop(columns=['Average User Rating'], inplace=False)  # Features
        Y = self.data['Average User Rating']  # Label
        self.data.to_csv("datapreprocessing/TrainingDataSet.csv", encoding='utf-8', index=False)
        return X, Y



    def training(self):
        def linear_regression(X, Y):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)
            cls = linear_model.LinearRegression()
            cls.fit(X_train, y_train)
            prediction = cls.predict(X_test)
            y_train_predicted = cls.predict(X_train)

            for i in X.columns.values:
                plt.scatter(X[i], Y)
                plt.xlabel(i, fontsize=20)
                plt.ylabel(Y.name, fontsize=20)
                pred = X[i]*cls.coef_[X.columns.get_loc(i)] + cls.intercept_
                plt.plot(X[i], pred, color='red', linewidth=3)
                plt.show()


            count = 0
            for i in range(len(y_test)):
                if (abs(np.asarray(prediction)[i] - np.asarray(y_test)[i]) < 1):
                    count += 1
            print(count / len(y_test))

            print('///////////////training/////////////////////////')
            print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_train), y_train_predicted))
            print('R2 Score :', cls.score(X_train, y_train))
            print('///////////////testing/////////////////////////')
            print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
            print('R2 Score :', cls.score(X_test, y_test))
            # true_average_rating = np.asarray(y_test)
            # predicted_average_rating = prediction
            # print(true_average_rating, predicted_average_rating)
            # print('acc :', r2_score(true_average_rating, predicted_average_rating))

            filename = 'model/linear.pkl'
            pickle.dump(cls, open(filename, 'wb'))

        # //////////////////////////polynomial_regression/////////////////////////////////////////

        def polynomial_regression(X, Y):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)
            poly_features = preprocessing.PolynomialFeatures(degree=2)

            # transforms the existing features to higher degree features.
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.fit_transform(X_test)
            # fit the transformed features to Linear Regression
            poly_model = linear_model.LinearRegression()
            poly_model.fit(X_train_poly, y_train)

            # predicting on training data-set
            y_train_predicted = poly_model.predict(X_train_poly)

            # predicting on test data-set
            prediction = poly_model.predict(X_test_poly)
            # print('Co-efficient of linear regression', poly_model.coef_)
            # print('Intercept of linear regression model', poly_model.intercept_)
            print('///////////////training/////////////////////////')
            print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_train), y_train_predicted))
            print('R2 Score :', poly_model.score(X_train_poly, y_train))
            print('///////////////testing/////////////////////////')
            print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
            print('R2 Score :', poly_model.score(X_test_poly, y_test))
            filename = 'model/poly.pkl'
            pickle.dump(poly_model, open(filename, 'wb'))

        X, Y = self.Preprocessing()
        print('///////////////linear_regression/////////////////////////')
        linear_regression(X, Y)
        print('//////////////polynomial_regression//////////////////////')
        polynomial_regression(X, Y)
