import numpy as np
import pandas as pd
# import seaborn as sns
from datetime import datetime
from sklearn import preprocessing , svm
import  os
from sklearn.preprocessing import StandardScaler

class Pre_Processing:
    def __init__(self):
        self.data     = pd.read_csv('Samples/appstore_games_classification.csv')
        self.dataTest = pd.read_csv('Samples/samples_predicting_mobile_game_success_test_set_classification.csv')

        self.languages = self.ExtractUniqe("Languages")
        self.genres = self.ExtractUniqe("Genres")

        self.columnName = []
        if os.path.exists('PreProcessing Data/TrainingDataSet.csv'):
            self.columnName = list(pd.read_csv('PreProcessing Data/TrainingDataSet.csv').columns.values)

        self.avgPrice = self.data["Price"].sum(skipna = True) / len(self.data["Price"])
        self.avgSize = self.data["Size"].sum(skipna = True) / len(self.data["Size"])

        self.avgYear = 0.0
        self.avgMonth = 0.0
        self.avgUsercount = 0.0
        self.avgApppurchases = 0.0

    def set_NULL(self):

        self.data["Current Version Release Date"] = self.data["Current Version Release Date"].map(
            lambda x: datetime.strptime(x, "%d/%m/%Y"))
        self.data["Current Version Release Year"] = self.data["Current Version Release Date"].map(lambda x: x.year)
        self.data["Current Version Release Month"] = self.data["Current Version Release Date"].map(lambda x: x.month)

        self.avgYear = self.data["Current Version Release Year"].sum(skipna = True) / len(self.data["Current Version Release Year"])
        self.avgMonth = self.data["Current Version Release Month"].sum(skipna = True) / len(self.data["Current Version Release Month"])

        self.avgUsercount = self.data["User Rating Count"].sum(skipna = True) / len(self.data["User Rating Count"])

        # self.data['In-app Purchases'].fillna(0, inplace=True)
        # self.data['In-app Purchases'] = self.data['In-app Purchases'].map(
        #     lambda x: sum(float(i) for i in str(x).split(",")))
        # self.avgApppurchases = self.data['In-app Purchases'].sum(skipna = True) / len(self.data['In-app Purchases'])

    def ExtractUniqe(self,name):
        tempdata = pd.read_csv('Samples/appstore_games_classification.csv')
        tempdata.drop_duplicates()
        # tempdata.dropna(subset=['Rate'] , inplace=True)

        returnunique = []

        if name == "Languages":
            tempdata[name].fillna('EN', inplace=True)
        if name == "Genres":
            tempdata[name].fillna('Games', inplace=True)

        tempdata[name] = tempdata[name].map(lambda x: x.lower())
        tempdata[name] = tempdata[name].map(lambda x: x.replace(" ", ""))
        tempdata[name] = tempdata[name].map(lambda x: x.split(","))

        distinictValues = set()
        for i in tempdata[name]:
            for j in i:
                distinictValues.add(j)

        for i in distinictValues:
            list_lng = []
            bool = 0
            for j in tempdata[name]:
                for k in j:
                    if (i == k): bool = 1
                list_lng.append(bool)
                bool = 0

            total = sum(list_lng)
            # if total > 1000:
            returnunique.append(i)
        return returnunique

    def extract_column_Training(self,data, name):
        distinictValues = set()
        for i in data[name]:
            for j in i:
                distinictValues.add(j)

        for i in distinictValues:
            list_lng = []
            bool = 0
            for j in data[name]:
                for k in j:
                    if (i == k): bool = 1
                list_lng.append(bool)
                bool = 0

            elementum = sum(list_lng)
            # if elementum > 1000:
            data.insert(data.shape[1], i, list_lng, True)
        return data

    def extract_column_Testing(self,data,name):
        distinictValues = set()

        # print(data[name])
        for i in data[name]:
            for j in i:
                distinictValues.add(j)
        for i in distinictValues:
            list_lng = []
            bool = 0
            for j in data[name]:
                for k in j:
                    if (i == k): bool = 1
                list_lng.append(bool)
                bool = 0
            # print(i)
            if i in self.languages :
                data.insert(data.shape[1], i, list_lng, True)
            if i in self.genres :
                data.insert(data.shape[1], i, list_lng, True)
        return data

    def check_column(self,data):
        testingCols = list(data.columns.values)
        # print(self.columnName)
        # print(testingCols)
        for tCol in testingCols :
            # print(tCol)
            if tCol not in self.columnName:
                data.drop(columns=[tCol] , inplace = True)
        # print(list(data.columns.values))
        # 'User Rating Count', 'Price', 'In-app Purchases',
        # 'Age Rating', 'Size', 'Rate',
        # 'Current Version Release Year', 'Current Version Release Month',

        for col in self.columnName :
            if col not in testingCols:
                if col == "User Rating Count" :
                    data.insert(data.shape[1], col, [self.avgUsercount]*data.shape[0], True)
                elif col == 'Price':
                    data.insert(data.shape[1], col, [self.avgPrice] * data.shape[0], True)
                # elif col == 'In-app Purchases:' :
                #     data.insert(data.shape[1], col, [self.avgApppurchases] * data.shape[0], True)
                elif col == 'Age Rating' :
                    data.insert(data.shape[1], col, [9] * data.shape[0], True)
                elif col == 'Size' :
                    data.insert(data.shape[1], col, [self.avgSize] * data.shape[0], True)
                elif col == 'Current Version Release Year':
                    data.insert(data.shape[1], col, [self.avgYear] * data.shape[0], True)
                elif col == 'Current Version Release Month':
                    data.insert(data.shape[1], col, [self.avgMonth] * data.shape[0], True)
                else :
                    data.insert(data.shape[1], col, [0] * data.shape[0], True)

        return data

    def PreProcessing_Trainig(self):
        # print(data.shape) #(17006,18)
        # print(data.isna().sum())
        # print(self.data.duplicated().value_counts())
        # print(self.data.columns)
        # print(data.info())
        # columns has NULL Values
        # subtitle = 11745 , User Rating count = 9445 , Price = 24 , In-app Purchases = 9323
        # Language = 60 , size = 1 , Rate = 9445
        # print(data.duplicated().value_counts()) #162 Row Duplicates

        # print(data.shape) #shape in this moment (16844,11)


        self.data = self.data.drop_duplicates()

        self.data.dropna(how='all', axis='columns', inplace=True)
        self.data.dropna(how='all', axis='rows', inplace=True)

        self.data.drop(columns = ['URL', 'ID', 'Name', 'Icon URL', 'Description', 'Subtitle', 'Primary Genre'], inplace=True)
        self.data.drop(columns = ['In-app Purchases'] , inplace = True)
        self.data.drop(columns = ['Developer'],inplace = True)
        # self.data.dropna(subset= ['Rate'], inplace=True)

        self.data["Price"].fillna(self.data["Price"].sum(skipna = True) / len(self.data["Price"]), inplace=True)


        self.data["Size"].fillna(self.data["Size"].sum(skipna = True) / len(self.data["Size"]) , inplace=True)
        self.data["Size"] = self.data["Size"].map(lambda x: round(x / (1024 * 1024), 2))

        self.avgUsercount = self.data["User Rating Count"].sum(skipna=True) / len(self.data["User Rating Count"])
        self.data['User Rating Count'].fillna(self.avgUsercount, inplace=True)

        self.data["Languages"].fillna('EN', inplace=True)
        self.data["Languages"] = self.data["Languages"].map(lambda x: x.lower())
        self.data["Languages"] = self.data["Languages"].map(lambda x: x.replace(" ", ""))
        self.data["Languages"] = self.data["Languages"].map(lambda x: x.split(","))

        self.data["Genres"].fillna('Games', inplace=True)
        self.data["Genres"] = self.data["Genres"].map(lambda x: x.lower())
        self.data["Genres"] = self.data["Genres"].map(lambda x: x.replace(" ", ""))
        self.data["Genres"] = self.data["Genres"].map(lambda x: x.split(","))

        self.data = self.extract_column_Training(self.data, "Languages")
        self.data.drop(columns=["Languages"], inplace=True)
        self.data = self.extract_column_Training(self.data, "Genres")
        self.data.drop(columns=["Genres"], inplace=True)

        # self.data['In-app Purchases'] = self.data['In-app Purchases'].map(lambda x: x.replace(" ", ""))
        # self.data['In-app Purchases'] = self.data['In-app Purchases'].map(lambda x:sum(float(i) for i in str(x).split(",")))
        # self.avgApppurchases = self.data["In-app Purchases"].sum(skipna=True) / len(self.data["In-app Purchases"])
        # self.data['In-app Purchases'].fillna(self.avgApppurchases , inplace=True)


        self.data["Original Release Date"] = self.data["Original Release Date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y"))
        self.data["Original Release Year"] = self.data["Original Release Date"].map(lambda x: x.year)
        self.data["Original Release Month"] = self.data["Original Release Date"].map(lambda x: x.month)

        self.data["Current Version Release Date"] = self.data["Current Version Release Date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y"))
        self.data["Current Version Release Year"] = self.data["Current Version Release Date"].map(lambda x: x.year)
        self.data["Current Version Release Month"] = self.data["Current Version Release Date"].map(lambda x: x.month)

        self.data.drop(columns=['Original Release Date', 'Current Version Release Date'], inplace=True)
        self.data.drop(columns=["Original Release Year", "Original Release Month"], inplace=True)

        self.avgYear = self.data["Current Version Release Year"].sum(skipna = True) / len(self.data["Current Version Release Year"])
        self.avgMonth = self.data["Current Version Release Month"].sum(skipna = True) / len(self.data["Current Version Release Month"])


        self.data["Age Rating"] = self.data["Age Rating"].map(lambda x: x.replace(" ",""))
        self.data["Age Rating"] = self.data["Age Rating"].map(lambda x: float(x[:-1]))

        # self.data["Rate"] = self.data["Rate"].map(lambda x: x.replace(" ", ""))
        self.data["Rate"] = self.data["Rate"].map(
                lambda x: 2 if x == 'High' else 1 if x == 'Intermediate' else 0 if x == 'Low' else 2)

        self.data.to_csv("PreProcessing Data/TrainingDataSet_BeforeNormalize.csv", encoding='utf-8', index=False)

        def Feature_Normalizer(X, cols):
            for c in cols:
                norm = preprocessing.MinMaxScaler(feature_range=(0, 1))
                norm.fit(X[c].values.reshape(-1, 1))
                X[c] = norm.transform(X[c].values.reshape(-1, 1))
            return X

        self.data = Feature_Normalizer(self.data, ['User Rating Count', 'Price','Age Rating',
                                            'Size',
                                            'Current Version Release Year',
                                            'Current Version Release Month'])



        Y = self.data['Rate']  # Label
        X = self.data.drop(columns=["Rate"], inplace=False)


        self.data.to_csv("PreProcessing Data/TrainingDataSet.csv", encoding='utf-8', index=False)
        print("Pre Processing in Training Data Done Sucessfuly & Saved")
        return X,Y

    def PreProcessing_Testing(self):
        self.set_NULL()

        self.dataTest.dropna(how='all', axis='columns', inplace=True)
        self.dataTest.dropna(how='all', axis='rows', inplace=True)

        self.dataTest.drop(columns=['URL', 'ID', 'Name', 'Icon URL', 'Description', 'Subtitle', 'Primary Genre'],
                           inplace=True)
        self.dataTest.drop(columns = ['In-app Purchases'])
        self.dataTest.drop(columns=['Developer'], inplace=True)

        self.dataTest["Price"].fillna(self.avgPrice, inplace=True)

        self.dataTest["Size"].fillna(self.avgSize, inplace=True)
        self.dataTest["Size"] = self.dataTest["Size"].map(lambda x: round(x / (1024 * 1024), 2))

        self.dataTest['User Rating Count'].fillna(self.avgUsercount, inplace=True)

        self.dataTest["Languages"].fillna('EN', inplace=True)
        self.dataTest["Languages"] = self.dataTest["Languages"].map(lambda x: x.lower())
        self.dataTest["Languages"] = self.dataTest["Languages"].map(lambda x: x.replace(" ", ""))
        self.dataTest["Languages"] = self.dataTest["Languages"].map(lambda x: x.split(","))

        self.dataTest["Genres"].fillna('Games', inplace=True)
        self.dataTest["Genres"] = self.dataTest["Genres"].map(lambda x: x.lower())
        self.dataTest["Genres"] = self.dataTest["Genres"].map(lambda x: x.replace(" ", ""))
        self.dataTest["Genres"] = self.dataTest["Genres"].map(lambda x: x.split(","))

        self.dataTest = self.extract_column_Testing(self.dataTest, "Languages")
        self.dataTest.drop(columns=["Languages"], inplace=True)
        self.dataTest = self.extract_column_Testing(self.dataTest, "Genres")
        self.dataTest.drop(columns=["Genres"], inplace=True)
        #
        # self.dataTest['In-app Purchases'].fillna(self.avgApppurchases, inplace=True)
        # # self.dataTest['In-app Purchases'] = self.dataTest['In-app Purchases'].map(lambda x: x.replace(" ", ""))
        # self.dataTest['In-app Purchases'] = self.dataTest['In-app Purchases'].map(
        #     lambda x: sum(float(i) for i in str(x).split(",")))
        # Adjust Date/Time Format
        self.dataTest["Original Release Date"] = self.dataTest["Original Release Date"].map(
            lambda x: datetime.strptime(x, "%d/%m/%Y"))
        self.dataTest["Original Release Year"] = self.dataTest["Original Release Date"].map(lambda x: x.year)
        self.dataTest["Original Release Month"] = self.dataTest["Original Release Date"].map(lambda x: x.month)

        self.dataTest["Current Version Release Date"] = self.dataTest["Current Version Release Date"].map(
            lambda x: datetime.strptime(x, "%d/%m/%Y"))
        self.dataTest["Current Version Release Year"] = self.dataTest["Current Version Release Date"].map(
            lambda x: x.year)
        self.dataTest["Current Version Release Month"] = self.dataTest["Current Version Release Date"].map(
            lambda x: x.month)

        self.dataTest.drop(columns=['Original Release Date', 'Current Version Release Date'], inplace=True)

        self.dataTest.drop(columns=["Original Release Year", "Original Release Month"], inplace=True)

        self.dataTest["Current Version Release Year"].fillna(self.avgYear, inplace=True)
        self.dataTest["Current Version Release Month"].fillna(self.avgMonth, inplace=True)

        # Delete the sign beside the age & convert from string to float

        self.dataTest["Age Rating"] = self.dataTest["Age Rating"].map(lambda x: x.replace(" ", ""))
        self.dataTest["Age Rating"].fillna('9+', inplace=True)
        self.dataTest["Age Rating"] = self.dataTest["Age Rating"].map(lambda x: float(x[:-1]))


        self.dataTest["Rate"] = self.dataTest["Rate"].map(lambda x: x.replace(" ", ""))
        self.dataTest["Rate"] = self.dataTest["Rate"].map(
            lambda x: 2 if x == 'High' else 1 if x == 'Intermediate' else 0 if x == 'Low' else x)

        self.dataTest = self.check_column(self.dataTest)

        data = pd.read_csv("PreProcessing Data/TrainingDataSet_BeforeNormalize.csv")
        data = data.drop(columns = ['Rate'],inplace = False)
        def Feature_Normalizer(X,data, cols):
            for c in cols:
                norm = preprocessing.MinMaxScaler(feature_range=(0, 1))
                norm.fit(data[c].values.reshape(-1, 1))
                X[c] = norm.transform(X[c].values.reshape(-1, 1))
            return X

        self.dataTest = Feature_Normalizer(self.dataTest,data, ['User Rating Count', 'Price', 'Age Rating',
                                                           'Size',
                                                           'Current Version Release Year',
                                                           'Current Version Release Month'])

        Y = self.dataTest['Rate']  # Label
        X = self.dataTest.drop(columns=["Rate"], inplace=False)

        self.dataTest.to_csv("PreProcessing Data/TestingDataSet.csv", encoding='utf-8', index=False)
        print("Pre Processing in Testing Data Done Sucessfuly & Saved")
        return X, Y
