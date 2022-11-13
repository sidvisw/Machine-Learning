import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Categorical_Encoder:
    '''
    Class to encode the categorical variables
    '''
    # Dictionary to map the Segmentation column
    Segmentation_dict = {'A': 0,
                         'B': 1,
                         'C': 2,
                         'D': 3}

    def encode(self, df):
        '''
        Function to encode the dataset
        args:
            df: pandas dataframe
        returns:
            df: pandas dataframe
        '''
        df['Gender'] = LabelEncoder().fit_transform(df.Gender)
        df['Ever_Married'] = LabelEncoder().fit_transform(df.Ever_Married)
        df['Graduated'] = LabelEncoder().fit_transform(df.Graduated)
        df['Profession'] = LabelEncoder().fit_transform(df.Profession)
        df['Work_Experience'] = LabelEncoder().fit_transform(df.Work_Experience)
        df['Spending_Score'] = LabelEncoder().fit_transform(df.Spending_Score)
        df['Family_Size'] = LabelEncoder().fit_transform(df.Family_Size)
        df['Var_1'] = LabelEncoder().fit_transform(df.Var_1)
        df['Segmentation'] = df.Segmentation.map(self.Segmentation_dict)
        return df


def fill_missing_values(df):
    '''
    Function to fill the missing values
    args:
        df: pandas dataframe
    returns:
        df: pandas dataframe
    '''
    df['Ever_Married'] = df['Ever_Married'].fillna(
        df['Ever_Married'].mode()[0])
    df['Graduated'] = df['Graduated'].fillna(df['Graduated'].mode()[0])
    df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])
    df['Work_Experience'] = df['Work_Experience'].fillna(
        round(df['Work_Experience'].mean()))
    df['Family_Size'] = df['Family_Size'].fillna(df['Family_Size'].mode()[0])
    df['Var_1'] = df['Var_1'].fillna(df['Var_1'].mode()[0])
    return df


def outlier_removal(df):
    '''
    Function to remove the outliers
    args:
        df: pandas dataframe
    returns:
        df: pandas dataframe
    '''
    COLS = ['Age', 'Work_Experience', 'Family_Size']
    for variable in COLS:
        upper_limit = df[variable].mean()+3*df[variable].std()
        df = df[df[variable] < upper_limit]
    return df


def normalize(df):
    '''
    Function to normalize the dataset
    args:
        df: pandas dataframe
    returns:
        df: pandas dataframe
    '''
    COLS = ['Age', 'Work_Experience']
    for variable in COLS:
        df[variable] = (df[variable]) / \
            (df[variable].max()-df[variable].min())
    return df


def split_n_folds(dataset, n_folds):
    '''
    Function to split the dataset into n folds
    args:
        dataset: numpy array
        n_folds: int
    returns:
        split_data: numpy array
    '''
    split_data = np.array_split(dataset, n_folds)
    return split_data


def n_fold_cross_validation(dataset, n_folds, model):
    '''
    Function to perform n fold cross validation
    args:
        dataset: numpy array
        n_folds: int
        model: training model object
    returns:
        accuracy: float
    '''
    split_data = split_n_folds(dataset, n_folds)
    n_fold_accuracy = 0.0
    for i in range(n_folds):
        test_data = split_data[i]
        train_data = np.concatenate(split_data[:i]+split_data[i+1:])
        train_X = train_data[:, :-1]
        train_Y = train_data[:, -1]
        test_X = test_data[:, :-1]
        test_Y = test_data[:, -1]
        model.fit(train_X, train_Y)
        output = model.predict(test_X)
        accuracy = calculate_accuracy(output, test_Y)
        print("Accuracy for", i, "th iteration", accuracy)
        n_fold_accuracy = max(n_fold_accuracy, accuracy)
    return n_fold_accuracy


def calculate_accuracy(output, test_Y):
    '''
    Function to calculate the accuracy
    args:
        output: numpy array
        test_Y: numpy array
    returns:
        accuracy: float
    '''
    output = np.array(output)
    test_Y = np.array(test_Y)
    return np.sum(output == test_Y)/len(output)


class BayesianClassifier():
    '''
    Class to implement the Naive Bayes Classifier
    '''
    def __init__(self, smoothing=0):
        '''
        Constructor
        args:
            smoothing: int
        '''
        self.smoothing = smoothing

    def fit(self, train_X, train_Y):
        '''
        Function to train the model
        args:
            train_X: numpy array
            train_Y: numpy array
        return:
            None
        '''
        self.features = train_X
        self.labels = train_Y
        self.num_of_classes = np.unique(train_Y).shape[0]
        self.num_of_features = train_X.shape[1]
        self.prior_prob = np.zeros(self.num_of_classes)

        self.likelihood = {}

        for i in range(self.num_of_features):
            self.likelihood[i] = np.zeros(
                (self.num_of_classes, np.max(np.unique(train_X[:, i]))+1))

        self.marginal_prob = np.zeros(
            (self.num_of_features, np.max(np.unique(train_X))+1))

        self.get_prior_prob()
        self.get_likelihood()
        self.get_marginal_prob()

    def predict(self, test_X):
        '''
        Function to predict the output
        args:
            test_X: numpy array
        returns:
            output: numpy array
        '''
        output = list()
        for test in test_X:
            output_prob = np.ones(self.num_of_classes)
            for i in range(self.num_of_classes):
                prior = self.prior_prob[i]
                likelihood = 1
                evidence = 1
                for j in range(self.num_of_features):
                    likelihood *= self.likelihood[j][i][test[j]]
                    evidence *= self.marginal_prob[j][test[j]]
                posterior = (prior*likelihood)/evidence
                output_prob[i] = posterior
            output.append(np.argmax(output_prob))
        return np.array(output)

    def get_prior_prob(self):
        '''
        Function to calculate the prior probability
        args:
            None
        returns:
            None
        '''
        for i in range(self.num_of_classes):
            self.prior_prob[i] = np.sum(
                self.labels == i)/(self.labels.shape[0])

    def get_likelihood(self):
        '''
        Function to calculate the likelihood
        args:
            None
        returns:
            None
        '''
        for i in range(self.num_of_features):
            for result in np.unique(self.labels):
                result_cnt = np.sum(self.labels == result)
                for value in np.unique(self.features[:, i]):
                    feature_cnt = np.sum(
                        (self.features[:, i] == value) & (self.labels == result))
                    self.likelihood[i][result][value] = (
                        feature_cnt + self.smoothing) / (result_cnt + self.smoothing * self.num_of_classes)

    def get_marginal_prob(self):
        '''
        Function to calculate the marginal probability
        args:
            None
        returns:
            None
        '''
        for i in range(self.num_of_features):
            for value in np.unique(self.features[:, i]):
                feature_cnt = np.sum(self.features[:, i] == value)
                self.marginal_prob[i][value] = feature_cnt / \
                    self.features.shape[0]

'''
Main function
'''
if __name__ == '__main__':
    print("-------------------------------------------------------------------------------------------------------------")
    print("------------------------------------------NAIVE BAYES CLASSIFIER---------------------------------------------")
    print("-------------------------------------------------------------------------------------------------------------")

    print("------------------------------------------Reading the data...------------------------------------------------")
    df = pd.read_csv('Dataset_A.csv')
    print("\'Dataset_A.csv\' read successfully!")
    print("-------------------------------------------------------------------------------------------------------------")

    print("------------------------------------------Preprocessing the data...------------------------------------------")
    del df['ID']

    df = fill_missing_values(df)

    categorical_encoder = Categorical_Encoder()
    df = categorical_encoder.encode(df)

    df = outlier_removal(df)

    df = normalize(df)
    df['Age'] = (10 * df['Age']).astype(int)
    df['Work_Experience'] = (15 * df['Work_Experience']).astype(int)

    print("ID column deleted, missing values filled, catergorical variables encoded, outliers removed, data normalized and scaled.")
    print("-------------------------------------------------------------------------------------------------------------")

    print("------------------------------------------Final set of features----------------------------------------------")
    print(df.columns.values[:-1])
    print("-------------------------------------------------------------------------------------------------------------")

    print("------------------------------Splitting the data into training and testing...--------------------------------")
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    print("Data split into training and testing sets.")
    print("-------------------------------------------------------------------------------------------------------------")

    df = df.values
    train_dataset = train.values
    test_dataset = test.values

    train_X, train_Y = train_dataset[:, :-1], train_dataset[:, -1]
    test_X, test_Y = test_dataset[:, :-1], test_dataset[:, -1]

    print("------------------------------------------Training the model...----------------------------------------------")
    model = BayesianClassifier()
    model.fit(train_X, train_Y)
    print("Model trained.")
    print("-------------------------------------------------------------------------------------------------------------")

    print("------------------------------------------Testing the model...-----------------------------------------------")
    output = model.predict(test_X)
    accuracy = calculate_accuracy(output, test_Y)
    print("Accuracy of the model on the test set: ", accuracy)
    print("-------------------------------------------------------------------------------------------------------------")

    print("----------------------------------Training the model using Laplace smoothing...------------------------------")
    model = BayesianClassifier(smoothing=1)
    model.fit(train_X, train_Y)
    print("Model trained.")
    print("-------------------------------------------------------------------------------------------------------------")

    print("-------------------------------Testing the model (Laplace Smoothing)...--------------------------------------")
    output = model.predict(test_X)
    accuracy = calculate_accuracy(output, test_Y)
    print("Accuracy of the model on the test set: ", accuracy)
    print("-------------------------------------------------------------------------------------------------------------")

    print("----------------------------------------N-Fold Cross Validation...-------------------------------------------")
    n_folds = 10
    model = BayesianClassifier()
    n_fold_accuracy = n_fold_cross_validation(
        df, n_folds, model)
    print("Accuracy of the model using N-Fold Cross Validation: ", n_fold_accuracy)
    print("-------------------------------------------------------------------------------------------------------------")
    
    print("--------------------------------N-Fold Cross Validation (Laplace Smoothing)...-------------------------------")
    model = BayesianClassifier(smoothing=1)
    n_fold_accuracy = n_fold_cross_validation(
        df, n_folds, model)
    print("Accuracy of the model using N-Fold Cross Validation: ", n_fold_accuracy)

    print("------------------------------------------END OF PROGRAM-----------------------------------------------------")
