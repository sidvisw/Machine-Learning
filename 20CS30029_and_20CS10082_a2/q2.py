# Machine Learning, Autumn 2022
# Assignment 2
# Q2: Supervised Learning
# Group 14: [Sidharth Vishwakarma(20CS10082), Kulkarni Pranav Suryakant(20CS30029)]

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import sys

INPUT_FILE = 'wine.data'
COLUMN_NAMES = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


# Read the dataset using pandas and convert it to numpy array for further processing
def readData():
    data = pd.read_csv(INPUT_FILE, header=None)
    data = data.to_numpy()
    return data

# normalize the data using standard scalar normalization technique
def standardScalarNormalisation(data):
    # get the mean and standard deviation of each column except the first column as it stores the class labels
    for columns in range(1, len(data[0])):
        data[:, columns] = (data[:, columns] - np.mean(data[:, columns])) / np.std(data[:, columns])
    return data

# split data function to split the data into training and testing data
def splitData(data):
    # split the data into 80% training and 20% testing data
    np.random.shuffle(data)
    trainingData = data[:int(len(data) * 0.8)]
    testingData = data[int(len(data) * 0.8):]
    return trainingData, testingData

# encode the categorical variable 
def encodeData(data):
    # In the A dataset categorical variable is given as 1, 2, 3
    # encode them as 0, 1, 2
    for i in range(len(data)):
        data[i][0] -= 1
    data[0] = data[0].astype(int)
    return data

def accuracyScore(testingLabels, predictedLabels):
    # Calculate the accuracy of the model
    correct = 0
    total = len(testingLabels)
    for i in range(len(testingLabels)):
        if predictedLabels[i] == testingLabels[i]:
            correct += 1
    accuracy = correct / total
    # print(correct, " out of ", len(testingLabels), " are correctly classified")
    return accuracy

# SVM classifier using sklearn with support for following kenrnels
# 1. Linear, 2. Qudratic, 3. Radiakl Basis Function
def SVMclassifier(trainingData, testingData, kernel, degree=2):
    # split the data into features and labels
    trainingFeatures, trainingLabels = trainingData[:, 1:], trainingData[:, 0]
    testingFeatures, testingLabels = testingData[:, 1:], testingData[:, 0]
    print("Training the SVM classifier...")
    # train the model
    if(kernel == 'linear'):
        model = svm.LinearSVC()
    else:
        if(kernel == 'quadratic'):
            model = svm.SVC(kernel='poly', degree=degree)
        else:
            model = svm.SVC(kernel=kernel)
    model.fit(trainingFeatures, trainingLabels)
    # predict the labels
    predictedLabels = model.predict(testingFeatures)
    # calculate the accuracy
    accuracy = accuracyScore(testingLabels, predictedLabels)
    print("Accuracy of the model: {:.2f}%".format(accuracy*100))
    print("")

# MLP classifier with
# 1. stochastic gradient descent solver 
# 2. learning rate of 0.001
# 3. batch size of 32
def MLPclassifier(trainingData, testingData, hiddenLayers, nodes, learningRate = 0.001, features = None):
    # split the data into features and labels
    trainingFeatures, trainingLabels = trainingData[:, 1:], trainingData[:, 0]
    testingFeatures, testingLabels = testingData[:, 1:], testingData[:, 0]
    if features != None:
        # sort indices of features
        features.sort()
        trainingFeatures = trainingFeatures[:, features]
        testingFeatures = testingFeatures[:, features]
    print("Training the MLP classifier...")
    # train the model
    solver = 'sgd'
    batch_size = 32
    layers = ()
    for i in range(hiddenLayers):
        layers += (nodes[i],)
    model = MLPClassifier(hidden_layer_sizes=layers, solver=solver, learning_rate_init=learningRate, batch_size=batch_size, max_iter=100000)
    model.fit(trainingFeatures, trainingLabels)
    # predict the labels
    predictedLabels = model.predict(testingFeatures)
    # calculate the accuracy
    accuracy = accuracyScore(testingLabels, predictedLabels)
    print("Accuracy of the model: {:.2f}%".format(accuracy*100))
    print("")
    return model, accuracy

# MLP classifier with
# 1. stochastic gradient descent solver 
# 2. learning rate of 0.001
# 3. batch size of 32
def MLPclassifierMod(trainingData, testingData, hiddenLayers, nodes, learningRate = 0.001, features = None):
    # split the data into features and labels
    trainingFeatures, trainingLabels = trainingData[:, 1:], trainingData[:, 0]
    testingFeatures, testingLabels = testingData[:, 1:], testingData[:, 0]
    if features != None:
        # sort indices of features
        features.sort()
        trainingFeatures = trainingFeatures[:, features]
        testingFeatures = testingFeatures[:, features]
    # train the model
    solver = 'sgd'
    batch_size = 32
    layers = ()
    for i in range(hiddenLayers):
        layers += (nodes[i],)
    model = MLPClassifier(hidden_layer_sizes=layers, solver=solver, learning_rate_init=learningRate, batch_size=batch_size, max_iter=100000)
    model.fit(trainingFeatures, trainingLabels)
    # predict the labels
    predictedLabels = model.predict(testingFeatures)
    # calculate the accuracy
    accuracy = accuracyScore(testingLabels, predictedLabels)
    return accuracy

def forwardSelectionMethod(trainingData, testingData, bestLayer):
    # get the number of features
    kFeatures = len(trainingData[0, 1:])
    selectedFeatures = []
    bestAccuracy = 0
    for i in range(kFeatures):
        currAccuracy = 0
        currFeature = -1
        for j in range(kFeatures):
            if j not in selectedFeatures:
                features = selectedFeatures + [j]
                accuracy = MLPclassifierMod(trainingData, testingData, len(bestLayer), bestLayer, features=features)
                if accuracy > currAccuracy:
                    currAccuracy = accuracy
                    currFeature = j
        if currAccuracy > bestAccuracy:
            bestAccuracy = currAccuracy
            selectedFeatures.append(currFeature)
            # print colunm names of selected features
            print("Current Accuracy: {:.2f}".format(bestAccuracy), end=" ")
            print("Best Features after %dth iteration: " % (i+1), end="")
            print([COLUMN_NAMES[i] for i in selectedFeatures])
            
        else:
            print("No improvement in accuracy after %dth iteration" % (i+1))
            print("Final Best Features: ", end="")
            print([COLUMN_NAMES[i] for i in selectedFeatures], end=" ")
            print("Final Best Accuracy: {:.2f}".format(bestAccuracy))
            print("")
            break

# Apply ensemble learning (max voting technique) using SVM with quadratic, SVM with radial
# basis function and the best accuracy model from part 3.
def ensembleLearning(trainingData, testingData, bestLayer):
    # split the data into features and labels
    trainingFeatures, trainingLabels = trainingData[:, 1:], trainingData[:, 0]
    testingFeatures, testingLabels = testingData[:, 1:], testingData[:, 0]
    print("Training the ensemble classifier...")
    # train the models
    model1 = svm.SVC(kernel='poly', degree=2)
    model1.fit(trainingFeatures, trainingLabels)
    model2 = svm.SVC(kernel='rbf')
    model2.fit(trainingFeatures, trainingLabels)
    model3 = MLPClassifier(hidden_layer_sizes=bestLayer, solver='sgd', learning_rate_init=0.001, batch_size=32, max_iter=100000)
    model3.fit(trainingFeatures, trainingLabels)
    # predict the labels
    predictedLabels1 = model1.predict(testingFeatures)
    predictedLabels2 = model2.predict(testingFeatures)
    predictedLabels3 = model3.predict(testingFeatures)
    # final prediction using max voting
    predictedLabels = []
    for i in range(len(testingLabels)):
        counts = [0, 0, 0]
        counts[int(predictedLabels1[i])] += 1
        counts[int(predictedLabels2[i])] += 1
        counts[int(predictedLabels3[i])] += 1
        predictedLabels.append(counts.index(max(counts)))
    # calculate the accuracy
    accuracy = accuracyScore(testingLabels, predictedLabels)
    print("Accuracy after applying Ensemble Learning: {:.2f}%".format(accuracy*100))
    print("")


def main():
    file = open('q2Output.txt','wt')
    sys.stdout = file
    ##### PART 1 #####
    print("################ Part 1: Standard Scalar Normalization, Sampling, Encoding ################")
    print("Reading the data from the file "+INPUT_FILE+"...")
    data = readData()
    print("Reading the data completed")
    print("Normalizing the data using standard scalar normalization technique...")
    data = standardScalarNormalisation(data)
    print("Normalization completed")
    print("Splitting the data into training and testing data...")
    trainingData, testingData = splitData(data)
    print("Splitting completed")
    print("Encoding the categorical variable...")
    trainingData = encodeData(trainingData)
    testingData = encodeData(testingData)
    print("Encoding completed")
    print("Data preprocessing completed\n")

    ##### PART 2 #####
    print("################################ Part 2: SVM Classifier ################################")
    # SVM classifier with linear kernel
    print("Training SVM classifier with linear kernel...")
    SVMclassifier(trainingData, testingData, 'linear')
    # SVM classifier with quadratic kernel
    print("Training SVM classifier with quadratic kernel..")
    SVMclassifier(trainingData, testingData, 'quadratic')
    # SVM classifier with radial basis function kernel
    print("Training SVM classifier with radial basis function kernel...")
    SVMclassifier(trainingData, testingData, 'rbf')

    ##### PART 3 #####
    print("################################ Part 3: MLP Classifier ################################")
    # MLP classifier with 1 hidden layer
    print("Training MLP classifier with 1 hidden layer with 16 nodes...")
    model_1, accuracy1 = MLPclassifier(trainingData, testingData, 1, [16])
    # MLP classifier with 2 hidden layers
    print("Training MLP classifier with 2 hidden layers with 256 and 16 nodes respectively...")
    model_2, accuracy2 = MLPclassifier(trainingData, testingData, 2, [256, 16])

    ##### PART 4 #####
    print("################ Part 4: Choosing Best Accuracy model from MLP Classifier ################")
    if(accuracy1 > accuracy2):
        print("MLP classifier with 1 hidden layer with 16 nodes has the best accuracy")
        bestLayer = [16]
    else:
        print("MLP classifier with 2 hidden layers with 256 and 16 nodes respectively has the best accuracy")
        bestLayer = [256, 16]
    
    print("Varying learning rate of the model...")
    learningRates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    accuracies = []
    for learningRate in learningRates:
        print("Training MLP classifier with learning rate: ", learningRate)
        model, accuracy = MLPclassifier(trainingData, testingData, len(bestLayer), bestLayer, learningRate)
        accuracies.append(accuracy)
    
    print("Plotting the graph for learning rate vs accuracy...")

    plt.plot(learningRates, accuracies, 'o-')
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Learning Rate vs Accuracy")
    plt.savefig("LearningRateVsAccuracy.png")
    print("Plotting completed")
    print("")

    ##### PART 5 #####
    print("################################ Part 5: Forward Selection Method ################################")
    print("Forward selection method running to select best features...")
    forwardSelectionMethod(trainingData, testingData, bestLayer)

    ##### PART 6 #####
    print("########################## Part 6: Ensemble Learning (Max Voting Technique) ##########################")
    print("Applying ensemble learning (max voting technique)...")
    ensembleLearning(trainingData, testingData, bestLayer)

    file.close()

if __name__ == '__main__':
    main()