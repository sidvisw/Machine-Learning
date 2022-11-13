# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys


class KMeans:
    '''
        Description: This class implements the model for the K-Means clustering algorithm
        Input: k - number of clusters
        Member Functions:
            fit - fits the model to the data
            predict - predicts the cluster for a given data point
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, data):
        # randomly initialize centroids
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[random.randint(0, data.shape[0]-1)]
        # iterate until convergence
        while True:
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
            # calculate euclidean distance between data and centroids
            for features in data:
                distances = [np.linalg.norm(
                    features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
            # store previous centroids
            prev_centroids = dict(self.centroids)
            # calculate new centroids
            for classification in self.classes:
                if (len(self.classes[classification])):
                    self.centroids[classification] = np.average(
                        self.classes[classification], axis=0)
            # check for convergence
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > 0.01:
                    optimized = False
            if optimized:
                break

    def predict(self, feature):
        distances = [np.linalg.norm(feature - self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances)) + 1
        return classification


def normalized_mutual_info_score(labels_true, labels_pred):
    '''
        Description: This function calculates the normalized mutual information score
        Input: labels_true - true labels
               labels_pred - predicted labels
        Output: nmi - normalized mutual information score
    '''
    # calculate mutual information
    mutual_info = mutual_information(labels_true, labels_pred)
    # calculate entropy
    entropy_true = entropy(labels_true)
    entropy_pred = entropy(labels_pred)
    # calculate normalized mutual information
    nmi = 2 * mutual_info / (entropy_true + entropy_pred)
    return nmi


def mutual_information(labels_1, labels_2):
    '''
        Description: This function calculates the mutual information
        Input: labels_1 - labels 1
               labels_2 - labels 2
        Output: mutual_info - mutual information
    '''
    mutual_info = 0
    # classes in the predicted labels with their frequency
    classes = np.unique(labels_2, return_counts=True)
    # number of classes in the true labels
    num_classes = len(np.unique(labels_1))
    # calculate mutual information
    for i in range(len(classes[0])):
        # probability of the class in the predicted labels
        p_class = np.zeros(num_classes)
        for j in range(len(labels_2)):
            if labels_2[j] == classes[0][i]:
                p_class[labels_1[j]-1] += 1
        p_class /= classes[1][i]
        # calculate the entropy in the class i
        Entropy = 0
        for j in range(num_classes):
            if p_class[j] != 0:
                Entropy += p_class[j] * np.log2(p_class[j])
        mutual_info += classes[1][i] / len(labels_1) * Entropy
    # return H(Y) - H(Y|C)
    return entropy(labels_1) + mutual_info


def entropy(labels):
    '''
        Description: This function calculates the entropy
        Input: labels - labels
        Output: entropy - entropy
    '''
    entropy = 0
    # calculate probability
    prob = np.unique(labels, return_counts=True)[1] / len(labels)
    # calculate entropy
    for p in prob:
        if p != 0:
            entropy += p * np.log2(p)
    return -entropy


if __name__ == '__main__':
    file = open('q1Output.txt','wt')
    sys.stdout = file
    print('--------------------------------------Reading and loading the dataset------------------------------------------')
    # importing or loading the dataset
    dataset = pd.read_csv('wine.data', header=None)
    print('---------------------------------------------------------------------------------------------------------------')

    print('----------------------------------------Split into features and labels-----------------------------------------')
    # distributing the dataset into two components features and labels
    features = dataset.iloc[:, 1:].values
    labels = dataset.iloc[:, 0].values
    print('---------------------------------------------------------------------------------------------------------------')

    print('-------------------------------------Peforming the standardization of data-------------------------------------')
    # performing preprocessing part
    sc = StandardScaler()

    sc.fit(features)
    features = sc.transform(features)
    print('---------------------------------------------------------------------------------------------------------------')

    print('-----------------------------------------Applying PCA on the dataset-------------------------------------------')
    # Applying PCA function on training and testing set of feature component
    pca = PCA(n_components=0.95)

    pca.fit(features)
    features = pca.transform(features)
    print('---------------------------------------------------------------------------------------------------------------')

    print('-------------------------------The variance explained by the principal components------------------------------')
    explained_variance = pca.explained_variance_ratio_
    for i in range(len(explained_variance)):
        print('Variance explained by principal component', i + 1, ':', explained_variance[i])
    print('---------------------------------------------------------------------------------------------------------------')

    print('-----------------------------Plot of Explained variance vs Number of components--------------------------------')
    # Plot of the PCA for Explained Variance vs Number of components
    plt.plot([i+1 for i in range(len(explained_variance))], np.cumsum(explained_variance * 100))
    plt.xticks([i+1 for i in range(len(explained_variance))])
    plt.axhline(y=95, linestyle='dashed')
    plt.text(4.5, 96, 'Threshold of 95%')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (%)')
    plt.title('Explained variance vs Number of components')
    plt.savefig('elbow_plot.png', dpi=100)
    plt.show()
    print('---------------------------------------------------------------------------------------------------------------')

    # creating object of KMeans class
    kmeans = KMeans(3)

    # fitting the model
    kmeans.fit(features)

    print('------------------------------------Cluster Representatives/Centriods-----------------------------------------')
    # the cluster representatives/centriod of the K-means clustering algorithm
    for centroid in kmeans.centroids:
        print(centroid+1, '-->', kmeans.centroids[centroid])
    print('---------------------------------------------------------------------------------------------------------------')

    # Plot the K vs normalised mutual information (NMI) graph
    print('-----------------------------------Plot of K vs normalised mutual information----------------------------------')
    nmi_scores = []
    for k in range(2, 9):
        km = KMeans(k)
        km.fit(features)
        predicted_labels = []
        for i in range(features.shape[0]):
            predicted_labels.append(km.predict(features[i]))
        nmi_scores.append(normalized_mutual_info_score(labels, predicted_labels))
    plt.figure(figsize=(10, 7))
    plt.plot(nmi_scores, range(2, 9))
    plt.xlabel('NMI')
    plt.ylabel('K')
    plt.title('K vs normalised mutual information', pad=15)
    plt.savefig('K_vs_NMI.png')
    plt.show()
    print('---------------------------------------------------------------------------------------------------------------')

    print('--------------------------------------------Optimal value of K-------------------------------------------------')
    # Optimal value of K
    print('Optimal value of K:', nmi_scores.index(max(nmi_scores)) + 2)
    print('---------------------------------------------------------------------------------------------------------------')
    file.close()