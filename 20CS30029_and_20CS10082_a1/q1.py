"""
    This file contains main function for Assignment-1-Q1
    Authors: Sidharth Vishwakarma (20CS10082)
             Kulkarni Pranav Suryakant (20CS30029)
"""
from cgi import test
from cmath import nan
from fileinput import close
from unicodedata import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from re import L
import random
import time
from turtle import pen

DATA_FILE = 'Dataset_A.csv'
OUT_FILE = 'output.txt'
TREE1_FILE = 'tree_before_pruning.txt'
TREE2_FILE = 'tree_after_pruning.txt'

def main():
    """
        This function is the main function for Assignment-1-Q1
    """
    f = open(OUT_FILE, 'w')
    tree_file_1 = open(TREE1_FILE, 'w')
    tree_file_2 = open(TREE2_FILE, 'w')

    data = read_data(DATA_FILE)
    print("---------- Data is loaded ----------", file=f)
    data = fill_missing_values(data)
    print("---------- Testing on 10 random splits ----------", file=f)
    accuracy1, best_tree, max_accuracy, validation_data, test_data = test_10_random_splits(data,f=f)
    print(f'Average Accuracy: {accuracy1}', file = f)
    print(f'Maximum Accuracy: {max_accuracy}', file = f)
    print("---------- Tree Before Pruning ----------", file=tree_file_1)
    best_tree.print_tree(best_tree.root,0,tree_file_1)
    best_tree.root.prune(best_tree, validation_data, max_accuracy)
    print("---------- Tree After Pruning ----------", file=tree_file_2)
    best_tree.print_tree(best_tree.root,0,tree_file_2)
    print(f'Accuracy after pruning: {get_accuracy(test_data, best_tree)}', file = f)

    f.close()
    tree_file_1.close()
    tree_file_2.close()

    # depth vs accuracy plot
    depth = np.arange(1, 11)
    accuracy = []
    for i in range(10):
        _, _, max_accuracy, _, _ = test_10_random_splits(data, depth=i)
        accuracy.append(max_accuracy)
    plt.plot(depth, accuracy)
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Depth')
    for i,j in zip(depth,accuracy):
        plt.annotate(str(round(j,2)),xy=(i,j))
    
    plt.savefig('accuracy_vs_depth.png')

        

if __name__ == "__main__":
    main()

def read_data(filename):
    """
        This function reads the data from the given file and returns a dataframe
    """
    return pd.read_csv(filename)

def fill_missing_values(data):
    """
        This function fills the missing values in the data with the mode of the attribute

        Parameters:
            data: The dataframe containing the data
        
        Returns:
            data: The dataframe with missing values filled with mode of the attribute and unique values adjusted to 10

    """
    for attr in data.columns[1:-1]:
        data[attr].fillna(data[attr].mode()[0], inplace=True)
    
    data['Age'] = pd.cut(data['Age'], bins = 10, labels = np.arange(10), right = False)
    data['Work_Experience'] = pd.cut(data['Work_Experience'], bins = 10, labels = np.arange(10), right = False)

    return data
    

def train_test_split(data, test_size=0.2):
    """
        This function splits the data into train and test data
    """
    train_data = data.sample(frac=1-test_size, random_state=np.random.randint(0, 100))
    test_data = data.drop(train_data.index)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    return train_data, test_data

def calc_entropy(test_cases):
    """
        This function calculates the entropy of the given data

        Parameters:
            test_cases: The test cases for which entropy is to be calculated

        Returns:
            entropy: The entropy of the given data
    """
    entropy = 0
    total = len(test_cases)
    for value in test_cases.unique():
        prob = len(test_cases[test_cases == value]) / total
        entropy += -prob * np.log2(prob)
    return entropy

def calc_info_gain(data, feature):
    """
        This function calculates the information gain of the given feature

        Parameters:
            data: The dataframe containing the data
            feature: The feature for which information gain is to be calculated
            test_cases: Indices of the test cases for which information gain is to be calculated

        Returns:
            info_gain: The information gain of the given feature
    """

    test_cases = data['Segmentation']
    info_gain = calc_entropy(test_cases)

    total = len(test_cases)
    distinct = data[feature].unique()

    for value in distinct:
        label = data[feature] == value
        test_cases = data['Segmentation'][label]
        info_gain -= (len(test_cases) / total) * calc_entropy(data['Segmentation'][label])
    return info_gain

def best_attribute(data):
    """
        This function returns the best attribute for the given data

        Parameters:
            data: The dataframe containing the data

        Returns:
            best_attr: The best attribute for the given data
    """
    info_gain = {}
    for attr in data.columns[1:-1]:
        info_gain[attr] = calc_info_gain(data, attr)
    return max(info_gain, key=info_gain.get), max(info_gain.values())

def get_accuracy(test_data, tree):
    """
        This function calculates the accuracy of the decision tree

        Parameters:
            test_data: The dataframe containing the test data
            tree: The decision tree
            root: The root of the decision tree

        Returns:
            accuracy: The accuracy of the decision tree
    """
    correct = 0
    for i in range(len(test_data)):
        if(tree.predict_segement(test_data.loc[i], tree.root) == test_data.iloc[i]['Segmentation']):
            correct += 1
    return correct / len(test_data)

def test_10_random_splits(data, depth = 5, min_samples = 50, f = None):
    """
        This function tests the decision tree with 10 random train and test splits

        Parameters:
            data: The dataframe containing the data

        Returns:
            accuracy: The average accuracy of the decision tree
            best_tree: The decision tree with the best accuracy
    """
    accuracy = 0
    max_accuracy = 0
    best_tree = None
    best_validation_data = None
    best_test_data = None

    for i in range(10):
        train_data, test_data = train_test_split(data)
        train_data, validation_data = train_test_split(train_data)
        tree = DecisionTree(depth, min_samples)
        tree.build_tree(train_data)
        val = get_accuracy(test_data, tree)
        if (f):
            print("Accuracy of split_{}:".format(i), val, file = f)
        accuracy += val
        if(val > max_accuracy):
            max_accuracy = val
            best_tree = tree
            best_validation_data = validation_data
            best_test_data = test_data

    return accuracy / 10, best_tree, max_accuracy, best_validation_data, best_test_data

class Node:
    """
        This class represents a node in the decision tree
    """
    def __init__(self, classifcation=None, attribute=None, attribute_value=None):
        """
            This function initializes the node
            
            attribute: The attribute of the node on which the data is split
            classification: The classification of the node if it is a leaf node,
                            else None
            children: The children of the node
        """
        self.attribute = attribute
        self.classifcation = classifcation
        self.attribute_value = attribute_value
        self.childern = []
    
    def is_leaf(self):
        """
            This function checks if the node is a leaf node or not

            Returns:
                True if the node is a leaf node, else False
        """
        return self.childern == []
    
    def subtree_node_count(self):
        """
            This function returns the number of nodes in the subtree rooted at
            the current node

            Returns:
                count: The number of nodes in the subtree
        """
        count = 1
        for child in self.childern:
            count += child.subtree_node_count()
        return count
    
    def prune(self, tree, validation_data, curr_accuracy):
        """
            This function prunes the subtree rooted at the current node
            1st children of current nodes are pruned and then the current node
            is pruned if accuracy after pruning is greater than accuracy before

            tree: The decision tree
            validation_data: The validation data for pruning
            curr_accuracy: The accuracy of the tree before pruning

            Returns:
                accuracy: The accuracy of the tree after pruning the subtree
                if accuracy reamins same, then the current accuracy is returned
        """
        if(self.is_leaf()):
            return curr_accuracy
        for child in self.childern:
            if child is not None:
                child.prune(tree, validation_data, curr_accuracy)
        
        children_temp = self.childern
        self.childern = []
        accuracy = get_accuracy(validation_data, tree)

        if(accuracy < curr_accuracy or tree.root.subtree_node_count() <= 10):
            self.childern = children_temp
            return curr_accuracy
        else:
            return accuracy
            
    
class DecisionTree:
    """
        This class represents the decision tree
    """
    def __init__(self, max_depth=5, min_samples=5):
        """
            This function initializes the decision tree

            max_depth: The maximum depth of the decision tree
            min_samples: The minimum number of samples in a node
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.depth = 0
    
    def build_tree(self, data, depth=0):
        """
            This function builds the decision tree

            data: The data on which the decision tree is built
            depth: The current depth of the tree

            Returns:
                node: The root node of the decision tree
        """
    
        # finding the attribute with the highest information gain
        best_attr, gain_value = best_attribute(data)
        
        if ((depth >= self.max_depth) or (len(data) <= self.min_samples) or (len(data['Segmentation'].unique()) == 1) or (gain_value == 0)):
            node = Node(classifcation=data['Segmentation'].mode()[0])
            return node
        
        new_node = Node(attribute=best_attr, classifcation=data['Segmentation'].mode()[0])
        # splitting the data based on the attribute
        for value in data[best_attr].unique():
            label = data[best_attr] == value
            new_node.childern.append(self.build_tree(data[label], depth+1))
            new_node.childern[-1].attribute_value = value
        
        self.depth = max(self.depth, depth)
        self.root = new_node
        return new_node
    
    def predict_segement(self, training_example, root):
        """
            This function predicts the class for the given data

            data: The data for which the class is to be predicted
            root: The root node of the decision tree

            Returns:
                Segmentation: The predicted class
        """
        if root is None:
            return None
        if root.is_leaf():
            return root.classifcation
        else:
            for child in root.childern:
                if training_example[root.attribute] == child.attribute_value:
                    return self.predict_segement(training_example, child)

    def print_tree(self, root, indent=0, f = None):
        """
            This function prints the decision tree

            root: The root node of the decision tree
            indent: The number of spaces to indent
        """
        if root is None:
            return
        if root.is_leaf():
            if f is None:
                print(' '*indent, root.classifcation, root.attribute_value)
            else:
                print(' '*indent, root.classifcation, root.attribute_value, file=f)
        else:
            if f is None:
                print(' '*indent, root.attribute, root.attribute_value)
            else:
                print(' '*indent, root.attribute, root.attribute_value, file=f)
            for child in root.childern:
                self.print_tree(child, indent+2, f)
                    