import sys
import random
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy import io
from tqdm import tqdm
from queue import Queue
from threading import Thread
import data_partitioning

data_lib = {}
fields = ["test_data", "training_data", "training_labels"]
for data_name in ["mnist"]:
    data = np.load(f"./data/{data_name}-data.npz")
    print("\nloaded %s data!" % data_name)
    data_lib[data_name] = data
def matrix_to_vector_form(matrix):
    """Take a 2 dimensional np.arr and return as vector
    Args: nxn np.arr matrix
    Returns: np.arr vector"""
    vector = []
    for row in matrix:
        vector.extend(row)
    vector = np.array(vector)
    return vector
    
def list_matrix_to_list_vector(list_matrix):
    list_vector = []
    for matrix in list_matrix:
        # print("matrix: ", matrix[0])
        list_vector.append(matrix_to_vector_form(matrix[0]))
    return list_vector
x_valid, y_valid = data_partitioning.create_validation_mnist_set()
x_valid = list_matrix_to_list_vector(x_valid)
x_train_set, y_train_set = data_partitioning.create_validation_mnist_set(size_set=10000)
moving_avg = 0
moving_
def train_data(c=0.1):
    #MNIST part a
    score = []
    global x_valid, y_valid, x_train_set, y_train_set

    mnist_data = data_lib["mnist"]
    
    x_train = x_train_set[0:10000]
    # x_train, y_train = data_partitioning.create_validation_mnist_set(size_set=i)
    
    x_train = list_matrix_to_list_vector(x_train)
    
    
    y_train = y_train_set[0:10000]
    
    training_model = svm.SVC(kernel= 'linear', random_state=1, c=c)
    training_model.fit(x_train, y_train)
    
    y_pred = training_model.predict(x_valid)
    score.append(accuracy_score(y_valid, y_pred))
    






