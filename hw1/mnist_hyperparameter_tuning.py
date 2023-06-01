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
x_valid, y_valid = data_partitioning.create_validation_mnist_set(size_set=20000)
x_valid = list_matrix_to_list_vector(x_valid)
x_train_set = x_valid[0:10000]
y_train_set = y_valid[0:10000]
x_valid = x_valid[10000:]
y_valid = y_valid[10000:]
moving_accuracy = []
x_train = x_train_set
# x_train = list_matrix_to_list_vector(x_train)
y_train = y_train_set


def train_data(d=0.1):
    #MNIST part a
    global moving_accuracy
    global x_valid, y_valid, x_train_set, y_train_set, x_train, y_train

    # mnist_data = data_lib["mnist"]
    
    
    # x_train, y_train = data_partitioning.create_validation_mnist_set(size_set=i)
    
    
    
    
    
    
    training_model = svm.SVC(kernel= 'linear', random_state=1, C=d)
    training_model.fit(x_train[:], y_train[:])
    
    y_pred = training_model.predict(x_valid)
    moving_accuracy.append(accuracy_score(y_valid, y_pred))
    
for i in tqdm(list(range(-20,1)), "running tests... "):
    if np.isnan(i):
        break
    exp = 2**i
    train_data(d=exp)
plt.plot(list(range(-20,1)), moving_accuracy)
plt.show()
    






