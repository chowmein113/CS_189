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
for data_name in ["spam"]:
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
        list_vector.append(matrix_to_vector_form(matrix))
    return list_vector
    
#spam part b
training_num = [4172]
score = []
# x_valid, y_valid = data_partitioning.create_validation_spam_set()
x_train_set, y_train_set = data_partitioning.create_validation_spam_set(size_set=4172)
divider = 4172 // 5
x_valid = x_train_set[:divider]
y_valid = y_train_set[:divider]
x_train_set = x_train_set[divider:]
y_train_set = y_train_set[divider:]
# print("x_valid: ", x_valid)
test_data = data_lib["spam"]["test_data"]
y_test_label = []
# x_valid = list_matrix_to_list_vector(x_valid)
for i in tqdm(training_num):
    
    
    x_train = x_train_set[0:i]
    # x_train, y_train = data_partitioning.create_validation_spam_set(size_set=i)
    
    # x_train = list_matrix_to_list_vector(x_train)
    
    
    y_train = y_train_set[0:i]
    training_model = svm.SVC(kernel= 'rbf', random_state=1, C=2**11)
    training_model.fit(x_train, y_train)
    
    y_test_label = training_model.predict(test_data)
    
import pandas as pd
file_name = "kaggle_spam.csv"
data_frame = pd.DataFrame([list(range(1, len(test_data) + 1)), y_test_label])
data_frame = data_frame.transpose()
data_frame.to_csv("results/"+file_name, index=False, header=None)






