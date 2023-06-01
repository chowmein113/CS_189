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
for data_name in ["cifar10"]:
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
    
#cfar part c
c_val = [2**(-3), 5000, 40]
score = []

x_train_set, y_train_set = data_partitioning.create_validation_cfar_set(size_set=50000)
x_valid = x_train_set[0:10000]
y_valid = y_train_set[0:10000]
x_train_set = x_train_set[10000:]
y_train_set = y_train_set[10000:]
# print("x_valid: ", x_valid)
# x_valid = list_matrix_to_list_vector(x_valid)
test_data = data_lib["cifar10"]["test_data"]
y_test_label = []
prev_score = 0
for i in tqdm(c_val, "training: "):
    
    
    x_train = x_train_set
    # print(x_train)
    # x_train, y_train = data_partitioning.create_validation_cfar_set(size_set=i)
    
    # x_train = list_matrix_to_list_vector(x_train)
    
    
    y_train = y_train_set
    training_model = svm.SVC(kernel= 'poly', random_state=1, C=i)
    training_model.fit(x_train, y_train)
    
    y_pred = training_model.predict(x_valid)
    curr_score = accuracy_score(y_valid, y_pred)
    score.append(curr_score)
    if curr_score > prev_score:
        y_test_label = training_model.predict(test_data)
    
plt.plot(c_val, score)
plt.show()
import pandas as pd
file_name = "kaggle_cifar.csv"
data_frame = pd.DataFrame([list(range(1, len(test_data) + 1)), y_test_label])
data_frame = data_frame.transpose()
data_frame.to_csv("results/"+file_name, index=False, header=None)






