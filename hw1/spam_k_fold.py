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
training_num = [100, 200, 500, 1000, 2000, 4172]
score = []
# x_valid, y_valid = data_partitioning.create_validation_spam_set()
x_train_set, y_train_set = data_partitioning.create_validation_spam_set(size_set=4172)
k = 5
subset_size = 4172 // 5
x_train_subsets = []
y_train_subsets = []
set_length = len(x_train_set)
for i in range(k + 1):
    start = i * subset_size
    end = (i + 1) * subset_size
    end = set_length if end > set_length else end
    x_train_subsets.append(x_train_set[start:end])
    y_train_subsets.append(y_train_set[start:end])
avg_scores = []
values = range(8, 13)
for c in tqdm(values, "going through c values"):
    for i in tqdm(range(len(x_train_subsets)), "testing different subsets..."):


        valid_x_set = x_train_subsets[i]
        valid_y_set = y_train_subsets[i]
        training_model = svm.SVC(kernel= 'rbf', random_state=1, C= 2**c)
        training_set_x = []
        training_set_y = []
        for r in (range(len(x_train_subsets))):
            if r != i:
                training_set_x.extend(x_train_subsets[r])
                training_set_y.extend(y_train_subsets[r])
        training_model.fit(training_set_x, training_set_y)

        y_pred = training_model.predict(valid_x_set)
        score.append(accuracy_score(valid_y_set, y_pred))
    avg_score = (sum(score)) / len(score)
    avg_scores.append(avg_score)
    score = []
        
plt.plot(list(values), avg_scores)
plt.show()  






