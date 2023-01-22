# This file is in scripts/load.py
import sys
import random
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io

data_sets = {}
fields = "test_data", "training_data", "training_labels"

for data_name in ["mnist", "spam", "cifar10"]:
    data_sets[data_name] = np.load(f"./data/{data_name}-data.npz")
    print("\nloaded %s data!" % data_name)
    
    

#part a
validation_size = 10000
data = data_sets.get("mnist")
key_arr = []
for i in range(len(data.get(fields[2]))):
    print("shuffle: ", i)
    key_arr.append(i)
#shuffle the data
random.shuffle(key_arr)

valid_mnist_set = {}
for i in range(validation_size):
    key = key_arr[i]
    print("key: ", i)
    valid_mnist_set[data.get("training_labels")[key]] = data.get("training_data")[key]
    

#part b
valid_spam_set = {}
data = data_sets.get("spam")
key_arr = []
validation_size = data.get(len("training_labels")) * 0.2
for i in range(len(data.get("training_labels"))):
    key_arr.append(i)
    print("shuffle: ", i)
random.shuffle(key_arr)

for i in range(validation_size):
    key = key_arr[i]
    print("key: ", i)
    k = data.get("training_labels")[key]
    v = data.get("training_data")[key]
    valid_spam_set[k] = v
    
#part c
valid_cfar_set = {}
data = data_sets.get("cifar10")
key_arr = []
validation_size = 5000

for i in range(len(data.get("training_labels"))):
    key_arr.append(i)
    print("shuffle: ", i)
random.shuffle(key_arr)

for i in range(validation_size):
    k = data.get("training_labels")
    v = data.get("training_data")
    valid_cfar_set[k] = v

print("cfar valid set ", len(valid_cfar_set))
print("spam valid set ", len(valid_spam_set))
print("mnist valid set ", len(valid_mnist_set))
    
    

    







