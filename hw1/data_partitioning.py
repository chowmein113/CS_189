# This file is in scripts/load.py
import sys
import random
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
from threading import Thread

data_sets = {}
fields = "test_data", "training_data", "training_labels"

for data_name in ["mnist", "spam", "cifar10"]:
    data_sets[data_name] = np.load(f"./data/{data_name}-data.npz")
    print("\nloaded %s data!" % data_name)
    
def make_validation_mnist(key_arr):
    global data
    global valid_mnist_labels
    global valid_mnist_data
    labels = data.get("training_labels")
    train_data = data.get("training_data")
    for key in key_arr:
        valid_mnist_labels.append(labels[key])
        valid_mnist_data.append(train_data[key])
    print("finished process")
def make_validation_spam(key_arr):
    global data
    global valid_spam_labels
    global valid_spam_data
    labels = data.get("training_labels")
    train_data = data.get("training_data")
    for key in key_arr:
        valid_spam_labels.append(labels[key])
        valid_spam_data.append(train_data[key])
    print("finished process")
def make_validation_cfar(key_arr):
    global data
    global valid_cfar_labels
    global valid_cfar_data
    labels = data.get("training_labels")
    train_data = data.get("training_data")
    for key in key_arr:
        valid_cfar_labels.append(labels[key])
        valid_cfar_data.append(train_data[key])
    print("finished process")
#part a
validation_size = 10000
data = data_sets.get("mnist")
key_arr = []
for i in range(len(data.get(fields[2]))):
    print("shuffle: ", i)
    key_arr.append(i)
#shuffle the data
random.shuffle(key_arr)

valid_mnist_labels = []
valid_mnist_data = []

num_threads = 1
data_size = validation_size
mod_size = data_size // num_threads
threads = []
for i in range(num_threads):
    start = i * mod_size
    end = (i + 1) * mod_size
    if end > data_size:
        end = data_size
    t = Thread(target=make_validation_mnist, args=(key_arr[start : end],),  daemon=True)
    t.start()
    threads.append(t)
[t.join() for t in threads]

    

    

#part b

data = data_sets.get("spam")
key_arr = []
validation_size = len(data.get("training_labels")) // 2
labels = data.get("training_labels")
for i in range(len(data.get("training_labels"))):
    key_arr.append(i)
    print("shuffle: ", i)
random.shuffle(key_arr)

valid_spam_labels = []
valid_spam_data = []

num_threads = 1
data_size = int(validation_size)
mod_size = data_size // num_threads
threads = []
for i in range(num_threads):
    start = i * mod_size
    end = (i + 1) * mod_size
    if end > data_size:
        end = data_size
    t = Thread(target=make_validation_spam, args=(key_arr[start : end],),  daemon=True)
    t.start()
    threads.append(t)
[t.join() for t in threads]
    
#part c
data = data_sets.get("cifar10")
key_arr = []
validation_size = 5000

for i in range(len(data.get("training_labels"))):
    key_arr.append(i)
    print("shuffle: ", i)
random.shuffle(key_arr)

valid_cfar_labels = []
valid_cfar_data = []

num_threads = 1
data_size = validation_size
mod_size = data_size // num_threads
threads = []
for i in range(num_threads):
    start = i * mod_size
    end = (i + 1) * mod_size
    if end > data_size:
        end = data_size
    t = Thread(target=make_validation_cfar, args=(key_arr[start : end],),  daemon=True)
    t.start()
    threads.append(t)
[t.join() for t in threads]


print("cfar valid set ", len(valid_cfar_labels), len(valid_cfar_data))
print("spam valid set ", len(valid_spam_labels), len(valid_spam_data))
print("mnist valid set ", len(valid_mnist_labels), len(valid_mnist_data))
print("labels: ", labels)
    
    

    







