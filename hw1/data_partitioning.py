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
from queue import *
from tqdm import tqdm

data_sets = {}
fields = "test_data", "training_data", "training_labels"

def process_queue():
    global data
    global queue
    global valid_mnist_data
    global valid_mnist_labels
    if queue.empty():
        sys.exit(1)
    try:
        key_arr = queue.get()
        make_validation_mnist(key_arr)
        queue.task_done()
    except:
        pass
    
        

for data_name in ["mnist", "spam", "cifar10"]:
    data_sets[data_name] = np.load(f"./data/{data_name}-data.npz")
    print("\nloaded %s data!" % data_name)
queue = Queue() 
data = []
valid_mnist_data = []
valid_mnist_labels = [] 
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
def create_validation_mnist_set(size_set=10000):
    validation_size = size_set
    global data_sets
    global data
    global valid_mnist_data
    global valid_mnist_labels
    data = data_sets.get("mnist")
    key_arr = []
    for i in tqdm(range(len(data.get(fields[2])))):
        # print("shuffle: ", i)
        key_arr.append(i)
    #shuffle the data
    random.shuffle(key_arr)

    valid_mnist_labels = []
    valid_mnist_data = []

    num_threads = 25
    data_size = validation_size
    mod_size = data_size // num_threads
    threads = []
    length = num_threads + 1 if num_threads > 1 else num_threads
    for i in range(length):
        start = i * mod_size
        end = (i + 1) * mod_size
        if end > data_size:
            end = data_size
        queue.put(key_arr[start:end])
    for i in range(length):
        t = Thread(target=process_queue, daemon=True)
        t.start()

    queue.join()
    return valid_mnist_data, valid_mnist_labels

    

    

#part b
valid_spam_labels = []
valid_spam_data = []
def create_validation_spam_set(size_set = 0):
    global data
    global valid_spam_data
    global valid_spam_labels
    data = data_sets.get("spam")
    key_arr = []
    validation_size = len(data.get("training_labels")) // 2 if size_set == 0 else size_set
    labels = data.get("training_labels")
    for i in tqdm(range(len(data.get("training_labels")))):
        key_arr.append(i)
        # print("shuffle: ", i)
    random.shuffle(key_arr)

    valid_spam_labels = []
    valid_spam_data = []

    num_threads = 1
    data_size = int(validation_size)
    mod_size = data_size // num_threads
    length = num_threads + 1 if num_threads > 1 else num_threads
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
    return valid_spam_data, valid_spam_labels
    
#part c
valid_cfar_labels = []
valid_cfar_data = []
def create_validation_cfar_set(size_set = 5000):
    global data
    global valid_cfar_data
    global valid_cfar_labels
    data = data_sets.get("cifar10")
    key_arr = []
    validation_size = size_set

    for i in tqdm(range(len(data.get("training_labels")))):
        key_arr.append(i)
        # print("shuffle: ", i)
    random.shuffle(key_arr)

    valid_cfar_labels = []
    valid_cfar_data = []

    num_threads = 1
    data_size = validation_size
    mod_size = data_size // num_threads
    threads = []
    length = num_threads + 1 if num_threads > 1 else num_threads
    for i in range(num_threads):
        start = i * mod_size
        end = (i + 1) * mod_size
        if end > data_size:
            end = data_size
        t = Thread(target=make_validation_cfar, args=(key_arr[start : end],),  daemon=True)
        t.start()
        threads.append(t)
    [t.join() for t in threads]
    return valid_cfar_data, valid_cfar_labels


if __name__ == '__main__':
    valid_mnist_data, valid_mnist_labels = create_validation_mnist_set()
    print("mnist valid set ", len(valid_mnist_labels), len(valid_mnist_data))
    valid_spam_data, valid_spam_labels = create_validation_spam_set()
    print("spam valid set ", len(valid_spam_labels), len(valid_spam_data))
    valid_cfar_data, valid_cfar_labels = create_validation_cfar_set()
    print("cfar valid set ", len(valid_cfar_labels), len(valid_cfar_data))
    
    
    

    







