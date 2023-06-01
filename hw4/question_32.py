import numpy as np
import scipy as sp
from question_2 import *
import random
import matplotlib.pyplot as plt
import tqdm
random.seed(10)


def stochastic_gradient_descent(X, y, w, epsilon, l2, rand_arr, update_lists = False, cost_func_vals=[], vals=[]):
    # global cost_func_vals
    # global vals
    iterations = 0
    repeat = True
    w_prev = 2 * w #arbitrary val so not equal on first run
    n = len(rand_arr)
    # zero_vec = np.zeros(len(w))
    while repeat:
        val = rand_arr.pop(0)
        xiT = X[val]
        yi = y[val]
        first = (yi - s_individual(w, xiT))
        # print("first: ", np.shape(first))
        second = l2 *  w - (n * first) * xiT          # print("second: ", np.shape(second))
        print(iterations)
        gradient = second
        w_prev = w
        w = w - epsilon * gradient
        if (cost_func_BGD(y, X, w, l2) > 0 or not np.array_equal(w_prev, w)) and iterations < len(X) - 1:
            iterations += 1
            if iterations % 50 == 0 and update_lists:
                cost_func_vals.append(cost_func_BGD(y, X, w, l2))
                vals.append(iterations)
        else:
            break
    return w
def cost_func_BGD(y, X, w, l2):
    vec1 = np.ones(len(y))
    sg = s(w, X)
    cost_func = -1 * y @ np.log(sg) + np.dot((y - vec1), np.log(vec1 - sg)) + l2 * np.linalg.norm(w) ** 2
    return cost_func
def preprocessing(X, y):
    means = []
    std_devs = []
    for i in range(len(X[0])):
        feature_points = np.array([row[i] for row in X])
        mean_i = np.mean(feature_points)
        std_dev_i = np.std(feature_points, ddof=1)
        means.append(mean_i)
        std_devs.append(std_dev_i)
    for i in range(len(X[0])):
        for row in X:
            row[i] -= means[i]
            row[i] /= std_devs[i]
    newX = np.c_[X, np.ones(len(X))]
    newy = np.array([i[0] for i in y])
    return newX, newy
def create_validation_set(X, y, validation_size):
    keys = list(range(len(X)))
    random.shuffle(keys)
    new_X = []
    new_y = []
    for key in keys:
        new_X.append(X[key])
        new_y.append(y[key])
    valid_X = np.array(new_X[:validation_size])
    valid_y = np.array(new_y[:validation_size])
    new_X = np.array(new_X[validation_size:])
    new_y = np.array(new_y[validation_size:])
    return new_X, new_y, valid_X, valid_y
            
def main():
    
    data = sp.io.loadmat("data")
    print(data.keys())
    X_sample = data["X"]
    y_label = data["y"]
    desc = data["description"]
    X_test = data["X_test"]
    
    print("X sample: ", X_sample)
    X_sample, y_label = preprocessing(X_sample, y_label)
    print("X sample after pp: ", X_sample)
    X, y, valid_X, valid_y = create_validation_set(X_sample, y_label, int(len(X_sample) / 5))
    cost_func_vals = []
    vals = []
    w = np.zeros(len(desc) + 1)
    w[12] = 0.01
    print("shape of X: ", np.shape(X))
    print("shape of w: ", np.shape(w))
    rand_arr = list(range(len(X)))
    random.shuffle(rand_arr)
    w = stochastic_gradient_descent(X, y, w, 0.00003, 0.1, rand_arr, update_lists=True, cost_func_vals=cost_func_vals, vals=vals)
    plt.title("L2 reg logistic los vs. iterations for WINE: ")
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function Value")
    print(cost_func_vals, vals)
    plt.plot(vals, cost_func_vals)
    plt.show()
    y_test = s(w, valid_X)
    for i in range(len(y_test)):
        if y_test[i] < 0.5:
            y_test[i] = 0
        else:
            y_test[i] = 1
    print(y_test)
    
    from sklearn.metrics import accuracy_score
    mean = accuracy_score(valid_y, y_test)
    
    print(mean)
    X_test, y_label = preprocessing(X_test, data["y"])
    y_test = s(w, X_test)
    for i in range(len(y_test)):
        if y_test[i] >= 0.5:
            y_test[i] = 1
        elif y_test[i] < 0.5:
            y_test[i] = 0
    print(y_test)
    print("X_test: ", X_test.shape)
    print("y_test: ", len(y_test))
    # from sklearn.metrics import accuracy_score
    import pandas as pd
    file_name = "g1d_kaggle.csv"
    data_frame = pd.DataFrame([list(range(1, len(y_test) + 1)), y_test])
    data_frame = data_frame.transpose()
    data_frame.to_csv("results/"+file_name, index=False, header=None)
if __name__ == "__main__":
    main()