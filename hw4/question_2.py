import numpy as np
import scipy as sp

    
    
def find_w(v, X, y):
    hessian_inv = find_hess_inv(hessian_J(v, X))
    return v - hessian_inv @ grad_J(v, X, y)
    
    
def grad_J(v, X, y):
    svec = s(v, X)
    # vec1 = np.array([1 for i in range(len(svec))])
    return -1 * X.T @ (y - svec)

def hessian_J(v, X):
    svec = s(v, X)
    vec1 = np.array([1 for i in range(len(svec))])
    return X.T @ (diag_s(X, v)) @ X
    
def omega(X, w):
    return np.diag(np.array([]))
def s(w, X):
    s = sp.special.expit(X.dot(w))
    for i in range(len(s)):
        if s[i] == 1.0:
            s[i] = 0.9999
    return s
def diag_s(X, w):
    d = []
    for x in X:
        si = s_individual(x, w)
        d.append(si * (1 - si))
    d = np.array(d)
    return np.diag(d)
def s_individual(x, w):
    return sp.special.expit(np.dot(x, w))
def find_hess_inv(H):
    L = np.linalg.inv(H)
    # P = (np.linalg.inv(L)).T
    return L

def main():
    X = np.array([[0.2, 3.1, 1.0], 
                  [1.0, 3.0, 1.0],
                  [-0.2, 1.2, 1.0],
                  [1.0, 1.1, 1.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    w0 = np.array([-1.0, 1.0, 0])
    print("s0: " + str(s(w0, X)))
    w1 = find_w(w0, X, y)
    print("w1: " + str(w1))
    s1 = s(w1, X)
    print("s1: " + str(s1))
    w2 = find_w(w1, X, y)
    print("w2: " + str(w2))
    
if __name__ == "__main__":
    main()