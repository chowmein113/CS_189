import numpy as np

import scipy as sp
import matplotlib.pyplot as plt



def get_l_norm(x, y, p):
    xp = np.abs(x) ** p
    yp = np.abs(y) ** p
    result = (xp + yp) ** (1 / p)
    return result
def main():
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    x, y = np.meshgrid(x, y)
    fig, axs = plt.subplots(3, 1, figsize=(7, 7))
    
    axs[0].set_title("part a with l and p = 0.5: ")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    f = get_l_norm(x, y, 0.5)
    axs[0].contourf(x, y, f, cmap='cividis')
    
    axs[1].set_title("part b with l and p = 1: ")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    f = get_l_norm(x, y, 1)
    axs[1].contourf(x, y, f, cmap='inferno')
    
    axs[2].set_title("part c with l and p = 2: ")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    f = get_l_norm(x, y, 2)
    axs[2].contourf(x, y, f, cmap='viridis')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()