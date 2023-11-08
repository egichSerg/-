import numpy as np
import matplotlib.pyplot as plt

f_history = list()
x_history_moment, y_history_moment = 0, 0

def opt_step_x(x, y, k, delta, xhm, yhm, f):
    nx, x_history_moment, y_history_moment = x, xhm, yhm
    f_nx = f(x + k*delta, y)
    dir = 1

    if f_nx < f_history[x_history_moment, y_history_moment]:
        x_history_moment += 1
        f_history[x_history_moment, y_history_moment] = f_nx
        nx = x + k*delta
    else:
        dir = -1
        nx = x - k*delta 
        f_nx = f(x - k*delta, y)
    
    step = k**2
    while f_nx < f_history[x_history_moment, y_history_moment]:
        nx = nx + dir*step*delta
        step *= k

    return [nx, x_history_moment, y_history_moment]

def opt_step_y(x, y, k, delta, f):
    pass

#program
f = lambda x, y : 5*x**2 + 5*y**2 + 8*x*y
f_vec = np.vectorize(f)

eps = 0.01
k = (np.sqrt(5) + 1) / 2
delta = eps / np.sqrt(2)

x, y = 1, 1



# abs(xn - xn-2) <= eps

