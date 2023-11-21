from inspect import currentframe, getframeinfo

import numpy as np
import matplotlib.pyplot as plt

f_history = [[]]
f_history_xy = {'x' : [], 'y' : []}

def opt_step_x(x, y, k, delta, prev, f):
    nx = x
    fprev = prev
    f_nx = f(x + k*delta, y)
    dir = 1

    if f_nx >= fprev:
        dir = -1
        f_nx = f(x - k*delta, y)
    
    f_history_xy['x'].append(nx)
    step = k
    res = fprev

    while f_nx < fprev:

        #accept previous step for y
        res = f_nx
        nx += dir*step*delta
        f_history_xy['x'].append(nx)

        #save f for new y
        fprev = f_nx

        #update y
        step *= k

        #calc f for next step
        f_nx = f(nx+ dir*step*delta, y)


    return [nx, res]

def opt_step_y(x, y, k, delta, prev, f):
    ny = y
    fprev = prev
    f_ny = f(x, y + k*delta)
    dir = 1

    if f_ny >= fprev:
        dir = -1
        f_ny = f(x, y - k*delta)  

    f_history_xy['y'].append(ny)
    step = k
    
    while f_ny < fprev:

        #accept previous step for y
        ny += dir*step*delta
        f_history_xy['y'].append(ny)

        #save f for new y
        fprev = f_ny

        #update y
        step *= k

        #calc f for next step
        f_ny = f(x, ny + dir*step*delta)
          

    return ny

def find_by_gold(history, delta, func, var, var_index):
    if len(history) < 3:
        frameinfo = getframeinfo(currentframe())
        print(f'Caught exception in file {frameinfo.filename} at line {frameinfo.lineno}\nfunction "{frameinfo.function}()"\nlen(history) must be at least 3 numbers!')
        exit(1)
    
    if var_index == 0:
        f = lambda x : func(var, x)
    else:
        f = lambda x : func(x, var)

    history_last_index = len(history) - 1
    a, b = sorted([history[history_last_index], history[history_last_index - 2]])
    x2 = history[history_last_index - 1]


    x1 = a + b - x2
    f1, f2 = f(x1), f(x2)
    while (abs(b-a) > delta):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 =  a + b - x2
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + b - x1
            f2 =f(x2)
    
    return (a+b)/2


def points_euclid_distance(p1, p2):
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    return np.sqrt((x2- x1)**2 + (y2 - y1)**2)


def main():
    f = lambda x, y : 5*x**2 + 5*y**2 + 8*x*y
    f_vec = np.vectorize(f)

    eps = 0.01
    k = (np.sqrt(5) + 1) / 2
    delta = eps / np.sqrt(2)

    x, y = 1, 1
    history_moment = 0

    f_history[history_moment] = [x, y, f(x, y)]

    # x = find_by_gold([0, 0.8, 1], 0.01, lambda x, y: -(x**2), y, 1)
    # print(x)

#loop start
    while True:
        x_star, fprev = opt_step_x(x, y, k, delta, f_history[history_moment][2], f)
        x = find_by_gold(f_history_xy['x'], delta, f, y, 1)
        print(x)

        y_star = opt_step_y(x, y, k, delta, fprev, f)
        y = find_by_gold(f_history_xy['y'], delta, f, x, 0)
        print(y)
    

        history_moment += 1
        f_history.append([x, y, f(x, y)])

        if (points_euclid_distance(f_history[history_moment], f_history[history_moment - 1]) < eps):
            break

#loop end

    print(f_history[history_moment])
    print(f_history_xy)


if __name__ == '__main__':
    main()



# abs(xn - xn-2) <= eps

