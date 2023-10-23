import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
from pyavltree import AVLTree

#cosmetic point - cross
def draw_cross(coords, size, a, b, N):
    size_in_steps = (b - a) / N * size
    X = np.arange(coords[0] - size_in_steps/2, coords[0] + size_in_steps/2, step=(b-a)/N)
    Y = np.full_like(X, coords[1])
    plt.plot(X, Y, color='red')

    Y = np.arange(coords[1] - size_in_steps/2, coords[1] + size_in_steps/2, step=(b-a)/N)
    X = np.full_like(Y, coords[0])
    plt.plot(X, Y, color='red')


#basis
f = np.vectorize(lambda x : np.cos(x))
f_der = np.vectorize(lambda x : np.abs(-np.sin(x)))
f_unvec = lambda x : np.cos(x)
# f = np.vectorize(lambda x : x**2 - 1)
# f_unvec = lambda x : x**2 - 1
# f_der = np.vectorize(lambda x : np.abs(2*x))


a, b, = 0, 4

N = 100
X = np.linspace(a, b, N)
Y = f(X)

L = max(f_der(X))

tre = AVLTree()
root = None

#helper function
K = lambda x, x0 : f(x0) - L*np.abs(x-x0)
k = np.vectorize(lambda x, x0: K(x, x0))

def p0xy(a, b):
    x0 = (0.5/L)*( f(a) - f(b) + L*(a+b) )
    y0 = 0.5*( f(a) + f(b) + L*(a-b) )
    return (y0, x0)

def pkxy(a, b, pkp):
    y0, x0 = pkp[:2]
    delta = 0.5/L * (f(x0) - y0)
    x1, x2 = x0 + delta, x0 - delta
    p1, p2 = 0.5 * (f(x1) + y0), 0.5 * (f(x2) + y0)
    return [(p1, x1,delta), (p2, x2, delta)]

def glopnp(xa, b, sigma):
    global root
    point = pkxy(a, b, p0xy(a, b)) #init to not double the calculations

    root = tre.insert_node(root, point[0])
    root = tre.insert_node(root, point[1])

    point = tre.avl_MinValue(root).value #bring to the common format

    while(2 * L * point[2] >= sigma):        
        points = pkxy(a, b, point)
        root = tre.delete_node(root, point)

        root = tre.insert_node(root, points[0])
        root = tre.insert_node(root, points[1])

        #plt.scatter(point[1], point[0])
        point = tre.avl_MinValue(root).value       
    
    point = tre.avl_MinValue(root).value
    return (point[1], f_unvec(point[1]))


sigma = 0.001

answer = glopnp(a, b, sigma)
print(answer)

draw_cross(answer, 10, a, b, N)
plt.plot(X, Y)
plt.show()