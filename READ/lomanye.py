import numpy as np
import matplotlib.pyplot as plt

#basis
f = np.vectorize(lambda x : np.cos(x)) #orig x**2 + np.exp(-x)
L = 1 #lipshits constant

#helper function
K = lambda x, x0 : f(x0) - L*np.abs(x-x0)
k = np.vectorize(lambda x, x0: K(x, x0))

#helper lines
def p0(x, a, b):
    x0 = (0.5/L)*( f(a) - f(b) + L*(a+b) )
    y0 = 0.5*( f(a) + f(b) + L*(a-b) )
    return (f(a) - L*(x-a) if x <= x0 else f(b) + L*(x-b), x0, y0)

def p1(x, a, b):
    p, x0, p_star = p0(x, a, b)
    delta = 0.5/L * (f(x0) - p_star)
    x1, x2 = x0 + delta, x0 - delta
    return (max(p, K(x, p_star)))

def p2(x, a, b):
    x0 = (0.5/L)*( f(a) - f(b) + L*(a+b) )
    p_star = 0.5*( f(a) + f(b) + L*(a-b) )
    delta = 0.5/L * (f(x0) - p_star)
    x1 = x0 - delta 
    x2 = x0 + delta
    return max(p1(x, a, b), K(x, p_star))

#main
a, b = 0, 1
x0 = np.random.uniform(a, b)

x = np.linspace(a, b, 100)
y = f(x)


pk0 = np.vectorize(lambda x, a, b : p0(x, a, b)[0])(x, a, b)
pk1 = np.vectorize(lambda x, a, b : p1(x, a, b))(x, a, b)
pk2 = np.vectorize(lambda x, a, b : p2(x, a, b))(x, a, b)

plt.plot(x, y)
plt.plot(x, pk0, label='p0')
plt.plot(x, pk1, label='p1')
plt.plot(x, pk2, label='p2')
plt.legend()
plt.show()
