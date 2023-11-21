import numpy as np

def vec_norm(vec : np.ndarray):
    return sum(vec**2)

def gradient_descent(f, f_grad, v0, alpha, eps, eps1):
    if alpha < eps:
        return v0
    
    grad = np.array(f_grad(*v0))
    if vec_norm(grad) < eps1:
        return v0
    
    v1 = v0 - grad * alpha
    if f(*v1) > f(*v0):
        return gradient_descent(f, f_grad, v0, alpha/2, eps, eps1)
    else:
        return gradient_descent(f, f_grad, v1, alpha, eps, eps1)

def main():
    f = lambda x, y : 5*x**2 + 5*y**2 + 8*x*y
    f_der_x = lambda x, y : 10*x + 8*y
    f_der_y = lambda x, y : 10*y + 8*x
    f_grad = (lambda x, y : (f_der_x(x, y), f_der_y(x, y)))
    v0 = np.array((1, 1))
    alpha, eps, eps1 = 1, 0.01, 0.001

    print(gradient_descent(f, f_grad, v0, alpha, eps, eps1))

if __name__ == '__main__':
    main()