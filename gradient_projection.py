import numpy as np

L2 = lambda vector : np.sqrt(sum(vector**2))


f = lambda x, y, z : np.exp(x) * (y**2 + z*y + z**2 + 1)
f_grad = lambda x, y, z: np.array([np.exp(x) * (y**2 + z*y + z**2 + 1), np.exp(x) * (2*y + z), np.exp(x) * (y + 2*z)])
u = lambda x, y, z : ((x-1)**2 + (y-2)**2 + (z-3)**2) <= 1
P = lambda x, x_center, radius : x if (u(*x)) else x_center + ((x - x_center) * (radius/L2(x - x_center)))


def main():
    alpha, eps = 1, 0.001
    Xh, Xh_chert, Xcenter, radius = np.array([0, 0, 0]), np.array([1, 2, 3]), np.array([1, 2, 3]), 1

    iter = 0
    while True:
        X1 = Xh - alpha * f_grad(*Xh)
        if f(*X1) > f(*Xh):
            alpha /= 2
            continue

        X1_chert = P(X1, Xcenter, radius)
        if L2(X1_chert - Xh) < eps:
            print(f'iteration {iter+1}, L2 = {L2(X1_chert - Xh)}')
            break
        else:
            print(f'iteration {iter+1}, L2 = {L2(X1_chert - Xh)}')
            Xh = X1_chert

        iter += 1
    print(f'\n\nresult:\n===========iterations: {iter+1}================\nX = {X1_chert}\nf = {f(*X1_chert)}\n=============================================')

if __name__ == '__main__':
    main()