import numpy as np
from collections import namedtuple

#вводим резалт
Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))


#метод Ньютона-Гаусса

def gauss_newton(y, f, j, x_0, k, tol):


    def costFUNCTION(O, H_0, y):
        global MU, Z
        Z = y[:, 0]
        MU = y[:, 1]
        return 1/2*(np.linalg.norm(MU - f(O, H_0, Z)))**2

    nfev = 0
    x = x_0
    costs = []
    step = np.linalg.norm(x)
    costs.append(costFUNCTION(x[0], x[1], y))  #массив ошибок
    delta_cost_array = costs[0]  #разницы между функцией потерь на n и n-1 шаге
    gradnorm = 1000


    while (delta_cost_array or gradnorm or step) >= tol: #придержимся заданной точности , которая регулируется параметром метода - tol
        Jacobi = j(x[0], x[1], Z, x_0)
        JacobiT = Jacobi.transpose()
        step = np.linalg.norm(k * np.dot(np.dot(np.linalg.inv(np.dot(JacobiT, Jacobi)), JacobiT), np.ravel(MU - f(x[0], x[1], Z))))
        x = x + k * np.dot(np.dot(np.linalg.inv(np.dot(JacobiT, Jacobi)), JacobiT), np.ravel(MU - f(x[0], x[1], Z)))
        costs.append(costFUNCTION(x[0], x[1], y))
        delta_cost_array = costs[-2] - costs[-1]
        nfev += 1
        gradnorm = np.linalg.norm(JacobiT @ np.ravel(MU - f(x[0], x[1], Z)))


    return Result(nfev=nfev, cost=costs, gradnorm=gradnorm, x=x)





#метод Левенберга-Марквардта

def lm(y, f, j, x_0, lmbd_0=1e-4, nu=2, tol=1e-4):

    def costFUNCTION(O, H_0, y):
        global MU, Z
        Z = y[:, 0]
        MU = y[:, 1]
        return 1/2*(np.linalg.norm(MU - f(O, H_0, Z)))**2

    nfev = 0
    lmbd_act = lmbd_0 #параметр метода на n-ом шаге


    x = x_0
    x_nu = x_0 #вспомогательный вектор


    step = np.linalg.norm(x)


    costs = []

    costs.append(costFUNCTION(x[0], x[1], y))  # массив ошибок


    delta_cost_array = costs[0]  # разницы между функцией потерь на n и n-1 шаге

    gradnorm = 1

    while (delta_cost_array or gradnorm or step) > tol:

        x_nu_actual = x_nu #вспомогательный вектор неизвестных параметров, который понадобился из-за условного оператора
        Jacobi = j(x[0], x[1], Z, x_0)
        JacobiT = Jacobi.transpose()
        residual = np.ravel(MU - f(x_nu[0], x_nu[1], Z))

        nfev += 1
        x_nu += np.dot(np.dot(np.linalg.inv(np.dot(JacobiT, Jacobi) + lmbd_act/nu * np.identity(len(x_0))), JacobiT), residual)

        cost_nu = [(costFUNCTION(x_nu[0], x_nu[1], y))]


        if cost_nu[-1] > costs[-1] and costs[-2] <= costs[-1]:
            lmbd_act, lmbd_prev = lmbd_act , lmbd_act
            x += np.dot(np.dot(np.linalg.inv(np.dot(JacobiT, Jacobi) + lmbd_act * np.identity(len(x_0))), JacobiT), residual)
            costs.append(costFUNCTION(x[0], x[1], y))

        elif cost_nu[-1] <= costs[-1]:
            lmbd_act, lmbd_prev = lmbd_act/nu, lmbd_act
            x += np.dot(np.dot(np.linalg.inv(np.dot(JacobiT, Jacobi) + lmbd_act * np.identity(len(x_0))), JacobiT),
                        residual)
            costs.append(costFUNCTION(x[0], x[1], y))

        else:
            lmbd_act, lmbd_prev = lmbd_act*nu, lmbd_act
            x_nu = x_nu_actual
            continue


        step = np.linalg.norm(np.dot(np.dot(np.linalg.inv(np.dot(JacobiT, Jacobi) + lmbd_act * np.identity(len(x_0))), JacobiT), residual))
        delta_cost_array = costs[-2] - costs[-1]
        gradnorm = np.linalg.norm(JacobiT @ residual)


    return Result(nfev=nfev, cost=costs, gradnorm=gradnorm, x=x)



Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов можельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива меньше nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""

