import numpy as np
import opt
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import json



#считываем данные
y = np.loadtxt("jla_mub.txt", delimiter=' ', dtype=np.float)

#базовые данные(первое приближение)

x_0 = [0.5, 50]



#космологическая функция определения расстояния

def f(O, H_0, z):
    def m(x):
        return 1 / (np.sqrt((1-O)*((1+x)**3)+O))

    listOfMUS = []
    for i in range(len(z)):
        g = scipy.integrate.quad(m, 0, z[i])
        k = np.log10(3 * (10 ** 11) / H_0 * (1 + z[i]) * g[0])
        listOfMUS.append(5 * k - 5)
    return listOfMUS



#матрица Якоби для нашего случая

def j(O, H_0, z, x_0):
    jacobi_matrix = np.zeros((len(z), len(x_0)))  # создаем будущую матрицу Якоби

    def partialDerO():  # частная производная по темной энергии

        def f(x):
            return 1 / (np.sqrt((1 - O) * ((1 + x) ** 3) + O))

        def f_derivative(x):
            return -(1 / 2) * (1 - ((1 + x)) ** 3) / (((1 - O) * ((1 + x) ** 3) + O) ** (3 / 2))

        return 5 / (np.log(10) * scipy.integrate.quad(f, 0, z[i])[0]) * (scipy.integrate.quad(f_derivative, 0, z[i])[0])

    def partialDerH_0(H_0):  # частная производная по постоянной Хаббла
        return round(-5 / (H_0 * np.log(10)), 10)

    for i in range(0, len(z)):  # заполняем матрицу Якоби частными производными
        jacobi_matrix[i, 0] = partialDerO()
        jacobi_matrix[i, 1] = partialDerH_0(H_0)
    return jacobi_matrix

#найдем оптимальные параметры метода для алгоритма Ньютона-Гаусса, предварительно выбрав массивы поиска
arrayCOST0 = []
for k in (0.1, 0.5, 1):
    for tol in (1e-3, 1e-4, 1e-5, 1e-6):
        arrayCOST0.append(opt.gauss_newton(y, f, j, x_0, k, tol)[1][-1])
for k in (0.1, 0.5, 1):
    for tol in (1e-3, 1e-4, 1e-6, 1e-7):
        if opt.gauss_newton(y, f, j, x_0, k, tol)[1][-1] == min(arrayCOST0):
            k, tol = k, tol
            break




#введём оптимальные параметры метода для алгоритма Левенберга-Марквардта, при которых хорошо видно, какой алгоритм сходится быстрее
# - Гаусса-Ньютона или Левенберга-Марквардта,предварительно вручную подобрав более менее подходящие


tolLM = 1e-6
nu = 1.4
lmbd_0 = 0.1




#выведем результаты в классе RESULT для обоих методов

print(opt.gauss_newton(y, f, j, x_0, k, tol))

print(opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM))



#перейдем к построению графиков. Начнем с графика mu-z

df = pd.read_csv('jla_mub.txt', sep=" ", header=1, names=["y", "x"])
xx, yy = df["x"], df["y"]


H0 = opt.gauss_newton(y, f, j, x_0, k=k, tol=tol)[3][1]
O0 = opt.gauss_newton(y, f, j, x_0, k=k, tol=tol)[3][0]

#первый график
plt.plot(np.array(yy), f(O0, H0, np.array(yy)), 'y', linestyle = '-', linewidth = 9, label = r'cosmological GAUSS')

H0 = opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM)[-1][1]
O0 = opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM)[-1][0]

#второй график
plt.plot(np.array(yy), f(O0, H0, np.array(yy)), 'c', linestyle = '-', linewidth = 5, label = r'cosmological LM')

#третий график
plt.plot(yy, xx, 'r', linestyle = ':', linewidth = 5, label = r'photometric')
plt.xlabel(r'redshift, z', labelpad = 10)
plt.ylabel(r'logarithmic distance, 5lg(d)-5', labelpad = 10)
plt.grid()
plt.title('comparison of photometric and cosmological methods')
plt.legend()

#сохраним и выведем первый график
plt.savefig('mu-z.png', dpi = 200)
plt.show()


#перейдем к построению второго графика COSTFUNCTION - ITERATION

#итерации(ось Х)
iterationsGAUSS = []
for i in range(len(opt.gauss_newton(y, f, j, x_0, k=k, tol=tol)[1])):
    iterationsGAUSS.append(i)
iterationsLM = []
for i in range(len(opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM)[1])):
    iterationsLM.append(i)

#функции ошибок(ось Y)
COSTfunctionGAUSS = opt.gauss_newton(y, f, j, x_0, k=k, tol=tol)[1]
COSTfunctionLM = opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM)[1]


plt.scatter(iterationsGAUSS, COSTfunctionGAUSS)
plt.plot(iterationsGAUSS, COSTfunctionGAUSS, 'r', linestyle = '--', linewidth = 1, label = r'Gauss-Newton')
plt.scatter(iterationsLM, COSTfunctionLM)
plt.plot(iterationsLM, COSTfunctionLM, 'c', linestyle = ':', linewidth = 1, label = r'Levenberge-Marquardt')
plt.xlabel(r'iteration', labelpad = 10)
plt.ylabel(r'costfunction of algorithm', labelpad = 10)
plt.grid()
plt.title('comparison of "costfunction-iteration" dependence for two algorithms')
plt.legend()

#сохраним и выведем второй график
plt.savefig('cost.png', dpi = 200)
plt.show()


#выводим результы в файл json и округляем переменные до минимального порядка входных данных, то есть до -10

data = {
  "Gauss-Newton": {"H0": round(opt.gauss_newton(y, f, j, x_0, k=k, tol=tol)[3][1], 10), "Omega": round(opt.gauss_newton(y, f, j, x_0, k=k, tol=tol)[3][0], 10), "nfev": round(opt.gauss_newton(y, f, j, x_0, k=k, tol=tol)[0], 10)},
  "Levenberg-Marquardt": {"H0": round(opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM)[-1][1], 10), "Omega": round(opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM)[-1][0], 10), "nfev": round(opt.lm(y, f, j, x_0, lmbd_0=lmbd_0, nu=nu, tol=tolLM)[0], 10)}
}

with open("parameters.json", "w") as write_file:
    json.dump(data, write_file)


