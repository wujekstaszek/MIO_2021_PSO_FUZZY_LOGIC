

from random import shuffle

import sys

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



def get_data_from_file(filepath: str):
    # pobieranie danych z pliku
    with open(filepath, 'r') as f:
        data = [[float(num) for num in line.split(',')] for line in f]

    return data


def get_test_and_learn_from_data(data: list):
    # przetasowanie zbioru
    data_cpy = data.copy()
    shuffle(data_cpy)

    n = len(data_cpy)//4 # liczba elementow testujacych

    # wybranie zbiorów testującego i uczącego
    test = data_cpy[0:n]
    learn = data_cpy[n:]

    return test, learn


if __name__ == '__main__':
    data = get_data_from_file('datasets/iris_dataset.txt')
    test, learn = get_test_and_learn_from_data(data)
    # print(data, '\n', test, '\n', learn)


    min_sepal_l = min(learn, key=lambda elem: elem[0])[0]
    max_sepal_l = max(learn, key=lambda elem: elem[0])[0]
    min_sepal_w = min(learn, key=lambda elem: elem[1])[1]
    max_sepal_w = max(learn, key=lambda elem: elem[1])[1]
    min_petal_l = min(learn, key=lambda elem: elem[2])[2]
    max_petal_l = max(learn, key=lambda elem: elem[2])[2]
    min_petal_w = min(learn, key=lambda elem: elem[3])[3]
    max_petal_w = max(learn, key=lambda elem: elem[3])[3]

    # New Antecedent/Consequent objects hold universe variables and membership functions
    eps = 0.1
    sepal_l = ctrl.Antecedent(np.arange(min_sepal_l, max_sepal_l + eps, eps), 'sepal length')
    sepal_w = ctrl.Antecedent(np.arange(min_sepal_w, max_sepal_w + eps, eps), 'sepal width')
    petal_l = ctrl.Antecedent(np.arange(min_petal_l, max_petal_l + eps, eps), 'petal length')
    petal_w = ctrl.Antecedent(np.arange(min_petal_w, max_petal_w + eps, eps), 'petal width')
    iris_class = ctrl.Consequent(np.arange(1, 3, 0.1), 'iris class')

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    parameter = 3
    sepal_l.automf(parameter)
    sepal_w.automf(parameter)
    petal_l.automf(parameter)
    petal_w.automf(parameter)
    iris_class.automf(parameter)

    # Custom membership functions can be built interactively with a familiar, Pythonic API
    # iris_class['1'] = fuzz.trimf(iris_class.universe, [1, 1, 5/3])
    # iris_class['2'] = fuzz.trimf(iris_class.universe, [5/3, 2, 7/3])
    # iris_class['3'] = fuzz.trimf(iris_class.universe, [7/3, 3, 3])

    sepal_l.view()
    sepal_w.view()
    petal_l.view()
    petal_w.view()
    iris_class.view()

    # fuzz.interp_membership()


    a = input('hej')

