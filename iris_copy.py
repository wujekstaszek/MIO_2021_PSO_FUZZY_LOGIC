

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

    words_dict = {'sepal length','sepal width','petal length','petal width'}
    var_list = ['sepal length','sepal width','petal length','petal width']
    result_word = "iris class"

    var_min_maxs_learn = []   # min i max dla kazdej zmiennej w zbiorze uczacym
    for i in range(len(var_list)):
            var_min_maxs_learn.append([min(learn, key=lambda elem: elem[i])[i],
                                       max(learn, key=lambda elem: elem[i])[i]])

    # New Antecedent/Consequent objects hold universe variables and membership functions
    eps = 0.1
    variables = []
    for i in range(len(var_list)):
        variables.append(ctrl.Antecedent(np.arange(var_min_maxs_learn[i][0], 
                                                   var_min_maxs_learn[i][1] + eps, 
                                                   eps), 
                                         var_list[i]))
    # sepal_l = ctrl.Antecedent(np.arange(min_sepal_l, max_sepal_l + eps, eps), 'sepal length')
    # sepal_w = ctrl.Antecedent(np.arange(min_sepal_w, max_sepal_w + eps, eps), 'sepal width')
    # petal_l = ctrl.Antecedent(np.arange(min_petal_l, max_petal_l + eps, eps), 'petal length')
    # petal_w = ctrl.Antecedent(np.arange(min_petal_w, max_petal_w + eps, eps), 'petal width')
    iris_class = ctrl.Consequent(np.arange(1, 4, 1), result_word)

    # variables = [sepal_l, sepal_w, petal_l, petal_w]

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    parameter = 7
    for var in variables:
        var.automf(parameter)
    # sepal_l.automf(parameter)
    # sepal_w.automf(parameter)
    # petal_l.automf(parameter)
    # petal_w.automf(parameter)
    # iris_class.automf(parameter)

    # Custom membership functions can be built interactively with a familiar, Pythonic API
    iris_class['1'] = fuzz.trimf(iris_class.universe, [1, 1, 5/3])
    iris_class['2'] = fuzz.trimf(iris_class.universe, [5/3, 2, 7/3])
    iris_class['3'] = fuzz.trimf(iris_class.universe, [7/3, 3, 3])

    # sl=0
    # sw=0
    # pl=0
    # pw=0
    # l1 = 101
    # l2 = 150
    # print(min(data[l1:l2], key=lambda elem: elem[0])[0])
    # print(max(data[l1:l2], key=lambda elem: elem[0])[0])
    # print(min(data[l1:l2], key=lambda elem: elem[1])[1])
    # print(max(data[l1:l2], key=lambda elem: elem[1])[1])
    # print(min(data[l1:l2], key=lambda elem: elem[2])[2])
    # print(max(data[l1:l2], key=lambda elem: elem[2])[2])
    # print(min(data[l1:l2], key=lambda elem: elem[3])[3])
    # print(max(data[l1:l2], key=lambda elem: elem[3])[3])
    # sl/=50
    # sw/=50
    # pl/=50
    # pw/=50

    # sepal_l.view()
    # sepal_w.view()
    # petal_l.view()
    # petal_w.view()
    # iris_class.view()

    words_values = ['dismal', 'poor', 'mediocre', 'average', 'decent', 'good', 'excellent']

    rules = []
    var_min_maxs = []   # min i max dla kazdej zmiennej
    for i in range(len(variables)):
            var_min_maxs.append([min(data, key=lambda elem: elem[i])[i],
                                 max(data, key=lambda elem: elem[i])[i]])
    for class_ind in range(1, int(max(data, key=lambda elem: elem[-1])[-1])+1):
        data_case = []  # dane z danej klasy
        for elem in data:
            if elem[-1] == class_ind:
                data_case.append(elem)
        val_borders = []    # granice przedzialow dla kazdej zmiennej
        for i in range(len(variables)):
            dist = var_min_maxs[i][1] - var_min_maxs[i][0]
            eps = dist / 6.0
            val_borders.append([var_min_maxs[i][0] + j*eps for j in range(len(words_values))])
        var_min_maxs_case = []   # min i max dla kazdej zmiennej danego case
        for i in range(len(variables)):
            var_min_maxs_case.append([min(data_case, key=lambda elem: elem[i])[i],
                                 max(data_case, key=lambda elem: elem[i])[i]])


        # print(val_borders)
        # print(len(words_values))

        words_borders = [] # lista granic slow dla kazdej zmiennej
        for i in range(len(variables)):
            borders = []
            borders.append([val_borders[i][0], val_borders[i][1]]) # granice dla slow good itp
            for j in range(1, len(words_values)-1):
                borders.append([val_borders[i][j-1], val_borders[i][j+1]])
            borders.append([val_borders[i][len(words_values)-2], val_borders[i][len(words_values)-1]])
            words_borders.append(borders)

        inner_rules = [[0 for _ in range(len(words_values))] for _ in range(len(variables))]
        
        for i in range(len(variables)):
            var_min = var_min_maxs_case[i][0]
            var_max = var_min_maxs_case[i][1]
            ind_min = 0
            ind_max = 0
            
            for j in range(len(words_values)):
                if words_borders[i][j][0] <= var_min and var_min <= words_borders[i][j][1]:
                    ind_min = j
                    break
            for j in range(len(words_values)):
                if words_borders[i][j][0] <= var_max and var_max <= words_borders[i][j][1]:
                    ind_max = j
            for j in range(ind_min, ind_max+1):
                inner_rules[i][j] = 1
        print(inner_rules)

        v1 = None
        vr = []
        for i in range(len(variables)):
            v1 = None
            for j in range(len(words_values)):
                if inner_rules[i][j]:
                    if v1 is None:
                        v1 = variables[i][words_values[j]]
                    else:
                        v1 = v1 | variables[i][words_values[j]]
            vr.append(v1)

        

        final_vr = vr[0]
        for i in range(1, len(vr)):
            final_vr = final_vr & vr[i]

        # print(final_vr)

        rules.append(ctrl.Rule(final_vr,iris_class[str(class_ind)]))


    # rule1 = ctrl.Rule((sepal_l['dismal'] | sepal_l['poor'] | sepal_l['mediocre'] | sepal_l['average']) & 
    #                     # (sepal_w['mediocre'] | sepal_w['average'] | sepal_w['decent']) & 
    #                     (petal_l['dismal'] | petal_l['poor']) &
    #                     (petal_w['dismal'] | petal_w['poor'] | petal_w['mediocre']), 
    #                     iris_class['1'])
    # rule2 = ctrl.Rule( ~(sepal_l['dismal'] | sepal_l['excellent']) & 
    #                     ~sepal_w['excellent'] & 
    #                     (petal_l['mediocre'] | petal_l['average'] | petal_l['decent'] | petal_l['good']) &
    #                     (petal_w['mediocre'] | petal_w['average'] | petal_w['decent'] | petal_w['good']), 
    #                     iris_class['2'])
    # rule3 = ctrl.Rule(~sepal_l['dismal'] & 
    #                     ~sepal_w['excellent'] & 
    #                     (petal_l['average'] | petal_l['decent'] | petal_l['good'] | petal_l['excellent']) &
    #                     (petal_w['average'] | petal_w['decent'] | petal_w['good'] | petal_w['excellent']), 
    #                     iris_class['3'])

    iris_class_ctrl = ctrl.ControlSystem(rules)

    iris_class_ctrl_sim = ctrl.ControlSystemSimulation(iris_class_ctrl, flush_after_run=len(data))

    inputs = {}
    for elem in test:
        i=0
        for word in words_dict:
            inputs[word]=elem[i]
            i+=1
        # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
        # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
        iris_class_ctrl_sim.inputs(inputs)

        # Crunch the numbers
        iris_class_ctrl_sim.compute()

        res = 0
        if iris_class_ctrl_sim.output[result_word] < 5/3:
            res = 1
        elif iris_class_ctrl_sim.output[result_word] < 7/3:
            res = 2
        else:
            res = 3

        print(res == elem[4])
        if not res == elem[4]:
            # dlaczego rozne? (mozliwe ze 2 i 3)
            pass

        # iris_class.view(sim=iris_class_ctrl_sim)

    a = input('exit')

