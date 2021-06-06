

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


def check_fl_on_test_set(sim: fuzz.control.controlsystem.ControlSystemSimulation, test: list, mf_fun_params: list, antecedents_names: list, consequent_name: str):
    count = 0
    for elem in test:
        for i in range(len(antecedents_names)):
            sim.input[antecedents_names[i]] = elem[i]

        # symulacja FL dla danego elementu
        sim.compute()

        # określenie do jakiej klasy należy dany element
        res = []
        for i in range(len(mf_fun_params)):
            if mf_fun_params[i][0] <= sim.output[consequent_name] and sim.output[consequent_name] <= mf_fun_params[i][2]:
                res.append(i+1)

        
        # zwiększenie liczby dobrych przypadków
        if elem[4] in res:
            count += 1
            # dlaczego rozne? (mozliwe ze 2 i 3)
        
        # wizualizacja
        # consequent.view(sim=sim)

    # wypisanie ile procentowo przypadków bylo poprawnie określonych przez FL
    print("Poprawność FL: {:.2f}%".format(count/len(test)*100))
    


if __name__ == '__main__':
    # przygotowanie zmiennych
    data = get_data_from_file('datasets/iris_dataset.txt')
    test, learn = get_test_and_learn_from_data(data)

    rule_values = ['dismal', 'poor', 'mediocre', 'average', 'decent', 'good', 'excellent']
    num_rule_values = len(rule_values)
    antecedents_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    num_of_antecedents = len(antecedents_names)

    num_of_classes = int(max(data, key=lambda elem: elem[-1])[-1])
    consequent_name = 'iris_class'

    mf_fun_params = [[1, 1, 5/3], [5/3, 2, 7/3], [7/3, 3, 3]]

    rules_indexes_by_hand = [[[1,1,1,1,0,0,0], [1,1,1,1,1,1,1], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0]],
                             [[1,1,1,1,1,1,0], [1,1,1,1,1,0,0], [0,0,1,1,1,1,0], [0,0,1,1,1,1,0]],
                             [[1,1,1,1,1,1,1], [1,1,1,1,1,1,0], [0,0,0,1,1,1,1], [0,0,0,1,1,1,1]]]

    minmax_of_antecedents = []
    for i in range(num_of_antecedents):
        minmax_of_antecedents.append([min(learn, key=lambda elem: elem[i])[i], max(learn, key=lambda elem: elem[i])[i]])
    
    minmax_of_data = []
    for i in range(num_of_antecedents):
        minmax_of_data.append([min(data, key=lambda elem: elem[i])[i], max(data, key=lambda elem: elem[i])[i]])
    # print(minmax_of_data, '\n')

    antecedents_intervals = []
    for i in range(num_of_antecedents):
        interval_eps = (minmax_of_data[i][1] - minmax_of_data[i][0]) / (num_rule_values-1)
        antecedents_intervals.append([minmax_of_data[i][0] + j * interval_eps for j in range(num_rule_values)])
    # print(antecedents_intervals, '\n')

    rule_values_intervals = []
    for i in range(num_of_antecedents):
        antecedent_rule_values_intervals = []

        antecedent_rule_values_intervals.append([antecedents_intervals[i][0], antecedents_intervals[i][1]])
        for j in range(num_rule_values-2):
            antecedent_rule_values_intervals.append([antecedents_intervals[i][j], antecedents_intervals[i][j+2]])
        antecedent_rule_values_intervals.append([antecedents_intervals[i][num_rule_values-2], antecedents_intervals[i][num_rule_values-1]])

        rule_values_intervals.append(antecedent_rule_values_intervals)
    # for elem in rule_values_intervals:
    #     print(elem)

    # poprzednicy i następnik
    eps = 0.1
    antecedents = []
    for i in range(num_of_antecedents):
        antecedents.append(ctrl.Antecedent(np.arange(minmax_of_antecedents[i][0], minmax_of_antecedents[i][1] + eps, eps), antecedents_names[i]))
    consequent = ctrl.Consequent(np.arange(1, num_of_classes+1, 1), consequent_name)

    # automatyczna funkcja przynależności
    for a in antecedents:
        a.automf(num_rule_values)
    # consequent.automf(num_rule_values)

    # specjalistyczna funkcja przynależności
    for class_index in range(num_of_classes):
        consequent[str(class_index+1)] = fuzz.trimf(consequent.universe, mf_fun_params[class_index])    

    # reguły indeksowane
    rules_indexes = []
    for r_ind in range(num_of_classes):
        all_rule_indexes = []

        rule_data = []
        for elem in learn:
            if elem[-1] == r_ind+1:
                rule_data.append(elem)
        minmax_of_rule_data = []
        for i in range(num_of_antecedents):
            minmax_of_rule_data.append([min(rule_data, key=lambda elem: elem[i])[i], max(rule_data, key=lambda elem: elem[i])[i]])

        for i in range(num_of_antecedents):
            antecedent_rules = [0 for _ in range(num_rule_values)]
            
            antecedent_min = minmax_of_rule_data[i][0] - 2 * 10**-1
            min_index = -1
            antecedent_max = minmax_of_rule_data[i][1] + 2 * 10**-1
            max_index = -1

            for j in range(num_rule_values):
                if min_index == -1 and rule_values_intervals[i][j][0] <= antecedent_min and antecedent_min <= rule_values_intervals[i][j][1]:
                    min_index = j
                if max_index == -1 and rule_values_intervals[i][j][0] <= antecedent_max and antecedent_max <= rule_values_intervals[i][j][1]:
                    max_index = j

            if min_index == -1:
                min_index = 0
            if max_index == -1:
                max_index = num_rule_values-1
            # if min_index > 0:
            #     min_index -= 1
            if max_index < num_rule_values-1:
                max_index += 1
            for j in range(min_index, max_index+1):
                antecedent_rules[j] = 1

            all_rule_indexes.append(antecedent_rules)

        rules_indexes.append(all_rule_indexes)

    # korekcja reguł indeksowanych
    for i in range(num_of_antecedents):
        for j in range(num_rule_values):
            rule_activated = False
            for k in range(num_of_classes):
                if rules_indexes[k][i][j]:
                    rule_activated = True
                    break
            if not rule_activated:
                for k in range(num_of_classes):
                    if (j-1 >= 0 and rules_indexes[k][i][j-1] == 1) or (j+1 <= num_rule_values-1 and rules_indexes[k][i][j+1] == 1):
                        rules_indexes[k][i][j] = 1
    
    # for i in range(len(rules_indexes_by_hand)):
    #     print(rules_indexes_by_hand[i])
    #     print(rules_indexes[i], '\n')
    
    # for elem in rules_indexes_by_hand:
    #     print(elem)
    # for elem in rules_indexes:
    #     print(elem)

    # reguły FuzzyLogic
    rules = []
    for r_ind in range(num_of_classes):
        indexes = rules_indexes[r_ind]

        expr_parts = []
        for i in range(num_of_antecedents):
            rr = indexes[i]
            expr = None
            for j in range(num_rule_values):
                if rr[j]:
                    if expr is None:
                        expr = antecedents[i][rule_values[j]]
                    else:
                        expr = expr | antecedents[i][rule_values[j]]
            expr_parts.append(expr)


        expr = expr_parts[0]
        for i in range(1,len(expr_parts)):
            expr = expr & expr_parts[i]
        
        rules.append(ctrl.Rule(expr, consequent[str(r_ind+1)]))

    # zbudowanie systemu na regułach FL
    system = ctrl.ControlSystem(rules)

    # symulacja FL
    sim = ctrl.ControlSystemSimulation(system, flush_after_run=len(test)+1)
    
    # sprawdzenie poprawności FL
    check_fl_on_test_set(sim, test, mf_fun_params, antecedents_names, consequent_name)

    a = input('exit')
