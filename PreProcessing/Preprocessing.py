import numpy as np
import pandas as pd
import random
from collections import defaultdict


class GridSearch:

    def __init__(self, model_function, cost_function, train_model, param_grid):
        self.__model_function = model_function
        self.__cost_function = cost_function
        self.__train_model = train_model
        self.__param_grid = param_grid
        self.__estimator = None
        self.__best = {}

    def fit(self, theta, X, y, *, scoring=None, best=False):
        estimator_dict = defaultdict(list)
        items = make_pairs(self.__param_grid.items())
        param_grid = []
        for item in items:
            param_grid = combi_element_it(param_grid, item)
        for params_set in param_grid:
            dict_param = dict(params_set)
            print("dict_param", dict_param)
            theta, costs = self.__train_model(theta, X, y, **dict_param, cost_return=True)
            estimator_dict['tested_parameters'].append(dict_param)
            print("theta shape ", theta.shape)
            estimator_dict['theta'].append(theta)
            estimator_dict['cost'].append(costs[-1])
            if scoring:
                score = None
                estimator_dict['scoring'].append(score)
            else:
                estimator_dict['scoring'].append(costs[-1])
            if best:
                if scoring:
                    if score >= max(estimator_dict['scoring']):
                        estimator_dict['best'] = {'best_params': dict_param,
                                                  'best_theta': theta,
                                                  'best_score': score}
                else:
                    if costs[-1] <= min(estimator_dict['cost']):
                        estimator_dict['best'] = {'best_params': dict_param,
                                                  'best_theta': theta,
                                                  'best_score': costs[-1]}
                        self.__best = estimator_dict['best']
        self.__estimator = estimator_dict

    def gridsearch_best(self):
        if self.__best:
            return self.__best
        else:
            raise "best option was set to false, hence best parameters were not recorded"

    def gridsearch_estimator(self):
        if self.__estimator:
            return self.__estimator
        else:
            raise "Estimator has not been initialized yet. Please check you've fit the model"


def split_data(data, percentage, target_loc):  # we could also use a target name
    nb_rows = int(len(data) * (percentage / 100))
    train_target = data.iloc[:nb_rows, target_loc]
    test_target = data.iloc[nb_rows:, target_loc]
    data = data.drop(data.columns[[target_loc]], axis=1)
    train_set = data.iloc[:nb_rows]
    test_set = data.iloc[nb_rows:]
    return train_set, test_set, train_target, test_target


def add_ones(array):  # maybe it would be better to add an optional parameter giving the possibility to work inplace
    if type(array) == pd.DataFrame:
        array_copy = array.copy()
        array_copy.insert(0, "Intercept", np.ones(len(array)), True)
        return array_copy
    else:
        return np.array([np.ones(len(array)), array])


def one_hot_encoding(data, columns_names, replace=False):
    new_array = pd.DataFrame()
    for name in columns_names:
        categories = data[name].unique()
        if replace:
            for category in categories:
                new_column = (data[name] == category).astype('float64')
                new_array[category] = new_column
                data[category] = new_column
            data.drop(columns=name, inplace=True)
        else:
            for category in categories:
                new_array[category] = data[name] == category
    return new_array


def dummy_encoding(data, columns_name, replace=False):
    if replace:
        new_array = one_hot_encoding(data, columns_name, replace=True)
        dropped_columns = [columns_name[i] for i in random.sample(range(len(new_array)), len(columns_name))]
        data.drop(columns=dropped_columns, inplace=True)
        new_array.drop(columns=dropped_columns, inplace=True)
    else:
        new_array = one_hot_encoding(data, columns_name, replace=False)
    return new_array


def combi_element_it(elements, iterable):
    if not iterable:
        return elements
    else:
        if not elements:
            elements = [()]
        new_elements = []
        for element in elements:
            for element_it in iterable:
                new_element = element + (element_it,)
                new_elements.append(new_element)
        return new_elements


def make_pairs(iter_dict):  # for each dict key, creates a tuple (key, element) for each element in the iterable value
    # associated with the key in the dict
    new_iter_dict = []
    for element in iter_dict:
        new_iter_dict.append([(element[0], element[1][i]) for i in range(len(element[1]))])
    return new_iter_dict