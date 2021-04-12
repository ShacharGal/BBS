import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error
from typing import Optional, Iterable, Any
from collections import namedtuple
from scipy.stats import pearsonr


glm = LinearRegression()
elastic = ElasticNetCV()
Features = namedtuple('Features', ['train_features', 'test_features'])
Splits = namedtuple('Features', ['train_indices', 'test_indices'])

transformer = PCA()


def decompose(training_data, num_components: int) -> np.array:
    """do docstring"""
    transformer.fit(training_data)
    components = transformer.components_.T[:, :num_components]
    return components


def extract_features(train_data, test_data, components) -> Features:
    """do docstring"""
    train_features = np.matmul(np.linalg.pinv(components), train_data.values.T).T
    test_features = np.matmul(np.linalg.pinv(components), test_data.values.T).T
    return Features(train_features, test_features)


class BBSPredictSingle:
    """do docstring"""
    data: pd.DataFrame
    target: Iterable
    groups: Iterable

    coefs: list
    components: list
    predicted: np.array

    def __init__(self, num_components: int, data: pd.DataFrame, target: Iterable, folds: int, groups: Iterable = None, model = glm):
        """
        :param num_components: hi hi
        :param data:
        :param target:
        :param folds:
        :param groups:
        """
        self.data = data
        self.target = target
        self.splitter = GroupKFold(n_splits=folds)
        self.groups = groups
        self.num_components = num_components
        self.model=model
        # perhaps add some test (e.g.,if data, target and groups have the same number of observations)

        self.coefs = []
        self.splits_ = []
        self.features = []
        self.components = []
        self.predicted = np.zeros(data.shape[0])
        self.stats = None # pearson r and mse for each iteration
        self.summary = None # mean and std for pearsons r and mse across folds

    
    def predict(self):
        """do docstring"""
        for fold, (train_indices, test_indices) in enumerate(self.splitter.split(self.data, self.target, self.groups)):
            self.splits_.append(Splits(train_indices, test_indices))
            # get relevent data slices for train-test split
            X_train, X_test = self.data.iloc[train_indices, :], self.data.iloc[test_indices, :]
            Y_train, Y_test = self.target[train_indices], self.target[test_indices]
            # extract features
            self.components.append(decompose(X_train, self.num_components))
            self.features.append(extract_features(X_train, X_test, self.components[fold]))
            # fit and predict
            self.model.fit(self.features[fold].train_features, Y_train)
            self.predicted[test_indices] = self.model.predict(X_test)

        # i want to asses the prediction accuracy, which will be done by calc_stats method.
        # is this a good pythonic way to make sure stats are calculated after the prediction process is done?
        self.calc_stats(self)

    def calc_stats(self):
        # this is a method i would want to run after self.predict() finished running
        # it should put values in self.stats and self.summary
        stats = {'r': [], 'mse': []}
        for fold in self.splits_:
            real_vals = self.target[fold.test_indices]
            pred_vals = self.predicted[fold.test_indices]
            stats['r'].append(pearsonr(real_vals, pred_vals)[0])
            stats['mse'].append(mean_squared_error(real_vals, pred_vals))
        self.stats = pd.DataFrame(stats)
        self.summary = self.stats.describe().loc[['mean', 'std'], :]
