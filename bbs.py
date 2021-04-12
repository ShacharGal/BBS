import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from typing import Optional, Iterable, Any
from collections import namedtuple
from sklearn.linear_model import LinearRegression, ElasticNetCV
from scipy.stats import pearsonr

glm = LinearRegression()
elastic = ElasticNetCV()
Features = namedtuple('Features', ['train_features', 'test_features'])
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
        self.features=[]
        self.components = []
        self.predicted = np.zeros(data.shape[0])
        self.scores_ = []
    
    def predict(self):
        """do docstring"""
        for fold, (train_arr, test_arr) in enumerate(self.splitter.split(self.data, self.target, self.groups)):
            # get relevent data slices for train-test split
            X_train, X_test = self.data.iloc[train_arr, :], self.data.iloc[test_arr, :]
            Y_train, Y_test = self.target[train_arr], self.target[test_arr]
            # extract features
            self.components.append(decompose(X_train, self.num_components))
            self.features.append(extract_features(X_train, X_test, self.components[fold]))
            # fit, predict and assess
            self.model.fit(self.features[fold].train_features, Y_train)
            self.predicted[test_arr] = self.model.predict(X_test)
            self.scores_.append(pearsonr(Y_test, self.predicted[test_arr])[0])



            
            
# further functions/classes/stuff in this module would be another kind of prediction\classification process,
# similar to "BBSPredictSingle" but with some changes or added steps. these should have a similar API
# i would also want to have some function/class to wrap the prediction functions to perform
# a permutation test (doesnt really matter what it is, just that it needs to use the various prediction objects