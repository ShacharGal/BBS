import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error
from typing import Optional, Iterable, Any
from scipy.stats import pearsonr
import warnings


glm = LinearRegression()
elastic = ElasticNetCV()
transformer = PCA()


def decompose(training_data, num_components: int) -> np.array:
    """do docstring"""
    transformer.fit(training_data)
    components = transformer.components_.T[:, :num_components]
    return components


def extract_features(train_data, test_data, components):
    """do docstring"""
    train_features = np.matmul(np.linalg.pinv(components), train_data.values.T).T
    test_features = np.matmul(np.linalg.pinv(components), test_data.values.T).T
    return train_features, test_features


class BBSpredict:
    """do docstring"""
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
        self.num_components = num_components
        self.model = model
        # this should probably be done differently?
        if groups == None:
            self.groups = np.arange(data.shape[0]) # each subject is a group, i.e, no grouping constrains on the splitter
        else:
            self.groups = groups

        # gilad what do you think about this? is this good practice?
        self._validate_inputs()

        self.coefs = []
        self.splits_ = {'train':[], 'test':[]}
        self.features = {'train':[], 'test':[]}
        self.components = []
        self.predicted = np.zeros(data.shape[0])

        self.stats = None # pearson r and mse for each iteration
        self.summary = None # mean and std for pearsons r and mse across folds
        self.contribution_map = None
        self.permutations_ = None

    def _validate_inputs(self):
        # "data" could be a dataframe or a list of dataframes. we validate for both
        if isinstance(self.data, pd.DataFrame) and self.data.shape[0] != len(self.target):
            warnings.warn("number of samples in data and target is not the same. please check your data.")
        if isinstance(self.data, list) and self.data[0].shape[0] != len(self.target):
            warnings.warn("number of samples in data and target is not the same. please check your data.")
        if len(self.target) != len(self.groups):
            warnings.warn("number of samples in target and 'groups' is not the same. please check your data.")
        if self.num_components > len(self.target):
            warnings.warn("requested number of components is larger than sample size. will result in error during pca")

    def calc_stats(self):
        """do docstring"""
        # this is a method i would want to run after self.predict() finished running
        # it should put values in self.stats and self.summary
        stats = {'r': [], 'mse': []}
        for fold in range(self.splitter.n_splits):
            real_values = self.target[self.splits_['test'][fold]]
            pred_values = self.predicted[self.splits_['test'][fold]]
            stats['r'].append(pearsonr(real_values, pred_values)[0])
            stats['mse'].append(mean_squared_error(real_values, pred_values))
        self.stats = pd.DataFrame(stats)
        self.summary = self.stats.describe().loc[['mean', 'std'], :]

    def get_permutations_pvalue(self):
        """do docstring"""
        n_perm = self.permutations_.shape[1]
        r_perm = np.zeros(n_perm)
        for perm in range(n_perm):
            r_perm[perm] = pearsonr(self.permutations[:, perm], self.target)[0]

        r_real = pearsonr(self.predicted, self.target)[0]
        # return the number of permutations equal or higher than the real results
        # (the real result is considered as a permutation as well, thus the "+1" in the numerator and denominator)
        return (np.sum(r_perm >= r_real) + 1) / (n_perm + 1)

    def permutation_test(self, permutation_num):
        """do docstring"""
        # first, check if "predict()" was even ran
        if sum(self.predicted) == 0:
            warnings.warn("first run predict() on the data")
        permutations = np.zeros((self.data.shape[0], permutation_num - 1))
        for fold in range(self.splitter.n_splits):
            y_train = self.target[self.splits_['train'][fold]].copy()
            train_features = self.features['train'][fold]
            test_features = self.features['test'][fold]
            for perm in range(permutation_num):
                np.random.shuffle(y_train)
                self.model.fit(train_features, y_train)
                permutations[self.splits_['test'][fold], perm] = self.model.predict(test_features)

        self.permutations_ = permutations
        return self.get_permutations_pvalue()

class BBSPredictSingle(BBSpredict):
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
        self.num_components = num_components
        self.model = model
        # this should probably be done differently?
        if groups == None:
            self.groups = np.arange(data.shape[0]) # each subject is a group, i.e, no grouping constrains on the splitter
        else:
            self.groups = groups

        # gilad what do you think about this? is this good practice?
        self._validate_inputs()

        self.coefs = []
        self.splits_ = {'train':[], 'test':[]}
        self.features = {'train':[], 'test':[]}
        self.components = []
        self.predicted = np.zeros(data.shape[0])

        self.stats = None # pearson r and mse for each iteration
        self.summary = None # mean and std for pearsons r and mse across folds
        self.contribution_map = None
        self.permutations_ = None

    def _validate_inputs(self):
        if self.data.shape[0] != len(self.target):
            warnings.warn("number of samples in data and target is not the same. please check your data.")
        if self.data.shape[0] != len(self.groups):
            warnings.warn("number of samples in data and 'groups' is not the same. please check your data.")
        if self.num_components > self.data.shape[0]:
            warnings.warn("requested number of components is larger than sample size. will result in error during pca")

    def predict(self):
        """do docstring"""
        for fold, (train_indices, test_indices) in enumerate(self.splitter.split(self.data, self.target, self.groups)):
            # get relevent data slices for train-test split
            X_train, X_test = self.data.iloc[train_indices, :], self.data.iloc[test_indices, :]
            Y_train, Y_test = self.target[train_indices], self.target[test_indices]
            # extract features
            self.components.append(decompose(X_train, self.num_components))
            train_features, test_features = extract_features(X_train, X_test, self.components[fold])
            # fit and predict
            self.model.fit(train_features, Y_train)
            self.predicted[test_indices] = self.model.predict(test_features)
            # save some stuff
            self.splits_['train'].append(train_indices)
            self.splits_['test'].append(test_indices)
            self.features['train'].append(train_features)
            self.features['test'].append(test_features)
            self.coefs.append(self.model.coef_)

        # i want to asses the prediction accuracy, which will be done by calc_stats method.
        # is this a good pythonic way to make sure stats are calculated after the prediction process is done?
        self.calc_stats()

        # this is a similar thing - i want this to run after prediction is done
        self.build_contribution_map()

    def build_contribution_map(self):
        """do docstring"""
        contribution_map = np.zeros((self.data.shape[1]))
        for fold in range(self.splitter.n_splits):
            # weight components with their related beta values
            weighted_task_comps = self.components[fold] * self.coefs[fold]
            # sum over components to get a single weighted map for this fold
            summed_weighted_task_comps = np.sum(weighted_task_comps, axis=1)
            # add the weighted components to the rest of the weighted components.
            contribution_map += summed_weighted_task_comps
        self.contribution_map = contribution_map

#class BBSpredictMulti(BBSPredictSingle):
