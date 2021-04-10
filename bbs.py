import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from typing import Optional, Iterable, Any
from collections import namedtuple

Features = namedtuple('Features', ['train_features', 'test_features'])

transformer = PCA()


def decompose(training_data, num_components: int) -> np.array:
    transformer.fit(training_data)
    components = transformer.components_.T[:, :num_components]
    return components


def extract_features(train_data, test_data, components) -> Features:
    train_features = np.matmul(np.linalg.pinv(components), train_data.values.T).T
    test_features = np.matmul(np.linalg.pinv(components), test_data.values.T).T
    return Features(train_features, test_features)


# so this is the basic building block
# the rest of the functions or classes that will be included in the module
# will probably interact with this class somehow.
# class BasisBrainSet:
#     """receives train and test data, to create bbs features ....."""
#     num_components: int
#     components: Optional[np.array]
#     train_features: Optional[np.array]
#     test_features: Optional[np.array]
#
#     def __init__(self, num_components: int):
#         self.num_components = num_components
#         self.components: np.array = None
#         self.train_features: np.array = None
#         self.test_features: np.array = None


# the idea here was to create a class that uses the BBS class for feature extraction
# and wraps it in the whole prediction process:
# a k-fold cross-validation process, where in each iteration we (1) extract features
# (2) fit a model to predict some target measure, and (3) save some parameters from steps 1+2

# im specifically disoriented as to the manner in which the BBS class should be used here:
# should it inherit its properties and expand on them? or should it use an instance of it..
# i started going down the inheritance road but fast enough i got confused.
class BBSPredictSingle:
    """
    This class is built from science stuff, and used to predict and display data.
    Used in this and that module.
    """
    data: pd.DataFrame
    target: Iterable
    splitter: Any
    groups: Iterable
    coefs: np.array
    predicted: np.array
    
    def __init__(self, num_components: int, data: pd.DataFrame, target: Iterable, folds: int, groups: Iterable = None):
        """
        Lol
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
    
    def predict(self):
        """
        Doees blabla and return this
        :return:
        """
        for train_arr, test_arr in self.splitter.split(self.data, self.target, self.groups):
            ...
            
            
# further functions/classes/stuff in this module would be another kind of prediction\classification process,
# similar to "BBSPredictSingle" but with some changes or added steps. these should have a similar API
# i would also want to have some function/class to wrap the prediction functions to perform
# a permutation test (doesnt really matter what it is, just that it needs to use the various prediction objects
