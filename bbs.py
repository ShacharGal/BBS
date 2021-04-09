import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold


transformer = PCA()
# so this is the basic building block
# the rest of the functions or classes that will be included in the module
# will probably interact with this class somehow.
class BasisBrainSet:
    """receives train and test data, to create bbs features ....."""
    def __init__(self, num_components: int):
        self.num_components = num_components
        self.components: np.array
        self.train_features: np.array
        self.test_features: np.array

    def _decompose(self, data):
        transformer.fit(data)
        self.components = transformer.components_.T[:, :self.num_components]

    def extract_features(self, train_data, test_data):
        self._decompose(train_data) # is this legit?
        self.train_features = np.matmul(np.linalg.pinv(self.components), train_data.values.T).T
        self.test_features = np.matmul(np.linalg.pinv(self.components), test_data.values.T).T


# the idea here was to create a class that uses the BBS class for feature extraction
# and wraps it in the whole prediction process:
# a k-fold cross-validation process, where in each iteration we (1) extract features
# (2) fit a model to predict some target measure, and (3) save some parameters from steps 1+2

# im specifically disoriented as to the manner in which the BBS class should be used here:
# should it inherit its properties and expand on them? or should it use an instance of it..
# i started going down the inheritance road but fast enough i got confused.
class BBSPredictSingle(BasisBrainSet):
    def __init__(self, num_components:int, data:pd.DataFrame, target, folds:int, groups=None):
        BasisBrainSet.__init__(self, num_components)
        self.data = data
        self.target = target
        self.splitter = GroupKFold(n_splits=folds)
        self.groups = groups

        self.coefs: np.array
        self.predicted: np.aray
        
    def extract_features(self, train_data, test_data):
        return super().extract_features(train_data, test_data)
    
    def predict(self):
        for train_arr, test_arr in self.splitter.split(self.data, self.target, self.groups):
            bbs = BasisBrainSet()
            

# further functions/classes/stuff in this module would be another kind of prediction\classification process,
# similar to "BBSPredictSingle" but with some changes or added steps. these should have a similar API
# i would also want to have some function/class to wrap the prediction functions to perform
# a permutation test (doesnt really matter what it is, just that it needs to use the various prediction objects



