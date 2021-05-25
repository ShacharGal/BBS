import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from typing import Iterable
from scipy.stats import pearsonr
import warnings
import pickle
import nibabel as nib
import sys

#TODO: think about saving a model and using it on an individual subject

transformer = PCA()


def from_pickle(pickle_path):
    """open a bbs object that was pickled"""
    with open(pickle_path, 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model


def to_pickle(bbs_object, pickle_path, with_data=False):
    """save a bbs object to pickle"""
    if not with_data:
        bbs_object.data = []
    with open(pickle_path, 'wb') as pickle_out:
        pickle.dump(bbs_object, pickle_out)


def match_dfs_by_ind(df_list, *target):
    """
    matches dataframes by their indices
    returns all dataframes with the intersection of all indices
    """
    all_indices = [list(df.index) for df in df_list]
    if target:
        traget_df = target[0] # its ok because if target exists, i expect only one value.
        all_indices.append(traget_df.index)
    in_all = list(set(all_indices[0]).intersection(*all_indices))
    if target:
        return [df[df.index.isin(in_all)] for df in df_list], traget_df[traget_df.index.isin(in_all)]
    else:
        return [df[df.index.isin(in_all)] for df in df_list]


def decompose(training_data, num_components: int) -> np.array:
    """
    returns a set number of principle components
    """
    transformer.fit(training_data)
    components = transformer.components_.T[:, :num_components]
    return components


def extract_features(train_data, test_data, components):
    """
    regresses components against individual data to yield components
    """
    train_features = np.matmul(np.linalg.pinv(components), train_data.values.T).T
    test_features = np.matmul(np.linalg.pinv(components), test_data.values.T).T
    return train_features, test_features


class BBSpredict:
    """the base class for bbs models"""
    def __init__(self, num_components: int, data: pd.DataFrame, target: Iterable, folds: int, model, groups: Iterable = None):
        """

        @param num_components: number of components to extract (will also be the number of features)
        @param data: a participantsXdata_size dataframe
        @param target: the score to predict, for all participants
        @param folds: number of cross-validation iterations by which to perform prediction
        @param model: an sklearn model object (such as LinearRegression)
        @param groups: information regarding dependence of samples (such as family relation between participants).
        """

        self.data = data
        self.target = target
        self.splitter = GroupKFold(n_splits=folds)
        self.num_components = num_components
        self.model = model

        if groups is not None:
            self.groups = groups
        else:
            self.groups = np.arange(len(self.target)) # each subject is a group, i.e, no grouping constrains on the splitter

        self._validate_inputs()

        self.coefs = []
        self.splits_ = {'train': [], 'test': []}
        self.features = {'train': [], 'test': []}
        self.components = []
        self.predicted = np.zeros(len(target))

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
        """calculate measurements of prediction quality"""
        stats = {'r': [], 'mse': []}
        for fold in range(self.splitter.n_splits):
            real_values = self.target[self.splits_['test'][fold]]
            pred_values = self.predicted[self.splits_['test'][fold]]
            stats['r'].append(pearsonr(real_values, pred_values)[0])
            stats['mse'].append(mean_squared_error(real_values, pred_values))
        self.stats = pd.DataFrame(stats)
        self.summary = self.stats.describe().loc[['mean', 'std'], :]

    def get_permutations_pvalue(self):
        """use the permutation data to yield significance of prediction"""
        n_perm = self.permutations_.shape[1]
        r_perm = np.zeros(n_perm)
        for perm in range(n_perm):
            r_perm[perm] = pearsonr(self.permutations_[:, perm], self.target)[0]

        r_real = pearsonr(self.predicted, self.target)[0]
        # return the number of permutations equal or higher than the real results
        # (the real result is considered as a permutation as well, thus the "+1" in the numerator and denominator)
        return (np.sum(r_perm >= r_real) + 1) / (n_perm + 1)

    def permutation_test(self, permutation_num):
        """perform permutation test to get recreate a null distribution of models"""
        # first, check if "predict()" was even ran
        if sum(self.predicted) == 0:
            warnings.warn("first run predict() on the data")
        # do permutation test
        permutations = np.zeros((len(self.target), permutation_num - 1))
        for fold in range(self.splitter.n_splits):
            y_train = self.target[self.splits_['train'][fold]].copy()
            train_features = self.features['train'][fold]
            test_features = self.features['test'][fold]
            for perm in range(permutation_num-1):
                np.random.shuffle(y_train)
                self.model.fit(train_features, y_train)
                permutations[self.splits_['test'][fold], perm] = self.model.predict(test_features)

        self.permutations_ = permutations
        return self.get_permutations_pvalue()


class BBSPredictSingle(BBSpredict):
    """
    a class to perform prediction using the bbs pipeline with a single map as an input for each participant
    """
    data: pd.DataFrame
    target: Iterable
    groups: Iterable

    coefs: list
    components: list
    predicted: np.array

    def predict(self):
        """predict individual traits from a single map for each participant"""
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

        self.calc_stats()
        self.build_contribution_map()

    def build_contribution_map(self):
        """create summed weighted maps that show each vertex's contribution to the prediction model"""
        contribution_map = np.zeros((self.data.shape[1]))
        for fold in range(self.splitter.n_splits):
            # weight components with their related beta values
            weighted_task_comps = self.components[fold] * self.coefs[fold]
            # sum over components to get a single weighted map for this fold
            summed_weighted_task_comps = np.sum(weighted_task_comps, axis=1)
            # add the weighted components to the rest of the weighted components.
            contribution_map += summed_weighted_task_comps
        self.contribution_map = contribution_map

    def save_contribution_map(self, filename):
        template = nib.load(f'{sys.path[0]}/misc/Smask.dtseries.nii')
        mask = np.asanyarray(template.dataobj)
        mask[mask == 1] = self.contribution_map
        to_save = nib.cifti2.cifti2.Cifti2Image(mask, template.header)
        nib.save(to_save, f'{filename}.dtseries.nii')



class BBSpredictMulti(BBSPredictSingle):
    """
    a class to perform prediction using the bbs pipeline with multiple maps as an input for each participant
    """

    data: list # list of dataframes
    target: Iterable
    groups: Iterable

    coefs: list
    components: list
    predicted: np.array

    def __init__(self, num_components: int, data: list, target: Iterable, folds: int, model, final_feature_number: int, groups: Iterable = None, map_names = None):
        """
        @param num_components: number of components to extract from each data-set in each fold
        @param data: a list of participantsXmap_size dataframes.
        @param target: the score to predict, for all participants
        @param folds: number of cross-validation iterations by which to perform prediction
        @param model: an sklearn model object (such as LinearRegression or ElasticNet)
        @param final_feature_number:
        @param groups: information regarding dependence of samples (such as family relation between participants).
        @param map_names: for documentation's sake
        """
        super().__init__(num_components, data, target, folds, model, groups)
        self.final_feature_number = final_feature_number
        self.components = [[] for map in range(len(self.data))]  # each map's components wil be added to a specific list
        self.map_names = map_names  # for documentation sake. should be a list of strings. optional
        self.masks_per_fold = []

    def corr_analysis(self, features, target):
        feat_num = features.shape[1]
        corrs = np.zeros(feat_num)
        for feat in range(feat_num):
            corrs[feat] = pearsonr(features[:, feat], target)[0]
        mask = abs(corrs) >= np.sort(abs(corrs))[feat_num - self.final_feature_number]
        return mask

    def predict(self):
        """predict individual traits from multiple maps for each participant"""
        for fold, (train_indices, test_indices) in enumerate(self.splitter.split(self.data[0], self.target, self.groups)):
            train_features = np.zeros([len(train_indices), self.num_components * len(self.data)])
            test_features = np.zeros([len(test_indices), self.num_components * len(self.data)])
            Y_train, Y_test = self.target[train_indices], self.target[test_indices]

            for i in range(len(self.data)):
                curr_data = self.data[i]

                # get relevant data slices for train-test split
                X_train, X_test = curr_data.iloc[train_indices, :], curr_data.iloc[test_indices, :]
                # extract features
                self.components[i].append(decompose(X_train, self.num_components))
                curr_train_features, curr_test_features = extract_features(X_train, X_test, self.components[i][fold])
                start = i * self.num_components
                end = start + self.num_components
                train_features[:, start:end] = curr_train_features
                test_features[:, start:end] = curr_test_features

            mask = self.corr_analysis(train_features, Y_train)
            self.masks_per_fold.append(mask)

            final_train_features = train_features[:, mask]
            final_test_features = test_features[:, mask]

            self.model.fit(final_train_features, Y_train)
            self.predicted[test_indices] = self.model.predict(final_test_features)
            # save some stuff
            self.splits_['train'].append(train_indices)
            self.splits_['test'].append(test_indices)
            self.features['train'].append(final_train_features)
            self.features['test'].append(final_test_features)
            self.coefs.append(self.model.coef_)

        self.calc_stats()