import numpy as np
import pandas as pd
import bbs
from sklearn.linear_model import LinearRegression, ElasticNetCV

glm = LinearRegression()
elastic = ElasticNetCV(l1_ratio=0.01, n_alphas=50, tol=0.001, max_iter=5000)

# read and mri data
orig_mat_1 = np.genfromtxt('../bbs_prediction/data/100_training/WM_09_s4_z_masked_orig.csv', delimiter=',')
orig_mat_2 = np.genfromtxt('../bbs_prediction/data/100_training/Lang_03_s4_z_masked_pred.csv', delimiter=',')
with open('../bbs_prediction/data/100_training/test_subjlist.txt', 'r') as f:
    subjects = [line.rstrip('\n') for line in f]
data_1 = pd.DataFrame(data=orig_mat_1, index=subjects)
data_2 = pd.DataFrame(data=orig_mat_2, index=subjects)

# read behavioral data
hcp_df = pd.read_csv('../bbs_prediction/hcp_dataframe_with_g.csv')
hcp_df['Subject'] = hcp_df['Subject'].apply(str)
hcp_df = hcp_df.set_index('Subject')
hcp_df = hcp_df[['g_efa']].dropna()

# make sure the same subjects appear in all dataframes
dfs, target = bbs.match_dfs_by_ind([data_1, data_2], hcp_df)

# test BBSPredictSingle
bbs_single = bbs.BBSPredictSingle(data=dfs[0], target=target['g_efa'], num_components=75,
                                  folds=10, model=glm)
bbs_single.predict()
print(bbs_single.summary)

# test BBSPredictMulti
bbs_multi = bbs.BBSpredictMulti(data=dfs, num_components=150, target=target['g_efa'], folds=10,
                                model=elastic, map_names=['wm', 'lang'], final_feature_number=150)
bbs_multi.predict()
print(bbs_multi.summary)

bbs_multi.permutation_test(100)
