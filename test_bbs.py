import numpy as np
import pandas as pd
import bbs

# read in data

orig_mat = np.genfromtxt('../bbs_prediction/data/100_training/WM_09_s4_z_masked_pred.csv', delimiter=',')
with open('../bbs_prediction/data/100_training/test_subjlist.txt', 'r') as f:
    subjects = [line.rstrip('\n') for line in f]
data = pd.DataFrame(data=orig_mat, index=subjects)

hcp_df = pd.read_csv('../bbs_prediction/hcp_dataframe_with_g.csv')
hcp_df['Subject'] = hcp_df['Subject'].apply(str)
hcp_df = hcp_df.set_index('Subject')
hcp_df = hcp_df[['g_efa']]
hcp_df = hcp_df.dropna()

all_indices = [list(data.index), list(hcp_df.index)]
in_all = list(set(all_indices[0]).intersection(*all_indices))
hcp_df = hcp_df[hcp_df.index.isin(in_all)]
data = data[data.index.isin(in_all)]

bbs_model = bbs.BBSPredictSingle(data=data, target=hcp_df['g_efa'], num_components=75, folds=10)
bbs_model.predict()
print(bbs_model.summary)