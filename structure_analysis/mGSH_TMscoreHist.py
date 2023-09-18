### Code for constructing a histogram figure of TM-align scores for

# import modules
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

## Load our datasets for plotting

data_path = rf'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs'

local_df = pd.read_csv(rf'{data_path}\local_TMalign.csv', index_col=0)
global_df = pd.read_csv(rf'{data_path}\global_TMalign.csv', index_col=0)

# for our plot, we have two desired datasets: local TM-scores for alignments of SLC25 tunnel residues, and global TM-score for whole SLC25 structure alignments
# Simplest method for plotting is to combine these dfs into a single df with a group label to sepcify the source dataset (e.g., 'Global' and 'Local')

local_flat = pd.DataFrame(data=local_df.to_numpy().ravel(), columns=['TM-Score']) 	# get flattened DFs
global_flat = pd.DataFrame(data=global_df.to_numpy().ravel(), columns=['TM-Score'])

local_flat['Group'] = 'Local' 	# add group label column
global_flat['Group'] = 'Global'

# Now we have them in any easy format for concatting and plotting
tmScores_df = pd.concat([local_flat, global_flat])
tmScores_df.reset_index(inplace=True, drop=True)
print(f'tmScores_df:\n{tmScores_df}')

# Now plot the the histogram
fig, ax = plt.subplots(figsize=(10, 6))
# hist = sns.displot(tmScores_df.iloc[:250,:], x='TM-Score', hue='Group', kind='hist', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
sns.histplot(tmScores_df, x='TM-Score', hue='Group', palette=['mediumseagreen', 'rebeccapurple'], element='step', binwidth=0.05, binrange=[0,1], ax=ax)
ax.set_yscale('log')
fig.savefig(rf'{data_path}\tmScore_hist_log.png')