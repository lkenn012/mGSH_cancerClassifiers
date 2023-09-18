### Code for constructing a histogram figure of TM-scores to compare local SLC25 structure alignments based on CAVER residues and full structure alignments
### TM-scores can be found in supplementary data file

##
## import modules
##
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

## Load our datasets for plotting

data_path = "path" 	## !! PLACEHOLDER !! replace this with the path to your data directory

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

# Now plot the the histogram
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(tmScores_df, x='TM-Score', hue='Group', palette=['mediumseagreen', 'rebeccapurple'], element='step', binwidth=0.05, binrange=[0,1], ax=ax)
ax.set_yscale('log')
fig.savefig(rf'{data_path}\tmScore_hist_log.png')
