### Functions to be used in classifier code to format data, build, run and, evaluate classifier models


##
## import modules
##

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr, distributions

##
## Define methods for classifier models
##

# define function to get p-val from spearman corr when correlatin transcripts (more efficient method, such as calling spearmanr() directly?)
def spearmanr_pval(x,y):
	return spearmanr(x,y)[1]


# define function for thresholding spearman correlations by significance; If pval > {pval_threshold}, replace corr value with 0
def threshold_corr(corr_df, pval_df, pval_threshold=0.05):

	threshold_df = corr_df.where(pval_df.values < pval_threshold, other=0)

	return threshold_df


# define function to fisher (i.e. arctanh) transform correlation rho values for normalization.
# used in feat_preprocess()
def fisher(data):
	return np.arctanh(data) 

