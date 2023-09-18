## Code for calculating evaluation metrics from classifier model results
## This code, given a csv file with CV predictions across genes for a given classifier, along with corresponding true values, will split data by each iteration (i.e. each group of CVs), and calculate average CV ROC, PRCs, and MCC for each iteration, which can then be plotted (mGSH_rocPlots.py).


##
## import modules
##

import glob
import os
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc, matthews_corrcoef

# for plotting and labelling
import re
import matplotlib.pyplot as plt
import seaborn as sns

# define function to get the ROC values from the predicted labels for a given model CV
def get_cvROC(cv_preds, true_labels, base_rate, method='roc'):

	# pred labels contain NaN for genes which were not used this CV, need to drop these
	preds = cv_preds.dropna()

	# get corresponding true labels
	trues = true_labels.loc[preds.index]

	# may be interested in getting ROC or PRC curves, methods are the same regardless
	if method =='roc':
		# now get the ROC values for this CV
		fpr, tpr, thresh = roc_curve(y_true=trues, y_score=preds)

		interp_tpr = np.interp(base_rate, fpr, tpr) # this will interpolate values to get a consistent number of values across CV folds for averaging
		interp_tpr[0] = 0 	# initial point at 0
		return interp_tpr

	# PRC values
	elif method == 'prc':
		prec, rec, thresh = precision_recall_curve(y_true=trues, probas_pred=preds)
		interp_prec = np.interp(base_rate[::-1], rec[::-1], prec[::-1]) 	# interpolate values over our base rate for consistency, need to reverse orders for proper interpolation
		interp_prec = interp_prec[::-1] 	# reverse order again to preserve plotting order

		return interp_prec

	# otherwise return error
	elif method != 'prc' or method != 'roc':
		raise ValueError(rf"\'method\' must be either \'prc\' or \'roc\', {method} was given.")


# define method to get MCC values for an array of predicted labels and true labels
def get_MCC(pred_scores, true_labels, threshold=0.5):

	# pred labels contain NaN for genes which were not used this CV, need to drop these
	relev_preds = pred_scores.dropna()

	# get corresponding true labels
	relev_trues = true_labels.loc[relev_preds.index]

	# MCC requires labels rather than probabilities, convert probs to labels by 'threshold'
	pred_labels = np.where(relev_preds >= threshold, 1, 0)

	# print(f'initial scores:\n{relev_preds}\nConverted to labels:\n{pred_labels}')

	# calculate and return mcc values
	return matthews_corrcoef(y_true=relev_trues, y_pred=pred_labels)


# define method for plotting curves, which calculates average y & std values, and can optionally plot individual curves
# Given all TPRs, create a ROC curve which plots all ROC curves
def plot_curve(all_ys, mean_baserate, plot_modelNames=False):

	# Need to combine all data into a single DF for plotting
	if plot_modelNames:
		roc_df = pd.DataFrame(data=all_ys, index=plot_modelNames, cols=mean_baserate)
	else:
		roc_df = pd.DataFrame(data=all_ys, cols=mean_baserate)
		
	# plot ROCs
	fig, ax = plt.subplots()

	'''
 	Palettes used in paper ROC curve, where "plot_modelNames" is like ['hybrid with 5 PC feats', 'hybrid with 14 PC feats', ..., 'transcriptomics-only with 30 PC feats', 'transcriptomics-only with 50 PC feats']
	mito palette: ["paleturquoise", "turquoise", "lightseagreen", "cadetblue", ...]
 	GSH palette: ["plum", "orchid", "darkviolet", "rebeccapurple"...]
  	transport palette: ["sandybrown", "orange", "darkorange", "peru", ...]

	NOTE: transcriptomics-only models actually have 1 or 2 extra PC components than what is specified, since the knowledge-based components are replaced with transcriptomics feats, as described in methods
   	'''
	
	sns.lineplot(data=roc_df.T, ax=ax, palette=["plum", "orchid", "darkviolet", "rebeccapurple", "gainsboro", "silver", "slategrey", "dimgray"], linewidth=1.75, dashes=False)
	
	
	plt.plot([0,100], [0,1], color='black', linestyle='--', label='50% chance')
	
	ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))
	
	plt.tight_layout()
	sns.move_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5))

	return fig, ax


# def main function for calculating evaluation metrics from model results
def main():

	# load data
	data_path = "path" 	## !!PLACEHOLDER!! replace with the path for your data directory
	f_nameRegex = ".+_Results_.+.csv" 	## !!PLACEHOLDER!! replace with a regex pattern for specific models to evaluate, the given Regex will match any 'Results' csv produced from 'run_classifierModel.py' containing test label predictions
	
	f_names = [f for f in os.listdir(data_path) if re.match(rf'{f_nameRegex}', f)] 	# get list of file name containing model results

	# Calculate evaluation metrics for each model in our list
	for model in f_names:
		data = pd.read_csv(rf'{results_path}\{model}', index_col=0) 	# load data as df

		# pull avg. predictions (not needed) and true labels, then split df into sub_dfs by # of CVs (5)
		data_T = data.T
		trueLabels = data_T.pop(data_T.columns[-1]) 	# pop last row which is True label
		predLabels = data_T.pop(data_T.columns[-1]) 	# pop the (now) last row which is avg. pred label
		data = data_T.T

		iter_dfs = np.array_split(data, len(data.index) / 5) 	# split data frame into n chunks, where n = total length/ number of CVs per iterations.

		iter_TPRs = []
		iter_precs = []
		iter_mccs = []

		mean_fpr = np.linspace(0, 1, 100) 	# this generates 100 values to use as common FPR values across CVs and iterations, this is necessary since the number of FPR/TPR values will vary
		for iteration in iter_dfs:

			# get ROC and PRC values for each CV (i.e. row)
			roc = iteration.apply(lambda x: get_cvROC(cv_preds=x, true_labels=trueLabels, base_rate=mean_fpr), axis=1)

			avg_tpr = roc.mean() 	# boostrap average values
			iter_TPRs.append(avg_tpr)


			# PRC values
			prc = iteration.apply(lambda x: get_cvROC(cv_preds=x, true_labels=trueLabels, base_rate=mean_fpr, method='prc'), axis=1)

			avg_prec = prc.mean() 	# boostrap average values
			iter_precs.append(avg_prec)

			# matthews correlation coefficient values for each
			mccs = iteration.apply(lambda x: get_MCC(pred_scores=x, true_labels=trueLabels), axis=1)

			avg_mcc = mccs.mean() 	# boostrap average values
			iter_mccs.append(avg_mcc)

		# Get average ROC/PRC values from our iteration values
		mean_TPR = np.array(iter_TPRs).mean(axis=0)
		mean_precision = np.array(iter_precs).mean(axis=0)

		mean_mcc = np.array(iter_mccs).mean(axis=0)
		print(f'~~~~~~~~~~~\n{model} model results:')
		print(f'avg. mcc:\n{mean_mcc}')

		mean_auc = auc(x=mean_fpr, y=mean_TPR)
		print(f'average AUROC over iterations: {mean_auc}')

		mean_auprc = auc(x=mean_fpr, y=mean_precision)
		print(f'average AUROC over iterations: {mean_auprc}')
		
		# Now can plot a ROC curve from these data
		roc_fig, roc_ax = plot_curve(all_ys=iter_TPRs, mean_baserate=mean_fpr, plot_modelNames=False)

		# Save fig
		roc_fig.savefig(rf'{folder_path}\ROC_classifier_models_1600dpi.png', bbox_inches='tight', dpi=1600)
		

# run main
main()
