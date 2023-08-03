## Code for calculating ROC and PRC curves from classifier model results
## This code, given a csv file with CV predictions across genes for a given classifier, along with corresponding true values, will split data by each iteration (i.e. each group of CVs), and calculate average CV ROC and PRCs for each iteration, which can then be plotted.

# import modules
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
#		print(f'precision:\n{prec}, length: {len(prec)}')
#		print(f'recall:\n{rec}, length: {len(rec)}')
		interp_prec = np.interp(base_rate[::-1], rec[::-1], prec[::-1]) 	# interpolate values over our base rate for consistency, need to reverse orders for proper interpolation
		interp_prec = interp_prec[::-1] 	# reverse order again to preserve plotting order
# 		print(f'interpolated precision:\n{interp_prec}, length: {len(interp_prec)}')

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
def plot_curve(all_ys, mean_baserate, plot_all=False, plot_std=False, plot_color='blue'):

	# first, given an array of all y (TPR or Precision) values, calculate maverage values for plotting
	mean_y = np.mean(all_ys, axis=0)
	if plot_std:
		mean_std = np.std(all_ys, axis=0)

	# Also get average AUC for adding to plot
	mean_auc = auc(x=mean_baserate, y=mean_y)

	# define plot params:
	sns.set_theme(style='dark')
	individ_color = sns.set_hls_values(plot_color, l=0.75)

	# plot individual lines, if desired
	if plot_all:
		for y in all_ys:
			plt.plot(mean_baserate, y, color=individ_color, alpha=0.75, linewidth=0.3)

	# plot mean curve and optionally std
	plt.plot(mean_baserate, mean_y, color=plot_color, label=f'average curve (AUC = {"{0:0.3f}".format(mean_auc)})')

	if plot_std:
		plt.fill_between(mean_baserate, mean_y - mean_std, mean_y + mean_std, color=plot_color,alpha=0.3)
	
	return

# def main
def main():

	# load data
	# path = r'C:\Users\lkenn\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code'
	path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code' 	# laptop path

	results_path = rf'{path}\outputs' 	# path to our results folder
	f_names = [f for f in os.listdir(results_path) if re.match(r'(?!ROC)+SVM.+_highCon_Results_26-06.+.csv', f)]
	print(f'f_names:\n{f_names}')

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

			# # get ROC and PRC values for each CV (i.e. row)
			# roc = iteration.apply(lambda x: get_cvROC(cv_preds=x, true_labels=trueLabels, base_rate=mean_fpr), axis=1)

			# avg_tpr = roc.mean() 	# boostrap average values
			# iter_TPRs.append(avg_tpr)


			# PRC values
			prc = iteration.apply(lambda x: get_cvROC(cv_preds=x, true_labels=trueLabels, base_rate=mean_fpr, method='prc'), axis=1)

			avg_prec = prc.mean() 	# boostrap average values
			iter_precs.append(avg_prec)

			# # matthews correlation coefficient values for each
			# mccs = iteration.apply(lambda x: get_MCC(pred_scores=x, true_labels=trueLabels), axis=1)

			# avg_mcc = mccs.mean() 	# boostrap average values
			# iter_mccs.append(avg_mcc)

		# Get average ROC/PRC values from our iteration values
		# mean_TPR = np.array(iter_TPRs).mean(axis=0)
		# print(f'avg. tprs:\n{mean_TPR}')
		mean_precision = np.array(iter_precs).mean(axis=0)
		# print(f'avg. precision:\n{mean_precision}')

		# mean_mcc = np.array(iter_mccs).mean(axis=0)
		print(f'~~~~~~~~~~~\n{model} model results:')
		# print(f'avg. mcc:\n{mean_mcc}')

		# mean_auc = auc(x=mean_fpr, y=mean_TPR)
		# print(f'average AUROC over iterations: {mean_auc}')

		mean_auprc = auc(x=mean_fpr, y=mean_precision)
		print(f'average AUROC over iterations: {mean_auprc}')

		# # print(f'\n~~~~~~~~~~~\n')
		# roc_df = pd.DataFrame(data=[mean_fpr, mean_TPR], index= ['FPR', 'TPR'])
		# roc_df.loc['mean AUC'] = mean_auc
		# print(f'roc_df:\n{roc_df}')
		# roc_df.to_csv(rf'{path}\outputs\ROC_{model}.csv')

		# # now can plot ROC/PRC curves
		# roc_ax = plot_curve(all_ys=iter_TPRs, mean_baserate=mean_fpr, plot_all=False, plot_std=True, plot_color='rebeccapurple') 	# From our individual ROC values, can plot all these data on a curve

		# # add chance line
		# plt.plot([0.01, 0.99], [0.01, 0.99], color='black', linestyle='--', linewidth=1, label='Random chance') 	# add 50% line
		# lines, labels = plt.gca().get_legend_handles_labels()
		# plt.legend(lines, labels) 	# add legend to plot

		# model_name = re.search(r"[^_]*_[^_]*_[^_]*_[^_]*_", model).group() 	# this regex contains all model information
		# plt.title(f'{model_name} ROC')	
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')

		# plt.savefig(rf'{path}\outputs\figures\{model_name}_ROC2.png')
		# plt.clf()

		# prc_ax = plot_curve(all_ys=iter_precs, mean_baserate=mean_fpr, plot_all=True, plot_std=True, plot_color='rebeccapurple')

		# # add 50% line
		# plt.plot([0.01, 0.99], [0.50, 0.50], color='black', linestyle='--', linewidth=1, label='Random chance') 	# this line corresponds to that off a randomly guessing classifier (0.50 is used since that is the proportion of true positives)
		# lines, labels = plt.gca().get_legend_handles_labels()
		# plt.legend(lines, labels) 	# add legend to plot


		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')

		# plt.title(f'{model_name} PRC')
		# plt.savefig(rf'{path}\outputs\figures\{model_name}_PRC.png')
		# plt.clf()

# Data of shape cols = [ENSG00000137275, ENSG00000181610, ..., Test Genes] and rows = [iter 1 CVs, iter 2 CVs, .... iter n CVs, avg. predicted label, true label]
# For calculating ROC/PRC values, need to split rows to contain CV preds for each iteration, and a series for true labels
# Since columns correspond to test genes across ALL iterations, many genes will hve no predictions for a given iteration (because negative genes are randomly selected). Need to remove all empty values
# probably most sense to split "iteration DF" into series for each CV, then remove all missing values for that CV and calculate ROC/PRC values.
# Then get average ROC/PRC values for each iteration + AUROC/AUPRC
# Maybe save these data into sheets of excel file? So {Model}_ROCPRC.xlsx, with rows = TPR,FPR,Thresh,AUC for each iteration

# Plot individual ROC/PRC curves (small linewidth + high alpha), then calculate average TPRs, FPRs, AUCs, STD and plot averge ROC/PRC + STD

# main function for loading ROC files and plotting
def main_2():

	folder_path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs'
	rocFile_names = [f for f in os.listdir(folder_path) if re.match(r'ROC_RF_.+GSH_highCon_Results.+.csv', f)]


	# now can plot ROC/PRC curves
	roc_ax = plot_curve(all_ys=iter_TPRs, mean_baserate=mean_fpr, plot_all=False, plot_std=True, plot_color='rebeccapurple') 	# From our individual ROC values, can plot all these data on a curve

	# add chance line
	plt.plot([0.01, 0.99], [0.01, 0.99], color='black', linestyle='--', linewidth=1, label='Random chance') 	# add 50% line
	lines, labels = plt.gca().get_legend_handles_labels()
	plt.legend(lines, labels) 	# add legend to plot

	model_name = re.search(r"[^_]*_[^_]*_[^_]*_[^_]*_", model).group() 	# this regex contains all model information
	plt.title(f'{model_name} ROC')	
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	plt.savefig(rf'{path}\outputs\figures\{model_name}_ROC2.png')
	plt.clf()

# run main
main()