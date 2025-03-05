## Kennedy et al. 2024
#
# Code for calculating various metrics from classifier model results across dataset and diseases.

# This code, given a csv file with CV predictions across genes for a given classifier, along with 
# corresponding true values, will split data by each iteration (i.e. each group of CVs), and 
# calculate average accuracy, precision, recall, F1 score, Matthew's correlation coefficient for 
# each iteration.

# import modules
import glob
import os
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc, matthews_corrcoef, f1_score, accuracy_score, precision_recall_fscore_support

# for plotting and labelling
import re
import matplotlib.pyplot as plt
import seaborn as sns

# define method to get MCC values for an array of predicted labels and true labels
def get_MCC(pred_scores, true_labels, threshold=0.5):

	# pred labels contain NaN for genes which were not used this CV, need to drop these
	relev_preds = pred_scores.dropna()

	# get corresponding true labels
	relev_trues = true_labels.loc[relev_preds.index]

	# MCC requires labels rather than probabilities, convert probs to labels by 'threshold'
	pred_labels = np.where(relev_preds >= threshold, 1, 0)

	# calculate and return mcc values
	return matthews_corrcoef(y_true=relev_trues, y_pred=pred_labels)

# define method to get F1 and precision recall values for an array of predicted labels and true labels
def get_F1_prec_rec(pred_scores, true_labels, threshold=0.5):

	# pred labels contain NaN for genes which were not used this CV, need to drop these
	relev_preds = pred_scores.dropna()

	# get corresponding true labels
	relev_trues = true_labels.loc[relev_preds.index]

	# F1 requires labels rather than probabilities, convert probs to labels by 'threshold'
	pred_labels = np.where(relev_preds >= threshold, 1, 0)

	# calculate and return mcc values
	return precision_recall_fscore_support(y_true=relev_trues, y_pred=pred_labels, average='binary')


# define method to get accuracy values for an array of predicted labels and true labels
def get_accuracy(pred_scores, true_labels, threshold=0.5):

	# pred labels contain NaN for genes which were not used this CV, need to drop these
	relev_preds = pred_scores.dropna()

	# get corresponding true labels
	relev_trues = true_labels.loc[relev_preds.index]

	# accuracy requires labels rather than probabilities, convert probs to labels by 'threshold'
	pred_labels = np.where(relev_preds >= threshold, 1, 0)

	# calculate and return mcc values
	return accuracy_score(y_true=relev_trues, y_pred=pred_labels)


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

	'''
	Summary: this code loads all  results from our different model types (Both CCLE and TCGA) 
	(Note: No plotting or output file are produced by this code)
	'''

	# load data
	results_path = rf'outputs' 	# !!PLACEHOLDER!! path to our results folder containing all model outputs
	model_fs = [ r'[a-z]+_TCGA_SKCM_PCA_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv', r'[a-z]+_SKIN_transPC_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv', 
	 			r'[a-z]+_TCGA_PAAD_PCA_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv', r'[a-z]+_TCGA_LIHC_PCA_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv', 
				r'[a-z]+_LIVER_transPC_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv', r'[a-z]+_PANCREAS_transPC_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv', 
				r'[a-z]+_TCGA_allDisease_PCA_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv', 
				r'[a-z]+_ccleTranscriptomics_PCA_RF_[0-9]+_Results_[0-9]+-[0-9]+-[0-9]+.csv'
				]
	f_names = [f for f_name in model_fs for f in os.listdir(results_path) if re.match(f_name, f)]
	print(f'f_names:\n{f_names}')

	tprs, precs, mccs, f1s, accs = [], [], [], [], []
	for model in f_names:
		data = pd.read_csv(rf'{results_path}\{model}', index_col=0) 	# load data as df

		# pull avg. predictions (not needed) and true labels, then split df into sub_dfs by # of CVs (5)
		data_T = data.T
		trueLabels = data_T.pop(data_T.columns[-1]) 	# pop last row which is True label
		predLabels = data_T.pop(data_T.columns[-1]) 	# pop the (now) last row which is avg. pred label
		data = data_T.T

		iter_dfs = np.array_split(data, len(data.index) / 5) 	# split data frame into n chunks, where n = total length/ number of CVs per iterations.

		# Create empty lists for each metric
		iter_TPRs, iter_precs, iter_mccs, iter_f1s, iter_accs = [], [], [], [], []

		mean_fpr = np.linspace(0, 1, 100) 	# this generates 100 values to use as common FPR values across CVs and iterations, this is necessary since the number of FPR/TPR values will vary
		for iteration in iter_dfs:

			# matthews correlation coefficient values for each
			mcc = iteration.apply(lambda x: get_MCC(pred_scores=x, true_labels=trueLabels), axis=1)

			avg_mcc = mcc.mean() 	# boostrap average values
			iter_mccs.append(avg_mcc)

			scores =  iteration.apply(lambda x: get_F1_prec_rec(pred_scores=x, true_labels=trueLabels), axis=1)
			scores = np.array(scores)

			prc = np.array([iteration[0] for iteration in scores])
			recall = np.array([iteration[1] for iteration in scores])
			f1 = np.array([iteration[2] for iteration in scores])

			avg_prc = prc.mean()
			iter_precs.append(avg_prc)
			avg_recall = recall.mean() 	# boostrap average values
			iter_TPRs.append(avg_recall)
			avg_f1 = f1.mean() 	# boostrap average values
			iter_f1s.append(avg_f1)

			# matthews correlation coefficient values for each
			acc = iteration.apply(lambda x: get_accuracy(pred_scores=x, true_labels=trueLabels), axis=1)

			avg_acc = acc.mean() 	# boostrap average values
			iter_accs.append(avg_acc)

		# Get average ROC/PRC values from our iteration values
		mean_TPR = np.array(iter_TPRs).mean(axis=0)
		std_TPR = np.array(iter_TPRs).std(axis=0)

		# Compute other metrics
		mean_precision = np.array(iter_precs).mean(axis=0)
		std_precision = np.array(iter_precs).std(axis=0)

		mean_mcc = np.array(iter_mccs).mean(axis=0)
		std_mcc = np.array(iter_mccs).std(axis=0)

		mean_f1 = np.array(iter_f1s).mean(axis=0)
		std_f1 = np.array(iter_f1s).std(axis=0)

		mean_acc = np.array(iter_accs).mean(axis=0)
		std_acc = np.array(iter_accs).std(axis=0)

		# print outputs
		print(f'~~~~~~~~~~~\n{model} model results:')
		print(f'avg. precision:\n{mean_precision}, std: {std_precision}')
		print(f'avg. recall/TPR:\n{mean_TPR}, std: {std_TPR}')
		print(f'avg. f1:\n{mean_f1}, std: {std_f1}')
		print(f'avg. acc:\n{mean_acc}, std: {std_acc}')
		print(f'avg. mcc:\n{mean_mcc}, std: {std_mcc}')
		print(f'~~~\n')


		# Add to our full lists
		tprs.append([mean_TPR, std_TPR])
		precs.append([mean_precision, std_precision])
		mccs.append([mean_mcc, std_mcc])
		f1s.append([mean_f1, std_f1])
		accs.append([mean_acc, std_acc])


	dfs = []
	metrics = ['Precision', 'Recall', 'F-1 score', 'Accuracy', 'MCC']
	for i, metric in enumerate([precs, tprs, f1s, accs, mccs]):
		temp = pd.DataFrame(data=metric, index=f_names, columns=['Mean metric', 'STD'])
		print(f'temp: {temp}')

		vals = []
		for f_regex in model_fs:
			grouped = temp.index.str.contains(f_regex, regex=True)
			print(f'grouoped for: {f_regex}\n{grouped}')
			avg_val = temp.loc[grouped].mean(axis=0)
			print(f'avg_val.mean(): {avg_val}')
			vals.append(avg_val)

		temp = pd.DataFrame(data=vals, index=model_fs, columns=['Mean metric', 'STD'])
		temp.loc[:,'Metric'] = metrics[i]
		temp.loc[:,'Classifier'] = ['Mitochondrial', 'Transporter', 'Mitochondrial', 'Transporter', 'Mitochondrial', 'Transporter', 'Mitochondrial', 'Transporter']  	# for grouping together related models (i.e., hybrid and non-hybrids) in the plot
		temp.loc[:,'Model type'] = ['TCGA', 'TCGA', 'CCLE', 'CCLE', 'TCGA', 'TCGA', 'TCGA', 'TCGA']
		print(f'temp: {temp}')
		dfs.append(temp)

	metric_df = pd.concat(dfs)
	print(f'metric_df:\n{metric_df}')
	metric_df.to_csv(rf'RF_meanMetrics.csv')

	# plot the data:
	# Draw a nested barplot by classifier type and metric
	# Create a FacetGrid to handle different measurements
	g = sns.FacetGrid(metric_df, col='Classifier', sharey=False)

	# Map a barplot to each facet
	g.map_dataframe(sns.barplot, x='Model type', y='Mean metric', hue='Metric', errorbar='se', palette='colorblind')
	g.add_legend()

	g.despine(left=True)

	# We want to add error bars to each plot based on standard deviation of metrics over iterations.
	for i, ax in enumerate(g.axes.flat):
		x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
		y_coords = [p.get_height() for p in ax.patches]
		ax.set_ylim(0,1.0)

		# For each subplot, determine the error values to add to the bars
		n_values = len(metric_df)
		std = metric_df['STD']

		# Because of the ordering in the dataframe, selecting the appropriate error values will need to be done directly
		# Data is ordered as [model1_hybrid, model1_nonHybrid, model2_hybrid,...] for each metric. Thus, we want to grab the
		# nth and nth+1 values for each metric for our group.
		errs = []
		for j in range(len(metrics)):
			metric_idx = int(j*(n_values/len(metrics)))
			errs.append(metric_df['STD'].iloc[metric_idx])  	# hybrid metric
			errs.append(metric_df['STD'].iloc[metric_idx+1])  	# non-hybrid metric

		ax.errorbar(x=x_coords, y=y_coords, yerr=errs, fmt='none', c='k')

	g.set_axis_labels('Classifier', 'Performance metric', fontsize=14)
	g.savefig(rf'RF_performance_barplot.png', dpi=600, bbox_inches='tight')


	return

##############
#	RUN MAIN()
##############
main()