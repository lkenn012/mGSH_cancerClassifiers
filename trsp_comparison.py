## This code plots mGSH mitochondrial prediction as function of MitoCarta Scores
## We want to compare out models classifications to those of MitoCarta, and identify any outlier predictions

# import modules
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import math

import mGSH_classifierComparisons as classifer_tests
import scipy.stats as stats

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
import re

# define function to get model predicted probabilities and corresponding MitoCarta scores for a list of genes

# define function to calculate to convert MitoCarta logOdds to probabilities (probability = exp(logOdds) / (1 - exp(logOds)))
def logOdds2prob(odds):
	print(f'odds: {odds}')
	return math.exp(odds)/(1-math.exp(odds)) 



# Define function to plot mdoel results as scatter plot
def plot_Results(results):

	# plot all data on one plot
	fig, ax = plt.subplots(1,1)
	ax = sns.scatterplot(ax=ax,
		data=mod_results, 
		x='TrSSP prediction', 
		y='Predicted label', 
		hue='True label', # groups
		palette=["rebeccapurple","darkkhaki"], # group color
		linewidth=0, # dot outline
		alpha=0.75) 	# dot transparency

	# # Formatting
	# ax.set(title='Mitochondrial classification probability vs. MitoCarta2.0 score', ylabel='Mitochondrial classification probability', xlabel='MitoCarta2.0 probability score')

	# # saved file
	# sns.move_legend(ax, bbox_to_anchor=(1.0, 1), loc='upper left') 	# move legend

	# # plot each group as subplot
	# mito_df = mod_results.loc[mod_results['True label'] == 'Mito']
	# nonMito_df = mod_results.loc[mod_results['True label'] == 'Non-Mito'] 	# get grouped data
	
	# fig, axes = plt.subplots(1,2)

	# sns.scatterplot(ax=axes[0],
	# 	data=mito_df,
	# 	x='MitoCarta2.0_Score',
	# 	y='Predicted label', 
	# 	color='darkcyan',
	# 	linewidth=0,
	# 	alpha=0.75,
	# 	label='Mito')

	# sns.scatterplot(ax=axes[1],
	# 	data=nonMito_df,
	# 	x='MitoCarta2.0_Score',
	# 	y='Predicted label', 
	# 	color='darkkhaki',
	# 	linewidth=0,
	# 	alpha=0.75,
	# 	label='Non-Mito')


# Define function to compute and then create a confusion matrix plot
def confusion_plot(pred_labels, true_labels, plot_ax, colorbar=True, confusion_palette=sns.color_palette("light:b", as_cmap=True)):

	confusion = confusion_matrix(y_true=true_labels, y_pred=pred_labels) 	# get confusion matrix

	confusion_df = pd.DataFrame(data=confusion, index=['Non-Transporter', 'Transporter'], columns= ['Non-Transporter', 'Transporter'])
	print(f'confusion_df:\n{confusion_df}')

	return sns.heatmap(confusion_df, 
		cmap=confusion_palette,
		cbar=colorbar,
		fmt='g', 
		annot=True, 
		ax=plot_ax, 
		vmin=0, 
		vmax=len(true_labels) # specify max color value as # of samples
		) 	# return a plot of the confusion matrix

# define function to get several resuolts files and take average predicted label
def get_meanResults(results_path, fName_regex):
	f_names = [f for f in os.listdir(rf'{results_path}') if re.match(fName_regex, f)]
	print(f'f_names:\n{f_names}')
	all_results = []
	for name in f_names:
		res_df = pd.read_csv(rf'{results_path}\{name}', index_col=0)
		all_results.append(res_df.T['Average predicted label'])

	print(f'res_df:\n{res_df}')
	avg_df = pd.DataFrame(data=all_results)
	print(f'avg_df:\n{avg_df}')
	mean_label = avg_df.mean(axis=0)
	print(f'mean_labels:\n{mean_label}')
	mean_df = mean_label.to_frame()
	mean_df['True label'] = res_df.loc['True label']
	print(f'mean_df:\n{mean_df}')

	return mean_df


# define main
def main():
	
	# load data
	# path = r'C:\Users\lkenn\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code'
	path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code' 	# laptop path

	f_name = r'(?!ROC)+RF_[0-9]+_transp_highCon_Results_20-06.+.csv'
	mod_results = get_meanResults(results_path=rf'{path}\outputs', fName_regex=f_name)
	print(f'mod_results:\n{mod_results}')

	# f_name = 'RF_14_noTrSSP_transp_highCon_Results_19-06-2023' 	# transporter model results

	TrSSPMito_annots = pd.read_csv(rf'{path}\data\mitocarta_trssp_scores.csv', index_col=0) 	# load DF containing IDs & cols for mitocarta, trssp scores

	trssp_scores = TrSSPMito_annots.loc[:,'TrSSP prediction'] 	# get mitocarta scores
	print(f'TrSSP scoresa:\n{trssp_scores}')
	# mod_results = pd.read_csv(rf'{path}\outputs\{f_name}.csv', index_col=0) 	# load model results
	# mod_results = mod_results.T

	# mod_results.drop('ENSG00000182446', axis=0, inplace=True) 	# This ID has missing value, remove

	# get the mitocarta scores for genes we have probabilities for
	pred_mitocarta = trssp_scores.loc[mod_results.index]
	print(f'TrSSP scores for model:\n{pred_mitocarta}')

	mod_results['TrSSP prediction'] = trssp_scores.loc[mod_results.index]
	print(f'trssp value counts:\n{mod_results["TrSSP prediction"].value_counts()}')
	print(f'true value counts:\n{mod_results["True label"].value_counts()}')

	# mod_results['True label'] = np.where(mod_results['True label'] == 1.0, 'Transporter', 'Non-Transporter')

	# mod_results['MitoCarta2.0_Score'] = mod_results['MitoCarta2.0_Score'].apply(lambda x: logOdds2prob(x)) 	# convert logOdds scores to probabilities (error: float division by zero)
	# print(f'full mod results:\n{mod_results}')

	# mito_df = mod_results.loc[mod_results['True label'] == 'Mito']
	# nonMito_df = mod_results.loc[mod_results['True label'] == 'Non-Mito']

	# ## Plot results
	# plot_Results(results=mod_results)

	# # Formatting
	# plt.title('Transporter classification probability vs. TrSSP score')
	# plt.ylabel('Transporter classification probability')

	# # saved file
	# handles, labels = ax.get_legend_handles_labels()
	# fig.legend(handles, labels, loc='upper left')

	# plt.savefig(f'TrSSP_vs_mod_{f_name}_group.png', bbox_inches='tight')


	print(f'model results:\n{mod_results}')

	# ## Comparing our classifier model vs. TrSSP predictions using independence tests

	# Replace class probabilities with class labels
	mod_results['Predicted label'] = np.where(mod_results.iloc[:,0] >= 0.75, 1, 0) 	# convert to class labels with threshold =0.50

	print(f'model results:\n{mod_results}')

	results_df = mod_results.loc[mod_results['TrSSP prediction'] != -1] 	# drop rows where there is no TrSSP prediction (= -1)

	print(f'final mod results:\n{results_df}')

	# Now want to calculate the chi-squared indepence test of our values
	contingency = classifer_tests.get_contingency(results_df, 'Predicted label', 'TrSSP prediction', 'True label')
	print(f'contingency:\n{contingency}')
	print(f'Contingency table:\n\tClassifier Wrong, Classifier right')
	print(f'TrSSP wrong: {contingency[0][0]} \t|\t {contingency[0][1]}')
	print(f'TrSSP right: {contingency[1][0]} \t|\t {contingency[1][1]}')

	chi2_results = classifer_tests.chi2_test(results_df, 'Predicted label', 'TrSSP prediction', 'True label')

	print(f'chi2 statistic: {chi2_results[0]}, chi2 p-value: {chi2_results[1]}')

	# Also try t-test
	ttest_results = stats.ttest_ind(results_df['Predicted label'], results_df['TrSSP prediction'])

	print(f'T-test statistics:')
	print(f'Classifier vs. TrSSP:\nt-test statistic: {ttest_results[0]}, t-test p-value: {ttest_results[1]}\n')

	ttest_results = stats.ttest_ind(results_df['Predicted label'], results_df['True label'])

	print(f'Classifier vs. GO terms:\nt-test statistic: {ttest_results[0]}, t-test p-value: {ttest_results[1]}\n')

	ttest_results = stats.ttest_ind(results_df['TrSSP prediction'], results_df['True label'])

	print(f'TrSSP vs. GO terms:\nt-test statistic: {ttest_results[0]}, t-test p-value: {ttest_results[1]}')

	## Code for getting confusion matrix plots
	fig, axes = plt.subplots(1,2, figsize=(16,8), tight_layout=True)

	# Plot each classifier (e.g., my classifier and trssp)
	trssp_confuse = confusion_plot(pred_labels=results_df['TrSSP prediction'], true_labels=results_df['True label'], plot_ax=axes[0], colorbar=False, confusion_palette=sns.light_palette("#BF6517", as_cmap=True))
	classifier_confuse = confusion_plot(pred_labels=results_df['Predicted label'], true_labels=results_df['True label'], plot_ax=axes[1], colorbar=False, confusion_palette=sns.light_palette("#BF6517", as_cmap=True))

	# Format subplots
	axes[0].set_title('TrSSP')
	axes[1].set_title('Transporter Classifier')

	# Format figure
	axes[1].set_yticks([]) 	# remove redundant y-axis
	fig.supxlabel('Predicted Class', fontsize=24)
	fig.supylabel('True Class', fontsize=24)
	fig.suptitle('Transport classification confusion matrices', fontsize=30)

	fig.savefig('confusion_transp_meanRF_HighCon_thresh75.png')

	# # Test with same values using ConfusionMatrixDisplay() to validate that matrices align
	# ConfusionMatrixDisplay.from_predictions(y_pred=results_df['TrSSP prediction'], y_true=results_df['True label'], normalize='true')
	# plt.savefig('MatDisplay_TrSSP_norm_rf14_HighConAnnots.png')

	# ConfusionMatrixDisplay.from_predictions(y_pred=results_df['Predicted label'], y_true=results_df['True label'], normalize='true')
	# plt.savefig('MatDisplay_transportereClassifier_norm_rf14_HighConAnnots.png')

# Run main
main()