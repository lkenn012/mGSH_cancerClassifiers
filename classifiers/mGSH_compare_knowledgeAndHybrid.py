## Code for comparing hybrid models produced here to existing knowledge-based models (MitoCarta vs. TrSSP)
## Applies several statistical tests of independence and produces a confusion matrix between a given knowledge-based model and the predictions from a given hybrid classifier produced in this work (or the average predictions of several classifiers)

##
## import modules
##

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import mGSH_classifierComparisons as classifer_tests 	# statistical tests for comparisons of models
import scipy.stats as stats

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
import re


##
## Define methods for generating confusion matrices
##

# define method for getting results from a list of file names, or a regex to extract files names.
# Average results from these files are determined by 'get_meanResults()' for use in model comparisons
def get_resultsFiles(results_path, f_names, fName_regex=False):

	# Optionally we can specify a regex to extract files, say all RF models (have 'RF' in file name), or all models from a specific date
	if fName_regex:
		f_names = [f for f in os.listdir(rf'{results_path}') if re.match(fName_regex, f)]

	# Given a list of file names of model results, extract the predicted labels from those files for averaging predictions by several models
	all_results
	for name in f_names:
		res_df = pd.read_csv(rf'{results_path}\{name}', index_col=0)
		all_results.append(res_df.T['Average predicted label'])

	# return all predicted labels and the corresponding true labels
	return all_results, res_df.loc['True label']
	
# define function to get several results files and take average predicted label
def get_meanResults(results_path, f_names, fName_regex=False):
	
	model_preds, true_labels = get_resultsFiles(results_path, f_names, fName_regex) 	# get the results (predicted labels) from our models of interest for averaging for further analysis

	# convert list of preds to df for getting average labels
	avg_df = pd.DataFrame(data=model_preds) 	
	mean_label = avg_df.mean(axis=0)

	# format as DF and add true labels
	mean_df = mean_label.to_frame()
	mean_df['True label'] = true_labels

	# return the df of average predicted labels over our models of interest
	return mean_df

# define function to run several stat tests between our predictions and the existing model predictions
def test_preds(pred_df, exist_preds, classifer_preds, true_labels):
	
	# get contingency table (chi-squared indepence test) of results
	contingency = classifer_tests.get_contingency(pred_df, classifer_preds, exist_preds, true_labels)
	print(f'Contingency table:\n\tClassifier Wrong, Classifier right')
	print(f'Existing model wrong: {contingency[0][0]} \t|\t {contingency[0][1]}')
	print(f'existing model right: {contingency[1][0]} \t|\t {contingency[1][1]}')	

	# chi-squared test statistic and p-value
	chi2_results = classifer_tests.chi2_test(pred_df, classifer_preds, exist_preds, true_labels)
	print(f'chi2 statistic: {chi2_results[0]}, chi2 p-value: {chi2_results[1]}')

	# Also try student's t-test
	ttest_results = stats.ttest_ind(results_df[classifer_preds], results_df[exist_preds])
	print(f'T-test statistics:')
	print(f'Classifier vs. Existing:\nt-test statistic: {ttest_results[0]}, t-test p-value: {ttest_results[1]}\n')

	return

# Define function to compute and then create a confusion matrix plot from given true and predicted labels
def confusion_plot(pred_labels, true_labels, plot_ax, colorbar=True, confusion_palette=sns.color_palette("light:b", as_cmap=True)):

	confusion = confusion_matrix(y_true=true_labels, y_pred=pred_labels) 	# get confusion matrix array
	confusion_df = pd.DataFrame(data=confusion, index=['Non-Transporter', 'Transporter'], columns= ['Non-Transporter', 'Transporter']) 	# convert to dataframe

	# Return a seaborn heatmap based on the confusion matrix data, with formatting
	return sns.heatmap(confusion_df, 
		cmap=confusion_palette,
		cbar=colorbar,
		fmt='g', 
		annot=True, 
		ax=plot_ax, 
		vmin=0, 
		vmax=len(true_labels) # specify max color value as # of samples
		) 	# return a plot of the confusion matrix
	
# define main function for running analysis
def main():
	
	# load data
	data_path = "path" 	# !! PLACEHOLDER !! replace with the path to where data (classifier results and TrSSP/MitoCarta data) is located

	model_fs = ["Model results file 1 here", "..."] 	# !! PLACEHOLDER !! optional, replace with a list of file names to include in analysis
	model_reg = "Model results file REGEX here" 	# !! PLACEHOLDER !! optional, replace with a regex expression to specfiy files for analysis or False if no REGEX
	
	mod_results = get_meanResults(results_path=rf'{data_path}', f_names=model_fs, fName_regex=False)

	TrSSPMito_annots = pd.read_csv(rf'{data_path}\mitocarta_trssp_scores.csv', index_col=0) 	# load DF containing IDs & cols for mitocarta, trssp scores

	trssp_scores = TrSSPMito_annots.loc[:,'TrSSP prediction'] 	# get column of existing model scores we wish to compare to, e.g. TrSSP prediction

	# get the existing model scores for genes we have also predicted
	pred_mitocarta = trssp_scores.loc[mod_results.index]
	mod_results['TrSSP prediction'] = trssp_scores.loc[mod_results.index]

	# Replace class probabilities from classifiers with class labels
	mod_results['Predicted label'] = np.where(mod_results.iloc[:,0] >= 0.75, 1, 0) 	# convert to class labels with threshold =0.75

	results_df = mod_results.loc[mod_results['TrSSP prediction'] != -1] 	# drop rows where there is no TrSSP prediction (= -1)

	## Comparing our classifier model vs. TrSSP predictions using independence tests
	test_preds(pred_df=results_df, exist_preds='TrSSP prediction', classifier_preds='Predicted label', true_labels='True label')

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

# Run main
main()
