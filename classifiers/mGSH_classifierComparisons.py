## Code for comparing binary classifier models
## statistical test methods are called in mGSH_compare_knowledgeAndHybrid.py for comparisons of hybrid classifier models produced here to the existing, knowledge-based, MitoCarta, or TrSSP


##
## import modules
##
import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency 	# method for chi-squared test

import matplotlib.pyplot as plt
import seaborn as sns


##
## Define methods for statistical tests
##

# Define a function to construct a contingency table of classifier results to be compared using McNemar's test
def get_contingency(results, model_1, model_2, ground_truth):

	# have a DF containing results, along with column names corresponding to model_1, _2, and ground-truth labels for each samples
	# From this, construct a 2X2 contingency table of the form:
	#			mod_1 wrong | mod_1 right
	# 			 ___________ ____________
	#	mod_2 	|			|			 |
	#	wrong	|			|			 |
	#  ------	--------------------------
	#	mod_2 	|			| 			 |
	#	right 	|___________|____________|
	#


	conting = np.zeros(shape=(2,2)) 	#  create an empty dataframe to hold our contingency table

	# Now need to get the values for our contingency table

	# Get columns of whether model labels are correct
	mod1_correct = np.where(results[model_1] == results[ground_truth], 1, 0) 	# 1 if correct, 0 if incorrect
	mod2_correct = np.where(results[model_2] == results[ground_truth], 1, 0)

	print(f'mo1_correct and mod_2 correct:\n{mod1_correct}\n{mod2_correct}')

	# Now fill in values for the contingency table
	conting[0][0] =  np.sum(np.logical_and(mod1_correct == 0, mod2_correct == 0))	# count where both models are wrong
	conting[0][1] =  np.sum(np.logical_and(mod1_correct == 1, mod2_correct == 0))	# count where model 1 is right and model 2 is wrong
	conting[1][0] =  np.sum(np.logical_and(mod1_correct == 0, mod2_correct == 1))	# count where model 2 is right and model 1 is wrong
	conting[1][1] =  np.sum(np.logical_and(mod1_correct == 1, mod2_correct == 1))	# count where both models are right

	# return the filled in contingency table
	return conting

## Define a function for Chi-squared test to compare un-paired data (this is used when comparing our classifier & MitoCarta scores,  for example)
def chi2_test(results_df, labels_1, labels_2, groundTruth_labels):

	# from a results df which contains: rows= samples, columns = [model labels, other labels, ground-truth labels]. Construct a contingency table to use in chi-squared test
	results_table = get_contingency(results_df, labels_1, labels_2, groundTruth_labels)

	# now can calculate Chi-Squared test
	chi_results = chi2_contingency(results_table)

	# return the chi-squared test statitic & p-value
	return chi_results.statistic, chi_results.pvalue



# define test case
def test():

	test = pd.DataFrame({'mod_1': [1,1,1,1,1,0,0,0,0,0],
		'mod_2': [1,1,1,1,0,1,0,0,0,0],
		'true_labels': [1,1,1,1,1,1,1,1,1,1]
		}
		)

	print(f'Test_df:\n{test}')

	# mod_1_correct = [true, true, true, true, true, false, false, false, false, false] 	(represented as 1/0 in get_contingency())
	# mod_2_correct = [true, true, true, true, false, true, false, false, false, false]

	# Should result in a contingency table which looks like:
	#			mod_1 wrong | mod_1 right
	# 			 ___________ ____________
	#	mod_2 	|			|			 |
	#	wrong	|	  4 	|	  1 	 |
	#  ------	 ------------------------
	#	mod_2 	|	  1 	| 	  4 	 |
	#	right 	|___________|____________|

	test_contingency = get_contingency(results=test, model_1='mod_1', model_2='mod_2', ground_truth='true_labels')

	print(f'Test_df contingency table:\n{test_contingency}')

	# Now want to test the chi_2 test
	chi2_result = chi2_test(results_df=test, labels_1='mod_1', labels_2='mod_2', groundTruth_labels='true_labels')

	print(f'chi_2 results - statistic, p-value\n\t\t{chi2_result[0]}, {chi2_result[1]}')

# Run test code
test()
