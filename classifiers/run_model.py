## Code for running a "generic" classifier model using the framework developed here.
## In command-line, users can specify the labeled data for the GO term (or other function) of interest, and feature data to get classificatioon probabilites.
## See README for detailed information.


# import modules
import argparse
import re
import os
import openpyxl

import pandas as pd
import numpy as np

import random
from sklearn.model_selection import train_test_split, StratifiedKFold

# All ML modules that were tested, all default params used with random forest set as default
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datetime import datetime # for output formatting

import main_mGSH as mGSH	# python file containing functions for building the model

import multiprocessing as mp
from functools import partial

"""
User arguments to parse for constructing and running model.
Contains arguments for feature data selections, labeled samples, and constructing/running classifier models
"""
parser = argparse.ArgumentParser(description='Generate and run classifier')
parser.add_argument('--model-name', type=str, default='',
	help='Optional, a model name or identifier to attach to any output files.')
parser.add_argument('--data-path', type=str, default=None,
	help='Optional, file directory to data for use in model.')
parser.add_argument('--labeled-genes', type=str, default='',
	help='A csv or excel file containing a list of sample names (rows) corresponding to positive samples for classification.')
parser.add_argument('--feature-data', type=str, default='',
	help='A csv or excel file name containing feature data (columns) for samples (rows), such as transcriptomics data across cell lines.')
parser.add_argument('--select-features', type=int, default=None,
	help='Optional, whether to select a subset of columns from \"feature-data\" to use in model construction (e.g., 10 will use first 10 columns).')
parser.add_argument('--other-features', type=str, default=None,
	help='Optional, another csv or excel file containing feature data to use, such as a categorical score for genes from another database.')
parser.add_argument('--other-feat-type', choices=['categorical', 'correlation', 'numeric'], default=None, 
	help='Type of feature data for \"other-features\":\n categorical, correlation, numeric. Determines preprocessing steps, use numeric when no preprocessing is needed.')
parser.add_argument('--iterations', type=int, default=20,
	help='Number of classifiers and randomly generated training/test sets to generate. Default = 20, >100 recommended for confident predictions.')
parser.add_argument('--cpu-cores', type=int, default=1,
	help='Number of CPU cores to run script over which determines runtime. Default is 1, but 4 or more is often not a problem.')

args = parser.parse_args()


##
## load data to be used in model
##

# Specify path to data
# Check a few different possible paths to identify where data is found
current_dir = os.getcwd() 	# get current directory

if args.data_path:
	if os.path.exists(os.path.join(current_dir, f'{args.data_path}')): 		# Check full directory path
		data_path = os.path.join(current_dir, f'{args.data_path}')
	elif os.path.exists(os.path.join(current_dir, '..', f'{args.data_path}')): 		# check if path from parent
		data_path = os.path.join(current_dir, '..', f'{args.data_path}')
	else:
		print(f'No existing directory found for data_path: {args.data_path} (Is this correct?). Trying current directory instead.')
		data_path = ''
else:
	data_path = ''

# Get and select feature data for use in models
if re.search(r'\.csv$', args.feature_data):
	feat_df = pd.read_csv(rf'{data_path}\{args.feature_data}', index_col=0)
elif re.search(r'\.(xlsx|xls)$', args.feature_data):
	feat_xl = pd.ExcelFile(rf'{data_path}\{args.feature_data}', engine='openpyxl')
	feat_df = pd.read_excel(feat_xl, sheet_name=0, index_col=0)
else:
	raise ValueError("Invalid file type for feature-data. Only excel (.xlsx, .xls) or csv (.csv) files are accepted.")

# labeled genes/samples for the positive class
if re.search(r'\.csv$', args.labeled_genes):
	labeled_df = pd.read_csv(rf'{data_path}\{args.labeled_genes}')
elif re.search(r'\.(xlsx|xls)$', args.labeled_genes):
	labeled_xl = pd.ExcelFile(rf'{data_path}\{args.labeled_genes}', engine='openpyxl')
	labeled_df = pd.read_excel(labeled_xl, sheet_name=0)
else:
	raise ValueError("Invalid file type for labeled-genes. Only excel (.xlsx, .xls) or csv (.csv) files are accepted.")

if args.other_features:
	if re.search(r'\.csv$', args.other_features):
		other_df = pd.read_csv(rf'{data_path}\{args.other_features}', index_col=0)
	elif re.search(r'\.(xlsx|xls)$', args.other_features):
		other_xl = pd.ExcelFile(rf'{data_path}\{args.other_features}', engine='openpyxl')
		other_df = pd.read_excel(other_xl, sheet_name=0, index_col=0)
	else:
		raise ValueError("Invalid file type for other-features. Only excel (.xlsx, .xls) or csv (.csv) files are accepted.")
else:
	other_df = None

print(f'All data has been loaded, preparing data for classifier model and training models...')
##
## define all necessary methods for classifier methods
##

# define function to generate train/test splits composed of our labeled genes from GO terms and a radnom selection of all other genes, for each iteration of our model
def split_trainTest(class1_genes, all_geneDF, n_iters, n_splits=5):

	# get unlabeled genes
	all_genes = all_geneDF.index.tolist() 	# list of all genes
	class1_geneList = class1_genes.iloc[:,0].tolist()
	unlabled_genes = list(set(all_genes) - set(class1_geneList)) 	# get the difference between the labeled class1 genes and all other genes in our data
	
	# For each split,
	training_sets = []

	for i in range(n_iters):
		class0_geneList = random.sample(unlabled_genes, len(class1_geneList))

		labeled_geneList = np.array(class1_geneList + class0_geneList) 	# all labeled Gene IDs
		gene_labels = np.array([1]*len(class1_geneList) + [0]*len(class0_geneList)) 	# corresponding labels

		# Get balanced K-fold splits for the data
		k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
		splits = [split for split in k_fold.split(labeled_geneList, gene_labels)] 	# should give 5 80/20 training/test splits

		training_sets.append([labeled_geneList, gene_labels, splits])

	return training_sets


# define function which will take the labeled data from our excel tables and format + remove intersecting genes to feed into model
def format_labeledGenes(pos_geneDF, neg_geneDF):

	pos_geneList = pos_geneDF.iloc[:,0].tolist() 	# convert to lists from df for removing overlap
	neg_geneList = neg_geneDF.iloc[:,0].tolist()

	# Get intersecting genes and remove overlap from each list
	common_genes = set(pos_geneList).intersection(neg_geneList)
	formatted_posGenes = list(set(pos_geneList) - common_genes)
	formatted_negGenes = list(set(neg_geneList) - common_genes)

	return formatted_posGenes, formatted_negGenes

# define function to format outputs from models into DFs for analysis and saving
def format_results(results, results_info, true_targets, model_info, model_infoHeader):

	formatDF_list = [] 	# to hold list of model outputs for each iteration

	# iterate over results tables to format
	for model_option, results_array in enumerate(results):

		# Want to add headers to output containing information for specific model outputs
		col_headers = ['True labels', f'{model_infoHeader} ({model_info[model_option]})', f'avg. result']
		results_avg = np.mean(results_array, axis=1)

		results_data = [pd.Series(results_info), pd.Series(true_targets), pd.Series(results_array), pd.Series(results_avg)] 	# get results as lsit of series to concat

		# Format into proper df
		results_df = pd.concat(results_data, axis=1)
		results_df.set_index(keys=0, inplace=True)
		results_df.columns = [col_headers]

		formatDF_list.append(results_df)

	return formatDF_list

# defube a function to select and format our feature data for feeding into the model
def build_feats(data, feat_subset, other_data, other_dtype):

	# Our model will only use the first N PC components which explain some p of the total explained variance (N=14 explains p=0.95 variance, N=100 explains p=0.99 variance (for CCLE transcriptomics))
	feat_componentDF = data.iloc[:,:feat_subset]

	if other_data:
		# other data to one hot if categorical
		if other_dtype == 'categorical':
			oneHot_other = pd.get_dummies(other_data, prefix=other_data.name)

			return pd.concat([feat_componentDF, oneHot_other], axis=1)

		# fisher transform correlation columns
		elif other_dtype == 'correlation':
			fisher_corr = pd.DataFrame(data= mGSH.fisher(other_data), index=other_data.index, columns=other_data.columns)

			return pd.concat([feat_componentDF, fisher_corr], axis=1)

		# else, data are numeric then no pre-processing is needed.
		elif other_dtype == 'numeric':
			return pd.concat([feat_componentDF, other_data], axis=1)
	else:
		return feat_componentDF

# define run_model function for running the model with our PCA feat inputs
def run_CVmodel(trainingData_splits, model_alg, feat_data, pos_geneIDs):

	# Extract the data we are using, for clarity
	labeled_geneIDs = trainingData_splits[0]
	labels = trainingData_splits[1]
	kfolds = trainingData_splits[2]

	# Need to create empty DFs for gene predictions, which will get updated each run
	cols = [f'CV {i}' for i, data in enumerate(kfolds)] 	# column headers for each CV
	labeled_predDF = pd.DataFrame(data=np.nan, index=feat_data.index, columns=cols)

	# Create a column for true labels (which is = 1 for our positive genes, and 0 elsewhere)
	labeled_predDF['True label'] = 0
	labeled_predDF.loc[labeled_predDF.index.isin(pos_geneIDs.squeeze().tolist()), 'True label'] = 1

	# Now create similar df but for our unlabeled gene predictions
	unlabeled_predDF = pd.DataFrame(data=np.nan, index=feat_data.index, columns=cols) 	# Can use all genes as index since it will only be filled with genes NOT in our labeled genes for a given iter.

	# Need to iterate over each K-fold split
	for i, K_fold in enumerate(kfolds):

		# First get unlabeled feat for applying our model to after
		unlabeled_geneList = [gene for gene in feat_data.index.tolist() if gene not in labeled_geneIDs] 	# get list of all NOT labeled genes
		unlabeled_data = feat_data.loc[unlabeled_geneList] 	# get corresponding features

		# MISSING ROWS IN DATA
		unlabeled_data.dropna(inplace=True)

		# Now need to get feature data for labeled genes
		labeled_data = feat_data.loc[labeled_geneIDs]

		# Split by our K-fold indexes in k_fold: [[1,5,8,9,10...], [2,3,4,6,7,...]] = [train_idxs, test_idxs]
		x_train = labeled_data.iloc[K_fold[0]]
		y_train = labels[K_fold[0]]

		x_test = labeled_data.iloc[K_fold[1]]
		y_test = labels[K_fold[1]]

		# Now train the model
		model_alg.fit(x_train.values, y_train)

		y_score = model_alg.predict_proba(x_test.values) 	# probabilities for class [1,0] for use in ROC curve

		# update dataframe with test predictions for evaluation
		pred_S = pd.Series(index=x_test.index, data=y_score[:, 1]) # hold predictions in series to update DF

		labeled_predDF.loc[pred_S.index, labeled_predDF.columns[i]] = pred_S	 # add pred scores for this CV

		valid_scores = model_alg.predict_proba(unlabeled_data.values) 	# get the predictions for our unlabeled data
 
		pred_S = pd.Series(index=unlabeled_data.index, data=valid_scores[:, 1]) # hold predictions in series to update DF
		unlabeled_predDF.loc[pred_S.index, unlabeled_predDF.columns[i]] = pred_S	 # add pred scores for this CV

		print(f'labeled_predDF:\n{labeled_predDF}\nunlabeled pred:\n{unlabeled_predDF}')

	return labeled_predDF, unlabeled_predDF 	# return predictions for this iteration

# define main() which runs model using LOOCV, for a specified ML model, and feat gene distribution and bootstrapping
# build outputs after
def main(posLabel_genes, ML_alg, main_feats, num_feats, other_feats=None, other_type=None, boot_iters=2, jobs=1, alg_name=''):

	'''
	posLabel_genes: specifies the gene symbols for our positive class (annotated by GO term of interest)
 	ML_alg: method, specifies the sklearn ML algorithm method to call for model training/testing
  	main_feats: dataframe, specifies the object containing the primary feature data (e.g., CCLE transcriptomics PCs)
  	num_feats: int, specifies the number of principal components from CCLE transcriptomics to use  as features in the model
   	other_feats: dataframe, specifies another dataset to be used as features in the model (e.g., MitoCarta scores)
   	other_type: string, specifies the type of data for other (categorical, numeric, correlation) which informs preprocessing
   	boot_iters: int, number of bootstrap iterations for training and testing classifier models with {boot_iters} randomly selected negative gene sets,
    jobs: int, number of processes, aka cpu cores, to use in multiprocessing for decreasing model runtime
    alg_name: str, specifies a string to attach to output files for specifying model parameters
	'''
	
	# Check used for multiprocessing
	if __name__ == '__main__':

		# construct the feature data DF to feat into our model
		feat_df = build_feats(data=main_feats, 
			feat_subset=num_feats, 		# For Principal components (or other) features, only a subset of data may be needed for use in model features
			 other_data=other_feats,  	# or = mitocartaScores
			 other_dtype=other_type
			 )

		# Check and clean data so that there are no missing data used in the model
		if feat_df.isnull().any(axis=1).any():
			feat_df.drop(feat_df[feat_df.isnull().any(axis=1)].index, inplace=True)

		# Given a list of genes in our positive class, need to randomly select some negative genes from the remaining gene list
		split_trainingGenes = split_trainTest(class1_genes=posLabel_genes, n_iters=boot_iters, all_geneDF=feat_df) 	# have a balanced list of gene sets for each boostrap iteration: ['training/test genes', 'labels', 'K-fold split idxs'] for each iteration

		# Framework for model is boostrap iterations over cross-validated positive and random negative gene training sets 
		# Using multi-processing can paralellize these runs over our labeled genes

		# Multiprocessing over boostrap iterations
		# The number of processes to use in the pool is dependent on the machine and number of the CPU cores.
		with mp.Pool(processes=jobs) as pool: 
			test_preds, unlabeled_preds = zip(*pool.map(
				partial(
				run_CVmodel, model_alg=ML_alg, 
				feat_data=feat_df,
				pos_geneIDs=posLabel_genes),
			 split_trainingGenes)) 	# iterate over each boostrap with information from 'split_trainingGenes'

			pool.close() 	# close pool


		true_labels = test_preds[0]['True label']

		# remove True labels from each df, will add back later
		for df in test_preds:
			df.drop('True label', axis=1, inplace=True)

		results_df = pd.concat(test_preds, axis=1)  # results from all iterations into one final results df
		results_df['Average predicted label'] = results_df.mean(axis=1)
		results_df['True label'] = true_labels
		
		# Save outputs
		date = datetime.today().strftime('%d-%m-%Y') 	# get a string of the current date via strftime()
		out_path = rf'{data_path}\outputs'

		# results for unlabeled data
		unknown_df = pd.concat(unlabeled_preds, axis=1)  # results from all iterations into one final results df
		unknown_df['Average predicted label'] = unknown_df.mean(axis=1)

		# Save results for unlabeled genes
		unknown_df['Average predicted label'].to_csv(rf'{out_path}\{alg_name}_unknownResults_{date}.csv')

		# Save results of model 
		results_df.T.to_csv(rf'{out_path}\{alg_name}_Results_{date}.csv')


##############
#	RUN MAIN()
##############

main(posLabel_genes=labeled_df, 
	ML_alg=RandomForestClassifier(), 
	main_feats=feat_df, 
	num_feats=args.select_features, 
	boot_iters=args.iterations, 
	jobs=args.cpu_cores, 
	alg_name=args.model_name
	)
