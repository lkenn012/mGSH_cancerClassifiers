## Kennedy et al. 2024
#
# main code for generating classifier model, running, outputting results (for mGSH transporter prediction)
# The code employ multiomics-based PCA components as features, with a relevant GO term (glutathione metabolism,
# mitochondia localization, or transmembrane transport) to identify training/test genes (class 1) and random 
# selection of equal length from the remaining genes (class 0)
# Training is boot strapped to account for the random selection of class 0 genes, and each iteration uses 10-cross fold validation 

# The specifics below runs a GSH term classifier and uses transcriptomics PCs as features, without GSH & GSSG metabolomics as features


# import modules
import pandas as pd
import numpy as np

import random
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.naive_bayes import GaussianNB 	# ML modules, all default params initially
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datetime import datetime # for output formatting

import main_mGSH as mGSH	# python file containing functions for building the model

import multiprocessing as mp
from functools import partial


##
## load data to be used in model
##

data_path = r'path' 		## ENTER PATH TO ccleTranscriptomics_PCA.csv

# CCLE data	
ccle_genePCADF = pd.read_csv(rf'{data_path}\ccleTranscriptomics_PCA.csv', index_col=0)
ccle_GSHCorrDF = pd.read_csv(rf'{data_path}\data\GSH_spearman.csv', index_col=0) 	# GSH correlations

# Load labeled genes by GO annotation to be used as positive class for model
labeled_data = pd.ExcelFile(rf'{data_path}\mGSH_labeledGenes_HighConAnnots.xlsx')
mitoGenes_df = pd.read_excel(labeled_data, sheet_name='Mitochondrial Ensembl')
transpGenes_df = pd.read_excel(labeled_data, sheet_name='Transporter Ensembl')
gshGenes_df = pd.read_excel(labeled_data, sheet_name='GSH Ensembl')


##
## define all necessary methods for classifier methods
##

# define function to generate train/test splits composed of our labeled genes from GO terms and a radnom selection of all other genes, for each iteration of our model
def split_trainTest(class1_genes, n_iters, n_splits=5, all_geneDF=ccle_genePCADF):

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
		print(f'results_array:\n{results_array}')
		results_avg = np.mean(results_array, axis=1)

		results_data = [pd.Series(results_info), pd.Series(true_targets), pd.Series(results_array), pd.Series(results_avg)] 	# get results as lsit of series to concat
		print(f'results_data:\n{results_data}')

		results_df = pd.concat(results_data, axis=1)
		print(f'results_df:\n{results_df}')
		results_df.set_index(keys=0, inplace=True)
		print(f'results_df w/ index names:\n{results_df}')
		results_df.columns = [col_headers]
		print(f'results_df w/ index & col names:\n{results_df}')


		# results_df = pd.DataFrame(data=results_data, index=results_info, columns=col_headers)
		# print(f'Result df for {model_info[model_option]}')
		formatDF_list.append(results_df)

	return formatDF_list

# defube a function to select and format our feature data for feeding into the model
def build_PC_feats(pc_data, num_components, other_data, other_categorical=False, other_corr=False):

	# Our model will only use the first N PC components which explain some p of the total explained variance (N=14 explains p=0.95 variance, N=100 explains p=0.99 variance (for 03-22 PCA data))
	feat_componentDF = pc_data.iloc[:,:num_components]

	# other data to one hot if categorical
	if other_categorical:
		oneHot_other = pd.get_dummies(other_data, prefix=other_data.name)

		return pd.concat([feat_componentDF, oneHot_other], axis=1)

	# fisher transform correlation columns
	elif other_corr:
		fisher_corr = pd.DataFrame(data= mGSH.fisher(other_data), index=other_data.index, columns=other_data.columns)

		return pd.concat([feat_componentDF, fisher_corr], axis=1)

	# else just concat data
	else:
		return pd.concat([feat_componentDF, other_data], axis=1)

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

	print(f'!!!\nlabels count:\n{labeled_predDF.loc[:,"True label"].value_counts()}')
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
def main(posLabel_genes, ML_alg, num_components, boot_iters=2, alg_name=''):

	'''
	posLabel_genes: specifies the gene symbols for our positive class (annotated by GO term of interest)
 	ML_alg: method, specifies the sklearn ML algorithm method to call for model training/testing
  	num_components: int, specifies the number of principal components from CCLE transcriptomics to use  as features in the model
   	boot_iters: int, number of bootstrap iterations for training and testing classifier models with {boot_iters} randomly selected negative gene sets,
    alg_name: str, specifies a string to attach to output files for specifying model parameters
	'''
	
	# Check used for multiprocessing
	if __name__ == '__main__':

		# construct the feature data DF to feat into our model
		feat_df = build_PC_feats(pc_data=ccle_genePCADF, 
			num_components=num_components, 		# explains 95% of variance
			 other_data=None,  	# or = mitocartaScores
			 other_corr=False
			 )

		# Check and clean data so that there are no missing data used in the model
		if feat_df.isnull().any(axis=1).any():
			feat_df.drop(feat_df[feat_df.isnull().any(axis=1)].index, inplace=True)

		# Given a list of genes in our positive class, need to randomly select some negative genes from the remaining gene list
		split_trainingGenes = split_trainTest(class1_genes=posLabel_genes, n_iters=boot_iters, all_geneDF=feat_df) 	# have a balanced list of gene sets for each boostrap iteration: ['training/test genes', 'labels', 'K-fold split idxs'] for each iteration
		print(f'length of splits (should = boot_iters): {len(split_trainingGenes)}')

		# Framework for model is boostrap iterations over cross-validated positive and random negative gene training sets 
		# Using multi-processing can paralellize these runs over our labeled genes

		# Multiprocessing over boostrap iterations
		# The number of processes to use in the pool is dependent on the machine and number of the CPU cores.
		with mp.Pool(processes=1) as pool: 		# default processes=1; 4 should be managable on most machines
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

		print(f'results df:\n{results_df}')
		
		# Save outputs
		date = datetime.today().strftime('%d-%m-%Y') 	# get a string of the current date via strftime()
		out_path = rf'{data_path}\outputs'

		# results for unlabeled data
		print(f'unlabeled_preds:\n{unlabeled_preds}')
		unknown_df = pd.concat(unlabeled_preds, axis=1)  # results from all iterations into one final results df
		unknown_df['Average predicted label'] = unknown_df.mean(axis=1)

		# Save results for unlabeled genes
		unknown_df['Average predicted label'].to_csv(rf'{out_path}\{alg_name}_unknownResults_{date}.csv')

		# Save results of model 
		results_df.T.to_csv(rf'{out_path}\{alg_name}_Results_{date}.csv')


##############
#	RUN MAIN()
##############

## Code to run models with different parameters.
## Below are all model parameters (algorithm and feature combinations) used in this paper.
## Uncommented runs correspond to random forest classifier models, which are the best performing framework.

# main(posLabel_genes=gshGenes_df, ML_alg=GaussianNB(), num_components=7, boot_iters=100, alg_name='NB_5')
# main(posLabel_genes=gshGenes_df, ML_alg=GaussianNB(), num_components=16, boot_iters=100, alg_name='NB_14')
# main(posLabel_genes=gshGenes_df, ML_alg=GaussianNB(), num_components=32, boot_iters=100, alg_name='NB_30')
# main(posLabel_genes=gshGenes_df, ML_alg=GaussianNB(), num_components=52, boot_iters=100, alg_name='NB_50')

# main(posLabel_genes=gshGenes_df, ML_alg=SVC(probability=True), num_components=7, boot_iters=100, alg_name='SVM_5')
# main(posLabel_genes=gshGenes_df, ML_alg=SVC(probability=True), num_components=16, boot_iters=100, alg_name='SVM_14')
# main(posLabel_genes=gshGenes_df, ML_alg=SVC(probability=True), num_components=32, boot_iters=100, alg_name='SVM_30')
# main(posLabel_genes=gshGenes_df, ML_alg=SVC(probability=True), num_components=52, boot_iters=100, alg_name='SVM_50')

# main(posLabel_genes=gshGenes_df, ML_alg=DecisionTreeClassifier(), num_components=7, boot_iters=100, alg_name='DT_5')
# main(posLabel_genes=gshGenes_df, ML_alg=DecisionTreeClassifier(), num_components=16, boot_iters=100, alg_name='DT_14')
# main(posLabel_genes=gshGenes_df, ML_alg=DecisionTreeClassifier(), num_components=32, boot_iters=100, alg_name='DT_30')
# main(posLabel_genes=gshGenes_df, ML_alg=DecisionTreeClassifier(), num_components=52, boot_iters=100, alg_name='DT_50')


main(posLabel_genes=gshGenes_df, ML_alg=RandomForestClassifier(), num_components=7, boot_iters=100, alg_name='RF_5')
main(posLabel_genes=gshGenes_df, ML_alg=RandomForestClassifier(), num_components=16, boot_iters=100, alg_name='RF_14')
main(posLabel_genes=gshGenes_df, ML_alg=RandomForestClassifier(), num_components=32, boot_iters=100, alg_name='RF_30')
main(posLabel_genes=gshGenes_df, ML_alg=RandomForestClassifier(), num_components=52, boot_iters=100, alg_name='RF_50')
