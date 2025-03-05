## Kennedy et al. 2024
#
# Compute pearson correlations for predicted annotations between different models
# For each task, gene annotations are correlated between pairs of models
# Results are returned as a csv file

# import modules
import pandas as pd
import numpy as np
import re
import os
from scipy.stats import pearsonr


# Define file names and model types for accessing and comparing model outputs
RESULTS_PATH = rf'outputs'  # !!PLACEHOLDER!! path to our results folder containing all model outputs
MODEL_TYPES = ['mito', 'transp']
MODEL_FS = [ r'TCGA_SKCM_PCA_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv', r'TCGA_PAAD_PCA_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv', 
			r'TCGA_LIHC_PCA_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv',  r'SKIN_transPC_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv',
			r'LIVER_transPC_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv', r'PANCREAS_transPC_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv', 
			r'[a-z]+_TCGA_allDisease_PCA_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv', 
			r'[a-z]+_ccleTranscriptomics_PCA_RF_[0-9]+_unknownResults_[0-9]+-[0-9]+-[0-9]+.csv'
			]

# Define method for loading and output files and collecting the annotation probabilities
def get_modelProbs(model_task, model_fs, f_path):
	'''
	model_task (str): 	classification task for model
	model_fs (array): 	Model output file name
	f_path (str): 		Path to output folder
	'''

	all_model_probs = []

	# For each model, get the average probability for each gene
	for results_name in model_fs:
		f_names = [f for f in os.listdir(f_path) if re.match(f'{model_task}_{results_name}', f)]
		dfs = [pd.read_csv(rf'{f_path}\{f}', index_col=0) for f in f_names]

		all_res = pd.concat(dfs, axis=1)
		all_res['mean'] = all_res.mean(axis=1)

		all_model_probs.append(all_res)

	return all_model_probs

	
# Now for each of our classification task, get all relevant files
def main():

	for task in MODEL_TYPES:

		# Get all model predictions
		all_model_results = get_modelProbs(model_task=task, model_fs=MODEL_FS, f_path=RESULTS_PATH)

		# Compute pairwise correlations between the model predictions
		all_pears = []
		for idx_i, i in enumerate(all_model_results):
			temp_pears = []
			for idx_j, j in enumerate(all_model_results):

				# First, since we may be using different datasets, get the common indices
				common_genes = i.index.intersection(j.index)

				i_com = i.loc[common_genes]
				j_com = j.loc[common_genes]
				# print(f'i_com:\n{i_com}')
				# for i, get all genes where mean prediction x > 0.6 or x < 0.4
				confident_i = i_com[(i_com['mean'] > 0.6) | (i_com['mean'] < 0.4)]
				confident_j = j_com[(j_com['mean'] > 0.6) | (j_com['mean'] < 0.4)]

				# print(f'confident_i:\n{confident_i}')

				common_conf = confident_i.index.intersection(confident_j.index)
				# subset both by these ids
				confident_i = i.loc[common_conf]
				confident_j = j.loc[common_conf]
				# comput pearson correlation between these
				pears = pearsonr(confident_i['mean'], confident_j['mean'])

				temp_pears.append(pears)
			all_pears.append(temp_pears)

		# Save correlations between models for this task
		pears_df = pd.DataFrame(data=all_pears, index=model_fs, columns=model_fs)
		pears_df.to_csv(rf'{task}_modPreds_pearson.csv')

##############
#	RUN MAIN()
##############
main()