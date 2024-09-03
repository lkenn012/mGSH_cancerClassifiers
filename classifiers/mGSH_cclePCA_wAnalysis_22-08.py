## Code for converting CCLE transcriptomics data to Prinicpal components from PCA for use as features in models.


# import modules
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler

import re
import seaborn as sns
import matplotlib.pyplot as plt

# define functions

# define code to get group transcriptomics data, Standardize them and then perform PCA
def PCA_transcriptomics(trans_data):
	
	# get Standardized data (such that mean=0, sd=1) for dimensionality reduction
	cleanData = trans_data.dropna() 	# Check for nan values which cannot be handled by PCA

	stand_scaler = StandardScaler()
	standard_data = stand_scaler.fit_transform(cleanData)

	# define PCA params and fit data
	if standard_data.shape[1] < 100:
		print(f'Only {standard_data.shape[1]} samples available, fitting PCA with this many components instead of default 100')
		pca_alg = PCA(n_components=standard_data.shape[1]) 	# reduce ~ 1000 cell lines -> 100 PCs
		pca_components = pca_alg.fit_transform(standard_data) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	else:
		pca_alg = PCA(n_components=100) 	# reduce ~ 1000 cell lines -> 100 PCs
		pca_components = pca_alg.fit_transform(standard_data) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	PCA_df = pd.DataFrame(data=pca_components, index=trans_data.index) # , columns=['PCA dimension 1', 'PCA dimension 2'])

	# Also want 1st and 2nd rows to reflect the PC component variance, variance ratio
	component_df = pd.DataFrame([pca_alg.explained_variance_, pca_alg.explained_variance_ratio_], index=['Component Variance', 'Component Varaince Ratio'])

	PCA_finalDF = pd.concat([component_df, PCA_df])
	return PCA_finalDF


'''
For use of principal components in our downstream task of function annotation, we would like to evaluate the rbbustness of the components. One approach
to do this is by generating components from subsets of the data, training sets, and evaluating the error when the trained algorithm is applied to the test
subset. This can be done through random subsets, or subsets defined by our sample groups in the CCLE data - namely, tissue types.
'''

def tissue_split(trans_data: pd.DataFrame, tissue_ids=None, get_ids=False):

	'''
	trans_data (pandas DataFrame):	Dataframe containing CCLE transcriptomics data (genes x cell lines)
	tissue_ids (str or list, opt):	Optional tissue identifiers to subset data on, if not provided all unique tissues will be subset.
	get_ids (bool): 				Whether to return the tissue IDs used to subset trans_data
	'''

	# Check for tissue_ids and handle
	if tissue_ids:
		if isinstance(tissue_ids, list): 
			pass
		elif isinstance(tissue_ids, str):
			tissue_ids = [tissue_ids] 	# convert to list for handling
		else:
			raise ValueError(f'tissue_ids object must be of type \"str\" or \"list\". (tissue_ids: {tissue_ids})')

	else:
		# CCLE sample codes are of the form {ID_chars}_{tissue_chars}. We want to split these codes to get {tissue_chars} for subsetting the data
		sample_tissues = [re.findall(r'(?<=_)\w+', sample)[0] for sample in trans_data.columns.tolist()]
		tissue_ids = list(set(sample_tissues)) 	# convert to dict and back to get only the unique IDs
		print(f'unique tissues:\n{tissue_ids}')

	tissue_expressionDFs = []
	for tissue in tissue_ids:
		tissue_cols = trans_data.columns.str.contains(tissue)
		sub_df = trans_data.loc[:,tissue_cols]
		print(f'Tissue: {tissue}\n\tSub_df shape: {sub_df.shape}')
		tissue_expressionDFs.append(sub_df)

	if get_ids:
		return tissue_expressionDFs, tissue_ids
	else:	
		return tissue_expressionDFs




###
## 	MAIN CODE
###
def main():
	# load data
	data_path = "path" 	## !! PLACEHOLDER !! replace this with the path to your data directory
	data_path = r'C:\Users\lkenn\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\data'
	out_path = r'C:\Users\lkenn\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs_22-08-2024'	
	# load data to be summarized by PCA
	ccle_geneDF = pd.read_csv(rf'{data_path}\cleaned_common_ccleRNAseq.csv', index_col=0)
	
	# # Run PCA on transcriptomics data
	# pc_df = PCA_transcriptomics(trans_data=ccle_geneDF, pos_set=mitoGenes_df, neg_set=housekeeping_df)
	# pc_df.to_csv(f'{data_path}\ccleTranscriptomics_PCA.csv')

	# Get sub DFs for all tissue types
	tissue_dfs, tissues = tissue_split(trans_data=ccle_geneDF, get_ids=True)

	!!! CAN ADD THRESHOLD - E.G., ATLEAST 10 SAMPLES


	# Now, for each of these tissue-specific DFs, we can compute principal components.
	# After getting these components, we can compare the components by cosine similarity
	# and comparing the variance (skree) plots.
	# Save all these data for possible downstream use.
	explained_var, var_ratio = [], []
	for i, n in enumerate(tissue_dfs):
		tissue_pcDF = PCA_transcriptomics(trans_data=n)

		explained_var.append(tissue_pcDF.iloc[0])
		var_ratio.append(tissue_pcDF.iloc[1])

		# tissue_pcDF.to_csv(rf'{out_path}\{tissues[i]}_transPC.csv')

	# Another analysis of these components is by random subsets of the data. Generate 
	# n_iter subsets of size genes x n_samples and compute principal components for all.
	# Variance plot (i.e., variance between components) or similarity plots for each subset?

	# Format as DFs for plotting the variances in our data
	explained_varDF = pd.DataFrame(data=explained_var, index=tissues)
	var_ratioDF = pd.DataFrame(data=var_ratio, index=tissues)

	# explained_varDF.reset_index(inplace=True)
	# var_ratioDF.reset_index(inplace=True)

	# print(f'explained_varDF (shape: {explained_varDF.shape}):\n{explained_varDF}')
	# print(f'var_ratioDF (shape: {var_ratioDF.shape}):\n{var_ratioDF}')

	# # Convert to long form for plotting
	# explained_varDF = explained_varDF.melt(0, var_name='tissue', value_name='Var.')
	# print(f'melted df:\n{explained_varDF}')

	# var_ratioDF = var_ratioDF.melt(0, var_name='tissue', value_name='Var. ratio')
	# print(f'melted df:\n{var_ratioDF}')


	# plot each 
	var_plot = sns.lineplot(data=explained_varDF.T, linewidth=1)
	var_plot.set_ylabel('Explained Variance', fontsize=14)
	var_plot.set_xlabel('Principal Component', fontsize=14)
	var_plot.set_title('CCLE tissue-specific PCA Scree Plot (Explained Var.)', fontsize=17)

	var_plot.set_xticks(np.arange(0,101,10)) 	# format x axis labels to only show every 5th component
	plt.legend(fontsize=7)
	sns.move_legend(var_plot, bbox_to_anchor=(1.0, 1), loc='upper left') 	# move legend
	plt.savefig(rf'{out_path}\tissue_PCA_explainedVarPlot_22-08.png', bbox_inches='tight', dpi=600)
	plt.close()

	# plot each 
	ratio_plot = sns.lineplot(data=var_ratioDF.T, linewidth=1)
	ratio_plot.set_ylabel('Explained Variance Ratio', fontsize=14)
	ratio_plot.set_xlabel('Principal Component', fontsize=14)
	ratio_plot.set_title('CCLE tissue-specific PCA Scree Plot (Var. Ratio)', fontsize=17)

	ratio_plot.set_xticks(np.arange(0,101,10)) 	# format x axis labels to only show every 5th component
	plt.legend(fontsize=7)
	sns.move_legend(ratio_plot, bbox_to_anchor=(1.0, 1), loc='upper left') 	# move legend
	plt.savefig(rf'{out_path}\tissue_PCA_ratioVarPlot_22-08.png', bbox_inches='tight', dpi=600)

	ratio_plot.set_xticks(np.arange(0,101,2)) 	# format x axis labels to only show every 5th component
	ratio_plot.set_ylim(0,0.1)
	ratio_plot.set_xlim(0,6)
	plt.savefig(rf'{out_path}\tissue_PCA_ZOOMexplainedVarPlot_22-08.png', bbox_inches='tight', dpi=600)

	plt.close()

	return
	

# run main code
main()
test_list = ['DMS53_LUNG', 'SW1116_LARGE_INTESTINE', 'NCIH1694_LUNG', 'P3HR1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
sample_tissues = [re.findall(r'(?<=_)\w+', sample)[0] for sample in test_list]
print(sample_tissues)
unique_tissues = list(set(sample_tissues))	# convert to set and back to get only the unique IDs
# unique_tissues = list(unique_tissues) 	# convert to dict and back to get only the unique IDs

print(f'unique tissues:\n{unique_tissues}')

