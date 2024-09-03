## Kennedy et al. 2024
#
# Code for converting CCLE transcriptomics data to Prinicpal components from PCA for use as features in models.
# Includes code for robustness analysis of PCA components through comparison to components generated from random,
# and tissue-specific subsets of the data. Also includes code for outputting figures for this analysis.


# import modules
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

import re
import seaborn as sns
import matplotlib.pyplot as plt

# define functions

# define a function to take specified trnscriptomics data, standardize them and then perform PCA
def PCA_transcriptomics(trans_data):

	'''
	trans_data (pandas DataFrame):	Dataframe containing CCLE transcriptomics data (genes x cell lines)
	'''
	
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


# define a function to split CCLE transcriptomics data based on the tissue IDs in sample names
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


# define a function to compute the cosine similarities of the specifiied principal components of our data
def get_cosine(a, b, align=True, diagonal=True):

	'''
	a (pandas DataFrame):	Dataframe containing CCLE transcriptomics Principal componets
	b (pandas DataFrame):	Dataframe containing CCLE transcriptomics Principal componets
	align (bool): 			Whether to set principal components in a & b to the same direction (recommended)
	diagonal (bool):		Whether to return only the ith by ith component similarity
	'''

	# Due to the arbitrary directions in PCA, check that components point in the same direction
	# Dot product of components < 0 indicates different directions which should then be aligned.
	if align:
		for i, col in enumerate(a.columns):
			if np.dot(a.iloc[:,i], b.iloc[:,i]) < 0:
				a.iloc[:,i] = a.iloc[:,i] * -1

	# Depending on sample sizes, components may be unequal so need to compute across the common components
	if a.shape[1] > b.shape[1]:
		temp = cosine_similarity(a.iloc[:,:b.shape[1]].T, b.T) 	
	else:
		temp = cosine_similarity(a.T, b.iloc[:,:a.shape[1]].T)

	if diagonal:
		return temp.diagonal()
	else:
		return temp



###
## 	MAIN CODE
###
def main():

	'''
	Summary: This code loads a datafile containing relevant CCLE transcriptomics data used in classifier models. Computes
	PCA using all cell lines as features. These, "baseline" components are then compared to components derived from 
	tissue-types or from random subsets of the data.

	Pseudo-code:
		"Baseline" PCA:
			Load transcriptomics data
			Compute principal components, save file
			Plot and save explained variance for baseline
		Tissue-type PCA:
			Split transcripotmics on tissue IDs in sample names
			Compute principal components along each data split, save files
			Plot and save explained variance for baseline
			for all tisues:
				for all principal components:
					compute cosine similarity between a tissue-component_i & baseline-component_i
			Plot and save cosine similarities as heatmap
		Subset data PCAs:
			Specify random subset sizes = [100, 250, ...]
			for all subset sizes:
				for i in N_samples:
					randomly subset CCLE transcriptomics to predefined size
					compute principal components, save file
					compute similarity between subset-component_i & baseline-component_i
				compute average similarity over random sample similarities, for each component
			Plot and save similarities to baseline for each sample size
	'''


	### ARGUMENTS
	N_ITERS = 20  	# Number of random sampling
	SAMPLE_SIZES = [877, 850, 750, 500, 250, 100]  	# Random subset sizes

	# load data
	data_path = "path" 	## !! PLACEHOLDER !! replace this with the path to your data directory
	out_path = "output\path" ## !!PLACEHOLDER!! replace with the path for your data directory
	
	# load data to be summarized by PCA
	ccle_geneDF = pd.read_csv(rf'{data_path}\cleaned_common_ccleRNAseq.csv', index_col=0)
	
	# Run PCA on transcriptomics data
	base_pcDF = PCA_transcriptomics(trans_data=ccle_geneDF)
	cum_varDF = base_pcDF.iloc[1].cumsum()
	var_df = pd.DataFrame(data=[base_pcDF.iloc[1], cum_varDF], index=['Component', 'Cumulative'])

	# Plot explain variances
	scree = sns.lineplot(data=var_df.T, linewidth=1.5)
	scree.set_ylabel('Explained Variance Ratio', fontsize=14)
	scree.set_xlabel('Principal Component', fontsize=14)

	scree.set_xticks(np.arange(0,101,10)) 	# format x axis labels to only show every 5th component
	plt.tight_layout()
	plt.legend(fontsize=4)
	sns.move_legend(scree, bbox_to_anchor=(1.0, 1), loc='upper left') 	# move legend

	plt.savefig(rf'{out_path}\tissue_PCA_Scree.png', bbox_inches='tight', dpi=600)
	plt.close()

	base_pcDF.to_csv(f'{data_path}\ccleTranscriptomics_PCA.csv')

	# Get sub DFs for all tissue types
	tissue_dfs, tissues = tissue_split(trans_data=ccle_geneDF, get_ids=True)

	# Now, for each of these tissue-specific DFs, we can compute principal components.
	# After getting these components, we can compare the components by cosine similarity
	# and comparing the variance (skree) plots.
	explained_var, var_ratio = [], []
	for i, n in enumerate(tissue_dfs):
		tissue_pcDF = PCA_transcriptomics(trans_data=n)

		explained_var.append(tissue_pcDF.iloc[0])
		var_ratio.append(tissue_pcDF.iloc[1])

		tissue_pcDF.to_csv(rf'{out_path}\{tissues[i]}_transPC.csv')

	# Format as DFs for plotting the variances in our data
	explained_varDF = pd.DataFrame(data=explained_var, index=tissues)
	var_ratioDF = pd.DataFrame(data=var_ratio, index=tissues)

	# plot each 
	fig, axes = plt.subplots(1,2, figsize=(12,6))

	sns.lineplot(data=var_ratioDF.T, linewidth=1.5, ax=axes[0], legend=False)
	axes[0].set_ylabel('Explained Variance Ratio', fontsize=14)
	axes[0].set_xlabel('Principal Component', fontsize=14)
	axes[0].set_xticks(np.arange(0,101,10)) 	# format x axis labels to only show every 5th component

	sns.lineplot(data=var_ratioDF.T, linewidth=1.5, ax=axes[1])
	axes[1].set_ylabel('Explained Variance Ratio', fontsize=14)
	axes[1].set_xlabel('Principal Component', fontsize=14)

	axes[1].set_xticks(np.arange(0,101,10)) 	# format x axis labels to only show every 5th component
	plt.tight_layout()
	plt.subplots_adjust(wspace=0.3)
	plt.legend(fontsize=4)
	sns.move_legend(axes[1], bbox_to_anchor=(1.0, 1), loc='upper left') 	# move legend

	axes[1].set_xticks(np.arange(0,101,2)) 	# format x axis labels to only show every 5th component
	axes[1].set_ylim(0,0.08)
	axes[1].set_xlim(0,5)
	plt.savefig(rf'{out_path}\tissue_PCA_Scree.png', bbox_inches='tight', dpi=600)
	plt.close()

	'''
	Given the input data for k n x m arrays of tissue principal components and a "baseline" array of all_tissue components
		1. For the i in m components in baseline:
			Compute similarity between i_base and each i_tissue
			if i_tissue does not exist, return nan
		2. save results
		3. plot as heatmap of similarities
	'''

	# for each tissue subset principal components we want to compute similarities w.r.t. baseline
	cos_sims = []
	for i, tissue in enumerate(tissues):
		tissue_pcDF = tissue_dfs[i]
		tissue_pcDF = tissue_pcDF.iloc[2:] 	# skip first two rows containin variances

		# As a sanity check, insure both dfs are sorted in the same fashion
		base_pcDF.sort_index(inplace=True)
		tissue_pcDF.sort_index(inplace=True)

		if base_pcDF.index.equals(tissue_pcDF.index):
			print(f'Indices are equal')
		else:
			print(f'Not equal')

		temp = get_cosine(a=tissue_pcDF, b=base_pcDF, align=True, diagonal=True)
		cos_sims.append(temp)

	# Combine data into dataframe for plotting
	cos_sims_df = pd.DataFrame(data=cos_sims, index=tissues)

	heat = sns.heatmap(cos_sims_df, annot=False, cmap=sns.cubehelix_palette(as_cmap=True))
	heat.set_ylabel('Cell line type', fontsize=14)
	heat.set_xlabel('Principal Component', fontsize=14)
	heat.tick_params(axis='x', labelrotation=90)
	heat.set_title('CCLE tissue-specific PCA cosine similarity to baseline', fontsize=17)

	plt.savefig(rf'{out_path}\tissuePCA_cosSims_Heatmap.png', bbox_inches='tight', dpi=600)
	plt.close()

	# Plot just the first 10 components
	heat = sns.heatmap(cos_sims_df.iloc[:,:10], annot=False, cmap=sns.cubehelix_palette(as_cmap=True))
	heat.set_ylabel('Cell line type', fontsize=14)
	heat.set_xlabel('Principal Component', fontsize=14)
	heat.tick_params(axis='x', labelrotation=90)
	heat.set_title('CCLE tissue-specific PCA cosine similarity to baseline', fontsize=17)

	plt.savefig(rf'{out_path}\tissuePCA_cosSims_Heatmap_n10.png', bbox_inches='tight', dpi=600)
	plt.closer()

	'''
	Another option for  investigating the robustness of our principal components is to bootstrap many PCAs
	on random subsets of the data, and then comparing components to our components of the full dataset

	To compare the components, there are many possible methods and metrics that would work well. We simply
	consider the first N components for each dataset and measure cosine similarity, component-wise, with
	respect to baseline componets (full datasaet). We can report these in a table or as a plot
	'''
	sub_cos, mean_sims = [], []

	base_pcDF.sort_index(inplace=True)  	# sort so that order will be equal

	for n_samples in SAMPLE_SIZES:
		# create random seeds for sampling
		rng = np.random.default_rng(seed=n_samples)
		rand_states = rng.integers(low=1, high=100000, size=N_ITERS)
		rand_sims = []
		for n in range(N_ITERS):
			sub_df = ccle_geneDF.sample(n=n_samples, axis=1, random_state=rand_states[n])
			subPC_df = PCA_transcriptomics(sub_df)

			temp = get_cosine(a=subPC_df.iloc[2:,:], b=base_pcDF, align=True, diagonal=True)
			rand_sims.append(temp)

			subPC_df.to_csv(rf'{out_path}\rand_pcas\randPCA_n{n_samples}_{n+1}.csv')
		sub_cos.append(rand_sims)
		mean_sims.append(np.mean(rand_sims,axis=0))

	# Save these data
	for i, sub in enumerate(sub_cos):
		temp = pd.DataFrame(data=sub)
		temp.to_csv(rf'{out_path}\rand_pcas\randPCA_coSims_n{SAMPLE_SIZES[i]}.csv')


	# Now plot these sims
	plot_data = pd.DataFrame(data=mean_sims, index=SAMPLE_SIZES)

	line = sns.lineplot(data=plot_data.T)
	line.set_ylabel('Mean cosine similarity with baseline', fontsize=14)
	line.set_xlabel('Subset principal component', fontsize=14)
	line.set_title('CCLE random subset PCA cosine similarity to baseline', fontsize=17)
	sns.move_legend(line, bbox_to_anchor=(1.0, 1), loc='upper left')
	plt.savefig(rf'{out_path}\randSubPCA_cosSim_line.png', dpi=600, bbox_inches='tight')
	plt.close()

	return
	
# run main code
main()