## Code for converting CCLE transcriptomics data to Prinicpal components from PCA for use as features in models.


# import modules
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler

# groups genes into housekeeping, mito, and overlapping genes
# define function to format the related/unrelated genes as list and get subset the overlapping IDs
def get_geneSets(group1_genes, group2_genes):

	group1_list = group1_genes.iloc[:,0].tolist() 	# convert to lists from df for removing overlap
	group2_list = group2_genes.iloc[:,0].tolist()

	# Get intersecting genes and remove overlap from each list
	overlap_set = set(group1_list).intersection(group2_list)

	uniq_group1List = list(set(group1_list) - overlap_set) 	# set difference to get uniq lists
	uniq_group2List = list(set(group2_list) - overlap_set)

	return uniq_group1List, uniq_group2List, list(overlap_set) 	# return each list


# define code to get group transcriptomics data, Standardize them and then perform PCA
def PCA_transcriptomics(trans_data, pos_set, neg_set):

	# get the over-lap and unique positive (class 1genes), negative (clas 2 genes)
	uniq_pos, uniq_neg, overlap = get_geneSets(pos_set, neg_set)

	# subset transcriptomics data to contain only the subset lists
	transIDs = list(trans_data.index)

	# get relevant genes as those we have data for
	trans_relatedGenes = list(set(uniq_pos).intersection(transIDs))
	trans_unRelatedGenes = list(set(uniq_neg).intersection(transIDs))
	trans_overlapGenes = list(set(overlap).intersection(transIDs))
	trans_otherGenes = list(set(transIDs) - set(trans_relatedGenes) - set(trans_unRelatedGenes) - set(trans_overlapGenes))	# Also want to get the remainder genes, or genes not found in our groups
	print(f'trans_otherGenes:\nlen {len(trans_otherGenes)}\n{trans_otherGenes[:10]}')
	
	# for n samples, sample each list for n genes
	relevant_data = trans_data.loc[trans_relatedGenes + trans_unRelatedGenes + trans_overlapGenes + trans_otherGenes,:] 	# This gets relevant genes, and re-orders based on our selection order
	
	groups = ['Mito']*len(trans_relatedGenes) + ['Non-Mito']*len(trans_unRelatedGenes) + ['Overlap']*len(trans_overlapGenes) + ['Other']*len(trans_otherGenes) 	# get a corresponding list, where each group gene has group label (for plotting)

	print(f'Orig data shape: {trans_data.shape}\nSelected data shape: {relevant_data.shape}\n# of selected genes: {len(groups)}')
	
	# get Standardized data (such that mean=0, sd=1) for dimensionality reduction
	cleanData = relevant_data.dropna() 	# Check for nan values which cannot be handled by PCA

	stand_scaler = StandardScaler()
	standard_data = stand_scaler.fit_transform(cleanData)

	# define PCA params and fit data
	pca_alg = PCA(n_components=100) 	# reduce ~ 1000 cell lines -> 100 PCs
	pca_components = pca_alg.fit_transform(standard_data) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	PCA_df = pd.DataFrame(data=pca_components, index=relevant_data.index) # , columns=['PCA dimension 1', 'PCA dimension 2'])

	# Also want 1st and 2nd rows to reflect the PC component variance, variance ratio
	component_df = pd.DataFrame([pca_alg.explained_variance_, pca_alg.explained_variance_ratio_], index=['Component Variance', 'Component Varaince Ratio'])

	PCA_finalDF = pd.concat([component_df, PCA_df])

	PCA_finalDF[f'Groups'] = ['NA', 'NA'] + groups 	# variance rows have no groups
	return PCA_finalDF


###
## 	MAIN CODE
###
def main():
	# load data
	data_path = "path" 	## !! PLACEHOLDER !! replace this with the path to your data directory
	
	# Load related & unrelated genes
	labeled_data = pd.ExcelFile(rf'{data_path}\mGSH_labeledGenes_FULL.xlsx')
	housekeeping_df = pd.read_excel(labeled_data, sheet_name='Housekeeping Ensembl') 	# related genes
	mitoGenes_df = pd.read_excel(labeled_data, sheet_name='Mitochondrial Ensembl') 	# un related genes
	
	
	# load data to be summarized by PCA
	ccle_geneDF = pd.read_csv(rf'{data_path}\cleaned_common_ccleRNAseq.csv', index_col=0)
	
	# Run PCA on transcriptomics data
	pc_df = PCA_transcriptomics(trans_data=ccle_geneDF, pos_set=mitoGenes_df, neg_set=housekeeping_df)
	pc_df.to_csv(f'{data_path}\ccleTranscriptomics_PCA.csv')
	
# run main code
main()
