## Code for converting CCLE transcriptomics data to Prinicpal components from PCA for use as features in models.


# import modules
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler

# define functions

# define code to get group transcriptomics data, Standardize them and then perform PCA
def PCA_transcriptomics(trans_data):
	
	# get Standardized data (such that mean=0, sd=1) for dimensionality reduction
	cleanData = trans_data.dropna() 	# Check for nan values which cannot be handled by PCA

	stand_scaler = StandardScaler()
	standard_data = stand_scaler.fit_transform(cleanData)

	# define PCA params and fit data
	pca_alg = PCA(n_components=100) 	# reduce ~ 1000 cell lines -> 100 PCs
	pca_components = pca_alg.fit_transform(standard_data) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	PCA_df = pd.DataFrame(data=pca_components, index=trans_data.index) # , columns=['PCA dimension 1', 'PCA dimension 2'])

	# Also want 1st and 2nd rows to reflect the PC component variance, variance ratio
	component_df = pd.DataFrame([pca_alg.explained_variance_, pca_alg.explained_variance_ratio_], index=['Component Variance', 'Component Varaince Ratio'])

	PCA_finalDF = pd.concat([component_df, PCA_df])
	return PCA_finalDF


###
## 	MAIN CODE
###
def main():
	# load data
	data_path = "path" 	## !! PLACEHOLDER !! replace this with the path to your data directory
		
	# load data to be summarized by PCA
	ccle_geneDF = pd.read_csv(rf'{data_path}\cleaned_common_ccleRNAseq.csv', index_col=0)
	
	# Run PCA on transcriptomics data
	pc_df = PCA_transcriptomics(trans_data=ccle_geneDF, pos_set=mitoGenes_df, neg_set=housekeeping_df)
	pc_df.to_csv(f'{data_path}\ccleTranscriptomics_PCA.csv')
	
# run main code
main()
