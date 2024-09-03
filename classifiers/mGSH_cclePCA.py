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



'''
For use of principal components in our downstream task of function annotation, we would like to evaluate the robustness of the components. One approach
to do this is by generating components from subsets of the data, training sets, and evaluating the error when the trained algorithm is applied to the test
subset. This can be done through random subsets, or subsets defined by our sample groups in the CCLE data - namely, tissue types.
'''

def tissue_split(trans_data: pd.DataFrame, tissue_ids=None: str|list):

	'''
	trans_data (pandas DataFrame):	Dataframe containing CCLE transcriptomics data (genes x cell lines)
	tissue_ids (str or list, opt):	Optional tissue identifiers to subset data on, if not provided all unique tissues will be subset.
	'''

	# Check for tissue_ids and handle
	if tissue_ids:
		if isinstance(tissue_ids, list): 

		elif isinstance(tissue_ids, str):

		else:
			raise ValueError(f'tissue_ids object must be of type \"str\" or \"list\". (tissue_ids: {tissue_ids})')

	# CCLE sample codes are of the form {ID_chars}_{tissue_chars}. We want to split these codes to get {tissue_chars} for subsetting the data
	samples = trans_data.columns 
	test_list = ['DMS53_LUNG', 'SW1116_LARGE_INTESTINE', 'NCIH1694_LUNG', 'P3HR1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']
	sample_tissues = [re.match(r'^{\W}+_', sample) for sample in test_list]
	return
