## Pre-processing data for features in mGSH model

# import modules
import pandas as pd
import numpy as np

import sklearn.preprocessing as skpr
from sklearn.impute import KNNImputer

from  main_mGSH import spearmanr_pval, threshold_corr	# for generating correlations between GSH/GSSG and transcripts

# define a function to check given data for variables with standard deviation=0 (or ={threshold}) across samples, returns truncated DF with those rows removed
# Note: check_axis determines which axis std dev will be calculated across, so =0 gets a row of std dev corresponding to col data
def check_stdDev(data_df, check_axis=0, threshold=0):

	# create copy of data DF
	copyData = data_df.copy()

	# get std dev of {axis} as new coluimn/row, then select new DF containing Std Dev > threshold
	data_stdDev = copyData.std(axis=check_axis)

	if check_axis == 0:
		copyData.loc['Std Dev'] = data_stdDev
		thresh_check = copyData.loc['Std Dev'] > threshold
		trunc_dataDF = copyData.loc[:, thresh_check]
		trunc_dataDF.drop(index='Std Dev', inplace=True)

	else:
		copyData.loc[:, 'Std Dev'] = data_stdDev
		thresh_check = copyData.loc[:, 'Std Dev'] > threshold
		trunc_dataDF = copyData.loc[thresh_check]
		trunc_dataDF.drop(columns='Std Dev', inplace=True)

	return trunc_dataDF

# this function checks cols for NaNs and removes any with NaN > threshold
def clean_nan(data_df, threshold=0.3, na_str=None):

	# create copy of data DF
	copyData = data_df.copy()

	data_cols = list(copyData.columns)

	if na_str:
		copyData = copyData.replace(na_str, np.NaN)
	else:
		copyData = copyData.replace('NA', np.NaN)

	col_NANs = [copyData.loc[:, x].isna().sum() for x in data_cols]
	col_NANPct = [x / len(copyData.index) for x in col_NANs]

	copyData.loc['NaN %'] = col_NANPct
	na_thresholdCheck = copyData.loc['NaN %'] < threshold
	trunc_dataDF = copyData.loc[:, na_thresholdCheck]
	trunc_dataDF.drop(index='NaN %', inplace=True)	

	return trunc_dataDF

# this function returns, given an input data DF, the K-NN imputed DF
def impute_data(data_df):

	# create copy of data DF
	copyData = data_df.copy()

	imputer = KNNImputer() 	# default params, n_neighbours = 5, etc.
	knn_copyData = imputer.fit_transform(copyData)
	imputed_copyDF = pd.DataFrame(knn_copyData, index=copyData.index, columns=copyData.columns)

	return imputed_copyDF

# define main function, which will clean and impute the CCLE metabolomics & transcriptomics data used in mGSH classifier code
def main():

	# load data
	# path = r'\data'
	path = r'C:\Users\lkenn\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\data'
	# gene_df = pd.read_csv(rf'{path}\BroadInstitute-CCLE_RNAseq_rsem_genes_tpm_20180929_csv.csv', index_col=0)
	# metab_df = pd.read_excel(rf'{path}\41591_2019_404_MOESM2_ESM.xlsx', sheet_name='1-raw data', index_col=0, usecols='A:EG') 	# raw metab data with lipids removed

	# # process metabolomic data
	# print(f'Original df:\n{metab_df}')

	# # remove CCLs, then metabs with STD dev. = 0 across samples
	# temp_metabDF = check_stdDev(metab_df, check_axis=1, threshold=0)
	# temp_metabDF = check_stdDev(temp_metabDF, check_axis=0, threshold=0)

	# # remove CCLS, then metabs with >30% NAN samples from DF
	# temp_metabDF = clean_nan(temp_metabDF.T, threshold=0.30, na_str='NA')
	# temp_metabDF = clean_nan(temp_metabDF.T, threshold=0.30, na_str='NA')

	# # Finally, impute remaining NANs using K-NN and save cleaned data
	# cleaned_metabDF = impute_data(temp_metabDF)
	# print(f'Final metab df, KNN imputed:\n{cleaned_metabDF}')	

	# cleaned_metabDF.to_csv(rf'{path}\cleaned_ccle_metabolomics.csv')

	# generate spearman correlations for GSH, GSSG across transcripts
	# Also want to seperate transcriptomics data CCLs that are/aren't shared with metabolomics data
	# Common CCL data will be used in model training/testing, remaing CCL data can be used as validation set
	gene_df = pd.read_csv(rf'{path}\cleaned_ccle_rnaSeq.csv', index_col=0)
	metab_df = pd.read_csv(rf'{path}\cleaned_ccle_metabolomics.csv', index_col=0)

	print(f'DFs to be correlated:\nMetab:\n{metab_df}\ngene df:\n{gene_df}')

	# Now want only common CCLS for correlations
	common_CCLS = metab_df.index.intersection(gene_df.columns)
	print(f'\nCommon CCLS:\n{common_CCLS}')

	common_metabDF = metab_df.loc[common_CCLS]
	common_geneDF = gene_df.loc[:,common_CCLS].copy()
	uniq_geneDF = gene_df.drop(common_CCLS, axis=1)
	print(f'Common CCL data:\nMetab:\n{common_metabDF}\nGene:\n{common_geneDF}')

	# Now generate correlations between the data, and then threshold based on p-vals & set to absollute
	metab_cols = common_metabDF.loc[:, ['alpha-ketoglutarate', 'carnitine', 'glutamate']]
	print(f'metab cols:\n{metab_cols}')

	metab_corrs = metab_cols.apply(lambda x: common_geneDF.corrwith(x, axis=1, method='spearman'))
	print(f'Corrwith apply:\n{metab_corrs}')
	corr_pvals = metab_cols.apply(lambda x: common_geneDF.corrwith(x, axis=1, method=spearmanr_pval))

	threshMetab_corrs = threshold_corr(metab_corrs, corr_pvals)
	finalMetab_corrs = threshMetab_corrs.abs()
	print(f'Final metab corrs:\n{finalMetab_corrs}')

	finalMetab_corrs.to_csv(rf'{path}\nonGSH_metab_spearman.csv')
	# common_geneDF.to_csv(rf'{path}\cleaned_common_ccleRNAseq.csv')
	# uniq_geneDF.to_csv(rf'{path}\cleaned_unique_ccleRNAseq.csv')


	# # process transcriptomics data as with metabolomics
	# print(f'Original df:\n{gene_df}')

	# # remove CCLs, then transcripts with STD dev. = 0 across samples
	# temp_geneDF = check_stdDev(gene_df, check_axis=1, threshold=0)
	# temp_geneDF = check_stdDev(temp_geneDF, check_axis=0, threshold=0)

	# # remove CCLS, then transcripts with >30% NAN samples from DF
	# temp_geneDF = clean_nan(temp_geneDF.T, threshold=0.30, na_str='NA')
	# temp_geneDF = clean_nan(temp_geneDF.T, threshold=0.30, na_str='NA')

	# # Finally, impute remaining NANs using K-NN and save cleaned data
	# cleaned_geneDF = impute_data(temp_geneDF)
	# print(f'Final gene df, KNN imputed:\n{cleaned_geneDF}')		

	# cleaned_geneDF.to_csv(rf'{path}\cleaned_ccle_rnaSeq.csv')

main()



### !! TO DO !!
###	normalize data? Fisher transform after correlation generated, so no norm needed.

####																								  ####
##	Source code for above, taken from preporcessing.py and CCLE_metab_SLC_NBclassifier_GET_FEATURES.py 	##
####																								  ####

# ## Need to examine proteomics data and manage NaN values
# ## Check amount of NaNs per protein, and NaNs per CCL. Consider removing these pns/ccls from data if NaNs >25%
# def nan_and_imputeData(prot_data):

# 	print(f'Initial prot_data, including NaN row/cols:\n{prot_data}\nnew DF shape: {prot_data.shape}')

# 	# First, remove NaN cols (CCLs) to preserve protein data
# 	prot_cols = list(prot_data.columns)

# 	ccl_NANs = [prot_data.loc[:, x].isna().sum() for x in prot_cols] 	# get na counts across cols
# 	ccl_NANPct = [x / len(prot_data.index) for x in ccl_NANs] 		# get NaN as % of col length

# 	drop_idxs = []
# 	for idx, percent in enumerate(ccl_NANPct):
# 		if percent >= 0.30:
# 			drop_idxs.append(idx)
# 	prot_data.drop(prot_data.columns[drop_idxs], axis=1, inplace=True)	# remove these col indexes from prot df

# 	# repeat for rows w/ update DF
# 	prot_rows = list(prot_data.index)

# 	pn_NANs = [prot_data.iloc[x].isna().sum() for x in range(0,len(prot_rows))]	# get na counts across rows note: use iloc due to duplicate labels in rows

# 	pn_NANPct = [x / len(prot_cols) for x in pn_NANs] 	# get percent of axis as nans

# 	# if nan % is above threshold (e.g. 30%) drop these cols/rows from data
# 	drop_idxs = []
# 	for idx, percent in enumerate(pn_NANPct):
# 		if percent >= 0.30:
# 			drop_idxs.append(idx)
# 	prot_data.drop(prot_data.index[drop_idxs], inplace=True)	# drop these row indexes from prot data

# 	print(f'\nNew prot_data, with nan row/cols removed:\n{prot_data}\nnew DF shape: {prot_data.shape}')

# 	# now that we have nan vectors removed, need to impute remaining nan values in rows
# 	# use K-NN algorithm from sci-kit for imputing these values (better than mean imputation)

# 	imputer = KNNImputer() 	# default params, n_neighbours = 5, etc.
# 	knn_protData = imputer.fit_transform(prot_data)
# 	imputed_protDF = pd.DataFrame(knn_protData, index=prot_data.index, columns=prot_data.columns)

# 	return imputed_protDF
 

# 	#Read in the input file name as pd df
# 	raw_df = pd.read_csv(filename, index_col=0, na_filter=False)
# 	raw_df = raw_df.replace('NA',np.NaN) #Replace NA with np.NaN for sklearn functions
# 	#sklearn methods return np_arrays, get df index and column to be used in output
# 	df_indexes = list(raw_df.index.values)
# 	df_columns = list(raw_df.columns.values)

# 	#impute missing values using k-Nearest Neighbours, sklearn.KNNImputer
# 	imputer = KNNImputer()
# 	imputed_array = imputer.fit_transform(raw_df)

# 	#log scale data then normalize using np.log10() and skpr.normalize() functions
# 	log_array = np.log10(imputed_array)
# 	processed_array = skpr.normalize(log_array)

# 	#Create dataframe from the preprocessed data
# 	preprocessed_df = pd.DataFrame(data=processed_array, index=df_indexes, columns=df_columns)

# 	return(preprocessed_df)