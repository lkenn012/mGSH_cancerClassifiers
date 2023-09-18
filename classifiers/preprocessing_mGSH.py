## Pre-processing data for features in mGSH classifier models. Cleans and imputes missing values from CCLE transcriptomics and metabolomics
## Generate correlations across metabolomics and transcriptomics

##
## import modules
##

import pandas as pd
import numpy as np

import sklearn.preprocessing as skpr
from sklearn.impute import KNNImputer

from  main_mGSH import spearmanr_pval, threshold_corr	# for generating correlations between GSH/GSSG and transcripts

##
## Define methods for data preprocessing for classifier models
##

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
# default NaN threshold is 0.30 (used in paper), also provides optional argument for a NaN str to be replaced
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
# Also generate correlations between gene transcriptomics data and metabolite metabolomics data to be used in classifiers ('glutathione reduced', 'glutathione oxidized', 'alpha-ketoglutarate', 'carnitine', 'glutamate')
def main():

	# load data
	data_path = "path" 	# !! PLACEHOLDER !! replace with the path to your dat directory
	gene_df = pd.read_csv(rf'{data_path}\BroadInstitute-CCLE_RNAseq_rsem_genes_tpm_20180929_csv.csv', index_col=0) 	# CCLE raw transcriptomics data
	metab_df = pd.read_excel(rf'{data_path}\41591_2019_404_MOESM2_ESM.xlsx', sheet_name='1-raw data', index_col=0) 	# CCLE raw metab data

	# process metabolomic data

	# remove CCLs, then metabs with STD dev. = 0 across samples
	temp_metabDF = check_stdDev(metab_df, check_axis=1, threshold=0)
	temp_metabDF = check_stdDev(temp_metabDF, check_axis=0, threshold=0)

	# remove CCLS, then metabs with >30% NAN samples from DF
	temp_metabDF = clean_nan(temp_metabDF.T, threshold=0.30, na_str='NA')
	temp_metabDF = clean_nan(temp_metabDF.T, threshold=0.30, na_str='NA')

	# Finally, impute remaining NANs using K-NN and save cleaned data
	cleaned_metabDF = impute_data(temp_metabDF)

	# process transcriptomics data as with metabolomics

	# remove CCLs, then transcripts with STD dev. = 0 across samples
	temp_geneDF = check_stdDev(gene_df, check_axis=1, threshold=0)
	temp_geneDF = check_stdDev(temp_geneDF, check_axis=0, threshold=0)

	# remove CCLS, then transcripts with >30% NAN samples from DF
	temp_geneDF = clean_nan(temp_geneDF.T, threshold=0.30, na_str='NA')
	temp_geneDF = clean_nan(temp_geneDF.T, threshold=0.30, na_str='NA')

	# Finally, impute remaining NANs using K-NN and save cleaned data
	cleaned_geneDF = impute_data(temp_geneDF)
	print(f'Final gene df, KNN imputed:\n{cleaned_geneDF}')		

	cleaned_geneDF.to_csv(rf'{path}\cleaned_ccle_rnaSeq.csv') 	# optionally, save the cleaned data for future use
	cleaned_metabDF.to_csv(rf'{path}\cleaned_ccle_metabolomics.csv')
	

	# generate spearman correlations for metabolites of interest for classsifier model features
	# Also want to seperate transcriptomics data CCLs that are/aren't shared with metabolomics data

	# Now want only common CCLs for correlations
	common_CCLS = cleaned_metabDF.index.intersection(cleaned_geneDF.columns)

	common_metabDF = cleaned_metabDF.loc[common_CCLS]
	common_geneDF = cleaned_geneDF.loc[:,common_CCLS].copy()
	uniq_geneDF = cleaned_geneDF.drop(common_CCLS, axis=1)

	# Now generate correlations between the data, and then threshold based on p-vals & set to absollute
	metab_cols = common_metabDF.loc[:, ['glutathione reduced', 'glutathione oxidized', 'alpha-ketoglutarate', 'carnitine', 'glutamate']]

	# Now we can calculate spearman correlations for each column (metabolite) of data in metabolomics data, then threshold based on p-values
	metab_corrs = metab_cols.apply(lambda x: common_geneDF.corrwith(x, axis=1, method='spearman'))
	corr_pvals = metab_cols.apply(lambda x: common_geneDF.corrwith(x, axis=1, method=spearmanr_pval))

	threshMetab_corrs = threshold_corr(metab_corrs, corr_pvals)
	finalMetab_corrs = threshMetab_corrs.abs()

	finalMetab_corrs.to_csv(rf'{data_path}\ccle_metab_spearman.csv') 	# Save outputs for use in models


# Run code for preprocessing data
main()
