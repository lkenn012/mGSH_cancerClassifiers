## Kennedy et al. 2024
#
# Subsetting of TCGA RNAseq by tumor/disease type via ID mapping.

# import modules
import pandas as pd

'''
for validation of the function annotation framework, we want to train and evaluate predictions of models
 trained on TCGA data to compare to CCLE. We need to identify TCGA samples that are comparable to CCLE. 
We will select TCGA tumor types which have corressponding cell lines in the CCLE for TCGA subsets.
'''

# DONE load TCGA tissueSourceSite, diseaseStudy -> map IDs to names to symbols
# load JUST the column names for TCGA bulk data -> map sample names to disease symbols
# DEFINE a list of desired disease symbols -> select rnaseq samples with disease
# save subset DFs for further use.


# define a main function
def main():

	# For the whole TCGA dataset, we would like to get common cancer types to those in CCLE
	# Luckily, others have matched these already and provide the mapping (CCLE_meta.txt)
	# CCLE_meta.txt downloaded from: https://github.com/katharineyu/TCGA_CCLE_paper
	# We can pull only the TCGA disease types that appear in CCLE for fair comparison

	ccle_metaDF = pd.read_csv(rf'data\CCLE_meta.txt', index_col=0, delimiter='\t')
	print(f'ccle meta:\n{ccle_metaDF}')
	disease_ids = ccle_metaDF['disease'].unique()

	# load TCGA code tables obtained from: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables
	tcga_tissueDF = pd.read_csv(rf'data\tissueSourceSite.tsv', delimiter='\t')
	tcga_diseaseDF = pd.read_csv(rf'data\diseaseStudy.tsv', delimiter='\t')

	# join source codes to disease codes
	tcga_codesDF = pd.merge(tcga_tissueDF, tcga_diseaseDF, on=['Study Name'], how='inner')
	print(f'tcga_codesDF:\n{tcga_codesDF}')

	# Now we want to get subsets of TCGA RNAseq by abbreviation
	# First load the sample names so we can map abbreviations to IDs for subsetting.
	# !!! NOTE: Due to file size, the following file is not included in github repo
	# 			Downloaded from: https://gdc.cancer.gov/about-data/publications/pancanatlas
	tcga_sampleDF = pd.read_csv(rf'data\EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', delimiter='\t', header=None, nrows=1)
	print(f'tcga_sampleDF:\n{tcga_sampleDF}')

	# Sample IDs are "TCGA-{source}-{participant}-{sample}{vial}-{portion}{analyte}-{plate}-{center}"
	# We need only {source} to identify columns for our disease of interest
	tcga_sampleDF.loc['TSS Code'] = tcga_sampleDF.iloc[0].str.slice(5,7) 	# two character code
	tcga_sampleInfoDF = pd.merge(tcga_sampleDF.T, tcga_codesDF.loc[:,['TSS Code','Study Abbreviation']], how='left', on=['TSS Code'])

	# Given any cancer types of interest, get the corresponding sample IDs and RNAseq values
	all_disease = []
	disease_idxs = [0] 	# list of idxs for the diseases we want. Add 0 for gene_ids
	for abbr in disease_ids:
		print(f'running {abbr} disease...')
		disease_idxs = tcga_sampleInfoDF.index[tcga_sampleInfoDF['Study Abbreviation'] == abbr]
		disease_idxs = disease_idxs.append(pd.Index(data=[0])) 	# need to add first column containing gene ids
		disease_df = pd.read_csv(rf'data\EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', delimiter='\t', usecols=disease_idxs, index_col=0)

		# Note: gene IDs are given as {gene symbol}|{NCBI}, split these for downstream tasks
		disease_df.reset_index(inplace=True)
		disease_df[['Gene Symbol', 'NCBI']] = disease_df.loc[:,'gene_id'].str.split('|', n=1, expand=True)
		
		# Some formatting to keep columns organized, and remove redundant column
		disease_df.set_index('Gene Symbol', inplace=True)
		disease_df.drop('gene_id', axis=1, inplace=True)
		col = disease_df.pop('NCBI')
		disease_df.insert(0, 'NCBI', col)

		# Append to our list of all disease data
		all_disease.append(disease_df)

		# # Save output
		# disease_df.to_csv(rf'data\TCGA_{abbr}_rnaseq.csv')


	# We want to get a DF containg all data:
	combined_df = pd.concat(all_disease)
	print(f'combined_df:\n{combined_df}')
	disease_df.to_csv(rf'data\TCGA_allDisease_rnaseq.csv')

	return

##############
#	RUN MAIN()
##############
main()