### Code for getting DeepGOWeb annotations from a list of Ensembl gene IDs
### Calls the ensembl API to get transcript IDs and then protein FASTA sequences from a given gene ID, then calls DeepGOWeb API to annotate the sequences

# import modules
import requests
import time
import pandas as pd
import numpy as np

import get_FASTAseqs as ensgDB

# define function to call the DeepGOWeb server to annotate a given sequence with minimum classification threshold = {annot_threshold}
def get_DeepGO(FASTA, annot_threshold=0.3):

	headers = {'Content-Type': 'application/json'}
	data = {'version': '1.0.13', 'data_format': 'fasta', 'data': FASTA, 'threshold': annot_threshold}
	r = requests.post('https://deepgo.cbrc.kaust.edu.sa/deepgo/api/create', json=data, headers=headers)

	return r.json() 	# Return as json

# define a function to check deepGO annotation set for a specific GO term for calculating sensitivity 
def get_GOAnnot(go, annotations, go_class=None):

	# given a json returned from deepGOWeb get the predicted functions
	try:
		go_annots = annotations['predictions'][0]['functions']
	except KeyError as e:
		if str(e) == '\'predictions\'': # get this error when no protein sequence is available for our original gene
			return 'NA'
			
	# check the probability of the annotation of our id of interest and return it
	if go_class:
		for go_group in go_annots:
			if go_group['name'] == go_class:
				for functions in go_group['functions']:
					if functions[0] == go:
						return functions[2]
						
	# if no GO class is specified check all functions
	else:
		for go_group in go_annots:
			for functions in go_group['name']['functions']:
				if functions[1] == go:
					return functions[2]
					
	# if the go term of interest is not found, return probability = 0
	return 0
	
# define function to take in a list of genes for a given GO term, from our annotation file, and get corresponding FASTA sequence and deepgo annotations
def annotate_Genes(annot_ids, n_samples=False):

	# Optionally, if there are many genes for our GO term of interest, may want to annotate a subset due to slow speed of DeepGOWeb
	if n_samples:
		annot_ids = np.random.choice(annot_ids, n_samples)
		

	# Iterate over our list of annot_ids
	annots = [] 	# holder for our annotations
	for ensg_id in annot_ids:

		time.sleep(1) 	# delays API calls
		canonTrans_id = ensgDB.get_canonicalTranscript(ensg_id.strip()) 	# get transcript ID

		# check if esng_id is found in ensembl, use corresponding canonical transcript to get fasta
		if canonTrans_id:
			fasta = ensgDB.get_proteinSeq(canonTrans_id) 	# get FASTA sequence
			deepGO_annot = get_DeepGO(fasta) 	# get deepGO annotation
			
			annots.append({'Ensembl gene ID': ensg_id, 'DeepGO annotation': deepGO_annot}) 	# save annotations

		else:
			print(f' **   ID not found in ensembl: {ensg_id}')

	# return the annotated file as a dataframe
	return pd.DataFrame(annots)

# define function for getting deepGO annotations from a list of ensembl gene IDs, and calculate deepGO annotation sensitivity for term of interest
def deepGOAnnots_fromIDs(ensgIDs, go_interest, annot_samples=False, class_thresh = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]):

	# Get the DeepGO annotations from our list of ids
	annot_df = annotate_Genes(annot_ids=ensgIDs, n_samples=annot_samples)

	# Now, calculate the performance of DeepGO for annotating our term of interest
	GO_probs = []
	for i, annot in annot_df.iterrows():
		GO_probs.append(get_GOAnnot(go=go_interest[0], annotations=annot[1], go_class=go_interest[1])) 	# check for annotations for our go term of interest

	# add this annotation as a new field in our annotated df
	annot_df[f'{go_interest[0]} prob.'] = GO_probs

	# sensitivity calculation
	annotated = [i for i in GO_probs if i != 'NA'] 	# get all annotated genes

	sensitivities = []
	for thresh in class_thresh:
		annotPred_count = len([i for i in annotated if i > thresh]) 	# count annotations above threshold
		sensitivities.append({thresh: annotPred_count/len(annotated)})
		
	# return the deepGO annotations, and corresponding sensitivities for our go term of interest
	return annot_df, sensitivities



##
## DEFINE AND RUN MAIN for an example of deepGO annotation
##

# This is an example method to run the functions from this file to annotate a set of Gene IDs by DeepGOWeb, and then evaluate the sensitivity for a specified GO term of interest.
def main():
	
	## load annotated ensembl IDs for GO terms of interest as pandas DF
	data_path = 'path' 	# Define the path to the annotation file or another xlsx file with ENSG IDs
	
	ensg_ids = pd.ExcelFile(rf"{data_path}\mGSH_labeledGenes_HighConAnnots.xlsx")
	goTerm_ids = pd.read_excel(ensg_ids, sheet_name='GSH Ensembl') 	# GSH annotated IDs

	## Specify the GO terms and class of interest for quantifying DeepGOPlus annotation performance, below are those used for the reported analysis.
	go_term = ('GO:0006749', "Biological Process") 	# glutathione metabolic process
	# go_term = ('GO:0005739', "Cellular Component") 	# mitochondrial localization
	# go_term = ('GO:0022857', "Molecular Function") 	# transmembrane transporter activity

	annotated_df, annot_sensitivity = deepGOAnnots_fromIDs(ensgIDs=goTerm_ids.iloc[:,0].to_list(), 	# convert pandas object to array
					    go_interest=go_term, 
					    annot_samples=False
					   )

	## Now have our final results, the DeepGOWeb annotated genes: {annotated_df}, and the DeepGOWeb sensitivity for our specified GO term: {annot_sensitivity}

## RUN MAIN
main()
