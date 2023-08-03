### Code for getting DeepGO annotations from a list of Ensembl gene IDs

# import modules
import requests
import time
import pandas as pd
import numpy as np

import get_FASTAseqs as ensgDB

# load ensembl IDs
path = r"C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code"
ensg_ids = pd.ExcelFile(rf"{path}\data\mGSH_labeledGenes_HighConAnnots.xlsx")


# define function to call the DeepGO web server to predict a given sequence
def get_DeepGO(FASTA, annot_threshold=0.3):

	headers = {'Content-Type': 'application/json'}
	data = {'version': '1.0.13', 'data_format': 'fasta', 'data': FASTA, 'threshold': annot_threshold}
	r = requests.post('https://deepgo.cbrc.kaust.edu.sa/deepgo/api/create', json=data, headers=headers) # , allow_redirects=False)

	result = r.json()
	return result

# define a function to check deepGO annotation set for a specific GO term for calculating sensitivity 
def deepGO_sensitivity(go, annotations, go_class=None):

	# given a json return from deepGO get the predicted functions
	try:
		go_annots = annotations['predictions'][0]['functions']
	except KeyError as e:
		if str(e) == '\'predictions\'': # get this error when no protein sequence is available for our original gene
			return 'NA'

	if go_class:
		for go_group in go_annots:
			if go_group['name'] == go_class:
				for functions in go_group['functions']:
					if functions[0] == go:
						return functions[2]

	else:
		for go_group in go_annots:
			for functions in go_group['name']['functions']:
				if functions[1] == go:
					return functions[2]
	return 0

# define main function to get FASTA seqs and then DeepGO annotations
def main():

	# specify annotations of interest
	annot_ids = pd.read_excel(ensg_ids, sheet_name='GSH Ensembl')
	print(f'annot_ids:\n{annot_ids}')

	# annot_ids = annot_ids.sample(n=150, random_state=42)
	# print(f'random subset of 150 IDs:\n{annot_ids}')


	annots = []
	# Iterate over our list of annot_ids
	for ensg_id in annot_ids.iloc[:,0].to_list():

		time.sleep(1)
		canonTrans_id = ensgDB.get_canonicalTranscript(ensg_id.strip()) 	# get transcript ID

		# check if esng_id is found in ensembl, use corresponding canonical transcript to get fasta
		if canonTrans_id:

			fasta = ensgDB.get_proteinSeq(canonTrans_id) 	# get FASTA sequence

			deepGO_annot = get_DeepGO(fasta) 	# get deepGO annotation

			annots.append({'Ensembl gene ID': ensg_id, 'DeepGO annotation': deepGO_annot})

		else:
			print(f' **   ID not found in ensembl: {ensg_id}')

	# convert to DF and save as pkl file for analysis
	annot_df = pd.DataFrame(annots)
	annot_df.to_pickle(rf'{path}\outputs\gsh_deepGOAnnots.pkl')

# function for checking annotations for a specific go term, when we already have an annotation file
def main2():

	## Specify parameters of interest
	#

	go_interest = ('GO:0005739', "Cellular Component") 	# mitochondrial localization
	# go_interest = ('GO:0006749', "Biological Process") 	# glutathione metabolic process
	# go_interest = ('GO:0022857', "Molecular Function") 	# transmembrane transporter activity

	f_name = 'randMito_deepGOAnnots' 	# mitochondrial term annotations
	# f_name = 'gsh_deepGOAnnots' 	# gsh term annotations
	# f_name = 'randTransp_deepGOAnnots' 	# transporter term annotations



	# load data
	annot_df = pd.read_pickle(rf'{path}\outputs\{f_name}.pkl')

	GO_probs = []
	for i, annot in annot_df.iterrows():

		GO_probs.append(deepGO_sensitivity(go=go_interest[0], annotations=annot[1], go_class=go_interest[1])) 	# check for gsh metabolic process go term

	print(f'GO probs:\n{GO_probs}')

	annot_df[f'{go_interest[0]} prob.'] = GO_probs

	# sensitivity calculation
	annotated = [i for i in GO_probs if i != 'NA']

	gshPred_count = len([i for i in annotated if i > 0.50])
	print(f'GO sensitivity: {gshPred_count / len(annotated)}')

	print(f'annot_df:\n{annot_df}')
	annot_df.to_csv(rf'{path}\outputs\{f_name}_wConfidence.csv')
	return



##
## RUN MAIN CODE
##

# main()
main2()