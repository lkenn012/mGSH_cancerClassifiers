### Code for conducting 3D structural comparisons using PyMOL for comparisons of known mGSH/non-mGSH SLC25 transporters with predicted transporters
# 	SLC25 structures are aligned via the TM-align algorithm implemented in the TMalign module (https://zhanggroup.org/TM-align)

##
## import modules
##

import pymol as pm
import os
import re

import pandas as pd

from psico.fitting import tmalign 	# (https://pymolwiki.org/index.php/TMalign)


##
## Define methods for structure alignment
##

# define function to load and name proteins in PyMol for structural alignment
def load_structs(pn_1, pn_2, path):

	# get names as structure file name without the file suffix (.pdb)
	pn1_name = pn_1[:-4]
	pn2_name = pn_2[:-4]

	# load pymol objects of structures 	
	pm.cmd.load(f'{path}\{pn_1}', pn1_name)
	pm.cmd.load(f'{path}\{pn_2}', pn2_name)

	return pn1_name, pn2_name

# define function to align two structures using pm.cmd.super and return the RMSD values
def super_RMSD(mobile_pn, target_pn):
	return pm.cmd.super(mobile=mobile_pn, target=target_pn)[0]

# given two structure files, align the structures in pymol
def struct_align(mobile_files, target_file, file_path):

	super_results = {}
	for struct_file in mobile_files:
		mobile_name, target_name = load_structs(struct_file, target_file,  file_path) 	# load structs

		super_results[mobile_name] = super_RMSD(mobile_name, target_name) 	# get structural alignment score (RMSD)

		pm.cmd.delete(mobile_name) 	# delete pymol objects after screen complete
		pm.cmd.delete(target_name)

	return pd.DataFrame.from_dict(super_results, orient='index', columns=[target_name]) # return alignment results as 1D DF with col=target and rows=mobile


# define function to get the list of residues for a given gene name from a data file
def get_pnRes(pn_name, res_data):

	pn_res = res_data.loc[pn_name] 	# gets list of residues for specified protein 

	return '+'.join(str(x) for x in pn_res) 	# returns residues as string of "res1+res2+res3+...+resN" which is formatted for pymol arg


# define function to make selections for pymol structures for specified residues to be used in a local strucutral alignment
def select_residues(mobile_obj, mobile_residues, target_obj, target_residues):

	selected_mobileName = f'{mobile_obj}_tunnel'
	selected_targetName = f'{target_obj}_tunnel'

	# create selections for each protein to be aligned
	selected_mobile = pm.cmd.select(selected_mobileName, f'{mobile_obj} and resi {mobile_residues}') 
	selected_target = pm.cmd.select(selected_targetName, f'{target_obj} and resi {target_residues}') 

	return selected_mobileName, selected_targetName


# given two structure files, align the structures in pymol using tmA
def struct_TMalign(mobile_files, target_file, file_path, align_res=pd.Series()):

	tm_results = {}
	for struct_file in mobile_files:
		mobile_name, target_name = load_structs(struct_file, target_file,  file_path) 	# load structs

		# if we have a list of specific residues to align, need to select those residues for alignment
		if not align_res.empty and not align_res.isin([True, False]).all():

			# Need to extract just the gene name for match to residue data
			print(f'mobile_name: {mobile_name}')
			mobile_res = get_pnRes(pn_name=re.findall(r"^SLC25A[0-9]+_[a-zA-Z]+", mobile_name)[0], res_data=align_res) 	# get the desired residues in an acceptable format for pymol
			target_res = get_pnRes(pn_name=re.findall(r"^SLC25A[0-9]+_[a-zA-Z]+", target_name)[0], res_data=align_res) 	# get the desired residues in an acceptable format for pymol

			# Now get residues for alignment for each structure and create selection of pymol objects
			mobile_name, target_name = select_residues(mobile_obj=mobile_name, mobile_residues=mobile_res, target_obj=target_name, target_residues=target_res)

		tm_results[mobile_name] = tmalign(mobile_name, target_name) 	# get structural alignment score (RMSD)

		pm.cmd.show('mesh', mobile_name)
		pm.cmd.show('mesh', target_name)
		pm.cmd.delete(mobile_name) 	# delete pymol objects after screen complete
		pm.cmd.delete(target_name)

	return pd.DataFrame.from_dict(tm_results, orient='index', columns=[target_name]) # return alignment results as 1D DF with col=target and rows=mobile NOTE: TM-align result is normalized to target sequence length



#### 
### MAIN
##
def main():

	# laod data
	data_path = "path" 	## !! PLACEHOLDER !! replace this with the path to your data directory
	struct_path = f'{data_path}\SLC25_structures' 	# path for specific models of interest

	local_alignResDF = pd.read_pickle(rf"{data_path}\SLC25_tunnelRes.pkl") 	# load data for each SLC25 proteins tunnel residues for local alignment (obtained from raw CAVER ouputs via mGSH_CAVER_resis.py)

	struct_fs = []
	# pulls all file in structure folder
	for f in os.listdir(struct_path):
		struct_fs += [f]

	# now, for each of our targets we can align the rest of our structures using pymol 'super' command
	align_results = []
	for target in struct_fs:
		screen = [struct for struct in struct_fs if struct != target] 	# screen against all other structures

		align_results += [struct_TMalign(screen, target, struct_path, local_alignResDF)] 	# tm alignment with specified residues to align	

	results_df = align_results[0].join(align_results[1:])
	results_df.to_csv(rf'{data_path}\SLC25_localTMalign.csv')
	
	# Get out!
	pymol.cmd.quit()

###
# run main()
main()
