## Code to extract interior/tunnel residues from caver results for SLC25 proteins for local structure alignments

# import modules
import os
import re
import pandas as pd

# define function to read the caver outputs for a given protein and extract the tunnel-lining residues
def get_residues(outputPath, tunnel_num=1):

	# initialize holers for residue information
	tunnels = []
	current_tunnel = []

	# open file containing our residues
	with open(rf'{outputPath}\analysis\residues.txt') as resiF: 	# within the caver output folder, the analysis folder contains 'residues' file with the info we need
		for line in resiF:

			# Check if we are at line with new tunnel residue information
			if re.match("^== Tunnel cluster", line):


				# If we already have tunnel data, add it to our list (excluding "== Tunnel cluster ==" line)
				if current_tunnel:
					tunnels.append(current_tunnel[1:])

				# Initialize new list
				current_tunnel = [line.strip()]

			# If we are not at new tunnel, append line (if it is data)
			elif re.match("  [A-Z]", line):
				current_tunnel.append(re.split("  +", line.strip()))  # columns of data are seperated by 4-6 spaces, this regex splits by "2 or more spaces"

		# Add final tunnel data
		if current_tunnel:
			tunnels.append(current_tunnel[1:])

	print(f'tunnels:\n{tunnels}')
	print(f'first tunnel:\n{tunnels[tunnel_num-1]}')

	# Since there can be multiple tunnels identified by CAVER, we will need to select one tunnel to get residues for (specified by 'tunnel_num', default = first tunnel)
	residues = [int(re.findall(r'[0-9]+', line[1])[0]) for line in tunnels[tunnel_num-1]]

	return residues

###
##	MAIN
#

def main():

	# define path to caver results
	data_path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs\PyMol'

	caver_results = [name for name in os.listdir(rf'{data_path}\caver_output')]
	print(f'Caver output folders:\n{caver_results}')
	tunnel_clust = [1]*len(caver_results) 	# since there can be multiple identified tunnels, specify which one we want residues for (tunnel 1 is usually best)

	# Based on 'SLC25_tunnelResidues.xlsx' notes, may want to select different tunnel for some proteins
	tunnel_clust[caver_results.index('SLC25A10_AF')] = 3
	tunnel_clust[caver_results.index('SLC25A19_AF')] = 2
	tunnel_clust[caver_results.index('SLC25A20_AF')] = 2
	tunnel_clust[caver_results.index('SLC25A24_AF')] = 2

	print(f'Caver output folders:\n{caver_results}\nDesired tunnels:\n{tunnel_clust}')

	tunnel_residues = []
	for i, out in enumerate(caver_results):
		tunnel_residues.append(get_residues(rf'{data_path}\caver_output\{out}', tunnel_clust[i])) 	# pull residue data for specified tunnel in protein

	res_df = pd.Series(data=tunnel_residues, index=caver_results)
	print(f'results df:\n{res_df}')
	res_df.to_pickle(rf'{data_path}\SLC25_tunnelRes.pkl')

main()