## Code for generating figures from mGSH outputs

# import modules
import pandas as pd
import numpy as np
from ast import literal_eval
import os

import re

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime # for output formatting

import main_mGSH as mGSH	# python file containing functions for building the model


# define function to pull our data and generate a roc plot of each set of values
def main(data_path, rocFile, plt_title='ROC curve'):

	# load data from out file
	roc_df = pd.read_pickle(rf'{data_path}\{rocFile}.pkl') #pkl object to retain list formatting

	# plot ROC curve
	fig, ax = plt.subplots() 	# generate plot axis

	roc_curve = mGSH.plot_ROC(plot_ax=ax, # generate roc plot
		plot_data=roc_df,
		plot_x='FPR',
		plot_y='TPR',
		title=plt_title)

	# Save plot
	date = datetime.today().strftime('%d-%m-%Y') 	# get a string of the current date via strftime()

	# fig.savefig(rf'{data_path}\figures\{rocFile}Plot_{date}.png')
	fig.savefig(rf'{data_path}\{plt_title}_{date}.png')

	fig.clf()

# main() funct which can handle plotting multiple rocs on single axis
# define function to pull our data and generate a roc plot of each set of values
def main_2(data_path, rocFiles, model_params=None, plt_title='ROC curve'):

	# # load data from out file
	# roc_dfs = [pd.read_pickle(rf'{data_path}\{f}.pkl') for f in rocFiles] # load roc data(s)
	# print(f'roc_dfs:\n{roc_dfs[0]}\n{roc_dfs[1]}')
	# plot ROC curve
	fig, ax = plt.subplots() 	# generate plot axis

	roc_dfs = []
	for i,f in enumerate(rocFiles):
		rocDF = pd.read_pickle(rf'{data_path}\{f}.pkl')

		# Add column at end specifiying how to differentiate models for plot styling
		if model_params:
			rocDF['Model specification'] = model_params[i]

		# Also need create column from index, which specifies number of feature genes in a given model
		feature_genes = [mod_feats.split()[1] for mod_feats in rocDF.index.tolist()] # split string to get the number of features (REGEX not workign)

		# Now use this as column
		rocDF['Feature genes'] = feature_genes

		roc_dfs.append(rocDF)

	final_rocDF = pd.concat(roc_dfs) 	# concat dfs into one to plot on one

	# plot these data on roc
	if model_params:
		roc_curve = mGSH.plot_ROC(plot_ax=ax, # generate roc plot
			plot_data=final_rocDF,
			plot_x='FPR',
			plot_y='TPR',
			title=plt_title,
			plot_style='Model Specification')

	else:
		roc_curve = mGSH.plot_ROC(plot_ax=ax, # generate roc plot
			plot_data=final_rocDF,
			plot_x='FPR',
			plot_y='TPR',
			title=plt_title)

				# for roc in roc_dfs:
	# 	roc_curve = mGSH.plot_ROC(plot_ax=ax, # generate roc plot
	# 		plot_data=roc,
	# 		plot_x='FPR',
	# 		plot_y='TPR',
	# 		title=plt_title)

	# Save plot
	date = datetime.today().strftime('%d-%m-%Y') 	# get a string of the current date via strftime()

	# fig.savefig(rf'{data_path}\figures\{rocFile}Plot_{date}.png')
	fig.savefig(rf'{data_path}\figures\{plt_title}_{date}.png')

	fig.clf()



# main() funct which can handle plotting multiple rocs on single axis
# define function to pull our data and generate a roc plot of each set of values
def main_3(data_path, rocFiles, model_params=None, plt_title='ROC curve'):

	# # load data from out file
	# roc_dfs = [pd.read_pickle(rf'{data_path}\{f}.pkl') for f in rocFiles] # load roc data(s)
	# print(f'roc_dfs:\n{roc_dfs[0]}\n{roc_dfs[1]}')

	# plot ROC curve
	fig, ax = plt.subplots() 	# generate plot axis
	sns.set_theme()

	roc_dfs = []
	for i,f in enumerate(rocFiles):
		rocDF = pd.read_csv(rf'{data_path}\{f}')

		# rename idxs for plotting with model details
		new_names = {}
		for mod_name in rocDF.index.tolist():
			if model_params:
				new_names[mod_name] = f'{mod_name.split(" ", 1)[1]} ({model_params[i]})' # edit column names for plotting to be more informative
			else:
				new_names[mod_name] = f'{mod_name.split(" ", 1)[1]}' # edit column names withoout additional information

		rocDF.rename(index=new_names, inplace=True)
		print(f'init rocDF:\n{rocDF}')
		rocDF.drop(rocDF.index[[-1,-2,-3]], axis=0, inplace=True)
		rocDF['Group'] = ['A', 'B', 'C'] # groups for coloring the plots
		print(f'final rocDF:\n{rocDF}')

		roc_dfs.append(rocDF)

	# # need to modify the first df values for early models
	# print(f'initial df:\n{roc_dfs[0]}')
	# off_one = lambda lst: [1 - element for element in lst]

	# roc_dfs[0]['TPR'] = roc_dfs[0]['TPR'].apply(off_one)
	# roc_dfs[0]['FPR'] = roc_dfs[0]['FPR'].apply(off_one)
	# roc_dfs[0]['AUC'] = 1 - roc_dfs[0]['AUC']
	# print(f'fixed df:\n{roc_dfs[0]}')


	mGSH.plot_ROC(plot_ax=ax, # generate roc plot
		plot_data=roc_dfs,
		plot_x='FPR',
		plot_y='TPR',
		title=plt_title,
		plot_style=True,
		plot_palette=["orange", "sandybrown", "peru", "lightgray", "darkgray", "dimgray"])	# mito palette: ["turquoise", "lightseagreen", "cadetblue", ...], # GSH palette: ["violet", "mediumpurple", "darkviolet",...], transport palette: ["orange", "sandybrown", "peru", ...]
	# 	# new_names = [f'{mod_feats.split(" ", 1)[1]} ({model_params[i]})' for mod_feats in rocDF.index.tolist()] # split string to get the number of features (REGEX not workign)
	# 	# print(f'new_names:\n{new_names}')
	# 	# rocDF.rename(new_names, inplace=True)

	# 	print(rocDF)
	# 	exit()
	# 	roc_dfs.append(rocDF)

	# final_rocDF = pd.concat(roc_dfs) 	# concat dfs into one to plot on one

	# # plot these data on roc
	# if model_params:
	# 	roc_curve = mGSH. (plot_ax=ax, # generate roc plot
	# 		plot_data=final_rocDF,
	# 		plot_x='FPR',
	# 		plot_y='TPR',
	# 		title=plt_title,
	# 		plot_style='Model Specification')

	# else:
	# 	roc_curve = mGSH.plot_ROC(plot_ax=ax, # generate roc plot
	# 		plot_data=final_rocDF,
	# 		plot_x='FPR',
	# 		plot_y='TPR',
	# 		title=plt_title)

				# for roc in roc_dfs:
	# 	roc_curve = mGSH.plot_ROC(plot_ax=ax, # generate roc plot
	# 		plot_data=roc,
	# 		plot_x='FPR',
	# 		plot_y='TPR',
	# 		title=plt_title)

	# Save plot
	date = datetime.today().strftime('%d-%m-%Y') 	# get a string of the current date via strftime()

	ax.get_legend().remove()
	# fig.savefig(rf'{data_path}\figures\{rocFile}Plot_{date}.png')
	fig.savefig(rf'{data_path}\{plt_title}_{date}.png')

	fig.clf()






# # run main with our desired output file path
# folder_path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs'
# # rocFile_name = 'NB_modelROC_13-12-2022_2'
# rocFile_name = 'SVC_mitoModelROC_17-02-2023'
# main(data_path=folder_path, rocFile=rocFile_name, plt_title='SVC_mitoModel ROC plot')

# # try multiple 
# folder_path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs'
# rocFile_names = ['SVC_GSHModelROC_23-02-2023', 'SVC_noGSHModelROC_23-02-2023']
# model_specs = ['with metabolomics', 'no metabolomics'] 	# specifies the differences between the models (if any) for how to differentiate in plot
# main_2(data_path=folder_path, 
# 	rocFiles=rocFile_names, 
# 	model_params=model_specs,
# 	plt_title='SVC_gshoModel ROC plot'
# 	)

# try multiple 


folder_path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs'
rocFile_names = [f for f in os.listdir(folder_path) if re.match(r'ROC_RF_.+GSH_highCon_Results.+.csv', f)]
print(f'rocFile_names:\n{rocFile_names}')
exit()
# rocFile_names = ['RF_transportModelROC_02-03-2023', 'RF_noTrsspTransportModelROC_10-03-2023'] 	# Mito (in Mito Models 17-02): ['RF_mitoModelROC_16-02-2023', 'RF_mitoModel_noCartaROC_17-02-2023'], GSH: ['RF_GSHModelROC_22-02-2023', 'RF_noGSHModelROC_23-02-2023']
model_specs = ['with metabolomics', 'without metabolomics'] 	# specifies the differences between the models (if any) for how to differentiate in plot
main_3(data_path=folder_path, 
	rocFiles=rocFile_names, 
	model_params=model_specs,
	plt_title='RF GSH model ROCs_manuscript'
	)