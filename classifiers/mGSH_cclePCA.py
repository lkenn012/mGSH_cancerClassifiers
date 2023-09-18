## This code is for analyzing and plotting CCLE transcriptomics correlations between groups (related/unrelated genes from GO) in mGSH models

## Get each dataset
## Normalize datasets (z-score, FC, log2FC)
## Combine datasets


# import modules
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA  		# stats and ML modules
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

import umap 	# another dimension reduction technique

import seaborn as sns 	# for plotting
import matplotlib.pyplot as plt

from random import sample
import functools

from main_mGSH import fisher 	# For fisher transformation of correlations for PCA

# group correlations into housekeeping, mito, and overlapping genes
# define function to format the related/unrelated genes as list and get subset the overlapping IDs
def get_geneSets(related_genes, unrelated_genes):

	rel_list = related_genes.iloc[:,0].tolist() 	# convert to lists from df for removing overlap
	unrel_list = unrelated_genes.iloc[:,0].tolist()

	# Get intersecting genes and remove overlap from each list
	overlap_set = set(rel_list).intersection(unrel_list)

	uniq_relList = list(set(rel_list) - overlap_set) 	# set difference to get uniq lists
	uniq_unrelList = list(set(unrel_list) - overlap_set)

	return uniq_relList, uniq_unrelList, list(overlap_set) 	# return each list

# Hierarch of all related, unrelated, overlapping genes?
def hierarch_map(data, mapping=None, axis=None):

	if axis: 	# assumes row mapping

		lut = dict(zip(mapping.unique(), ["darkcyan", "darkkhaki"])) 	#, "mediumseagreen"]))
		color_map = mapping.map(lut)

		hierarch = sns.clustermap(data,
			yticklabels=False,
			xticklabels=False, 
			row_cluster=False, # select which axis to cluster along (default is both axes)
			col_colors=color_map,
			method='ward') 	# ward method minimizes cluster variance

	else: 	# no mapping
		hierarch = sns.clustermap(data,
			yticklabels=False,
			xticklabels=False, 
			method='ward') 	# ward method minimizes cluster variance

	return hierarch

# Define function for creating dendrogram plots from hierarchical clustering
def hierarch_clustering(data, clusters=False, dist_thresh=0, leaf_labelDict=None):

	# can specify distance or cluster threshold, distance_threshold=0 creates full dnedrogram which we can use to select # of clusters for clustering
	if clusters:
		hierarch = AgglomerativeClustering(distance_threshold=None, n_clusters=clusters) 
	else:
		hierarch = AgglomerativeClustering(distance_threshold=dist_thresh, n_clusters=None) # can specify distance or cluster threshold, distance_threshold=0 creates full dnedrogram which we can use to select # of clusters for clustering

	# fit clustering
	clusters = hierarch.fit(data)

	# Create dendrogram plot
	# dendrogram uses model children_ and distances_ features
	dend_data = np.column_stack([clusters.children_, clusters.distances_, [0] * len(clusters.distances_)]).astype(float)	# (re: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html)
	print(f'dendrogram linkage matrix:\n{dend_data}')
	print(f'dend_data shape: {dend_data.shape}')

	# plot dendrogram of clusters with scipy.cluster.hierarchy
	if leaf_labelDict:
		label_ids = list(leaf_labelDict.keys()) 	# keys contain gene ids we wish to show
		dend_results = dendrogram(dend_data, labels=label_ids,leaf_rotation=90, leaf_font_size=2)	# full plot

		# get current axes to color the leaf labels
		ax = plt.gca()
		leaf_labels = ax.get_xmajorticklabels()

		for leaf in leaf_labels:
			leaf.set_color(leaf_labelDict[leaf.get_text()]) # define leaf color based on our dict with color info
	else:
		dend_results = dendrogram(dend_data) 	# full plot

	# # options for truncated plots
	# dendrogram(dend_data, truncate_mode='lastp') 	# truncated plot

	# if leaf_labelDict:
	# 	label_ids = list(leaf_labelDict.keys()) 	# keys contain gene ids we wish to show
	# 	print(f'label_ids:\n{label_ids}')
	# 	dend_results = dendrogram(dend_data, truncate_mode='level', p=4, labels=label_ids, leaf_rotation=45, leaf_font_size=6)	# truncate by level

	# 	# # get current axes to color the leaf labels
	# 	# ax = plt.gca()
	# 	# leaf_labels = ax.get_xmajorticklabels()
	# 	# print(f'leaf_labels:\n{leaf_labels}')

	# 	# for leaf in leaf_labels:
	# 	# 	print(f'leaf:\n{leaf}')
	# 	# 	leaf.set_color(leaf_labelDict[leaf.get_text()]) # define leaf color based on our dict with color info

	# # else:
	# # 	dend_results = dendrogram(dend_data, truncate_mode='level', p=4)	# truncate by level

	return dend_results

# given a dict of dicts containing group correlations, combine each dict into a df for plotting
def combine_groups(dicts):

	dfs = []
	for group, corrs in dicts.items():
		print(f'---\n{group} group for combine_groups():')
		df = pd.DataFrame.from_dict(corrs) 	# convert dict to df where rows=corrs, columns=groups
		print(f'---\ndf from corr dict:\n{df}')		
		group_df = df.stack().reset_index() 	# this converts df into col_0=level col_1=group, col_2=corrs
		group_df.columns = [0,'Group', group] 	# rename columns

		print(f'---\nformatted df:\n{group_df.iloc[:,[1,2]]}')
		dfs.append(group_df.iloc[:,[1,2]].set_index('Group')) 	# add to list of group dfs but remove first column (just has information from stacking)

	return dfs 	

	# now combine each group df into single group

# Given groups of genes, return dfs containing correlations across samples of all groups and a gorup identifier column
def group_sampleCorrs(data, group_dict, sample_size):

	# Create subset dict with only randomly sampled ids
	sample_dict = {}
	for group, ids in group_dict.items():
		sample_dict[group] = sample(ids, sample_size)

	grouped_samples = {} 	# holder for results

	for group_1, ids_1 in sample_dict.items():
		group_corrs = {}

		for group_2, ids_2 in sample_dict.items():
			temp = data.loc[ids_1, ids_2].apply(lambda x: x.tolist(), axis=1) 	# for subset [group_2 rows, group_1 columns], convert each column to list
			temp = [item for items in temp for item in items] 	# concat values into single list
			print(f'--\ntemp in group_sampleCorrs:\n(Should be list of {group_1} corrs for each gene in {group_1} (len = sample size * sample size = {sample_size}*{sample_size}))')
			print(f'\n{temp}\n---')
			group_corrs[group_2] = temp 	# all group_1 - group_2 correlations

		grouped_samples[f'{group_1} genes'] = group_corrs

	# now grouped_samples should be:
	# {'group 1':
	#		{'group 1 corrs': corrs},
	#		{'group 2 corrs': corrs}
	#	'group 2':
	#		{'group 1 corrs': corrs},
	#		{'group 2 corrs': corrs}
	# }

	dfs = combine_groups(grouped_samples)
	concat_df = pd.concat(dfs,axis=1,sort=False).reset_index()
	return concat_df

def main():

	# get the over-lap and unique related, unrelated gene lists
	uniq_related, uniq_unrelated, overlap = get_geneSets(mitoGenes_df, housekeeping_df)

	# subset corr df to contain only the subset lists
	print(f'Orig corrs shape: {ccle_geneCorrDF.shape}')
	ccle_corrIDs = list(ccle_geneCorrDF.columns)

	# relevant_genes = uniq_related + uniq_unrelated + overlap
	relevant_genes = uniq_related + uniq_unrelated

	correlated_relevantGenes = list(set(relevant_genes).intersection(ccle_corrIDs))


	# get overlap of this list with ccle corrs columns, these are genes we have calculate correlations for
	relevant_corrDF = ccle_geneCorrDF.loc[correlated_relevantGenes, correlated_relevantGenes]

	# delete corr file from memory
	l = [ccle_geneCorrDF]
	del ccle_geneCorrDF
	del l

	print(f'Only relevant subset: {relevant_corrDF.shape}\n# of relevant genes: {len(relevant_genes)}')

	## For labelling groups
	##

	# label_dict = {'Mito':uniq_related,
	# 	'Non-Mito': uniq_unrelated,
	# 	'Overlap': overlap} 	# create dict describing how to label our genes

	# label_dict = {'Mito':uniq_related,
	# 	'Non-Mito': uniq_unrelated}

	# # # map labels to our corr data 
	# relevant_corrDF['Gene Group'] = 'NA'

	# for label, ids in label_dict.items():
	# 	relevant_corrDF['Gene Group'] = np.where(relevant_corrDF.index.isin(ids), label, relevant_corrDF['Gene Group']) 	# create gene group column and assign labels where ids match (for a given idx)

	# gene_groups = relevant_corrDF['Gene Group']
	# relevant_corrDF.drop('Gene Group', axis=1, inplace=True)

	## End of group labeling
	##

	# ## For ploting clustermap
	# ## 
	# relevant_corrDF.index.name = 'CCLE Transcriptomics Correlations'
	# print(f'rel corrs for plotting:\n{relevant_corrDF.head()}')

	# print(f'indexes: {relevant_corrDF.index[:10]}\nCols: {relevant_corrDF.columns[:10]}')
	# # hierarch of relevant genes
	# hierarch_plot = hierarch_map(data=relevant_corrDF, 
	# 	axis='Row', 
	# 	mapping=gene_groups)

	# hierarch_plot.savefig(rf'{path}\outputs\figures\Test_mitoHierarch_08-03-2023_noOverlap.png')

	# ## End of clustermap
	# ##

	## For plotting hierarchical dendrogram
	##

	# generate leaf labels for plot from sample of each set
	leaf_colors = {}

	# for g in sample(uniq_related, 10): 	# random selection of 10 genes
	# 	leaf_colors[g] = 'darkcyan' 	# label colors based on group
	# for g in sample(uniq_unrelated, 10):
	# 	leaf_colors[g] = 'darkkhaki'

	correlated_relGenes = list(set(uniq_related).intersection(ccle_corrIDs))
	correlated_unRelGenes = list(set(uniq_unrelated).intersection(ccle_corrIDs))

	for g in correlated_relGenes: 	# random selection of 10 genes
		leaf_colors[g] = 'darkcyan' 	# label colors based on group
	for g in correlated_unRelGenes:
		leaf_colors[g] = 'darkkhaki'

	# print(f'leaf_color[0]:\n{leaf_colors["(0)"]}')
	# exit()

	plt.title(f'Mito related/un-related genes (CCLE correlations)')
	hierarch_results = hierarch_clustering(relevant_corrDF, clusters=False, leaf_labelDict=leaf_colors)

	plt.savefig(rf'{path}\outputs\figures\Test_mitoDendrogram_Grouped_09-03-2023.png')

	## End of dendrograms
	##


# define function for generating similar hierarchical plots of groups, but as histogram and boxplots
def group_BoxPlots():

	# get the over-lap and unique related, unrelated gene lists
	uniq_related, uniq_unrelated, overlap = get_geneSets(mitoGenes_df, housekeeping_df)

	# subset corr df to contain only the subset lists
	print(f'Orig corrs shape: {ccle_geneCorrDF.shape}')
	ccle_corrIDs = list(ccle_geneCorrDF.columns)

	# get relevant genes as those we have correlations for
	correlated_relatedGenes = list(set(uniq_related).intersection(ccle_corrIDs))
	correlated_unRelatedGenes = list(set(uniq_unrelated).intersection(ccle_corrIDs))
	correlated_overlapGenes = list(set(overlap).intersection(ccle_corrIDs))

	# for n = 50, sample each list for n genes
	groups = {'Mito': correlated_relatedGenes,
		'Non-Mito': correlated_unRelatedGenes,
		'Overlap': correlated_overlapGenes} 	# create dict describing how to label our genes

	groupedDF = group_sampleCorrs(data=ccle_geneCorrDF, group_dict=groups, sample_size=80)

	# now can plot this as boxplot based on groups
	group_long = pd.melt(groupedDF, 'Group', var_name='Correlated Group', value_name='CCLE Transcriptomics Spearman rho')

	# group_long.replace(0, np.nan, inplace=True) 	# Remove all 0 (i.e. non-significant) correlations from data

	# box plot with overlapping swarm plot
	sns.boxplot(data=group_long, x='Correlated Group', y='CCLE Transcriptomics Spearman rho', hue='Group', palette={"Mito": "white", "Non-Mito": "white", "Overlap": "white"}, showfliers=False)
	sns.stripplot(data=group_long, 
		x='Correlated Group', 
		y='CCLE Transcriptomics Spearman rho', 
		hue='Group', 
		palette={"Mito": "darkcyan", "Non-Mito": "darkkhaki", "Overlap": "mediumseagreen"}, 
		dodge=True,
		jitter=0.10,
		alpha=0.5
		)

	# Edit plot for proper formatting 
	
	# set each group label to match data color
	ax = plt.gca()

	ax.get_xticklabels()[0].set_color("darkcyan")
	ax.get_xticklabels()[1].set_color("darkkhaki")
	ax.get_xticklabels()[2].set_color("mediumseagreen")

	# Only include legend for stripplot, not box plot
	h,l = ax.get_legend_handles_labels()
	ax.legend(h[3:], l[3:],loc='center left', bbox_to_anchor=(1,0.5))

	# save plot
	plt.tight_layout()
	plt.savefig(rf'{path}\outputs\figures\mitoBoxDot_NoAbs_14-03-2023.png')


# define function for plotting dimension reduction results (Component 1, 2 of PCA, t-SNE, UMAP)
def plot_Components(comp_data, label_palette, plot_ax, labels=False): 	# Takes data of form samples x [Component_1, Component_2, *LABELS] (*LABELS col informed by 'labels' arg, which will be used to color the data in plot)

	# if we have group labels to color our data by
	if labels:
		scatt = sns.scatterplot(data=comp_data,
			x=comp_data.columns[1], 	# Component 1 column name
			y=comp_data.columns[2], 	# Component 2 column name
			hue=labels,
			palette=label_palette,
			ax=plot_ax,
			alpha=0.35
			)
	else:
		scatt = sns.scatterplot(data=comp_data,
			x =comp_data.columns[1],
			y=comp_data.columns[2],
			ax=plot_ax,
			alpha=0.35
			)	

	return scatt

# define code to get group correlations, transform them and then perform and plot PCA
def PCA_byGroup():

	# get the over-lap and unique related, unrelated gene lists
	uniq_related, uniq_unrelated, overlap = get_geneSets(mitoGenes_df, housekeeping_df)

	# subset corr df to contain only the subset lists
	ccle_corrIDs = list(ccle_geneCorrDF.columns)

	# get relevant genes as those we have correlations for
	correlated_relatedGenes = list(set(uniq_related).intersection(ccle_corrIDs))
	correlated_unRelatedGenes = list(set(uniq_unrelated).intersection(ccle_corrIDs))
	correlated_overlapGenes = list(set(overlap).intersection(ccle_corrIDs))

	# for n samples, sample each list for n genes
	groups = {'Mito': correlated_relatedGenes,
		'Non-Mito': correlated_unRelatedGenes,
		'Overlap': correlated_overlapGenes} 	# create dict describing how to label our genes

	relevant_corrs = ccle_geneCorrDF.loc[:,correlated_relatedGenes + correlated_unRelatedGenes + correlated_overlapGenes] 	# only want corrs for relevant (i.e. group) genes
	print(f'rel corrs:\n{relevant_corrs}')
	
	groups = ['Mito']*len(correlated_relatedGenes) + ['Non-Mito']*len(correlated_unRelatedGenes) + ['Overlap']*len(correlated_overlapGenes) 	# get a corresponding list, where each group gene has group label (for plotting)

	# get fisher transformed for correlations (necessary for PCA)
	fisher_corrs = fisher(relevant_corrs)

	# fisher transforms ranges [-inf, inf], where corrs range [-1,1]
	fisher_corrs.replace([np.inf, -np.inf], np.nan, inplace=True) 	# Replace infs with nans, then drop all nan rows. Alternatively can replace infs & nans with a numeric value and retain rows
	cleanFisher = fisher_corrs.dropna() # Drop rows with all NAN which cannot be handled by PCA

	print(f'shape before ({fisher_corrs.shape}) and after removing NAN rows ({cleanFisher.shape})')

	## CODE for RUNNING PCA
	##

	# define PCA params and fit data
	pca_alg = TSNE(n_components=2)
	princip_componenets = pca_alg.fit_transform(cleanFisher.T) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	PC_df = pd.DataFrame(data=princip_componenets, index=cleanFisher.columns, columns=['t-SNE component 1', 't-SNE component 2'])
	print(f't-SNE component df:\n{PC_df.head()}\nShape: {PC_df.shape}')

	PC_df[f'Groups'] = groups

	print(f't-SNE component df:\n{PC_df.head()}\nShape: {PC_df.shape}')

	PC_df.to_csv('mitoCorrs_tSNE.csv')

	## CODE for LOADING PCA DATA
	##
	PC_df = pd.read_csv('mitoCorrs_tSNE.csv')

	fig, ax_pca = plt.subplots() 	# define figure and axis objects for plotting

	pca_plot = plot_Components(PC_df, 
		labels='Groups', 
		label_palette={"Mito": "darkcyan", "Non-Mito": "darkkhaki", "Overlap": "mediumseagreen"},
		plot_ax=ax_pca
		)

	# format plot
	sns.move_legend(pca_plot, loc='center left', bbox_to_anchor=(1,0.5)) 	# edit legend position

	plt.title(f'CCLE transcriptomics t-SNE (Spearman rho (no abs.))')

	# Save plot
	plt.tight_layout()
	fig.savefig(rf'{path}\outputs\figures\mitoGroupCorrs_tSNE_noAbs_14-03-2023.png')

	return

# define code to get group correlations, transform them and then perform and plot UMAP
def UMAP_byGroup():

	# get the over-lap and unique related, unrelated gene lists
	uniq_related, uniq_unrelated, overlap = get_geneSets(mitoGenes_df, housekeeping_df)

	# subset corr df to contain only the subset lists
	ccle_corrIDs = list(ccle_geneCorrDF.columns)

	# get relevant genes as those we have correlations for
	correlated_relatedGenes = list(set(uniq_related).intersection(ccle_corrIDs))
	correlated_unRelatedGenes = list(set(uniq_unrelated).intersection(ccle_corrIDs))
	correlated_overlapGenes = list(set(overlap).intersection(ccle_corrIDs))

	# for n samples, sample each list for n genes
	groups = {'Mito': correlated_relatedGenes,
		'Non-Mito': correlated_unRelatedGenes,
		'Overlap': correlated_overlapGenes} 	# create dict describing how to label our genes

	relevant_corrs = ccle_geneCorrDF.loc[:,correlated_relatedGenes + correlated_unRelatedGenes + correlated_overlapGenes] 	# only want corrs for relevant (i.e. group) genes
	
	groups = ['Mito']*len(correlated_relatedGenes) + ['Non-Mito']*len(correlated_unRelatedGenes) + ['Overlap']*len(correlated_overlapGenes) 	# get a corresponding list, where each group gene has group label (for plotting)

	# get fisher transformed for correlations (necessary for PCA)
	fisher_corrs = fisher(relevant_corrs)

	# fisher transforms ranges [-inf, inf], where corrs range [-1,1]
	fisher_corrs.replace([np.inf, -np.inf], np.nan, inplace=True) 	# Replace infs with nans, then drop all nan rows. Alternatively can replace infs & nans with a numeric value and retain rows
	cleanFisher = fisher_corrs.dropna() # Drop rows with all NAN which cannot be handled by PCA

	print(f'shape before ({fisher_corrs.shape}) and after removing NAN rows ({cleanFisher.shape})')

	## CODE for RUNNING UMAP
	##

	# define UMAP params and fit data
	umap_alg = umap.UMAP()
	umap_components = umap_alg.fit_transform(cleanFisher.T) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	UMAP_df = pd.DataFrame(data=umap_components, index=cleanFisher.columns, columns=['UMAP dimension 1', 'UMAP dimension 2'])
	print(f'UMAP dimension df:\n{UMAP_df.head()}\nShape: {UMAP_df.shape}')

	UMAP_df[f'Groups'] = groups

	UMAP_df.to_csv('mitoCorrs_UMAP.csv')

	## CODE for LOADING PCA DATA
	##
	# UMAP_df = pd.read_csv('mitoCorrs_UMAP.csv')

	# fig, ax_umap = plt.subplots() 	# define figure and axis objects for plotting

	# umap_plot = plot_Components(UMAP_df, 
	# 	labels='Groups', 
	# 	label_palette={"Mito": "darkcyan", "Non-Mito": "darkkhaki", "Overlap": "mediumseagreen"},
	# 	plot_ax=ax_umap
	# 	)

	# # format plot
	# sns.move_legend(umap_plot, loc='center left', bbox_to_anchor=(1,0.5)) 	# edit legend position

	# plt.title(f'CCLE transcriptomics UMAP (Spearman rho (no abs.))')

	# # Save plot
	# plt.tight_layout()
	# fig.savefig(rf'{path}\outputs\figures\mitoGroupCorrs_UMAP_noAbs_14-03-2023.png')


# define code to get group transcriptomics data, Standardize them and then perform and plot UMAP
def UMAP_transcriptomics(trans_data, pos_set, neg_set):

	# get the over-lap and unique related, unrelated gene lists
	uniq_pos, uniq_neg, overlap = get_geneSets(pos_set, neg_set)

	# subset transcriptomics data to contain only the subset lists
	transIDs = list(trans_data.index)

	# get relevant genes as those we have data for
	trans_relatedGenes = list(set(uniq_pos).intersection(transIDs))
	trans_unRelatedGenes = list(set(uniq_neg).intersection(transIDs))
	trans_overlapGenes = list(set(overlap).intersection(transIDs))
	trans_otherGenes = list(set(transIDs) - set(trans_relatedGenes) - set(trans_unRelatedGenes) - set(trans_overlapGenes))	# Also want to get the remainder genes, or genes not found in our groups
	print(f'trans_otherGenes:\nlen {len(trans_otherGenes)}\n{trans_otherGenes[:10]}')
	# for n samples, sample each list for n genes
	relevant_data = trans_data.loc[trans_relatedGenes + trans_unRelatedGenes + trans_overlapGenes + trans_otherGenes,:] 	# This gets relevant genes, and re-orders based on our selection order
	
	groups = ['Mito']*len(trans_relatedGenes) + ['Non-Mito']*len(trans_unRelatedGenes) + ['Overlap']*len(trans_overlapGenes) + ['Other']*len(trans_otherGenes) 	# get a corresponding list, where each group gene has group label (for plotting)

	print(f'Orig data shape: {trans_data.shape}\nSelected data shape: {relevant_data.shape}\n# of selected genes: {len(groups)}')
	# get Standardized data (such that mean=0, sd=1) for dimensionality reduction

	# Since data has already been cleaned and imputed, should not be any NAN values, but check just in case
	print(f'data shape before ({relevant_data.shape})')
	cleanData = relevant_data.dropna() # Drop rows with all NAN which cannot be handled by PCA
	print(f' and after removing NAN rows ({cleanData.shape})')

	stand_scaler = StandardScaler()
	standard_data = stand_scaler.fit_transform(cleanData)

	# print(f' standard data (shape= {standard_data.shape}\n{standard_data[:5]}')

	# ## CODE for RUNNING UMAP
	# ##

	# # define UMAP params and fit data
	# umap_alg = umap.UMAP(n_components=100) 	# reduce ~ 1000 cell lines -> 100 UMAP embeddings
	# umap_components = umap_alg.fit_transform(standard_data) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	# print(f'umap_components:\n{umap_components[:5]}\nshape: {umap_components.shape}')
	# UMAP_df = pd.DataFrame(data=umap_components, index=relevant_data.index) # , columns=['UMAP dimension 1', 'UMAP dimension 2'])
	# print(f'UMAP dimension df:\n{UMAP_df.head}\nShape: {UMAP_df.shape}')

	# UMAP_df[f'Groups'] = groups

	# UMAP_df.to_csv('mitoTranscriptomics_UMAP.csv')


	# Repeat with PCA
	# define PCA params and fit data
	pca_alg = PCA(n_components=100) 	# reduce ~ 1000 cell lines -> 100 UMAP embeddings
	pca_components = pca_alg.fit_transform(standard_data) 	# fit_transform takes rows as samples and cols as features, so this will result in gene-wise PCA

	print(f'pca_components:\n{pca_components[:5]}\nshape: {pca_components.shape}')
	PCA_df = pd.DataFrame(data=pca_components, index=relevant_data.index) # , columns=['PCA dimension 1', 'PCA dimension 2'])
	print(f'PCA dimension df:\n{PCA_df.head()}\nShape: {PCA_df.shape}')


	# Also want 1st and 2nd rows to reflect the PC component variance, variance ratio
	component_df = pd.DataFrame([pca_alg.explained_variance_, pca_alg.explained_variance_ratio_], index=['Component Variance', 'Component Varaince Ratio'])

	print(f'PC component variance df:\n{component_df}')

	PCA_finalDF = pd.concat([component_df, PCA_df])

	PCA_finalDF[f'Groups'] = ['NA', 'NA'] + groups 	# variance rows have no groups
	print(f'final:\n{PCA_finalDF}')
	PCA_finalDF.to_csv('mitoTranscriptomics_PCA_Variance.csv')

	# # CODE for LOADING PCA DATA
	# #
	# UMAP_df = pd.read_csv('mitoTranscriptomics_UMAP.csv')

	# # Want to remove 'Other' rows as it makes plots crowded
	# print(f'UMAP_df:\n{UMAP_df.head()}\nShape: {UMAP_df.shape}')
	# UMAP_df = UMAP_df[UMAP_df['Groups'] != 'Other']

	# mitoUmap_df = UMAP_df[UMAP_df['Groups'] == 'Mito']
	# nonMitoUmap_df = UMAP_df[UMAP_df['Groups'] == 'Non-Mito']


	## Code for 3D plot

	# fig = plt.figure()
	# ax_3D = plt.axes(projection='3d')
	# ax_3D.scatter(xs=nonMitoUmap_df.iloc[:,1], ys=nonMitoUmap_df.iloc[:,2], zs=nonMitoUmap_df.iloc[:,3], color='darkkhaki', label='Non-Mito genes')
	# ax_3D.scatter(xs=mitoUmap_df.iloc[:,1], ys=mitoUmap_df.iloc[:,2], zs=mitoUmap_df.iloc[:,3], color='darkcyan', label='Mito genes')

	# ax_3D.set(title='CCLE transcriptomics UMAP', xlabel='UMAP dimension 1', ylabel='UMAP dimension 2', zlabel='UMAP dimension 3')
	# ax_3D.legend(loc='upper left', bbox_to_anchor=(-0.25,1))

	# plt.tight_layout()
	# fig.savefig(rf'{path}\outputs\figures\3D_test_ccleTrans_MitoGroupsOnly_UMAP_16-03-2023.png')


	# print(f'new UMAP shape: {UMAP_df.shape}')

	# fig, ax_umap = plt.subplots() 	# define figure and axis objects for plotting

	# umap_plot = plot_Components(UMAP_df, 
	# 	labels='Groups', 
	# 	label_palette={"Mito": "darkcyan", "Non-Mito": "darkkhaki", "Overlap": "mediumseagreen"},  # , "Other": "grey"},  	##  IF INCLUDING OTHER GENES
	# 	plot_ax=ax_umap
	# 	)

	# # format plot
	# sns.move_legend(umap_plot, loc='center left', bbox_to_anchor=(1,0.5)) 	# edit legend position

	# # plt.title(f'CCLE transcriptomics UMAP')
	# ax_umap.set(title='CCLE transcriptomics UMAP', xlabel='UMAP dimension 1', ylabel='UMAP dimension 2')

	# # Save plot
	# plt.tight_layout()
	# fig.savefig(rf'{path}\outputs\figures\ccleTrans_MitoGroupsOnly_UMAP_16-03-2023.png')

	return

# define function for generating 3D scatter plots (of reduced dimension embeddings)
# def 3D_scatter(data, dimensions):
# 	print()
# 	return

###
## 	MAIN CODE
###

# load data
path = r'C:\Users\lkenn\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code' 	# PC path
# path = r'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code' 	# laptop path

# ccle_geneCorrDF = pd.read_pickle(rf'{path}\data\common_ccleRNAseq_spearman.pkl') # load correlations data (NOTE: these correlations are thresholded (p<0.05 and abs.))
# ccle_geneCorrDF = pd.read_pickle(rf'{path}\data\common_mitoRNAseq_spearman_NO_ABS.pkl') # load correlations data (NOTE: these are thresholded but NOT ABSOLUTE (p<0.05))

# Load related & unrelated genes
labeled_data = pd.ExcelFile(rf'{path}\data\mGSH_labeledGenes_FULL.xlsx')
housekeeping_df = pd.read_excel(labeled_data, sheet_name='Housekeeping Ensembl') 	# related genes
mitoGenes_df = pd.read_excel(labeled_data, sheet_name='Mitochondrial Ensembl') 	# un related genes

# Run dimensionality reduction
# UMAP_byGroup()


# Want to run some analyses on the Transcriptomics data itself, rather than corrs (PCA and UMAP probably best for this since larger dataset)
ccle_geneDF = pd.read_pickle(rf'{path}\data\cleaned_common_ccleRNAseq.pkl')

# Run UMAP on transcriptomics data
UMAP_transcriptomics(trans_data=ccle_geneDF, pos_set=mitoGenes_df, neg_set=housekeeping_df)

### FOR PCA



### NOTE: Correlations contain many 0 (non-sig) correlations, and 1 (self) correlations
##		Fisher transform converts these to NaN, and inf corrs, respectively.
##	Two options:
##		1. PCA with rows containing NaN, inf corrs removed
##			1.b. PCA with NAN, inf replacement (mean/median)
##		2. Sparse dimensionality reduction with SVD


# long-form df containing [corrs, corr group] cols
# fisher transform corrs
# PCA on fisher corrs
	# get PC1, PC2, and explained variance
# plot scatter of PCA
# color dots based on 'corr group'
# save

# Also try t-SNE of above