### Functions to be used in classifier code to format data, build, run and, evaluate classifier models

## import module
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr, distributions


# define function to get p-val from spearman corr when correlatin transcripts (more efficient method, such as calling spearmanr() directly?)
def spearmanr_pval(x,y):
	return spearmanr(x,y)[1]


# define function for thresholding spearman correlations by significance; If pval > {pval_threshold}, replace corr value with 0
def threshold_corr(corr_df, pval_df, pval_threshold=0.05):

	threshold_df = corr_df.where(pval_df.values < pval_threshold, other=0)

	return threshold_df


# spearmanr p-value calculation from https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py
# rs = spearman coefficients (rhos)
# n = number of samples/observations
def pvalCalc_spearmanr(rs, n):
 
	t = rs * np.sqrt((n-2) / ((rs+1.0)*(1.0-rs))) 	# t-value

	prob = distributions.t.sf(np.abs(t),n-2)*2 	# calculate p-value from t

	return prob


# Based on code from: https://stackoverflow.com/questions/52371329/fast-spearman-correlation-between-two-pandas-dataframes/59072032#59072032
def get_corrs(corr_id, raw_data, corr_data, corr_filePath=None, abs_corr=True):

	if corr_id not in corr_data.columns:
		raw_ranks = raw_data.rank(axis=1) 	# convert to ranks for spearman calcs
		rawRow_rank = raw_ranks.loc[corr_id] 	# get transcriptomics data for selected genes

		if isinstance(rawRow_rank, pd.DataFrame):
			rawRow_rank = rawRow_rank.iloc[0,:]	# if multiple rows corresponding to id, select first instance so we have a series

		np_rowRank = rawRow_rank.to_numpy()
		np_rawRank = raw_ranks.to_numpy()

		# calculate spearman and p-vals
		spearm = corr(a=rawRow_rank.to_numpy().reshape(1,-1), b=raw_ranks.to_numpy())  	# reshape(1,-1) for proper dimensions, to_numpy() for numba
		spearm_p = pvalCalc_spearmanr(rs=spearm, n=len(rawRow_rank)) 	### NOTE: not numba

		thresh_corr = p_thresh(thresh_val=0.05, rhos=spearm, pvals=spearm_p)

		# Save the correlations as absolute or standard correlations
		if abs_corr:
			final_corr = pd.DataFrame(data=abs(thresh_corr).T, index=raw_data.index, columns=[corr_id])
		else:
			final_corr = pd.DataFrame(data=thresh_corr.T, index=raw_data.index, columns=[corr_id])

		# update and save corrs
		corr_data.loc[:,corr_id] = final_corr

	else:
		final_corr = corr_data[corr_id]

	return final_corr


# define function to get a randomly generated list of genes = [n * positive genes, n * negative genes] to be used as feature genes
def get_featGenes(pos_geneIDs, neg_geneIDs, n_samples=1, except_ID=False):

	# For LOOCV, we have a test gene determined that cannot be used as a feature, except_id removes that gene from feat gene selection process
	if except_ID:
		try:
			pos_geneIDs.remove(except_ID)
			list_tracker = 1 	# track which list ID was removed from to append after

		except:
			neg_geneIDs.remove(except_ID)
			list_tracker = 0

		pos_feats = random.sample(pos_geneIDs, n_samples) 
		neg_feats = random.sample(neg_geneIDs, n_samples)

		pos_geneIDs.append(except_ID) if list_tracker == 1 else neg_geneIDs.append(except_ID) 	# add ID back to which ever list it was removed from

	else:
		pos_feats = random.sample(pos_geneIDs, n_samples) 
		neg_feats = random.sample(neg_geneIDs, n_samples)

	feat_genes = pos_feats + neg_feats

	trunc_posIDs = list(set(pos_geneIDs) - set(pos_feats))
	trunc_negIDs = list(set(neg_geneIDs) - set(neg_feats))

	return feat_genes, trunc_posIDs, trunc_negIDs


# define function to balance positive and negative samples
# returns both lists with random samples removed such that len() of each is equal
def balance_samples(pos_samples, neg_samples, sample_size=False, except_ID=False):

	# If we have an ID to exclude from sampling, must remove from whichever list, do sampling, and add back in
	if except_ID:
		try:
			pos_samples.remove(except_ID)
			if not sample_size:
				if len(pos_samples)+1 <= len(neg_samples):
					balanced_neg = random.sample(neg_samples, len(pos_samples)+1) # need to sample one extra since except_ID removed from pos_samples
					balanced_pos = pos_samples
				else:
					balanced_neg = neg_samples
					balanced_pos = random.sample(pos_samples, len(neg_samples)-1) # sample one less since except_ID will be added back
			else:
				balanced_neg = random.sample(neg_samples, sample_size)
				balanced_pos = random.sample(pos_samples, sample_size-1) # sample one less since except_ID will be added back

			balanced_pos.append(except_ID)

		except:
			neg_samples.remove(except_ID)
			if not sample_size:
				if len(pos_samples) <= len(neg_samples)+1:
					balanced_neg = random.sample(neg_samples, len(pos_samples)-1) # sample one less since except_ID will be added back
					balanced_pos = pos_samples
				else:
					balanced_neg = neg_samples
					balanced_pos = random.sample(pos_samples, len(neg_samples)+1) # sample one more since neg_sample is truncated
			else:
				balanced_neg = random.sample(neg_samples, sample_size-1) # sample one less since neg_samples is truncated
				balanced_pos = random.sample(pos_samples, sample_size)
			balanced_neg.append(except_ID)

	# Now methods when except_ID must not be considered.
	# get equal-sized positive & negative samples either by some minimum size ({sample_size}), or min of both provided sets
	elif not sample_size:
		if len(pos_samples) <= len(neg_samples):
			if list_tracker == 1:
				balanced_neg = random.sample(neg_samples, (len(pos_samples) + 1))
				balanced_pos = pos_samples + [except_ID]
			elif list_tracker == -1:
				balanced_neg = random.sample(neg_samples, len(pos_samples))
				balanced_neg += [except_ID]
				balanced_pos = pos_samples	
		else:
			balanced_pos = random.sample(pos_samples, len(neg_samples))
			balanced_neg = neg_samples
	else:
		balanced_pos = random.sample(pos_samples, sample_size)
		balanced_neg = random.sample(neg_samples, sample_size)

	return balanced_pos, balanced_neg


# define function to one-hot encode categorical features (MitoCarta or TrSSP scores).
# used in feat_preprocess()
def one_hot(data, cat_cols):
	cat_data = data[cat_cols]
	oneHot_data = pd.get_dummies(cat_data, prefix=cat_cols)			### ! possibly can avoid this method by using 'columns=cat_cols' on original dataset, then split after for num_data

	return oneHot_data


# define function to fisher (i.e. arctanh) transform correlation rho values for normalization.
# used in feat_preprocess()
def fisher(data):
	return np.arctanh(data) 


# define function to preprocess feature data for model
# return feat_data with numeric (correlations) fisher transformed and categorical as one-hot encoded
def feat_preprocess(feat_data, categorical_cols=False):

	if categorical_cols:
		oneHot_cats = one_hot(feat_data, categorical_cols)

		# num_data = feat_data[~feat_data[categorical_cols]]
		num_data = feat_data.drop(categorical_cols, axis=1)
		fisher_data = fisher(num_data)

		return pd.concat([oneHot_cats, fisher_data], axis=1)
	else:
		return fisher(feat_data)


# define function to split dataframe (features) based on idxs (genes)
# returns the subset DF based on given idxs, and the remainder df (corresponds to train/test & validation)
def split_feats(df, idxs):

	copy = df.copy()
	# slice df based on idxs
	slice_df = copy.loc[idxs]
	trunc_df = copy.drop(idxs, axis=0)

	# return selected df, and remainder in correct orientation
	return slice_df, trunc_df

# define function for getting ROC curve values from true and predicted labels
def get_ROCvals(y, pred_y):

	false_pr, true_pr, thresholds = roc_curve(y, pred_y)

	auc_val = auc(false_pr, true_pr)

	return [false_pr, true_pr, thresholds, auc_val]

# define function for generating ROC curve from fp & tp rates
def plot_ROC(plot_ax=False, plot_x=False, plot_y=False, auc=False, title=False, plot_data=None, plot_style=None, plot_palette=None):

	sns.set_style('dark')

	# roc_plot = sns.lineplot(x=plot_x, y=plot_y, ax=plot_ax)
	# plt.legend(loc='lower right', labels=[f'ROC curve (area = {auc})'])
	if plot_data is not None:
		line_styles =['solid', 'dashed', 'dotted', 'dashdot'] 	# specify line styles (note: these are options in seaborn, may be a better way to specify)
		for i, data in enumerate(plot_data):
		# fpr_tprNP = plot_data.loc[:,['FPR', 'TPR', 'AUC']].to_numpy() 		### NOTE: Cannot calculate average ROC as len(fpr/tpr) varies over runs? Maybe need average per threshold isntead?
		# print(f'fpr_tprNP:\n{fpr_tprNP}')
		# print(f'FPR in df vs np:\n{plot_data["FPR"]}\n{fpr_tprNP[:,0]}')
		# for fpr in fpr_tprNP[:,0]:
		# 	print(f'fpr:\n{fpr}\nshape: {fpr.shape}')
		# fpr_mean = np.mean(fpr_tprNP[:,0], axis=0)
		# print(fpr_mean)
		# avgs = np.mean(fpr_tprNP, axis=0)
		# print(avgs)

		# plot_data.loc['Mean'] = plot_data.mean(axis=0)

			plot_AUC = data.loc[:,'AUC'] 	# AUC data affects plotting, want to use just for plot legend
			print(f'plot_AUC: {plot_AUC}')
			roc_data = data.drop('AUC', axis=1)
			print(f'roc data:\n{roc_data}')
			legend = []
			count=0
			for j, model_roc in roc_data.iterrows():
				name = model_roc.name

				model_rocDF = pd.DataFrame(data=[model_roc['FPR'], model_roc['TPR']], index=['FPR', 'TPR'])
				print(f'model_roc:\n{model_roc}\nmodel_roc DF:\n{model_rocDF}')

				# May have column of model features to specify on
				if plot_style:
					print(f' i * j:\n{i}, {j}')
					print(f'plot colors: {plot_palette[i*3+count]}')
					sns.set_theme()
					sns.set_style({"grid.linestyle": line_styles[i]})
					with sns.axes_style({"grid.linestyle": '-'}):
						sns.lineplot(data=model_rocDF.T, x='FPR', y='TPR', ax=plot_ax, label=f'{name} (AUC = {"{0:0.3f}".format(plot_AUC[j])})', color=plot_palette[i*3+count], legend=False, errorbar=None) #, style=line_styles[i]) 			#### NOTE: not assigning line styles
				else:
					sns.lineplot(data=model_rocDF.T, x='FPR', y='TPR', ax=plot_ax, label=f'{name} (AUC = {"{0:0.3f}".format(plot_AUC[j])})', color=plot_palette[i*3+count], legend=False, errorbar=None)

				count +=1 # for tracking palette

	# add info for 50% line which will be added to plot
	plt.plot([0,1], [0,1], color='black', linestyle='--', label='50% chance') 	# add 50% line
	lines, labels = plot_ax.get_legend_handles_labels()
	plt.legend(lines, labels) 	# add legend to plot

	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title(title)

	return plot_ax


## Example maion function for generating a GSH ML model
def main():

	# !!! Param to be added to main()
	model_alg = GaussianNB()

	# get feature genes based on our training gene lists
	feature_genes, new_posGenes, new_negGenes = get_featGenes(pos_geneList, neg_geneList, n)

	# get feature gene correlations
	feat_corrs = [get_corrs(gene_id, ccle_geneDataDF, ccle_geneCorrDF) for gene_id in feature_genes]

	# now need other features, for GSH that is correlations between genes and GSH, GSSG
	metab_feats = ccle_metabGeneCorrDF[['glutathione reduced', 'glutatihone oxidized']]		###	!!! THRESHOLD METAB CORRS 	!!!	###

	# concat feature data into one df
	feat_df = pd.concat([metab_feats, feat_corrs], axis=1)

	# want training set to contain equal positive and negative genes
	balanced_posGenes, balanced_negGenes = balance_samples(pos_geneList, neg_geneList)

	# generate input data list and target list (1 = positive, 0 = negative) for model
	x = balanced_posGenes + balanced_negGenes
	y = [1]*len(balanced_posGenes) + [0]*len(balanced_negGenes)


	# Have list of gene IDs to be used in training/test set, now need to get corresponding features (also gets remaining gene feats, i.e. validation set)
	x_data, valid_data = split_feats(feat_df, x) 

	# Now need to split training/test set, if necessary
	x_train, x_test, y_train, y_test = train_test_split(x_data, y, train_size=0.80)

	# need to preprocess features now (fisher transform & one-hot)
	proX_train = feat_preprocess(x_train)
	proX_test = feat_preprocess(x_test)

	# fit (param-specified) algothim and predict on test set to get output probabilities
	model_alg.fit(proX_train, y_train)
	y_score = model_alg.predict_proba(proX_test) 	# probability for use in ROC curve

	# generate fpr, tpr for ROC curve
	fpr, tpr, model_auc = get_ROCvals(y, y_score)

	# plot ROC curve
	fig, ax = plt.subplots()

	roc_curve = make_plot(ax, fpr, tpr, model_auc, title='Test ROC curve')

	# Save plot
	fig.savefig('test_ROC_2022-11-24_3.png')
	fig.clf()

###
###	Add pre-processing steps
###

### !! THRESHOLD & ABS METAB-GENE CORRS !! ###
### !! Should gene-gene corrs be over common CCLs, or all transcriptomics CCLs?
### !! transform categorical values after train-test split?
### !! Change mitocarta scores such that {1=Mito-localized, 0= nonMito-localized, -1= not in Human.MitoCarta3.0.xlsx}
### !! how to do boostrapping when iterating over [ml algs, # of feat genes, random feat gene selection, LOOCV of labeled genes] = (5 * 10 * 10 * ~130) iterations
### !! Remove overlapping genes between {positive genes} and housekeeping for use in features?
### !! Add ENSG ID version # removal to preproceesing_mGSH.py
