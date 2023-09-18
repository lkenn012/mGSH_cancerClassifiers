### Generate heatmap

# import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
data_path = rf'C:\Users\User\OneDrive\Desktop\School and Work\Programming\MastersPython\mGSH manuscript code\outputs'
heat_data = pd.read_csv(rf'{data_path}\local_TMalign.csv', index_col=0).iloc[:-1,:]

# format data for plotting
heat_data.fillna(1, inplace=True)
format_ids = heat_data.index.str.replace(r'[-.]\w+','')

heat_data.index = format_ids
heat_data.columns = format_ids
print(heat_data)
### regex to remove model IDs from names for plotting

plot_IDs = ['SLC25A1_AF', 'SLC25A3_AF', 'SLC25A10_AF', 'SLC25A11_AF', 'SLC25A13_AF', 'SLC25A24_AF', 'SLC25A37_AF', 'SLC25A43_AF', 'SLC25A50_AF', 'SLC25A39_AF', 'SLC25A40_AF', 'SLC25A39_homology_1okc', 'SLC25A40_homology_8hbv']

heatPlot_data = heat_data.loc[plot_IDs, plot_IDs]
print(heatPlot_data)

# plot data
fig, ax = plt.subplots(figsize=(12,10))
# sns.set(font_scale=5)
heat = sns.heatmap(data=heatPlot_data, ax=ax, annot=True, annot_kws={'fontsize':'large'}, vmin=0, vmax=1, cmap=sns.light_palette('seagreen', as_cmap=True))
# ax.set_xlabel(fontsize = 16)
# ax.set_xlabel(fontsize = 16)

plt.tight_layout()
fig.savefig(rf'{data_path}\local_TMAlign_heatmap.png', dpi=600)