# Information
Contains all code and data used for:
"Mapping relationships between glutathione and SLC25 transporters in cancers using hybrid machine learning models" - Luke Kennedy, Jagdeep K Sandhu, Mary-Ellen Harper, and Miroslava Cuperlovic-Culf

## classifiers
Contains all python scripts used in data pre-processing, classifier model training, testing, and evaluation used for this work.

## data
Contains all data files used in python code in this repository. 'info.txt' contains specific information for each data file.

## deepGOWeb
Contains all python scripts used for collecting required data (FASTA sequences) for calling DeepGOWeb API, and evaluation of DeepGO annotations and RF classifier annotations.

## structural_analysis
Contains all python scripts used for structural analysis of SLC25 proteins, and for generating corresponding output figures. CAVER was implemented via plugin through PyMol GUI and results are summarized in SLC25_tunnelRes.pkl, generated by mGSH_CAVER_resis.py from CAVER plugin.


## Environment information

The following version of Python and Python packages are used in the code provided here. All code was implemented on Windows with an installation time of 15 minutes. 

Python   --3.8.16

### Scientific programming & machine learning packages
- numpy   --1.23.5
- pandas   --1.5.2
- scikit-learn   --1.2.1
- scipy   --1.10.0

### Plotting packages
- matplotlib   --3.6.3
- matplotlib-venn   --0.11.9
- seaborn   --0.12.2

### Other packages
- pymol   --2.4.1
- pymol-psico   --4.2.2
- requests   --2.28.1

Installation information:
'''
pip install numpy
pip install pandas
pip install scikit-learn
pip install scipy
pip install matplotlib 
pip install matplotlib-venn
pip install seaborn
conda install -c schrodinger pymol
conda install -c speleo3 pymol-psico
pip install requests
'''

