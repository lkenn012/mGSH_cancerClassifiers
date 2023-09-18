# Code for generating venn diagrams of classifier False positives (FPs)
# Values for Venn diagrams are obtained from supplementary data

##
## import modules
##

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

# Create venn diagram based on MitoCarta FPs, random forest FPs and the overlapping FPs
venn2(subsets=(6,229,615), set_labels = ('MitoCarta False Positives', 'RF False Positives'), set_colors=('rebeccapurple', 'mediumseagreen'), alpha=0.50)
plt.tight_layout()
plt.savefig('mitoFP_venn.png', dpi=1600)

# Repeat for transporter classifier and TrSSP
venn2(subsets=(147,218,1477), set_labels =('TrSSP False Positives', 'RF False Positives'), set_colors=('peru', 'mediumseagreen'), alpha=0.60)
plt.tight_layout()
plt.savefig('transpFP_venn.png', dpi=1600)

## NOTE: no way to adjust positioning of labels in venn diagrams, must adjust manually for publication using Biorender, Adobe Illustrator, or some other software
