### generate venn diagrams

# import modules
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


# Define sets for venn diagram
# venn3(subsets=(6,206,615,0,0,23,0), set_labels = ('MitoCarta False Positives', 'RF False Positives', 'RF False Positives, 1+ evidence'))
# venn2(subsets=(6,229,615), set_labels = ('MitoCarta False Positives', 'RF False Positives'), set_colors=('rebeccapurple', 'mediumseagreen'), alpha=0.50)
# venn2(subsets=(0,206,23), set_labels = ('RF FPs, 1+ evidence', 'RF FPs, 0 evidence'), set_colors=('saddlebrown', 'mediumseagreen'), alpha=0.60)
# plt.tight_layout()
# plt.savefig('test FP venn diagram_RF-only.png', dpi=1600)

venn2(subsets=(147,218,1477), set_labels =('TrSSP False Positives', 'RF False Positives'), set_colors=('peru', 'mediumseagreen'), alpha=0.60)
plt.tight_layout()
plt.savefig('TrSSP_FP_venn.png', dpi=1600)