import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def box_plots_comparisons(AUC_vecs,names,plot_params):
    data=np.vstack(AUC_vecs)
    fig, ax = plt.subplots()
        
    ax.boxplot(data.T)

    if "title" in plot_params:
        plt.title(plot_params["title"])
    else:
        plt.title("AUC Performance Comparison for EDSS worsening")
    ax.set_xticklabels(names)
    if "legend" in plot_params:
        ax.legend(plot_params["legend"],loc="lower right")
    plt.savefig("../comparisons_results/box_plot.pdf")


