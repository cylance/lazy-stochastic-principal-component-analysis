# -*- coding: utf-8 -*-
from utils import *

def add_inner_title(title, loc, ax=None, size=None, **kwargs):
    if ax == None:
        ax = plt.gca()
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size, pad=0.7, borderpad=0.01, frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

fig, axs = plt.subplots(nrows=2, ncols=2)

row2name = {0: "subspace_distance", 1: "Steifel_distance"}
col2k = {0: 100, 1: 2000}
lab = {(0, 0): "(a)", (0, 1): "(b)", (1, 0): "(c)", (1, 1): "(d)"}
ticks = projection_algorithms
ims = [[None, None], [None, None]]

for i in range(2):
    for j in range(2):
        name, k = row2name[i], col2k[j]
        mat = np.loadtxt("%s/%s__k_%d.txt" % (name, name, k))
        im = axs[i][j].matshow(np.log10(mat))
        ims[i][j] = im
        plt.sca(axs[i][j])
        if i == 0:
            plt.xticks(np.arange(mat.shape[0]), ticks, rotation="vertical")
        else:
            plt.xticks([])
        if j == 0:
            plt.yticks(np.arange(mat.shape[0]), ticks)
        else:
            plt.yticks([])
        add_inner_title(lab[(i, j)], loc=2, ax=axs[i][j])
        fig.colorbar(ims[i][j], ax=axs[i][j])

        #if i == 0:
        #    if j == 1:
        #        cb = fig.colorbar(ims[i][j], ax=axs[i][j])
        #        cb.set_clim(vmin=-12, vmax=2)
        #else:
        #    if j == 1:
        #        cb = fig.colorbar(ims[i][j], ax=axs[i][j])
        #        cb.set_clim(vmin=2.265, vmax=2.310)
plt.subplots_adjust(wspace=0, hspace=0.1)
plt.savefig("subspace_distance.pdf", bbox_inches="tight")