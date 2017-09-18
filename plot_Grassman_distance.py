from utils import *

name = "Grassman_distance"

ks = [10, 100, 500, 1000, 2000]
for k in ks:
    mat = np.loadtxt("%s/%s__k_%d.txt" % (name, name, k))
    plt.matshow(np.log10(mat))
    plt.colorbar()
    plt.clim(-12, -2)
    if mat.shape[0] == len(projection_algorithms):
        ticks = projection_algorithms
    else:
        ticks = ["raw"] + projection_algorithms
    plt.yticks(np.arange(mat.shape[0]), ticks)
    plt.xticks(np.arange(mat.shape[0]), ticks, rotation="vertical")
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.title(r"$log_{10}(Grassman Distance)$ with $k=%d$" % k, fontsize=14)
    plt.savefig("%s__k_%s.pdf" % (name, k), bbox_inches="tight")
