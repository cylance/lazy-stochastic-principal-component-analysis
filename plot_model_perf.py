from utils import *

model_path = "run_model/"

model_names = ["linear", "rf10"]
title_name = {"linear": "Ordinary Linear Regression", "rf10": "Random Forest"}
ks = [10, 100, 500, 1000, 2000]
name_convert = {"RP": "RP",
                "PCA": "PCA",
                "SPCA": "SPCA",
                "SSPCA": "SSPCA"}  

for model_name in model_names:
    plt.figure()
    for projection_name in projection_algorithms:
        scores = np.loadtxt(os.path.join(model_path, "%s__%s.txt" % (model_name, projection_name)))
        plt.semilogx(ks, scores[1:], plt_styles[projection_name], label=name_convert[projection_name], mew=0)
    print "raw data performance with %s: %f" % (model_name, scores[0])
    if model_name != "linear":
        plt.plot([10, 2000], [scores[0], scores[0]], "--", color="gray", label="raw")
    plt.legend(loc=0)
    plt.title(title_name[model_name])
    plt.xticks(ks, ks)
    plt.xlim(8, 2500)
    plt.ylabel("RMSE")
    plt.xlabel("k")
    plt.savefig("%s.pdf" % model_name, bbox_inches="tight")



