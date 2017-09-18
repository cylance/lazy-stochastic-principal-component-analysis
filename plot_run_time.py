from utils import *

def parse_log(filename):
    rec = []
    with open(filename, "r") as f:
        for line in f:
            if "time:" in line:
                rec.append(eval(line.split("time:")[1]))
    return np.array(rec)

data_path = "run_time/"

name_convert = {"RP": "RP",
                "PCA": "PCA",
                "SPCA": "SPCA",
                "SSRPCA": "SSPCA"}

ks = [100, 300, 500, 800] + range(1000, 21000, 1000)
plt.figure()
for projection_name in projection_algorithms:
    rec_log = parse_log(os.path.join(data_path, "run_time__%s.log" % projection_name))
    #discard first data point due to the cool start issue
    rec_log = rec_log[:, 1:-1]
    print rec_log
    plt.plot(ks[:len(rec_log)], np.mean(rec_log, axis=1),
                 plt_styles[projection_name],
                 label=name_convert[projection_name], mew=0)
plt.legend(loc=0, framealpha=0.6)
plt.xlim((1000, 18000))
plt.title("Run Times")
plt.ylabel("Time (seconds)")
plt.xlabel("k")
plt.savefig("run_time.pdf", bbox_inches="tight")




