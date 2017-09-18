from utils import *

filename = "Grassman_distance"

np.random.seed(0)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(funcName)s: %(message)s',
                    filename="%s.log" % filename)

if not os.path.isdir(filename):
    os.mkdir(filename)

X_raw, y = get_raw_data()
names = projection_algorithms
lnames = len(names)

for k in [10, 100, 500, 1000, 2000]:
    logging.info("k=%d" % k)
    rec = [proj_V[name](X_raw, k) for name in names]
    diff_mat = np.zeros((lnames, lnames))
    for i in range(lnames):
        for j in range(i + 1, lnames):
            d = Grassman_distance(rec[i], rec[j])
            diff_mat[i, j] = d
            diff_mat[j, i] = d
    np.savetxt("%s/%s__k_%d.txt" % (filename, filename, k), diff_mat)


