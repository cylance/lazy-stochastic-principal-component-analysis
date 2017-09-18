from utils import *

filename = "subspace_distance"

np.random.seed(0)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(funcName)s: %(message)s',
                    filename="%s.log" % filename)

if not os.path.isdir(filename):
    os.mkdir(filename)

X_raw, y = get_raw_data()
names = projection_algorithms

def smart_reshape(A):
    if issparse(A):
        A = A.toarray()
    return A.reshape(-1)

for k in [10, 100, 500, 1000, 2000]:
    logging.info("k=%d" % k)
    #Chordal distance
    rec = np.array([smart_reshape(VVT(name, X_raw, k)) for name in names])
    diff_mat = squareform(pdist(rec))
    np.savetxt("%s/%s__k_%d.txt" % (filename, filename, k), diff_mat)


