from utils import *
import time

projection_name = sys.argv[1]

if not os.path.isdir("run_time"):
    os.mkdir("run_time")

np.random.seed(0)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(funcName)s: %(message)s',
                    filename="run_time/run_time__%s.log" % projection_name)

X_raw, y = get_raw_data()
ks = [100, 300, 500, 800] + range(1000, 21000, 1000)

for k in ks:
    logging.info("k=%d" % k)
    time_rec = []
    for i in range(5):
        beg = time.time()
        X = projection(projection_name, X_raw, k)
        dt = time.time() - beg
        time_rec.append(dt)
    logging.info("time: %s" % str(time_rec))