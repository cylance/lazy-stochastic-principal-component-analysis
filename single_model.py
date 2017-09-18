from utils import *

model_name = sys.argv[1]
projection_name = sys.argv[2]

if not os.path.isdir("run_model"):
    os.mkdir("run_model")

np.random.seed(0)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(funcName)s: %(message)s',
                    filename="run_model/%s__%s.log" % (model_name, projection_name))

X_raw, y = get_raw_data()

rec_score = []

X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y,
                                                            test_size=0.1,
                                                            random_state=0)
clf = make_model(model_name)
clf.fit(X_raw_train, y_train)
score = rmse(clf, X_raw_test, y_test)
rec_score.append(score)

for k in [10, 100, 500, 1000, 2000]:
    logging.info("k=%d" % k)
    X = projection(projection_name, X_raw, k)
    logging.info("after projection: %s" % str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    clf = make_model(model_name)
    clf.fit(X_train, y_train)
    score = rmse(clf, X_test, y_test)
    rec_score.append(score)

np.savetxt("run_model/%s__%s.txt" % (model_name, projection_name), rec_score)
