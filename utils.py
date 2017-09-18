import sys
import os
import logging
from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import issparse
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 9
    mpl.rcParams['lines.markeredgewidth'] = 1
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 18
except:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import safe_sparse_dot

np.random.seed(0)

TRAIN = "data/train.csv"
FEATURES = ["product_title", "search_term"]

#=====
#model
#=====
models = {"linear": LinearRegression,
          "rf10": RandomForestRegressor}

models_param = {
          "linear": {"n_jobs": -1},
          "rf10": {"n_estimators": 10, "n_jobs": -1, "random_state": 0}}

def make_model(model_name):
    return models[model_name](**models_param[model_name])

#======
#projection algorithms
#======

def RP_V(X, k):
    #random projection
    #overall O(M * k)
    transformer = SparseRandomProjection(n_components=k, random_state=0)
    #O(M * k)
    transformer.fit(X)
    V = transformer.components_.T
    return V

def PCA_V(X, k):
    #use the ARPACK wrapper in SciPy (scipy.sparse.linalg.svds)
    transformer = TruncatedSVD(n_components=k, algorithm="arpack", random_state=0)
    transformer.fit(X)
    V = transformer.components_.T
    return V

def SPCA_V(X, k):
    # RPCA or SPCA by vectors in im(X)
    #overall O(M * k + (c_1 * n + c_2 * p) * k^2)
    transformer = SparseRandomProjection(n_components=k, random_state=0)
    #O(Mk)
    Y = transformer.fit_transform(X).toarray()
    #O(nk^2)
    Q, R = np.linalg.qr(Y)
    #O(Mk)
    B = safe_sparse_dot(Q.T, X)
    #O(pk^2)
    U, S, V = np.linalg.svd(B)
    V = V[:k]
    return V.T

def SSPCA_V(X, k):
    # overall O(M * k + c_3 * p * k^2)
    transformer = SparseRandomProjection(n_components=k, random_state=0)
    #O(M * k)
    Y = transformer.fit_transform(X)
    #O(M * k)
    B = safe_sparse_dot(Y.T, X).toarray()
    #O(p * k^2)
    U, S, V = np.linalg.svd(B)
    V = V[:k]
    return V.T

proj_V = {
        "RP": RP_V,
        "PCA": PCA_V,
        "SPCA": SPCA_V,
        "SSPCA": SSPCA_V
        }

projection_algorithms = ["RP", "PCA", "SPCA", "SSPCA"]

plt_styles = {
            "RP": "ko-",
            "PCA": "yv--",
            "SPCA": "bd-.",
            "SSPCA": "r^--"
            }

def projection(name, X, k):
    V = proj_V[name](X, k)
    A = safe_sparse_dot(X, V)
    if issparse(A):
        return A.toarray()
    else:
        return A

#======
#measure of subspace distance
#======

def VVT(name, X, k):
    V = proj_V[name](X, k)
    return np.dot(V, V.T)

def smart_trace(X):
    if issparse(X):
        return X.diagonal().sum()
    else:
        return np.trace(X)

def cross2(V1, V2):
    return safe_sparse_dot(V1.T, V2)

def cross4(V1, V2):
    cross = cross2(V1, V2)
    return safe_sparse_dot(cross, cross.T)

def Steifel_distance(V1, V2):
    assert V1.shape == V2.shape
    k = V1.shape[1]
    cross = cross2(V1, V2)
    return 2 * k - 2 * smart_trace(cross)

def Grassman_distance(V1, V2):
    assert V1.shape == V2.shape
    k = V1.shape[1]
    return k - 0.5 * smart_trace(cross4(V1, V2)) - 0.5 * smart_trace(cross4(V2, V1))    

#======
#generate data
#======

def cooccurrence_terms(lst1, lst2):
    terms = [""] * len(lst1) * len(lst2)
    cnt =  0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + "X|X" + item2
            cnt += 1
    res = " ".join(terms)
    return res

def make_cooccurrence_matrix(df_all):
    """get coocurrence of product_title and search_term
    """
    df_tmp = pd.DataFrame()
    df_tmp["id"] = df_all["id"]
    for feat in FEATURES:
        df_tmp[feat + "_unigram"] = list(df_all[feat].apply(lambda x: x.split()))
    df_tmp["cooccurrence_terms"] = list(df_tmp.apply(lambda x: cooccurrence_terms(x["search_term_unigram"], x["product_title_unigram"]), axis=1))
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_mat = vectorizer.fit_transform(df_tmp["cooccurrence_terms"])
    return tfidf_mat

def get_raw_data():
    train = pd.read_csv(TRAIN, encoding="ISO-8859-1")
    y = train["relevance"].values
    X_raw = make_cooccurrence_matrix(train)
    logging.info("raw: %s, nnz_rate: %f" % (str(X_raw.shape), float(X_raw.nnz) / (X_raw.shape[0] * X_raw.shape[1])))
    return X_raw, y

def rmse(clf, X, y):
    score = np.sqrt(np.mean((clf.predict(X) - y) ** 2))
    logging.info("RMSE: %f" % score)
    return score

