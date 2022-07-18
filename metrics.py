from functools import partial
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import scipy.sparse.linalg
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing, check_X_y
import torch
import pandas
import os

def metric_helper(y_true, y_pred, metric):
    res = 0
    if len(y_pred.shape) != len(y_true.shape):
        if len(y_true.shape) < len(y_pred.shape):
            V = y_pred.shape[1]
            y_true = np.tile(np.array([y_true]).transpose(), (1, V))
        else:
            V = y_true.shape[1]
            y_pred = np.tile(np.array([y_pred]).transpose(), (1, V))
    if len(y_pred.shape) > 1:
        res = np.mean([metric(y_true[:, i], y_pred[:, i]) for i in range(y_pred.shape[1])])
    else:
        res = metric(y_true, y_pred)
    return res


def ari(y_true, y_pred, **kwargs):
    return metric_helper(y_true, y_pred, adjusted_rand_score)


def nmi(y_true, y_pred, **kwargs):
    return metric_helper(y_true, y_pred, normalized_mutual_info_score)

def davies_bouldin(y_pred, xtest, kernels, **kwargs):
    V = len(xtest)
    from kernels import LinearKernel
    if isinstance(kernels[0], LinearKernel):
        return np.mean([davies_bouldin_score(xtest[v], y_pred[:, v]) if len(np.unique(y_pred[:, v])) > 1 else 0.0 for v in range(V)])
    else:
        return np.mean([my_davies_bouldin_score(xtest[v], y_pred[:, v], kernels[v]) if len(np.unique(y_pred[:,v])) > 1 else 10.0 for v in range(V)])


def my_davies_bouldin_score(X, labels, kernel):
    # Adapted from https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/metrics/cluster/_unsupervised.py#L303
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(kernel(torch.from_numpy(cluster_k).t(), torch.from_numpy(np.array([centroid])).t()).numpy())

    centroid_distances = kernel(torch.from_numpy(centroids).t()).numpy()

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)


def blf(etest, y_pred, K, k, eta=0.5, **kwargs):
    """
    Compute BLF score.
    :param etest: score variables of test points, dimension [Ntest,k-1]
    :param y_pred: array of {-1,1} of assignments for test points, dimension [Ntest,V]
    :param K: kernel matrix, dimension [Ntrain,Ntest]
    :param eta: float in [0,1]
    :return: float
    """
    V = len(etest)
    return np.mean([blf_singleview(etest[v], y_pred[:,v], K[v], k, eta=eta) for v in range(V)])

def blf_singleview(etest, y_pred, K, k, eta=0.5):
    """
    Compute BLF score.
    :param etest: score variables of test points, dimension [Ntest,k-1]
    :param y_pred: array of {-1,1} of assignments for test points, dimension [Ntest]
    :param K: kernel matrix, dimension [Ntrain,Ntest]
    :param eta: float in [0,1]
    :return: float
    """
    assert len(y_pred.shape) == 1
    assert etest.shape[0] == y_pred.shape[0]
    c, counts_ = np.unique(y_pred, axis=0, return_counts=True)
    counts = np.ones((k,))
    counts[c] = counts_
    assert etest.shape[1] == k - 1
    if k == 2:
        Z = [np.array([etest[y_pred == p].flatten(), np.array([]) if k > 2 else np.sum(K[y_pred == p], axis=1)]).T for p in range(k)]
    else:
        Z = [etest[y_pred == p] for p in range(k)]
    Z = [Z[p] - np.mean(Z[p], axis=0) for p in range(k)]
    C = [1. / counts[p] * np.matmul(Z[p].T, Z[p]) for p in range(k)]
    zeta = [np.sort(np.real(scipy.sparse.linalg.eigs(C[p], k=k - 1, which="LM")[0]))[::-1] for p in range(k)]
    if k == 2:
        linefit = sum([zeta[p][0] / (zeta[p][0] + zeta[p][1]) - 0.5 for p in range(k)])
    else:
        linefit = sum([((k - 1.) / (k - 2.)) * (zeta[p][0] / sum(zeta[p]) - 1. / (k - 1.)) for p in range(k)]) / k
    balance = np.min(counts) / np.max(counts)
    res = eta * linefit + (1. - eta) * balance
    res = 0.0 if np.isnan(res) else res
    assert 0 <= res <= 1
    return res


def baf(etest, y_pred, etrain, k, **kwargs):
    """
    Compute BAF score.
    :param etest: score variables of test points, dimension [Ntest,k-1]
    :param y_pred: array of {-1,1} of assignments for test points, dimension [Ntest,V]
    :param K: kernel matrix, dimension [Ntrain,Ntest]
    :param eta: float in [0,1]
    :return: float
    """
    V = len(etest)
    return np.mean([baf_singleview(y_pred[:,v], etest[v], etrain[v], k) for v in range(V)])

def baf_singleview(y_pred, etest, etrain, k):
    """
    Compute BAF score.
    :param y_pred: array of {-1,1} of assignments for test points, dimension [Ntest]
    :param etest: score variables for test points, dimension [Ntest,k-1]
    :param etrain: score variables for training points, dimension [Ntrain,k-1]
    :return: float
    """
    assert len(y_pred.shape) == 1
    Ntrain, _ = etrain.shape
    Ntest = y_pred.shape[0]
    c, counts_ = np.unique(y_pred, axis=0, return_counts=True)
    counts = np.ones((k,))
    counts[c] = counts_
    c = list(range(k))
    from mvkscrkm import compute_alphaCenters
    etrainCenters = compute_alphaCenters(etrain, etrain)
    K = cosine_similarity(etest, etrainCenters) # [Ntest, k]
    assert K.shape == (Ntest, k)
    res = 1. / k * sum([1. / counts[i] * np.sum(np.max(K[y_pred == c[i]], axis=1)) for i in range(k)])
    assert -1 <= res <= 1
    return res


def silhouette(y_pred, Ktest, **kwargs):
    V = len(Ktest)
    return np.mean([silhouette_score(np.abs(Ktest[v]-np.diag(np.diag(Ktest[v]))), y_pred[:, v], metric="precomputed") if len(np.unique(y_pred[:,v])) > 1 else 0 for v in range(V)])

internal_metrics = list(zip(["ari", "nmi", "baf", "silhouette", "davies_bouldin"] + [("blf%03.2f" % i).replace(".","") for i in [0.05*j for j in range(21)]],
                   [ari, nmi, baf, silhouette, davies_bouldin] + [partial(blf, eta=i) for i in [0.05*j for j in range(21)]]))

external_metrics = list(zip(["ari", "nmi"], [ari, nmi]))