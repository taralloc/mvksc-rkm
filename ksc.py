import socket
import time
from argparse import Namespace
from datetime import datetime
from functools import partial
from typing import List
import scipy.sparse.linalg
import torch
import os
from definitions import OUT_DIR
from utils import my_sign
import utils
from dataloader import get_dataloader, get_dataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from kernels import kernel_factory
import numpy as np
import scipy.io
import hydra
from omegaconf import DictConfig, open_dict
import logging, sys
import omegaconf

def train_ksc_helper(X, kernel, k, kpca=False):
    """
    :param X: [N,d] matrix
    :param kernel: just one kernel
    :param k: number of clusters
    :return: function taking a [Ntest,d] matrix
    """
    from mvkscrkm import omega_mv, degree_matrix, centered_omegas, codebook, compute_alphaCenters, closest_code
    start = datetime.now()

    Omega = omega_mv([X], [X], [kernel])[0]
    Dadd, Dinv, dinv = degree_matrix([Omega])
    OmegaCentered = centered_omegas([Omega], Dinv, dinv)[0]
    Dadd = Dadd if not kpca else np.eye(Dadd.shape[0])
    try:
        eigenValues, H = scipy.sparse.linalg.eigs(OmegaCentered, k=k - 1, M=Dadd, which="LM", maxiter=200)
    except ValueError:
        eigenValues, H = np.zeros(k-1), np.zeros((OmegaCentered.shape[0], k-1))
    eigenValues, H = np.real(eigenValues), np.real(H)
    eigenVectors = H

    etrain = np.matmul(OmegaCentered, H)

    Ktest_mem = {}
    def f(x_test):
        hash_X = sum([hash(str(x_test))])
        if hash_X not in Ktest_mem:
            Ktest_mem[hash_X] = [omega_mv([x_test], [x_test], [kernel])[0]]
        Ktest = Ktest_mem[hash_X]

        Omega_test = omega_mv([X], [x_test], [kernel])[0]
        Dadd_test, Dinv_test, dinv_test = degree_matrix([Omega_test])
        OmegaCentered_test = centered_omegas([Omega_test], Dinv_test, dinv_test, Dinv, dinv)[0]
        etest = np.matmul(OmegaCentered_test, H)
        codebooks = [np.array(codebook(my_sign(etrain)))]
        alphaCenters = [compute_alphaCenters(H, etrain)]
        q = np.array([closest_code(my_sign(etest), codebooks[0], H, alphaCenters[0])]).transpose()
        return {"y_pred": q, "K": [OmegaCentered_test], "etest": [etest], "etrain": [etrain],
                "Ktest": Ktest, "xtest": x_test, "k": k}

    elapsed_time = datetime.now() - start
    return f, {"eigs": eigenValues, "H": eigenVectors, "train_time": elapsed_time.total_seconds()}

def train_ksc_cat(X, kernel, k, kpca=False):
    V = len(X)
    f_cat_helper, train_dict_cat = train_ksc_helper(np.concatenate(X, axis=1), kernel, k, kpca=kpca)

    def f_cat(x_test):
        res = f_cat_helper(np.concatenate(x_test, axis=1))
        res["y_pred"] = np.tile(res["y_pred"], (1,V))
        res["xtest"] = x_test
        res["kernels"] = [kernel for _ in range(V)]
        return res

    f_cat.method = "cat" if not kpca else "catkpca"
    return [f_cat], train_dict_cat

def train_ksc_view(X, kernels, k):
    V = len(X)
    assert len(kernels) == V
    f_views_helper = [train_ksc_helper(X[v], kernel, k) for v, kernel in zip(range(V), kernels)]

    def f_view(x_test, v):
        res = f_views_helper[v][0](x_test[v])
        res["y_pred"] = np.tile(res["y_pred"], (1,V))
        res["xtest"] = x_test
        res["kernels"] = kernels
        return res

    f_views = [partial(f_view, v=v) for v in range(V)]
    for v in range(V):
        f_views[v].method = f"view{v}"

    return f_views, utils.merge_dicts([x[1] for x in f_views_helper])

def best_single_view(eval_dict, val_metrics=None, test_metrics=None, traintest="train"):
    if test_metrics is None:
        test_metrics = ["ari", "nmi"]
    if val_metrics is None:
        val_metrics = ["blf000", "blf005", "blf010", "blf015", "blf020", "blf025", "blf030", "blf035", "blf040", "blf045", "blf050", "blf055", "blf060", "blf065", "blf070", "blf075", "blf080", "blf085", "blf090", "blf095", "blf100", "baf", "silhouette", "davies_bouldin"]
    def best_single_view_helper(x, test_metric, val_metric):
        if not f"{val_metric}_{traintest}_view0" in x.keys():
            return {}
        v = np.nanargmax([x.get(f"{val_metric}_{traintest}_view{v}", -np.inf) for v in range(10)])
        max_v = x[f"{val_metric}_{traintest}_view{v}"]
        if max_v == -np.inf or np.isnan(max_v):
            return {}
        res = {f"{val_metric}_{traintest}_bv": max_v,
               f"{test_metric}_{traintest}_bv_{val_metric}": x[f"{test_metric}_{traintest}_view{v}"]}
        return res
    from functools import reduce
    return reduce(utils.merge_two_dicts, [best_single_view_helper(eval_dict, test_metric, val_metric) for test_metric in test_metrics for val_metric in val_metrics])

@hydra.main(config_path='configs', config_name='config_ksc')
def main(args: DictConfig):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    label = "run002"
    created_timestamp = int(time.time())
    model_dir = OUT_DIR.joinpath(label)
    model_dir.mkdir()

    # Load Training Data
    trainloader = get_dataloader(Namespace(**utils.merge_two_dicts(args.dataset, {"train": True})))
    test_loader = get_dataloader(Namespace(**utils.merge_two_dicts(args.dataset, {"train": False})))
    x_train, y_train = get_dataset(trainloader)
    V = len(x_train)
    x_test, y_test = get_dataset(test_loader)

    # Define kernel
    if 'name' in args.kernels:
        kernels = [kernel_factory(args.kernels.name, args.kernels.args) for v in range(V)]
    else:
        kernels = [kernel_factory(kernel.name, kernel.args) for kernel in [t[1] for t in args.kernels.items()]]
    assert len(kernels) == V

    # k-fold cross-validation
    logging.info("Starting training")
    def training_fun(x):
        fs, train_dict = [], []

        if args.model.cat:
            fs_cat, train_dict_cat = train_ksc_cat(x, kernel=kernels[0], k=args.dataset.k)
            train_dict_cat = {f"{key}_cat": value for key, value in train_dict_cat.items()}
            fs += fs_cat
            train_dict.append(train_dict_cat)

        if args.model.view:
            fs_view, train_dict_view = train_ksc_view(x, kernels=kernels, k=args.dataset.k)
            train_dict_view = {f"{key}_view": value for key, value in train_dict_view.items()}
            fs += fs_view
            train_dict.append(train_dict_view)

        if args.model.kpca:
            fs_catkpca, train_dict_catkpca = train_ksc_cat(x, kernel=kernels[0], k=args.dataset.k, kpca=True)
            train_dict_catkpca = {f"{key}_catkpca": value for key, value in train_dict_catkpca.items()}
            fs += fs_catkpca
            train_dict.append(train_dict_catkpca)

        return fs, utils.merge_dicts(train_dict)
    eval_dict = utils.kfold_cv(training_fun, x_train, y_train, args.dataset.k_folds, x_test=x_test, y_test=y_test)

    # Update best single view metrics
    if args.model.view:
        eval_dict.update(best_single_view(eval_dict, traintest="train"))
        eval_dict.update(best_single_view(eval_dict, traintest="val"))
        eval_dict.update(best_single_view(eval_dict, traintest="test"))

    # Print result
    [eval_dict.pop(key) for key in ["eigs_cat", "H_cat", "eigs_catkpca", "H_catkpca", "eigs_view", "H_view", "fs_output"] if key in eval_dict]
    eval_dict.update({"timestamp": created_timestamp, "hostname": socket.getfqdn()})
    logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
    logging.info(f"Saved label: {label}")
    return eval_dict

if __name__ == '__main__':
    main()
