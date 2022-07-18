import socket
import time
from argparse import Namespace
from typing import List
import scipy.sparse.linalg
import torch
import os
import definitions
from definitions import OUT_DIR
from utils import my_sign
import utils
from dataloader import get_dataloader, get_dataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from kernels import kernel_factory
import numpy as np
import scipy.io
import hydra
from omegaconf import DictConfig
import logging, sys
import omegaconf

def codebook(e):
    """
    Finds the codebook for encoding matrix e
    :param e: N x (k-1) matrix of -1, 1 entries
    :return: list of the k most frequent encodings
    """
    k = e.shape[1] + 1
    c, counts = np.unique(e, axis=0, return_counts=True)
    return [t[0] for t in sorted(zip(c, counts), key=lambda x: -x[1])[:k]]

def closest_code(e, codebook, alphat=None, alphaCenters=None):
    """
    Finds closest encoding vector in codebook
    :param e: N x (k-1) matrix of -1, 1 entries
    :param codebook: list of the k codes of length (k-1)
    :return: array of length N, closest element in codebook to e
    """
    from sklearn.neighbors import DistanceMetric
    dist = DistanceMetric.get_metric('hamming')
    dist2 = DistanceMetric.get_metric('euclidean')
    d = dist.pairwise(e, np.array(codebook))
    qtrain = np.argmin(d, axis=1)
    if alphat is not None and alphaCenters is not None and d.shape[1] > 1 and qtrain.shape[0] <= alphat.shape[0]:
        #Break ties
        sorted_d = np.sort(d, axis=1)
        nidx = sorted_d[:,0] == sorted_d[:,1]
        if np.sum(nidx) > 0:
            nidx_test = nidx
            if nidx.shape[0] < alphat.shape[0]:
                nidx_test = np.concatenate([nidx, np.zeros((alphat.shape[0] - nidx.shape[0]), dtype=bool)])
            d2 = dist2.pairwise(alphat[nidx_test], alphaCenters)
            qtrain[nidx] = np.argmin(d2, axis=1)
    return qtrain

def assign_mean(etrain: List[np.ndarray], beta=None, **kwargs):
    """
    Mean decision rule.
    :param etrain: list of V N x (k-1) matrices of score variables for view v
    :param beta: weights for each view in the decision rule. If None, beta[i] = 1/V
    :return: new score variable as a list of V N x (k-1) matrices
    """
    from math import isclose
    N, V = etrain[0].shape[0], len(etrain)
    if beta is None:
        beta = np.array([1. / V for v in range(V)])
    else:
        if type(beta) == omegaconf.listconfig.ListConfig or type(beta) == list:
            beta = np.array(list(beta))
        else:
            if type(beta["value"]) == str and (beta["value"] == "None" or beta["value"] == "null"):
                beta["value"] = None
            if beta["value"] is not None:
                beta = np.array(list(beta["value"]))
            elif beta["value"] is None:
                beta = np.array([beta[f"beta{v + 1}"] for v in range(V) if beta[f"beta{v + 1}"] is not None])
                if len(beta) == 0:
                    beta = np.array([1. / V for v in range(V)])
                else:
                    beta = beta / sum(beta)
    if not isclose(sum(beta), 1.0):
        beta = beta / sum(beta)
    assert len(beta) == V
    assert isclose(sum(beta), 1.0)

    encoding_total = np.array(etrain)
    dim_array = np.ones((1, encoding_total.ndim), int).ravel()
    dim_array[0] = -1
    beta_reshaped = beta.reshape(dim_array)
    encoding_total = encoding_total * beta_reshaped
    encoding_total = np.sum(encoding_total, axis=0)
    return [encoding_total for v in range(V)]

def assign_uncoupled(etrain: List[np.ndarray], **kwargs):
    """
    Uncoupled decision rule.
    :param etrain: list of V N x (k-1) matrices of score variables for view v
    :return: new score variable as a list of V N x (k-1) matrices
    """
    N, V = etrain[0].shape[0], len(etrain)
    return [etrain[v] for v in range(V)]

def omega_mv(X1, X2, kernels):
    V = len(X1)
    # Build Omega for each view
    Omegas = []
    for v in range(V):
        Omegas_tmp = kernels[v](torch.from_numpy(X1[v]).t(), torch.from_numpy(X2[v]).t()).numpy()
        Omegas.append((0.5 * (Omegas_tmp + Omegas_tmp.transpose())) if Omegas_tmp.shape[0] == Omegas_tmp.shape[1] else Omegas_tmp.T)
    assert len(Omegas) == V
    return Omegas

def degree_matrix(Omegas):
    V = len(Omegas)
    Ntest, N = Omegas[0].shape
    # Compute the kernel matrix Omega, and the degree matrix D
    Dinv = []
    dinv = []
    Dadd = np.zeros((Ntest, Ntest))
    for v in range(V):
        d = np.sum(Omegas[v], axis=1)
        dinv.append(np.nan_to_num((1. / d)).reshape(Ntest, 1))
        Dinv.append(np.nan_to_num(np.diag(1. / d)))
        Dadd += np.diag(d)
    assert len(Dinv) == V
    assert len(dinv) == V
    return Dadd, Dinv, dinv

def centered_omegas(Omegas, Dinv, dinv, Dinvtrain=None, dinvtrain=None):
    Dinvtrain = Dinvtrain if Dinvtrain is not None else Dinv
    dinvtrain = dinvtrain if dinvtrain is not None else dinv
    V = len(Omegas)
    Ntest, N = Omegas[0].shape
    # Compute the centered kernel matrices
    OmegasCentered = []
    for v in range(V):
        md = np.eye(Ntest) - np.matmul(np.ones((Ntest, 1)), dinv[v].transpose()) / np.sum(dinv[v])
        kd = np.eye(N) - np.matmul(np.matmul(Dinvtrain[v], np.ones((N, 1))), np.ones((1, N))) / np.sum(dinvtrain[v])
        OmegasCentered.append(np.matmul(np.matmul(md, Omegas[v]), kd))
    assert len(OmegasCentered) == V
    return OmegasCentered

def compute_alphaCenters(alphat, etrain):
    assert alphat.shape[1] == etrain.shape[1]
    assert alphat.shape[0] == etrain.shape[0]

    N, d = alphat.shape
    k = d + 1
    alphaCenters = np.zeros((k, d))
    c, m, uniquecw = np.unique(my_sign(etrain), return_index=True, return_inverse=True, axis=0)
    cwsizes = np.zeros((len(m)))
    for i in range(len(m)):
        cwsizes[i] = np.sum(uniquecw == i)
    j= np.argsort(-cwsizes, kind='mergesort')
    if len(m) < k:
        k = len(m)
    qtrain = np.zeros((alphat.shape[0],))
    for i in range(k):
        qtrain[uniquecw == j[i]] = i + 1
    for i in range(k):
        alphaCenters[i] = np.median(alphat[qtrain == (i+1)], axis=0)
    return alphaCenters

def train_mvkscrkm(X, kernels, k, eta, assignment_args, rho=1, kappa=1.0):
    """
    :return: list of functions that take x as argument and return
             {"y_pred": list of V cluster assignment [Ntest],
              "K": list of V OmegaCentered_add [Ntrain,Ntest],
              "etest": list of V score variables of x [Ntest,k-1],
              "etrain": list of V score variables of training points X [Ntrain,k-1],
              "Ktest": list of V non-centered kernel matrices [Ntest,Ntest]}
    """
    assert 0 <= rho <= 1
    V = len(X)
    if type(kappa["value"]) == str and (kappa["value"] == "None" or kappa["value"] == "null"):
        kappa["value"] = None
    if type(kappa) == float or type(kappa) == int:
        kappa = [kappa for v in range(V)]
    elif type(kappa) == list:
        kappa = kappa
    else:
        if kappa["value"] is not None:
            kappa = kappa["value"]
            if type(kappa) == float or type(kappa) == int:
                kappa = [kappa for v in range(V)]
        elif kappa["value"] is None:
            kappa = [kappa[f"kappa{v+1}"] for v in range(V) if kappa[f"kappa{v+1}"] is not None]
    assert len(kappa) == V

    # Build Omega for each view
    Omegas = omega_mv(X, X, kernels)

    # Compute the kernel matrix Omega, and the degree matrix D
    Dadd, Dinv, dinv = degree_matrix(Omegas)

    # Compute the centered kernel matrices
    OmegasCentered = centered_omegas(Omegas, Dinv, dinv)
    OmegasCentered = [kappa[v] * OmegasCentered[v] for v in range(V)]
    OmegasCentered_add = rho * np.add.reduce(OmegasCentered) + (1.0 - rho) * np.multiply.reduce(OmegasCentered)

    # Build matrices
    R = Dadd
    L = 1 / eta * OmegasCentered_add
    eigenValues, H = scipy.sparse.linalg.eigs(L, k=k - 1, M=R, which="LM", maxiter=200)
    eigenValues = np.real(eigenValues)
    eigenVectors = H
    H = np.real(H)

    # Compute score variables
    etrain = []
    for v in range(V):
        etrain.append(np.matmul(OmegasCentered[v], H))

    def compute_etest(x_test):
        Omegas_test = omega_mv(X, x_test, kernels)
        Dadd_test, Dinv_test, dinv_test = degree_matrix(Omegas_test)
        OmegasCentered_test = centered_omegas(Omegas_test, Dinv_test, dinv_test, Dinv, dinv)
        etest = [np.matmul(OmegasCentered_test[v], H) for v in range(V)]
        return etest, OmegasCentered_test

    Ktest_mem = {}
    def f_generic(x_test, assign_f):
        hash_X = sum([hash(str(x_test[v])) for v in range(V)])
        if hash_X not in Ktest_mem:
            Ktest_mem[hash_X] = omega_mv(x_test, x_test, kernels)
        Ktest = Ktest_mem[hash_X]

        etest, OmegasCentered_test = compute_etest(x_test)
        etrainused = assign_f(etrain, **assignment_args)
        etestused = assign_f(etest, **assignment_args)
        codebooks = [np.array(codebook(my_sign(etrainused[v]))) for v in range(V)]
        alphaCenters = [compute_alphaCenters(H, etrainused[v]) for v in range(V)]
        q = np.array([closest_code(my_sign(etestused[v]), codebooks[v], H, alphaCenters[v]) for v in range(V)]).transpose()
        return {"y_pred": q, "K": OmegasCentered_test, "etest": etestused, "etrain": etrainused, "Ktest": Ktest, "xtest": x_test, "kernels": kernels,
                "k": k}

    f_uncoupled = lambda x: f_generic(x, assign_uncoupled)
    f_mean = lambda x: f_generic(x, assign_mean)
    f_uncoupled.method = "uncoupled"
    f_mean.method = "mean"
    return [f_uncoupled, f_mean], {"eigs": eigenValues, "H": eigenVectors}

@hydra.main(config_path='configs', config_name='config_rkm')
def main(args: DictConfig):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    label = "run000"
    created_timestamp = int(time.time())
    model_dir = OUT_DIR.joinpath(label)
    model_dir.mkdir()

    # Load Training Data
    trainloader = get_dataloader(Namespace(**utils.merge_two_dicts(args.dataset, {"train": True})))
    test_loader = get_dataloader(Namespace(**utils.merge_two_dicts(args.dataset, {"train": False})))
    x_train, y_train = get_dataset(trainloader)
    V = len(x_train)
    x_test, y_test = get_dataset(test_loader)

    # Define kernels for each view
    if 'name' in args.kernels:
        kernels = [kernel_factory(args.kernels.name, args.kernels.args) for v in range(V)]
    else:
        kernels = [kernel_factory(kernel.name, kernel.args) for kernel in [t[1] for t in args.kernels.items()]]
    assert len(kernels) == V

    # k-fold cross-validation
    training_fun = lambda x: train_mvkscrkm(x, kernels, k=args.dataset.k, eta=args.model.eta, assignment_args=args.model.assignment,
                                rho=args.model.rho, kappa=args.model.kappa)
    eval_dict = utils.kfold_cv(training_fun, x_train, y_train, args.dataset.k_folds, x_test=x_test, y_test=y_test)

    # Print result
    [eval_dict.pop(key) for key in ["eigs", "H", "fs_output"] if key in eval_dict]
    eval_dict.update({"timestamp": created_timestamp, "hostname": socket.getfqdn()})
    logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
    logging.info(f"Saved label: {label}")
    return eval_dict

if __name__ == '__main__':
    main()
