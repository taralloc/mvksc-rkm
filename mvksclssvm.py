import socket
import time
from argparse import Namespace
from datetime import datetime
from functools import partial
import scipy.sparse.linalg
import os
from definitions import OUT_DIR
from mvkscrkm import omega_mv, degree_matrix, centered_omegas, codebook, compute_alphaCenters, closest_code, assign_uncoupled, assign_mean
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
import torch

def compute_DOI(OmegasCentered, Dinv, gamma):
    V = len(OmegasCentered)
    N = OmegasCentered[0].shape[0]
    assert len(gamma) == len(Dinv) == V
    DOI = [np.eye(N) - (gamma[v]/N) * np.matmul(Dinv[v], OmegasCentered[v]) for v in range(V)]
    return DOI

def train_mvksclssvm(X, kernels, k, assignment_args, gamma):
    """
    :return: list of functions that take x as argument and return
             {"y_pred": list of V cluster assignment [Ntest],
              "K": list of V OmegaCentered_add [Ntrain,Ntest],
              "etest": list of V score variables of x [Ntest,k-1],
              "etrain": list of V score variables of training points X [Ntrain,k-1],
              "Ktest": list of V non-centered kernel matrices [Ntest,Ntest]}
    """
    V = len(X)
    N = X[0].shape[0]
    if type(gamma["value"]) == str and (gamma["value"] == "None" or gamma["value"] == "null"):
        gamma["value"] = None
    if type(gamma) == float or type(gamma) == int:
        gamma = [gamma for v in range(V)]
    elif type(gamma) == list:
        gamma = gamma
    else:
        if gamma["value"] is not None:
            gamma = gamma["value"]
            if type(gamma) == float or type(gamma) == int:
                gamma = [gamma for v in range(V)]
        elif gamma["value"] is None:
            gamma = [gamma[f"gamma{v+1}"] for v in range(V) if gamma[f"gamma{v+1}"] is not None]
    assert len(gamma) == V
    assert not np.any(np.array(gamma) < 0)

    # Build Omega for each view
    Omegas = omega_mv(X, X, kernels)

    # Compute the kernel matrix Omega, and the degree matrix D
    _, Dinv, dinv = degree_matrix(Omegas)

    # Compute the centered kernel matrices
    OmegasCentered = centered_omegas(Omegas, Dinv, dinv)

    # Compute DOI
    DOI = compute_DOI(OmegasCentered, Dinv, gamma)

    # Build matrices
    R = scipy.linalg.block_diag(*DOI)
    assert R.shape == (V*N, V*N)
    L = np.concatenate([np.concatenate(
        ([np.matmul(np.matmul(np.sqrt(Dinv[v]), np.sqrt(Dinv[v2])), OmegasCentered[v2]) for v2 in range(v) if v2 != v] if v > 0 else []) +
        [np.zeros((N,N))] +
        ([np.matmul(np.matmul(np.sqrt(Dinv[v]), np.sqrt(Dinv[v2])), OmegasCentered[v2]) for v2 in range(v,V) if v2 != v] if v < V-1 else []), axis=1)
        for v in range(V)], axis=0)
    assert L.shape == (V*N, V*N)
    eigenValues, H = scipy.sparse.linalg.eigs(L, k=k - 1, M=R, which="LM", maxiter=200)
    H = np.real(H)
    eigenValues = np.real(eigenValues)
    eigenVectors = H
    alphas = [H[v*N:(v+1)*N,:] for v in range(V)]

    # Compute score variables
    etrain = [np.matmul(OmegasCentered[v], alphas[v]) for v in range(V)]

    def compute_etest(x_test):
        Omegas_test = omega_mv(X, x_test, kernels)
        _, Dinv_test, dinv_test = degree_matrix(Omegas_test)
        OmegasCentered_test = centered_omegas(Omegas_test, Dinv_test, dinv_test, Dinv, dinv)
        etest = [np.matmul(OmegasCentered_test[v], alphas[v]) for v in range(V)]
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
        alphaCenters = [compute_alphaCenters(alphas[v], etrainused[v]) for v in range(V)]
        q = np.array([closest_code(my_sign(etestused[v]), codebooks[v], alphas[v], alphaCenters[v]) for v in range(V)]).transpose()
        return {"y_pred": q, "K": OmegasCentered_test, "etest": etestused, "etrain": etrainused, "Ktest": Ktest, "xtest": x_test, "kernels": kernels, "k": k}

    f_uncoupled = lambda x: f_generic(x, assign_uncoupled)
    f_mean = lambda x: f_generic(x, assign_mean)
    f_uncoupled.method = "uncoupled"
    f_mean.method = "mean"
    return [f_uncoupled, f_mean], {"eigs": eigenValues, "H": eigenVectors}

@hydra.main(config_path='configs', config_name='config_lssvm')
def main(args: DictConfig):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    label = "run001"
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
    logging.info("Starting training")
    training_fun = lambda x: train_mvksclssvm(x, kernels, k=args.dataset.k, assignment_args=args.model.assignment, gamma=args.model.gamma)
    eval_dict = utils.kfold_cv(training_fun, x_train, y_train, args.dataset.k_folds, x_test=x_test, y_test=y_test)

    # Print result
    [eval_dict.pop(key) for key in ["eigs", "H", "fs_output"] if key in eval_dict]
    eval_dict.update({"timestamp": created_timestamp, "hostname": socket.getfqdn()})
    logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
    logging.info(f"Saved label: {label}")
    return eval_dict

if __name__ == '__main__':
    main()
