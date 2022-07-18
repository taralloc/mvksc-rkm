import collections
import logging
import sys
from datetime import datetime
from torch.utils.data import Dataset
import numpy as np
import utils


def merge_two_dicts(x, y):
    # In case of same key, it keeps the value of y
    return {**x, **y}


def merge_dicts(list_of_dicts):
    from functools import reduce
    return reduce(merge_two_dicts, list_of_dicts)


def flatten_dict(d, parent_key='', sep='_', prefix='eval_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep, prefix=prefix).items())
        else:
            items.append((prefix + new_key, v))
    return dict(items)


def float_format(f: float) -> str:
    return "%+.4e" % f


def my_sign(x):
    return np.sign(x) + (x == 0)


def kfold_cv(training_fun, x_train, y_train, k_folds, x_test=None, y_test=None):
    eval_dict = {}
    if k_folds > 2:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds)
        eval_dicts = []
        for train, val in kf.split(x_train[0]):
            # Define training and validation sets
            x_train_cv = [x[train] for x in x_train]
            x_val_cv = [x[val] for x in x_train]
            y_train_cv = y_train[train]
            y_val_cv = y_train[val]
            assert x_train_cv[0].shape[0] >= x_val_cv[0].shape[0]  # training set must be larger or equal than validation set

            # Train
            fs = training_fun(x_train_cv)
            train_dict = {}
            if type(fs) == tuple:
                fs, train_dict = fs

            # Evaluate on validation set
            from metrics import internal_metrics
            fs_output = [(traintest, x, f(x[0]), f.method) for traintest, x in zip(["val"], [(x_val_cv, y_val_cv)]) for f in fs]
            eval_dict = {f"{metric[0]}_{traintest}_{f_method}": metric[1](y_true=y, **f_output)
                         for metric in internal_metrics
                         for traintest, (x, y), f_output, f_method in fs_output}
            eval_dicts.append(merge_two_dicts(eval_dict, train_dict))

            # Average performance across k runs
            eval_dict = {key: np.mean([d[key] for d in eval_dicts]) for key in eval_dicts[0].keys()}

    # Train on full dataset
    from metrics import external_metrics, internal_metrics
    start = datetime.now()
    fs = training_fun(x_train)
    train_dict = {}
    if type(fs) == tuple:
        fs, train_dict = fs
    elapsed_time = datetime.now() - start
    logging.info("Training complete in: " + str(elapsed_time))

    # Finally evaluate performance on full dataset and possibly on test set
    a = zip(["train"] + (["test"] if x_test is not None else []), [(x_train, y_train)] + ([(x_test,y_test)] if x_test is not None else []))
    fs_output = [(traintest, x, f(x[0]), f.method) for traintest, x in a for f in fs]
    eval_dict = merge_dicts([{"train_time": elapsed_time.total_seconds()}, eval_dict, train_dict,
                             {f"{metric[0]}_{traintest}_{f_method}": metric[1](y_true=y, **f_output)
                              for metric in external_metrics + internal_metrics
                              for traintest, (x, y), f_output, f_method in fs_output}])
    eval_dict["fs_output"] = fs_output

    return eval_dict