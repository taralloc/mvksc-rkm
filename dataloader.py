import math
import os
from argparse import Namespace
from typing import List
import numpy as np
import scipy.io
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from definitions import DATA_DIR, device, TensorType, Tensor
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset(loader):
    x, y = [], []
    for data, labels in loader:
        if type(data) == Tensor:
            data = data.to(device=device)
        elif type(data) == list:
            data = list(map(lambda x: x.to(device), data))
        labels = labels.to(device=device)
        x.append(data)
        y.append(labels)
    if x == []:
        return None, None
    elif type(x[0]) == Tensor:
        x = torch.cat(x).numpy()
    elif type(x[0]) == list:
        x = [torch.cat([x[i][v] for i in range(len(x))]).numpy() for v in range(len(x[0]))]
    y = torch.cat(y)
    return x, y.numpy()

def get_dataloader(args):
    args = transformation_factory(args)
    loader = get_dataloader_helper(args)
    loader = get_dataloader_subset(loader, args)
    return loader

def transformation_factory(args):
    def transformation_factory_helper(transformation):
        assert len(transformation.items()) == 1
        d = {'normalize': transforms.Normalize}
        name, args = list(transformation.items())[0]
        t = d[name](**args)
        return t
    if "post_transformations" in args:
        args.post_transformations = list(map(transformation_factory_helper, args.post_transformations))
    return args

def get_dataloader_subset(loader, args):
    N = args.N if args.train else args.Ntest
    if N >= 0 and N < len(loader.dataset):
        rng = np.random.default_rng(torch.initial_seed())
        indices = list(rng.choice(range(len(loader.dataset)), N, shuffle=False, replace=N>len(loader.dataset)))
        loader = DataLoader(torch.utils.data.Subset(loader.dataset, indices), batch_size=args.mb_size, pin_memory=False, num_workers=args.workers, shuffle=args.shuffle, worker_init_fn=seed_worker)
    return loader

def get_dataloader_helper(args):

    args_dict = vars(args)

    if "post_transformations" not in args_dict:
        args_dict["post_transformations"] = []
    if "pre_transformations" not in args_dict:
        args_dict["pre_transformations"] = []
    if "train" not in args_dict:
        args_dict["train"] = False
    if "dataset_name" not in args_dict:
        args_dict["dataset_name"] = args_dict["name"]

    print(f'Loading data for {args_dict["dataset_name"]}...')

    if args.dataset_name == 'toydataset3':
        return get_toydataset3_dataloader(args=args)

    elif args.dataset_name == 'toydataset2':
        return get_toydataset2_dataloader(args=args)

    elif args.dataset_name == 'data3Sources':
        return get_data3Sources_dataloader(args=args)

    elif args.dataset_name == 'reuters2':
        return get_reuters2_dataloader(args=args)

    elif args.dataset_name == 'reuters':
        return get_reuters_dataloader(args=args)

    elif args.dataset_name == 'ads':
        return get_ads_dataloader(args=args)

    elif args.dataset_name == 'kolenda':
        return get_kolenda_dataloader(args=args)

    elif args.dataset_name == 'nus':
        return get_nus_dataloader(args=args)

    elif args.dataset_name == 'game':
        return get_game_dataloader(args=args)

class MultiViewDataSet(Dataset):
    def __init__(self, x: List[Tensor], y):
        self.V = len(x)
        self.N = x[0].shape[0]

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return [self.x[v][index] for v in range(self.V)], self.y[index]

    def __len__(self):
        return self.N

def get_toydataset3_dataloader(args):

    print("Loading toydataset3.")

    def generate_view(N, rng, perc_pos=0.5, mean_pos=np.array((-1.5, 0)), std_pos=np.array([[1,0],[0,1]]), mean_neg=np.array((1.5, 0)), std_neg=np.array([[1,0],[0,1]])):
        N_pos = math.floor(N * perc_pos)
        N_neg = N - N_pos
        x = torch.cat([torch.from_numpy(rng.multivariate_normal(mean=mean_pos, cov=std_pos, size=(N_pos,))).type(TensorType),
                       torch.from_numpy(rng.multivariate_normal(mean=mean_neg, cov=std_neg, size=(N_neg,))).type(TensorType)])
        y = torch.cat([torch.zeros(N_pos), torch.ones(N_neg)]) + 1
        randperm = torch.from_numpy(rng.permutation(x.shape[0])).long()
        x = x[randperm]
        y = y[randperm]
        return x, y

    rng = np.random.default_rng(0)
    N_train, N_test, perc_pos = 1000, 300, 0.8
    x1, y1 = generate_view(N_train + N_test, rng, perc_pos=perc_pos, mean_pos=np.array((1,1)), mean_neg=np.array((2,2)), std_pos=np.array([[0.1,0],[0,0.3]]), std_neg=np.array([[1.5,0.4],[0.4,1.2]]))
    if args.train:
        x1, y1 = x1[:N_train], y1[:N_train]
    else:
        x1, y1 = x1[N_train:], y1[N_train:]
    x2, y2 = generate_view(N_train + N_test, rng, perc_pos=perc_pos, mean_pos=np.array((2,2)), mean_neg=np.array((1,1)), std_neg=np.array([[1,0.5],[0.5,0.9]]), std_pos=np.array([[0.3,0],[0,0.6]]))
    if args.train:
        x2, y2 = x2[:N_train], y2[:N_train]
    else:
        x2, y2 = x2[N_train:], y2[N_train:]

    X = [x1, x2]

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(X)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(X[view].numpy(), norm='l2')))
        X = X2

    dataset = MultiViewDataSet(X, torch.stack([y1,y2]).t())

    # all_transforms = transforms.Compose(
    #     args.pre_transformations + [transforms.ToTensor()] + args.post_transformations)
    # all_transforms = None
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    # _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader

def get_data3Sources_dataloader(args, path_to_data='./data'):
    """data3Sources dataloader."""

    print("Loading data3Sources.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('data3Sources.mat'))
    X = mat['X']
    y = mat['Y']
    y = np.array([y[0, 0], y[0, 1], y[0, 2]]).swapaxes(0,1).reshape(169,3)
    X = list(map(lambda x: torch.from_numpy(x.toarray()).type(TensorType), X[0].tolist()))

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(X)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(X[view].numpy(), norm='l2')))
        X = X2

    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(X, torch.from_numpy(y).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader

def get_reuters2_dataloader(args, path_to_data='./data'):
    """reuters2 dataloader."""

    print("Loading Reuters 2.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('reutersMulSubset.mat'))
    X = [mat['En'].toarray(), mat['Fr'].toarray(), mat['Gr'].toarray()]
    y = mat['GND'].flatten()
    X = list(map(lambda x: torch.from_numpy(x).type(TensorType), X))

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(X)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(X[view].numpy(), norm='l2')))
        X = X2

    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(X, torch.from_numpy(y).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader

def get_reuters_dataloader(args, path_to_data='./data'):
    """reuters dataloader."""

    print("Loading Large-scale Reuters.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('Reuters.mat'))
    X = [mat['X'][0,v].astype(np.float32).todense() for v in range(5)]
    y = mat['Y'].flatten()
    X = list(map(lambda x: torch.from_numpy(x).type(TensorType), X))

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(X)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(X[view].numpy(), norm='l2')))
        X = X2

    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(X, torch.from_numpy(y).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader


def get_ads_dataloader(args, path_to_data='./data'):
    """ads dataloader."""

    print("Loading Ads.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('ad_data.mat'))
    Xtrain, Xtest = mat['X'], mat['Xt']
    ytrain, ytest = mat['Y'], mat['Yt']
    split = 0
    Xtrain, Xtest = [Xtrain[0][split][0][0], Xtrain[0][split][0][1], Xtrain[0][split][0][2]], [Xtest[0][split][0][0], Xtest[0][split][0][1], Xtest[0][split][0][2]]
    Xtrain, Xtest = list(map(lambda x: torch.from_numpy(x).type(TensorType), Xtrain)), list(map(lambda x: torch.from_numpy(x).type(TensorType), Xtest))
    ytrain = ytrain[0, split].flatten()
    ytest = ytest[0, split].flatten()

    Xtrain = [torch.cat([Xtrain[i], Xtest[i]]) for i in range(len(Xtrain))]
    ytrain = np.concatenate([ytrain, ytest])

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(Xtrain)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(Xtrain[view].numpy(), norm='l2')))
        Xtrain = X2


    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(Xtrain if args.train else Xtest, torch.from_numpy(ytrain if args.train else ytest).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader

def get_kolenda_dataloader(args, path_to_data='./data'):
    """kolenda dataloader."""

    print("Loading Kolenda.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('Kolenda_data.mat'))
    Xtrain, Xtest = mat['Xs'], mat['Xts']
    ytrain, ytest = mat['Ys'], mat['Yts']
    split = 0
    Xtrain, Xtest = [Xtrain[0][split][0][0], Xtrain[0][split][0][1], Xtrain[0][split][0][2]], [Xtest[0][split][0][0], Xtest[0][split][0][1], Xtest[0][split][0][2]]
    Xtrain, Xtest = list(map(lambda x: torch.from_numpy(x.astype(float)).type(TensorType), Xtrain)), list(map(lambda x: torch.from_numpy(x.astype(float)).type(TensorType), Xtest))
    ytrain = ytrain[0, split].flatten()
    ytest = ytest[0, split].flatten()

    Xtrain = [torch.cat([Xtrain[i], Xtest[i]]) for i in range(len(Xtrain))]
    ytrain = np.concatenate([ytrain, ytest])

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(Xtrain)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(Xtrain[view].numpy(), norm='l2')))
        Xtrain = X2

    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(Xtrain if args.train else Xtest, torch.from_numpy(ytrain if args.train else ytest).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader

def get_nus_dataloader(args, path_to_data='./data'):
    """nus dataloader."""

    print("Loading NUS.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('NUSWIDE_dataset.mat'))
    Xtrain, Xtest = mat['Xs'], mat['Xts']
    ytrain, ytest = mat['Ys'], mat['Yts']
    split = 0
    Xtrain, Xtest = [Xtrain[0][split][0][v] for v in range(5)], [Xtest[0][split][0][v] for v in range(5)]
    Xtrain, Xtest = list(map(lambda x: torch.from_numpy(x).type(TensorType), Xtrain)), list(map(lambda x: torch.from_numpy(x).type(TensorType), Xtest))
    ytrain = ytrain[0, split].flatten()
    ytest = ytest[0, split].flatten()

    Xtrain = [torch.cat([Xtrain[i], Xtest[i]]) for i in range(len(Xtrain))]
    ytrain = np.concatenate([ytrain, ytest])

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(Xtrain)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(Xtrain[view].numpy(), norm='l2')))
        Xtrain = X2


    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(Xtrain if args.train else Xtest, torch.from_numpy(ytrain if args.train else ytest).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader

def get_game_dataloader(args, path_to_data='./data'):
    """game dataloader."""

    print("Loading Game.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('GameData.mat'))
    Xtrain, Xtest = mat['Xs'], mat['Xts']
    ytrain, ytest = mat['Ys'], mat['Yts']
    split = 0
    Xtrain, Xtest = [Xtrain[0][split][0][v] for v in range(3)], [Xtest[0][split][0][v] for v in range(3)]
    Xtrain, Xtest = list(map(lambda x: torch.from_numpy(x).type(TensorType), Xtrain)), list(map(lambda x: torch.from_numpy(x).type(TensorType), Xtest))
    ytrain = ytrain[0, split].flatten()
    ytest = ytest[0, split].flatten()

    Xtrain = [torch.cat([Xtrain[i], Xtest[i]]) for i in range(len(Xtrain))]
    ytrain = np.concatenate([ytrain, ytest])

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(Xtrain)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(Xtrain[view].numpy(), norm='l2')))
        Xtrain = X2


    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(Xtrain if args.train else Xtest, torch.from_numpy(ytrain if args.train else ytest).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader


def get_toydataset2_dataloader(args, path_to_data='./data'):
    """toydataset2 dataloader."""

    print("Loading toydataset2.")
    mat = scipy.io.loadmat(DATA_DIR.joinpath('synth3views_2clusters.mat'))
    X = mat['X']
    y = mat['truth'].flatten()
    X = list(map(lambda x: torch.from_numpy(x).type(TensorType), X[0].tolist()))

    if "normalize" in args and args.normalize:
        X2 = []
        from sklearn.preprocessing import normalize
        view_size = len(X)
        for view in range(view_size):
            X2.append(torch.from_numpy(normalize(X[view].numpy(), norm='l2')))
        X = X2

    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = MultiViewDataSet(X, torch.from_numpy(y).type(TensorType))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader
