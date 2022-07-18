import os
from wcmatch import pathlib
import torch
import sys
import numpy as np

torch.cuda.empty_cache()
Tensor = torch.Tensor
device = torch.device("cpu")
TensorType = torch.FloatTensor if "dataset=reuters" in sys.argv else torch.DoubleTensor
torch.set_default_tensor_type(TensorType)

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) # This is your Project Root
OUT_DIR = pathlib.Path("~/out/mvksc-rkm/").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT_DIR.joinpath('data').expanduser().absolute()