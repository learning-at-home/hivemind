import numpy as np
import torch


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)


DUMMY = torch.empty(0, requires_grad=True)
