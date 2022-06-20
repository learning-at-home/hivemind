import torch
import torch.nn.functional as F


@torch.jit.script
def orthogonalize_(matrix, eps: float = 1e-8):
    """Orthogonalize a 2d tensor in-place over the last dimension"""
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i]
        F.normalize(col, dim=0, eps=eps, out=col)
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest.addmm_(col[:, None], (col @ rest)[None, :], alpha=-1)


def get_flatten_greedy_dims(tensor: torch.Tensor, max_ndim: int = 2):
    """get dims to flatten tensor upto max_ndim dimensions by merging small axes together"""
    dims = list(tensor.shape)
    while len(dims) > max_ndim:
        squeeze_ix = min(range(len(dims) - 1), key=lambda i: dims[i] * dims[i + 1])
        squeezed_dim = dims.pop(squeeze_ix)
        dims[squeeze_ix] *= squeezed_dim
    return dims
