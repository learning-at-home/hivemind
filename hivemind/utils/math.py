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
