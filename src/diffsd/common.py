import torch

def batch_indexing(v: torch.Tensor, i: torch.IntTensor, dim: int = 1) -> torch.Tensor:
    """
    v: tensor (*B, m, *)
    i: int tensor (*B, *n)
    out: tensor (*B, *n, *)

    out[j, *k,...] = v[j, i[j, *k], ...]
    """
    if dim < 0:
        dim = len(v.shape) + dim
    B, [M, *rest] = v.shape[:dim], v.shape[dim:]
    Bi, N = i.shape[:dim], i.shape[dim:]

    assert B == Bi

    if not B:
        return v[i]
    
    Bsize = 1
    for d in B:
        Bsize *= d

    vv = v.view(-1, *rest)
    ii = i.view(Bsize, -1)
    ii = ii + torch.arange(Bsize, device=i.device)[:,None] * M

    return vv[ii].view(*B, *N, *rest)

def batch_grid_indexing(v, i, dim_start, dim_end=None):
    """
    v: tensor (*B, *m, *)
    i: int tensor (*B, *n, |*m|)
    out: tensor (*B, *n, *)
    
    out[j, *k, ...] = v[j, i[j, *k, 0], i[j, *k, 1], ... i[j, *k, |*m|], ...]
    """
    if dim_start < 0:
        dim_start = len(v.shape) + dim_start
    if dim_end is None:
        dim_end = len(v.shape)
    if dim_end < 0:
        dim_end = len(v.shape) + dim_end
    B, M, V = v.shape[:dim_start], v.shape[dim_start:dim_end], v.shape[dim_end:]
    Bi, N, lenM = i.shape[:dim_start], i.shape[dim_start:-1], i.shape[-1]
    
    assert B == Bi
    assert len(M) == lenM
    
    Bsize = 1
    for d in B:
        Bsize *= d
    
    Mcoeff = torch.ones(lenM, dtype=int)
    Msize = 1
    for d in range(lenM):
        Mcoeff[:d] *= M[d]
        Msize *= M[d]
    
    vv = v.view(-1, *V)
    ii = torch.matmul(i, Mcoeff).view(Bsize, -1)
    ii = ii + torch.arange(Bsize, device=i.device)[:,None] * Msize
    
    return vv[ii].view(*B, *N, *V)

def batch_indexing_set(v: torch.Tensor, i: torch.IntTensor, t, dim: int = 1) -> None:
    """
    v: tensor (*B, m, *)
    i: int tensor (*B, *n)
    t: tensor (*B, *n, *)
    out: tensor (*B, *n, *)

    v[j, i[j, *k], ...] = t[j, *k, ...]
    """
    if dim < 0:
        dim = len(v.shape) + dim
    B, [M, *rest] = v.shape[:dim], v.shape[dim:]
    Bi, N = i.shape[:dim], i.shape[dim:]

    assert B == Bi

    if not B:
        v[i] = t
        return
    
    Bsize = 1
    for d in B:
        Bsize *= d

    vv = v.view(-1, *rest)
    ii = i.view(Bsize, -1)
    ii = ii + torch.arange(Bsize, device=i.device)[:,None] * M

    vv[ii] = t

def batch_index_add_(v, i, s, dim: int = 1):
    """
    v: tensor (*B, m, *n)
    i: int tensor (*B, *k)
    s: tensor (*B, *k, *n)

    v[*B, i[*B, k], ...] += s[*B, k, ...]
    """
    if dim < 0:
        dim = len(v.shape) + dim
    B, M, N = v.shape[:dim], v.shape[dim], v.shape[dim+1:]
    Bi, K = i.shape[:dim], i.shape[dim:]
    value_dims = len(N)
    Bs, Ks, Ns = s.shape[:dim], s.shape[dim:-value_dims], s.shape[-value_dims:]

    assert B == Bi
    assert B ==  Bs
    assert K == Ks 
    assert N == Ns
    
    Bsize = 1
    for d in B:
        Bsize *= d
    
    Ksize = 1
    for d in K:
        Ksize *= d

    vv = v.view(Bsize * M, *N)
    ii = i.view(Bsize, Ksize)
    ss = s.view(Bsize * Ksize, *N)
    ii += torch.arange(Bsize, device=i.device)[:,None] * M

    vv.index_add_(0, ii.view(-1), ss)

def batch_index_reduce_(v, i, s, dim: int = 1, reduce = 'amin', include_self=True):
    """
    v: tensor (*B, m, *n)
    i: int tensor (*B, *k)
    s: tensor (*B, *k, *n)

    v[*B, i[*B, k], ...] += s[*B, k, ...]
    """
    if dim < 0:
        dim = len(v.shape) + dim
    B, M, N = v.shape[:dim], v.shape[dim], v.shape[dim+1:]
    Bi, K = i.shape[:dim], i.shape[dim:]
    value_dims = len(N)
    if value_dims == 0:
        Bs, Ks, Ns = s.shape[:dim], s.shape[dim:], N
    else:
        Bs, Ks, Ns = s.shape[:dim], s.shape[dim:-value_dims], s.shape[-value_dims:]

    assert B == Bi
    assert B ==  Bs
    assert K == Ks 
    assert N == Ns
    
    Bsize = 1
    for d in B:
        Bsize *= d
    
    Ksize = 1
    for d in K:
        Ksize *= d

    vv = v.view(Bsize * M, *N)
    ii = i.view(Bsize, Ksize)
    ss = s.view(Bsize * Ksize, *N)
    ii += torch.arange(Bsize, device=i.device)[:,None] * M

    vv.index_reduce_(0, ii.view(-1), ss, reduce, include_self=include_self)

def multibatch_mm(a, b):
    """
    a: tensor (*B, n, m)
    b: tensor (*B, m, p)
    out: tensor (*B, n, p) (a @ b)
    """
    *B, n, m = a.shape
    *Bb, mb, p = b.shape
    assert B == Bb and m == mb

    aa = a.reshape(-1, n, m)
    bb = b.reshape(-1, m, p)
    rr = torch.bmm(aa, bb)
    return rr.view(*B, n, p)