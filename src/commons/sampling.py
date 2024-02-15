import torch

from einops import rearrange, reduce, repeat

from .indexing import *

class UniformSurfaceSampler:
    def __init__(self, vf, return_barycentric=False):
        edge = vf[...,[1,2],:] - vf[...,[0,0],:]  # [*B, F, 2, 3]
        area_vec = torch.cross(edge[...,0,:], edge[...,1,:], dim=-1) # [*B, F, 3]
        area = torch.linalg.norm(area_vec, dim=-1) * 0.5 # [*B, F]
        
        area_cum = torch.cumsum(area, dim=-1) # [*B, F]
        prob_cum = area_cum / area_cum[...,[-1]]
        
        self.vf = vf
        self.batch_shape = self.vf.shape[:-3]
        self.prob_cum = prob_cum
        self.return_barycentric = return_barycentric
    
    def __iter__(self):
        return self
    
    def __next__(self):
        e = torch.rand((*self.batch_shape, 1), dtype=self.vf.dtype, device=self.vf.device) # [*B, 1]
        f_idx = torch.searchsorted(self.prob_cum, e) # i[*B, 1]
        face = batch_indexing(self.vf, f_idx, dim=-3) # [*B, 1, 3, 3]
        
        w = torch.rand((*self.batch_shape, 1, 3), dtype=self.vf.dtype, device=self.vf.device)
        w1 = w[...,:2].sum(dim=-1) >= 1
        w[w1,:] = 1 - w[w1,:]
        w[...,2] = 1 - w[...,0] - w[...,1]
        
        point = multibatch_mm(w[...,None,:], face).squeeze(-2)
        if self.return_barycentric:
            return point, f_idx, w # i[*B, 1], f[*B, 1, 3]
        # else
        return point # f[*B, 1, 3]
    
    def get(self, count):
        e = torch.rand((*self.batch_shape, count), dtype=self.vf.dtype, device=self.vf.device) # [*B, k]
        f_idx = torch.searchsorted(self.prob_cum, e) # i[*B, k]
        face = batch_indexing(self.vf, f_idx, dim=-3) # [*B, k, 3, 3]
        
        w = torch.rand((*self.batch_shape, count, 3), dtype=self.vf.dtype, device=self.vf.device)
        w1 = w[...,:2].sum(dim=-1) >= 1
        w[w1,:] = 1 - w[w1,:]
        # w1 = w[...,:2].sum(dim=-1, keepdim=True).floor()
        # w += w1 * (1 - 2*w) # [*B, k, 3]
        w[...,2] = 1 - w[...,0] - w[...,1]

        point = multibatch_mm(w[...,None,:], face).squeeze(-2)        
        if self.return_barycentric:
            return point, f_idx, w # i[*B, k], f[*B, k, 3]
        # else
        return point # f[*B, k, 3]