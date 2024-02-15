import math
import torch
from torch.autograd import Function

from .common import *

def face_info(v: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
    """
    v: float tensor (*B, 3, 3)
    out: 
     - face_inv: float tensor (*B, 2, 3)
     - face_sym: float tensor (*B, 3, 3)
     - face_obt: bool tensor (*B, 3)
    """
    orig_shape = v.shape[:-2]
    v = v.view(-1, 3, 3)
    edge_vec = v.roll(-1,1) - v # edge vectors (B, 3, 3) [0-1, 1-2, 2-0]
    A = edge_vec[...,0,:] # 0-1 edges (B, 3)
    B = -edge_vec[...,2,:] # 0-2 edges (B, 3)
    
    # AA/BB/AB dot prod. (B, )
    AA = (A * A).sum(dim=-1)
    BB = (B * B).sum(dim=-1)
    AB = (A * B).sum(dim=-1)

    det = AA*BB - AB*AB
    det[(0 <= det) & (det < 1e-10)] = 1e-10
    det[(-1e-10 < det) & (det < 0)] = -1e-10

    # face_inv (B, 2, 3)
    AA, BB, AB = AA[...,None], BB[...,None], AB[...,None]
    face_inv = torch.stack([BB*A - AB*B, AA*B - AB*A], dim=1)
    face_inv /= det[...,None,None]

    # face_sym = v^T @ v  (*, 3, 3)
    face_sym = torch.bmm(torch.transpose(v, 1, 2), v)

    edge_dot = (edge_vec * edge_vec.roll(1,1)).sum(dim=-1) # edge dot (negative) (B, F, 3)
    face_obt = edge_dot > 0

    face_inv = face_inv.view(*orig_shape, 2, 3)
    face_sym = face_sym.view(*orig_shape, 3, 3)
    face_obt = face_obt.view(*orig_shape, 3)
    return face_inv, face_sym, face_obt

def p2t_dist(
        v: torch.FloatTensor, 
        p: torch.FloatTensor, 
        face_inv: torch.FloatTensor,
        face_sym: torch.FloatTensor,
        face_obt: torch.BoolTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    v: float tensor (*B, 3, 3)
    p: float tensor (*B, 3)
    info_idx: int tensor ()
    face_inv: float tensor (*B, 2, 3)
    face_sym: float tensor (*B, 3, 3)
    face_obt: bool tensor (*, 3)
    out:
     - dist: float tensor (*B, )
     - w: float tensor (*B, 3) - barycentric coordinate of the closest point
    """
    # get barycentric coordinate
    orig_shape = v.shape[:-2]
    v = v.view(-1, 3, 3) # (B, 3, 3)
    p = p.view(-1, 3) # (B, 3)
    face_inv = face_inv.view(-1, 2, 3) # (B, 2, 3)
    face_sym = face_sym.view(-1, 3, 3) # (B, 3, 3)
    face_obt = face_obt.view(-1, 3) # (B, 3)
    
    p0 = p - v[:,0, :]
    w12 = torch.bmm(face_inv, p0[:,:,None]) # (B, 2, 1)
    w0 = 1 - w12.sum(-2, keepdim=True) # (B, 1, 1) 
    w = torch.cat([w0, w12], dim=-2).squeeze(-1) # (B, 3)

    # project w into triangle
    outside = torch.any(w < 0, dim=-1) # (B, )
    if torch.any(outside):
        w[outside] = proj_barycentric_coord(
            w[outside], 
            v[outside], 
            face_sym[outside],
            face_obt[outside]
        )

    # get distance
    cp = (w[...,None]*v).sum(-2) # (B, 3)
    disp = p - cp
    dist = (disp**2).sum(-1) ** 0.5 # (B, )

    dist = dist.view(*orig_shape)
    w = w.view(*orig_shape, 3)
    return dist, w

def proj_barycentric_coord(
        w: torch.FloatTensor,
        v: torch.FloatTensor,
        face_sym: torch.FloatTensor,
        face_obt: torch.FloatTensor
    ) -> torch.FloatTensor:
    """
    w: float tensor (B, 3)
    v: float tensor (B, 3, 3)
    face_sym: float tensor (B, 3, 3)
    face_obt: bool tensor (B, 3)
    out: float tensor (B, 3)
    """
    cp = (w[...,None]*v).sum(-2) # (B, 3)
    v0 = torch.zeros(w.shape[0], dtype=int)[:,None] # (B, 1)
    e = v.roll(1,1) - v # (B, 3, 3) [0-2, 1-0, 2-1]
    
    nw = w <= 0 # (B, 3), negative w
    obt_cond = ((cp.unsqueeze(-2) - v) * e).sum(-1) > 0 # (B, 3), DDOT(p, v[i-1], v[i] > 0)

    nw2 = nw.roll(-1,1) & nw.roll(-2,1) # other 2 are <= 0
    nw1 = nw & ~nw.roll(-1,1) & ~nw.roll(-2,1) # exactly 1 is <= 0
    
    obt_case = face_obt & obt_cond
    
    case0 = (nw2[:,0] & ~obt_case[:,0]) | (nw2[:,1] & obt_case[:,1]) | nw1[:,0]
    case1 = (nw2[:,1] & ~obt_case[:,1]) | (nw2[:,2] & obt_case[:,2]) | nw1[:,1]
    case2 = (nw2[:,2] & ~obt_case[:,2]) | (nw2[:,0] & obt_case[:,0]) | nw1[:,2]
    
    assert torch.all(case0 | case1 | case2)
    assert not torch.all((case0 & case1) | (case1 & case2) | (case2 & case0))

    v0[case0] = 0
    v0[case1] = 1
    v0[case2] = 2

    # (B, 1)
    v1 = (v0 + 1) % 3
    v2 = (v0 + 2) % 3

    a0 = batch_indexing(face_sym, v0, dim=-2) - batch_indexing(face_sym, v1, dim=-2)
    a0 = a0.squeeze(1) # (B, 3)
    a0_v0 = batch_indexing(a0, v0, dim=-1)
    a0_v1 = batch_indexing(a0, v1, dim=-1)
    k = ((w * a0).sum(-1, keepdim=True) - a0_v1) / (a0_v0 - a0_v1)
    
    t = torch.zeros_like(w)
    t_v0 = torch.clamp(k, 0, 1)
    batch_indexing_set(t, v0, t_v0, dim=-1)
    batch_indexing_set(t, v1, 1 - t_v0, dim=-1)
    batch_indexing_set(t, v2, 0, dim=-1)

    return t

def winding(
        v: torch.FloatTensor, 
        p: torch.FloatTensor   
    ) -> torch.FloatTensor:
    """
    v: float tensor (*B, 3, 3)
    p: float tensor (*B, 3)
    out: float tensor (*B, ) - winding number
    """
    orig_shape = v.shape[:-2]
    v = v.view(-1, 3, 3) # (B, 3, 3)
    p = p.view(-1, 3) # (B, 3)

    pv = v - p[:,None,:] # (B, 3, 3) - B * [A, B, C]
    l = (pv ** 2).sum(-1) ** 0.5 # (B, 3) - B * [|A|, |B|, |C|]
    dot = (pv.roll(1, 1) * pv.roll(2, 1)).sum(-1) # (B, 3) - B * [<BC>, <CA>, <AB>]

    det = torch.linalg.det(pv) # (B, )
    denom = l.prod(-1) + (dot * l).sum(-1) # (B, )

    w = torch.atan2(det, denom) * 0.5 / math.pi
    w = w.view(*orig_shape)
    return w

class SignedDistance(Function):
    @staticmethod
    def forward(
            ctx, 
            vf: torch.FloatTensor, 
            point: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        vf: int tensor (*B, F, 3, 3)
        points: float tensor (*B, P, 3)
        out:
         - dist: float tensor (*B, P)
         - w: float tensor (*B, P, 3)
         - min_face_idx: int tensor (*B, P)
        """
        print("TODO: fix incorrect result for corner cases")
        *batch_size, num_faces = vf.shape[:-2]
        *batch_size_p, num_points = point.shape[:-1]
        assert batch_size == batch_size_p
        
        face_inv, face_sym, face_obt = face_info(vf)
        
        # (BPF, ...)
        point_all = point.repeat_interleave(num_faces, dim=-2).view(-1, 3)

        repeat_param = [*[1 for _ in batch_size], num_points]
        vf_all = vf.repeat(*repeat_param, 1, 1).view(-1, 3, 3)
        face_inv_all = face_inv.repeat(*repeat_param, 1, 1).view(-1, 2, 3)
        face_sym_all = face_sym.repeat(*repeat_param, 1, 1).view(-1, 3, 3)
        face_obt_all = face_obt.repeat(*repeat_param, 1).view(-1, 3)

        dist_all, w_all = p2t_dist(vf_all, point_all, face_inv_all, face_sym_all, face_obt_all)
        dist_all = dist_all.view(*batch_size, num_points, num_faces) # (*B, P, F)
        w_all = w_all.view(*batch_size, num_points, num_faces, 3) # (*B, P, F, 3)

        dist, min_face_idx = torch.min(dist_all, dim=-1) # (*B, P)
        w = batch_indexing(w_all, min_face_idx, dim=-2) # (*B, P, 3)

        wn_all = winding(vf_all, point_all) # (BPF, )
        wn_all = wn_all.view(*batch_size, num_points, num_faces) # (*B, P, F)
        wn = wn_all.sum(-1) #(*B, P)
        inside = wn > 0.5
        dist[inside] *= -1

        ctx.save_for_backward(vf, point, dist, w, min_face_idx, inside)
        return dist, w, min_face_idx

    @staticmethod
    def backward(
            ctx, 
            grad_dist: torch.FloatTensor, 
            _grad_w: torch.FloatTensor, 
            _grad_min_face_idx: torch.IntTensor
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        grad_dist: float tensor (*B, P)
        out: 
         - grad_vf: float tensor (*B, F, 3, 3)
         - grad_points: float tensor (*B, P, 3)
        """
        # (*B, F, 3, 3) / (*B, P, 3) / (*B, P) / (*B, P, 3) / int (*B, P) / bool (*B, P)
        vf, point, dist, w, min_face_idx, inside = ctx.saved_tensors
        grad_vf = grad_points = None

        closest_face = batch_indexing(vf, min_face_idx, dim=-3) # (*B, P, 3, 3)
        cp = (w[...,None]*closest_face).sum(-2) # (*B, P, 3)
        disp = point - cp
        norm_disp = disp / dist[...,None] # dist contains inside/outside info
        
        grad_points = grad_dist[...,None] * norm_disp # (*B, P, 3)

        # (*B, P, 3, 3), vertex gradient contribution from point to the closest face
        grad_vf_p = multibatch_mm(w[..., None], -grad_points[..., None, :])

        grad_vf = torch.zeros_like(vf)
        batch_index_add_(grad_vf, min_face_idx, grad_vf_p, dim=-3)

        return grad_vf, grad_points

def signed_distance(vf, p, **kwargs):
    return SignedDistance.apply(vf, p)
