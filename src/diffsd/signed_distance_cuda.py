import numpy as np
from numba import cuda
import os
os.environ['NUMBAPRO_LIBDEVICE']='/usr/lib/nvidia-cuda-toolkit/libdevice/'
os.environ['NUMBAPRO_NVVM']='/usr/lib/x86_64-linux-gnu/libnvvm.so.3.1.0'

import math
import torch
from torch.autograd import Function

from .common import *

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@cuda.jit(device=True)
def dot3(v1, v2):
    result = 0
    for i in range(3):
        result += v1[i] * v2[i]
    return result

@cuda.jit(device=True)
def diffdot3(v1, v2, v):
    result = 0
    for i in range(3):
        result += (v1[i] - v[i]) * (v2[i] - v[i])
    return result

@cuda.jit
def face_info(vf, face_inv, face_sym, face_obt):
    b = cuda.grid(1)
    if b >= vf.shape[0]:
        return
    
    A = cuda.local.array(3, np.float64)
    B = cuda.local.array(3, np.float64)
    for i in range(3):
        A[i] = vf[b,1,i] - vf[b,0,i]
        B[i] = vf[b,2,i] - vf[b,0,i]
    AA = dot3(A, A)
    BB = dot3(B, B)
    AB = dot3(A, B)

    det = AA*BB - AB*AB
    if det >= 0:
        det = max(det, 1e-10)
    else:
        det = min(det, -1e-10)
    
    for i in range(3):
        face_inv[b,0,i] = (BB*A[i] - AB*B[i]) / det
        face_inv[b,1,i] = (AA*B[i] - AB*A[i]) / det
    
    T = vf[b,:,:]
    for i in range(3):
        for j in range(3):
            face_sym[b,i,j] = dot3(T[:,i], T[:,j])
    
    E = cuda.local.array((3,3), np.float64)
    for i in range(3):
        for d in range(3):
            E[i,d] = T[(i+1)%3,d] - T[i,d]
    
    if dot3(E[0,:], E[2,:]) > 0:
        face_obt[b,0] += 1
    if dot3(E[1,:], E[0,:]) > 0:
        face_obt[b,1] += 1
    if dot3(E[2,:], E[1,:]) > 0:
        face_obt[b,2] += 1

@cuda.jit(device=True)
def barycentric_projection(w, vf, face_sym, face_obt, cp):
    v0 = -1
    if w[1] <= 0 and w[2] <= 0:
        v0 = 0
        if face_obt[0] >= 1 and diffdot3(cp, vf[2,:], vf[0,:]) > 0:
            v0 = 2
    elif w[2] <= 0 and w[0] <= 0:
        v0 = 1
        if face_obt[1] >= 1 and diffdot3(cp, vf[0,:], vf[1,:]) > 0:
            v0 = 0
    elif w[0] <= 0 and w[1] <= 0:
        v0 = 2
        if face_obt[2] >= 1 and diffdot3(cp, vf[1,:], vf[2,:]) > 0:
            v0 = 1
    elif w[0] <= 0:
        v0 = 1
    elif w[1] <= 0:
        v0 = 2
    elif w[2] <= 0:
        v0 = 0
    
    v1 = (v0 + 1) % 3
    v2 = (v0 + 2) % 3
    
    a0 = cuda.local.array(3, np.float64)
    for i in range(3):
        a0[i] = face_sym[v0,i] - face_sym[v1,i]
    k = (dot3(w, a0) - a0[v1]) / (a0[v0] - a0[v1])
    k = min(max(k,0),1)
    w[v0] = k
    w[v1] = 1 - k
    w[v2] = 0

@cuda.jit(device=True)
def winding(vf, p):
    V = cuda.local.array((3,3), np.float64)
    for v in range(3):
        for d in range(3):
            V[v,d] = vf[v, d] - p[d]
    
    A = V[0,:]
    B = V[1,:]
    C = V[2,:]
    a = dot3(A,A) ** 0.5
    b = dot3(B,B) ** 0.5
    c = dot3(C,C) ** 0.5
    
    det = A[0] * (B[1]*C[2] - B[2]*C[1]) \
        - A[1] * (B[0]*C[2] - B[2]*C[0]) \
        + A[2] * (B[0]*C[1] - B[1]*C[0])
    denom = a*b*c + dot3(A, B)*c + dot3(A, C)*b + dot3(B, C)*a
    return math.atan2(det, denom) * 0.5 / math.pi

@cuda.jit
def p2t_dist(vf, p, face_inv, face_sym, face_obt, dist, _w, _winding):
    B = cuda.grid(1)
    if B >= vf.shape[0]:
        return

    p0 = cuda.local.array(3, np.float64)
    for i in range(3):
        p0[i] = p[B,i] - vf[B,0,i]
    
    w = cuda.local.array(3, np.float64)
    w[1] = dot3(face_inv[B,0,:], p0)
    w[2] = dot3(face_inv[B,1,:], p0)
    w[0] = 1 - w[1] - w[2]
    
    cp = cuda.local.array(3, np.float64)
    for i in range(3):
        cp[i] = dot3(vf[B,:,i], w)
    
    if w[0] < 0 or w[1] < 0 or w[2] < 0:
        barycentric_projection(w, vf[B,:,:], face_sym[B,:,:], face_obt[B,:], cp)
        
        # w changed, so recompute cp
        for i in range(3):
            cp[i] = dot3(vf[B,:,i], w)
    
    dist[B] = diffdot3(p[B,:], p[B,:], cp) ** 0.5
    for i in range(3):
        _w[B,i] = w[i]
    _winding[B] = winding(vf[B,:,:], p[B,:])

@cuda.jit
def backward_p2t_dist(vf, p, dist, min_face_idx, w, grad_dist, grad_vf, grad_points):
    batch_size, num_points = p.shape[0], p.shape[1]
    BP = cuda.grid(1)
    if BP >= batch_size * num_points:
        return
    
    B = BP // num_points
    P = BP % num_points
    
    cfi = min_face_idx[B,P]
    closest_face = vf[B,cfi, :, :]
    cp = cuda.local.array(3, np.float64)
    disp = cuda.local.array(3, np.float64)
    for i in range(3):
        cp[i] = dot3(closest_face[:,i], w[B,P,:])
        disp[i] = (p[B,P,i] - cp[i]) / dist[B,P]
        grad_points[B,P,i] = grad_dist[B,P] * disp[i]
    
    for v in range(3):
        for d in range(3):
            cuda.atomic.add(grad_vf, (B, cfi, v, d), -grad_points[B,P,d] * w[B,P,v])
    
    
class SignedDistanceCuda(Function):
    @staticmethod
    def forward(ctx, vf, points):
        *batch_size_mesh, num_faces = vf.shape[:-2]
        *batch_size_p, num_points = points.shape[:-1]
        assert batch_size_mesh == batch_size_p
        
        batch_size = 1
        for d in batch_size_mesh:
            batch_size *= d
        
        BLOCK = 1024
        
        vf = vf.detach().view(-1, 3, 3)
        face_inv = torch.zeros(batch_size*num_faces, 2, 3, dtype=vf.dtype, device=vf.device)
        face_sym = torch.zeros(batch_size*num_faces, 3, 3, dtype=vf.dtype, device=vf.device)
        face_obt = torch.zeros(batch_size*num_faces, 3, dtype=int, device=vf.device)
        
        face_info[((batch_size-1)//BLOCK + 1), (BLOCK)](vf, face_inv, face_sym, face_obt)
        
        vf = vf.view(-1, num_faces, 3, 3)
        vf_all = vf.view(-1, 1, num_faces, 3, 3)
        face_inv = face_inv.view(-1, 1, num_faces, 2, 3)
        face_sym = face_sym.view(-1, 1, num_faces, 3, 3)
        face_obt = face_obt.view(-1, 1, num_faces, 3)
        
#         points = points.detach().view(-1, num_points, 3)
#         points_all = points.repeat_interleave(num_faces, dim=-2).view(-1,3)
        points = points.detach().view(-1, num_points, 3)
        points_all = points.view(-1, num_points, 1, 3)
        points_all = points_all.repeat(1, 1, num_faces, 1).view(-1, 3)
        
        vf_all = vf_all.repeat(1,num_points,1,1,1).view(-1, 3, 3)
        face_inv_all = face_inv.repeat(1,num_points,1,1,1).view(-1,2,3)
        face_sym_all = face_sym.repeat(1,num_points,1,1,1).view(-1,3,3)
        face_obt_all = face_obt.repeat(1,num_points,1,1).view(-1,3)
        
        BPF = batch_size * num_points * num_faces
        
        dist_all = torch.zeros(BPF, dtype=vf.dtype, device=vf.device)
        w_all = torch.zeros(BPF, 3, dtype=vf.dtype, device=vf.device)
        winding_all = torch.zeros(BPF, dtype=vf.dtype, device=vf.device)
        
        # p2t_params = (vf_all, points_all, face_inv_all, face_sym_all, face_obt_all, dist_all, w_all, winding_all)
        # for param in p2t_params:
        #     print(param.device)
        # d_params = (cuda.as_cuda_array(t) for t in p2t_params)
        p2t_dist[((BPF-1)//BLOCK + 1), (BLOCK)](
            vf_all, points_all, 
            face_inv_all, face_sym_all, face_obt_all, 
            dist_all, w_all, winding_all
        )
        # p2t_dist[((BPF-1)//BLOCK + 1), (BLOCK)](*d_params)
        
        dist_all = dist_all.view(*batch_size_mesh, num_points, num_faces)
        w_all = w_all.view(*batch_size_mesh, num_points, num_faces, 3)
        winding_all = winding_all.view(*batch_size_mesh, num_points, num_faces)
        wn = winding_all.sum(-1)
        
        dist, min_face_idx = torch.min(dist_all, dim=-1)
        w = batch_indexing(w_all, min_face_idx, dim=-2)
        
        inside = wn > 0.5
        dist[inside] *= -1
        
        ctx.mark_non_differentiable(min_face_idx)
        ctx.save_for_backward(vf, points, dist, w, min_face_idx)
        ctx.batch_size_mesh = batch_size_mesh
        ctx.batch_size = batch_size
        ctx.num_points = num_points
        return dist, w, min_face_idx
    
    @staticmethod
    def backward(ctx, grad_dist, _grad_w, _grad_min_face_idx):
        BLOCK = 1024

        vf, points, dist, w, min_face_idx = ctx.saved_tensors
        batch_size_mesh = ctx.batch_size_mesh
        batch_size = ctx.batch_size
        num_points = ctx.num_points
        grad_vf = grad_points = None
        
        dist = dist.view(-1, num_points).detach()
        grad_dist = grad_dist.detach().view(-1, num_points)
        w = w.view(-1, num_points, 3).detach()
        min_face_idx = min_face_idx.view(-1, num_points)
        grad_vf = torch.zeros_like(vf)
        grad_points = torch.zeros_like(points)

        BP = batch_size * num_points
        backward_p2t_dist[((BP - 1)//BLOCK + 1), (BLOCK)](
            vf, points, 
            dist, min_face_idx, w, 
            grad_dist, grad_vf, grad_points
        )

        num_faces = vf.shape[-3]
        grad_vf = grad_vf.view(*batch_size_mesh, num_faces, 3, 3)
        grad_points = grad_points.view(*batch_size_mesh, num_points, 3)
        return grad_vf, grad_points

signed_distance_cuda = SignedDistanceCuda.apply
