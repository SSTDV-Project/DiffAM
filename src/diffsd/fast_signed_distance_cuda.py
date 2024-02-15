import numpy as np
from numba import cuda
import os
os.environ['NUMBAPRO_LIBDEVICE']='/usr/lib/nvidia-cuda-toolkit/libdevice/'
os.environ['NUMBAPRO_NVVM']='/usr/lib/x86_64-linux-gnu/libnvvm.so.3.1.0'

import math
import torch
from torch.autograd import Function

from einops import rearrange, reduce, repeat

from .common import *
from .torch_bvh import construct_bvh, sort_face_properties

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

@cuda.jit(device=True)
def winding_approx(dipole, q):
    pq = cuda.local.array(3, np.float64)
    for d in range(3):
        pq[d] = dipole[3+d] - q[d]
    
    w = dot3(dipole[:3], pq) / (dot3(pq, pq) ** 1.5) / (4 * math.pi)
    return w

@cuda.jit(device=True)
def bbox_range(p, bmin, bmax):
    dp = cuda.local.array(3, np.float64)
    for d in range(3):
        dp[d] = max(bmin[d] - p[d], 0, p[d] - bmax[d])
    dmin = dot3(dp, dp)
    
    for d in range(3):
        dp[d] = max((p[d] - bmin[d]) ** 2, (p[d] - bmax[d]) ** 2)
    dmax = dp[0] + dp[1] + dp[2]
    return dmin, dmax

@cuda.jit(device=True)
def bbox_dist(p, bmin, bmax):
    dp = cuda.local.array(3, np.float64)
    for d in range(3):
        dp[d] = p[d] - 0.5*(bmin[d] + bmax[d])
    dcenter = dot3(dp, dp)
    
    for d in range(3):
        dp[d] = bmax[d] - bmin[d]
    bdiam = dot3(dp, dp)
    return dcenter, bdiam

@cuda.jit(device=True)
def point2tri(p, B, f, vf, face_inv, face_sym, face_obt):
    p0 = cuda.local.array(3, np.float64)
    for i in range(3):
        p0[i] = p[i] - vf[B,f,0,i]
    
    w = cuda.local.array(3, np.float64)
    w[1] = dot3(face_inv[B,f,0,:], p0)
    w[2] = dot3(face_inv[B,f,1,:], p0)
    w[0] = 1 - w[1] - w[2]
    
    cp = cuda.local.array(3, np.float64)
    for i in range(3):
        cp[i] = dot3(vf[B,f,:,i], w)
    
    if w[0] < 0 or w[1] < 0 or w[2] < 0:
        barycentric_projection(w, vf[B,f,:,:], face_sym[B,f,:,:], face_obt[B,f,:], cp)
        
        # w changed, so recompute cp
        for i in range(3):
            cp[i] = dot3(vf[B,f,:,i], w)
    
    dist = diffdot3(p, p, cp)
    return dist, w[0], w[1], w[2]

@cuda.jit
def traverse_bvh_dist(p, vf, bvh_i, bvh_f, bvh_w, face_inv, face_sym, face_obt, cfi, _dist, w, sort2og):
    batch_size, num_points = p.shape[0], p.shape[1]
    BP = cuda.grid(1)
    if BP >= batch_size * num_points:
        return
    
    B = BP // num_points
    P = BP % num_points
    
    pos = cuda.local.array(3, np.float64)
    for d in range(3):
        pos[d] = p[B,P,d]
    
    best_cfi = -1
    best_dist = np.inf
    best_w = cuda.local.array(3, np.float64)
    wind = 0.0
    
    idx_stack = cuda.local.array(128, np.int32)
    idx_stack[0] = 0
    idx_stack_top = 0
    bound_min = cuda.local.array(3, np.float64)
    bound_max = cuda.local.array(3, np.float64)
    while idx_stack_top >= 0:
        bvh_idx = idx_stack[idx_stack_top]
        idx_stack_top -= 1
        
        is_leaf = (bvh_idx % 2)
        bvh_idx //= 2
        
        if is_leaf == 1:
            dist, w0, w1, w2 = point2tri(pos, B, bvh_idx, vf, face_inv, face_sym, face_obt)
            # print(P, bvh_idx, sort2og[B,bvh_idx], is_leaf, dist, best_dist, best_cfi, sort2og[B, best_cfi])
            if dist < best_dist:
                best_cfi = bvh_idx
                best_dist = dist
                best_w[0] = w0
                best_w[1] = w1
                best_w[2] = w2
            continue
        
        offset = (1 - is_leaf) * 6
        for d in range(3):
            bound_min[d] = bvh_f[B, bvh_idx, 0+offset+d]
            bound_max[d] = bvh_f[B, bvh_idx, 3+offset+d]

        bd_min, bd_max = bbox_range(pos, bound_min, bound_max)
        bcp, br4 = bbox_dist(pos, bound_min, bound_max)
        dist_skip = (bd_min > best_dist)
        # print(P, bvh_idx, sort2og[B,bvh_idx], is_leaf, bd_min, best_dist, best_cfi, sort2og[B, best_cfi])
        
        if not dist_skip:
            best_dist = min(best_dist, bd_max)
            left_first = (pos[0] + pos[1] + pos[2]) < 0
            idx_stack[idx_stack_top+1+left_first] = bvh_i[B, bvh_idx, 2] * 2 + bvh_i[B, bvh_idx, 4]
            idx_stack[idx_stack_top+2-left_first] = bvh_i[B, bvh_idx, 3] * 2 + bvh_i[B, bvh_idx, 5]
            idx_stack_top += 2

    cfi[B,P] = best_cfi
    _dist[B,P] = best_dist ** 0.5
    for d in range(3):
        w[B,P,d] = best_w[d]

@cuda.jit
def traverse_bvh_winding(p, vf, bvh_i, bvh_f, bvh_w, _dist):
    batch_size, num_points = p.shape[0], p.shape[1]
    BP = cuda.grid(1)
    if BP >= batch_size * num_points:
        return
    
    B = BP // num_points
    P = BP % num_points
    
    pos = cuda.local.array(3, np.float64)
    for d in range(3):
        pos[d] = p[B,P,d]
    
    wind = 0.0
    
    idx_stack = cuda.local.array(64, np.int32)
    idx_stack[0] = 0
    idx_stack_top = 0
    bound_min = cuda.local.array(3, np.float64)
    bound_max = cuda.local.array(3, np.float64)
    while idx_stack_top >= 0:
        bvh_idx = idx_stack[idx_stack_top]
        idx_stack_top -= 1
        
        is_leaf = (bvh_idx % 2)
        bvh_idx //= 2
        
        if is_leaf == 1:
            wind += winding(vf[B,bvh_idx,:,:], pos)
            continue
        
        offset = (1 - is_leaf) * 6
        for d in range(3):
            bound_min[d] = bvh_f[B, bvh_idx, 0+offset+d]
            bound_max[d] = bvh_f[B, bvh_idx, 3+offset+d]

        bd_center, b_diam = bbox_dist(pos, bound_min, bound_max)
        wind_approx = bd_center > b_diam
        
        if wind_approx:
            # use approx winding
            dipole_idx = (1 - is_leaf) * 6
            dipole = bvh_w[B, bvh_idx, dipole_idx:dipole_idx+6]
            wind += winding_approx(dipole, pos)
        else:
            idx_stack[idx_stack_top+1] = bvh_i[B, bvh_idx, 2] * 2 + bvh_i[B, bvh_idx, 4]
            idx_stack[idx_stack_top+2] = bvh_i[B, bvh_idx, 3] * 2 + bvh_i[B, bvh_idx, 5]
            idx_stack_top += 2
    
    if wind > 0.5:
        _dist[B,P] *= -1
        
@cuda.jit
def backward_p2t_dist(vf, p, dist, min_face_idx, w, sort2og, grad_dist, grad_vf, grad_points):
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
            face_idx = sort2og[B, cfi]
            cuda.atomic.add(grad_vf, (B, face_idx, v, d), -grad_points[B,P,d] * w[B,P,v])
    
class FastSignedDistanceCuda(Function):
    @staticmethod
    def forward(ctx, vf, points):
        *batch_size_mesh, num_faces = vf.shape[:-2]
        *batch_size_p, num_points = points.shape[:-1]
        assert batch_size_mesh == batch_size_p
        
        BLOCK = 1024
        
        vf = vf.detach().view(-1, num_faces, 3, 3)
        batch_size = vf.shape[0]
        
        bvh_i, bvh_f, bvh_w, bvh_aux, sort2og, sorted_vf = construct_bvh(vf)
        
        face_inv = torch.zeros(batch_size*num_faces, 2, 3, dtype=vf.dtype, device=vf.device)
        face_sym = torch.zeros(batch_size*num_faces, 3, 3, dtype=vf.dtype, device=vf.device)
        face_obt = torch.zeros(batch_size*num_faces, 3, dtype=int, device=vf.device)
        
        face_info[((batch_size*num_faces-1)//BLOCK + 1), (BLOCK)](sorted_vf.view(-1,3,3), face_inv, face_sym, face_obt)
        
        face_inv = face_inv.view(-1, num_faces, 2, 3)
        face_sym = face_sym.view(-1, num_faces, 3, 3)
        face_obt = face_obt.view(-1, num_faces, 3)
        
        points = points.detach().view(-1, num_points, 3)
        
        dist = torch.zeros(batch_size, num_points, dtype=vf.dtype, device=vf.device)
        w = torch.zeros(batch_size, num_points, 3, dtype=vf.dtype, device=vf.device)
        cfi = torch.zeros(batch_size, num_points, dtype=int, device=vf.device)
        
        BLOCK = 512
        
        traverse_bvh_dist[((batch_size*num_points-1)//BLOCK + 1), (BLOCK)](
            points, sorted_vf, bvh_i, bvh_f, bvh_w, face_inv, face_sym, face_obt, cfi, dist, w, sort2og)
        traverse_bvh_winding[((batch_size*num_points-1)//BLOCK + 1), (BLOCK)](
            points, sorted_vf, bvh_i, bvh_f, bvh_w, dist)

        ctx.mark_non_differentiable(cfi)
        ctx.save_for_backward(sorted_vf, points, dist, w, cfi, sort2og, )
        ctx.batch_size_mesh = batch_size_mesh
        ctx.batch_size = batch_size
        ctx.num_points = num_points
        
        og_cfi = batch_indexing(sort2og, cfi)
        
        dist = dist.view(*batch_size_mesh, num_points)
        w = w.view(*batch_size_mesh, num_points, 3)
        og_cfi = og_cfi.view(*batch_size_mesh, num_points)
        return dist, w, og_cfi
    
    @staticmethod
    def backward(ctx, grad_dist, _grad_w, _grad_min_face_idx):
        BLOCK = 1024

        sorted_vf, points, dist, w, min_face_idx, sort2og = ctx.saved_tensors
        batch_size_mesh = ctx.batch_size_mesh
        batch_size = ctx.batch_size
        num_points = ctx.num_points
        grad_vf = grad_points = None
        
        dist = dist.view(-1, num_points).detach()
        grad_dist = grad_dist.detach().view(-1, num_points)
        w = w.view(-1, num_points, 3).detach()
        min_face_idx = min_face_idx.view(-1, num_points)
        grad_vf = torch.zeros_like(sorted_vf)
        grad_points = torch.zeros_like(points)

        BP = batch_size * num_points
        backward_p2t_dist[((BP - 1)//BLOCK + 1), (BLOCK)](
            sorted_vf, points, 
            dist, min_face_idx, w, sort2og, 
            grad_dist, grad_vf, grad_points
        )

        num_faces = sorted_vf.shape[-3]
        grad_vf = grad_vf.view(*batch_size_mesh, num_faces, 3, 3)
        grad_points = grad_points.view(*batch_size_mesh, num_points, 3)
        return grad_vf, grad_points

fast_signed_distance_cuda = FastSignedDistanceCuda.apply
