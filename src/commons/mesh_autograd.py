import torch
from torch.autograd import Function

from einops import rearrange, reduce, repeat

from .indexing import *

class InternalAngle(Function):
    @staticmethod
    def forward(ctx, v, f):
        """
        v: f[*B, v, 3]
        f: i[*B, f, 3]
        out: f[*B, f, 3] - internal angles in radian
        """
        *batch_shape, num_vertices = v.shape[:-1]
        *batch_shape_f, num_faces = f.shape[:-1]
        assert batch_shape == batch_shape_f
        
        v = v.view(-1, num_vertices, 3)
        f = f.view(-1, num_faces, 3)
        batch_size = v.shape[0]
        
        vf = batch_indexing(v, f, dim=1) # f[B, f, 3, 3] / dim2 : (A, B, C)
        
        edges = torch.roll(vf, -1, dims=2) - vf # f[B, f, 3, 3] /  dim2 : (B-A, C-B, A-C)
        l = torch.linalg.norm(edges, dim=3) # f[B, f, 3], edge length
        
        n_edges = edges / l[...,None] # normalized edges, f[B, f, 3, 3]
        
        cos_out = (n_edges * -torch.roll(n_edges, 1, dims=2)).sum(dim=3) # f[B, f, 3]
        out = torch.acos(cos_out)
        out = out.view(*batch_shape, num_faces, 3)
        
        ctx.save_for_backward(v, f, out, edges)
        ctx.batch_shape = batch_shape
        ctx.num_vertices = num_vertices
        ctx.num_faces = num_faces
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: f[*B, f, angle_idx=3]
        grad_v: f[*B, v, 3]
        """
        grad_v = grad_f = None
        
        v, f, out, edges = ctx.saved_tensors
        batch_shape = ctx.batch_shape
        batch_size = v.shape[0]
        num_vertices = ctx.num_vertices
        num_faces = ctx.num_faces
        
        out = out.view(-1, num_faces, 3)
        
        grad_out = grad_out.view(-1, num_faces, 3)
        
        grad_v = torch.zeros_like(v)
        # grad_out_v = torch.zeros_like(edges)
        grad_out_v = torch.zeros((batch_size, num_faces, 3, 3, 3), dtype=v.dtype, device=v.device)
        # f[B, F, angle_idx=3, vertex_idx=3, xyz=3]
        
        f_normal_raw = torch.cross(edges[:,:,0,:], edges[:,:,1,:])
        f_normal = f_normal_raw / torch.linalg.norm(f_normal_raw, dim=-1, keepdim=True) # f[B, f, 3]
        
        # computing da/dA
        for angle in range(3):
            a = angle
            b = (angle + 1) % 3
            c = (angle + 2) % 3 # vertex index
            u = edges[:,:,a,:]
            v = -edges[:,:,c,:] # edge b-a, c-a; f[B,f,3]
            grad_out_v[:,:,angle,b,:] = -torch.cross(f_normal, u) / (u ** 2).sum(-1, keepdim=True)
            grad_out_v[:,:,angle,c,:] = torch.cross(f_normal, v) / (v ** 2).sum(-1, keepdim=True)
            grad_out_v[:,:,angle,a,:] = -(grad_out_v[:,:,angle,b,:] + grad_out_v[:,:,angle,c,:])
        
        # dL/dA = dL/da @ da/dA
        ## B f a; B f a 3 3 -> B f 3 3
        grad_fv = torch.bmm(
            grad_out.view(-1, 1, 3), # B f a -> (B f) 1 a
            grad_out_v.view(-1, 3, 9) # B f a 3 3 -> (B f) a (3 3)
        ) # (B f) 1 (3 3)
        grad_fv = grad_fv.view(-1, num_faces, 3, 3) # B f 3 3
        
        # fv to v
        batch_index_add_(grad_v, f, grad_fv, dim=1)
        
        grad_v = grad_v.view(*batch_shape, num_vertices, 3)
        return grad_v, grad_f

class FaceNormal(Function):
    @staticmethod
    def forward(ctx, v, f):
        """
        v: f[*B, v, 3]
        f: i[*B, f, 3] - 2-manifold vertex position/face index triplet
        out: f[*B, f, 3] - face normal at the point
        """
        *batch_shape, num_vertices = v.shape[:-1]
        *batch_shape_f, num_faces = f.shape[:-1]
        assert batch_shape == batch_shape_f
        
        v = v.view(-1, num_vertices, 3)
        f = f.view(-1, num_faces, 3)
        batch_size = v.shape[0]
        
        out = torch.zeros((batch_size, num_faces, 3), dtype=v.dtype, device=v.device)
        
        vf = batch_indexing(v, f, dim=1) # f[B, f, 3, 3] / dim2 : (A, B, C)
        
        edges = torch.roll(vf, 1, dims=2) - torch.roll(vf, -1, dims=2) # f[B, f, 3, 3] /  dim2 : (C-B, A-C, B-A)
        f_normal = torch.cross(edges[:,:,0,:], edges[:,:,1,:])
        area2 = torch.linalg.norm(f_normal, dim=-1) # f[B f]
        f_normal /= area2[...,None] # f[B, f, 3]
        
        ctx.save_for_backward(v, f, f_normal, edges, area2)
        ctx.batch_shape = batch_shape
        ctx.num_vertices = num_vertices
        ctx.num_faces = num_faces
        
        return f_normal.view(*batch_shape, num_faces, 3)
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: f[*B, f, 3]
        grad_v: f[*B, v, 3]
        """
        grad_v = grad_f = None
        
        v, f, out, edges, area2 = ctx.saved_tensors
        batch_shape = ctx.batch_shape
        batch_size = v.shape[0]
        num_vertices = ctx.num_vertices
        num_faces = ctx.num_faces
        
        grad_out = grad_out.view(-1, num_faces, 3)
        
        grad_v = torch.zeros_like(v)
        # grad_out_v = torch.zeros_like(edges)
        grad_out_v = torch.zeros((batch_size, num_faces, 3, 3, 3), dtype=v.dtype, device=v.device)
        # f[B, F, vi=3, dN/dA=(3,3)]
        
        # computing dN/dA
        for vi in range(3):
            eN = torch.cross(edges[:,:,vi,:], out) # f[B f 3]
            grad_out_v[:,:,vi,:,:] = multibatch_mm(eN[...,None], out[...,None,:])
        grad_out_v /= area2[...,None,None,None]
        
        # dL/dA = dL/dN @ dN/dA
        ## B f dL/dN=3; B f vi=3 dN/dA=(3,3) -> B f vi=3 dL/dA=3
        grad_fv = multibatch_mm(
            repeat(grad_out, 'B f d -> B f 3 1 d').contiguous(),
            grad_out_v
        ).view(batch_size, num_faces, 3, 3)
        
        # fv to v
        batch_index_add_(grad_v, f, grad_fv, dim=1)
        
        grad_v = grad_v.view(*batch_shape, num_vertices, 3)
        return grad_v, grad_f

class AngleWeightedNormal(Function):
    """
    How NOT to create VxF matrix?
    """
    @staticmethod
    def forward(ctx, N, a, f, num_verts):
        """
        N: f[*B, f, N=3] - face normals
        a: f[*B, f, vi=3] - internal angle
        f: i[*B, f, vi=3] - face index
        num_verts: int - num vertices
        vaN: f[*B, num_verts, N=3] - angle-multiplied normal sum
        va: f[*B, num_verts] - internal angle sum
        """
        batch_shape = N.shape[:-2]
        
        aN = multibatch_mm(a[...,None], N[...,None,:]) # f[*B f vi N]
        
        # vertex internal angle sum
        va = torch.zeros((*batch_shape, num_verts), dtype=N.dtype, device=N.device) 
        batch_index_add_(va, f, a, dim=-1)
        
        # vertex angle-multiplied normal sum
        vaN = torch.zeros((*batch_shape, num_verts, 3), dtype=N.dtype, device=N.device) 
        batch_index_add_(vaN, f, aN, dim=-2)
        
        ctx.save_for_backward(N, a, f, vaN, va)
        ctx.num_verts = num_verts
        
        return vaN, va
    
    @staticmethod
    def backward(ctx, grad_vaN, grad_va):
        """
        grad_vaN: f[*B, v, N=3]
        grad_va: f[*B, v]
        grad_N: f[*B, f, N=3]
        grad_a: f[*B, f, vi=3]
        """
        grad_N = grad_a = grad_f = grad_num_verts = None
        
        N, a, f, vaN, va = ctx.saved_tensors
        num_verts = ctx.num_verts
        
        grad_f_vaN = batch_indexing(grad_vaN, f, dim=-2) # f[*B, f, vi=3, N=3]
        
        grad_f_vN = a[...,None] * grad_f_vaN # f[*B, f, vi=3, N=3]
        grad_N = grad_f_vN.sum(-2) # f[*B, f, N=3]
        
        grad_va_a = batch_indexing(grad_va, f, dim=-1)
        grad_vaN_a = (grad_f_vaN * N[...,None,:]).sum(-1) # ... f vi N1, ... f 1 N2 -> ... f vi dot(N1, N2)
        grad_a = grad_va_a + grad_vaN_a
        
        return grad_N, grad_a, grad_f, grad_num_verts

    
internal_angle = InternalAngle.apply
face_normal = FaceNormal.apply
angle_weighted_normal = AngleWeightedNormal.apply

def vertex_normal(v, f):
    """
    v: f[*B, v, 3]
    f: i[*B, f, 3] - 2-manifold vertex position/face index triplet
    out: f[*B, v, 3] - angle weighted vertex (pseudo-)normal
    """
    N = face_normal(v, f) # f[*B f N=3]
    a = internal_angle(v, f) # f[*B f vi=3]
    
    vaN, va = angle_weighted_normal(N, a, f.clone(), v.size(-2))
    vN = vaN / va[...,None]
    vNN = vN / torch.linalg.norm(vN, dim=-1, keepdim=True)
    return vNN

def surface_normal(v, f, w, fi):
    """
    v: f[*B, v, 3]
    f: i[*B, f, 3] - 2-manifold vertex position/face index triplet
    w: f[*B, p, 3] - barycentric coordinate 
    fi: i[*B, p] - barycentric triangle index
    out: f[*B, p, 3] - weighted normal
    """
    
    vN = vertex_normal(v, f) # f[*B v N=3]
    
    pf = batch_indexing(f, fi, dim=-2) # i[*B p vi=3]
    pvN = batch_indexing(vN, pf, dim=-2) # f[*B p vi=3 N=3]
    
    # f[*B p 1 w=3] @ f[*B p vi=3 N=3] -> f[*B p 1 N=3]
    sN = multibatch_mm(w[...,None,:], pvN).squeeze(-2)
    sNN = sN / torch.linalg.norm(sN, dim=-1, keepdim=True)
    return sNN
