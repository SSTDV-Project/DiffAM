import numpy as np
from numba import cuda
import os
os.environ['NUMBAPRO_LIBDEVICE']='/usr/lib/nvidia-cuda-toolkit/libdevice/'
os.environ['NUMBAPRO_NVVM']='/usr/lib/x86_64-linux-gnu/libnvvm.so.3.1.0'

import math
import torch
import torch.nn.functional as F
from torch.autograd import Function

from einops import rearrange, reduce, repeat

from .common import *

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def to_morton_codes(c):
    """
    c: f[B, f, pos=3] / 0 <= c <= 1
    out: f[B, f] : morton code of c
    """
    c10 = torch.clamp(c * 1024, min=0.0, max=1023.0).to(torch.int32)
    
    c30 = (c10 * 0x00010001) & 0xFF0000FF
    c30 = (c30 * 0x00000101) & 0x0F00F00F
    c30 = (c30 * 0x00000011) & 0xC30C30C3
    c30 = (c30 * 0x00000005) & 0x49249249
    out = c30[...,0]*4 + c30[...,1]*2 + c30[...,2]
    return out

@cuda.jit(device=True)
def lcp(sorted_mcode, B, i, j): # length of common prefix
    if j < 0 or j >= sorted_mcode.shape[1]:
        return -1
    ci = sorted_mcode[B, i]
    cj = sorted_mcode[B, j]
    len_common_prefix = cuda.clz(ci ^ cj)
    if ci == cj:
        len_common_prefix += cuda.clz(i ^ j)
    return len_common_prefix

@cuda.jit
def construct_brt_kernel(sorted_mcode, bvh_i, bvh_aux): # binary radix tree
    B, i = cuda.grid(2)
    if B >= sorted_mcode.shape[0] or i >= sorted_mcode.shape[1] - 1:
        return
    
    # Direction of the range
    lcp_pos = lcp(sorted_mcode, B, i, i+1)
    lcp_neg = lcp(sorted_mcode, B, i, i-1)
    d = lcp_pos - lcp_neg
    lcp_min = 0
    if d < 0:
        d = -1
        lcp_min = lcp_pos
    else:
        d = 1
        lcp_min = lcp_neg
    
    # Binary searching the other end
    l_max = 2
    while lcp(sorted_mcode, B, i, i + l_max*d) > lcp_min:
        l_max *= 2
    l = 0
    t = l_max // 2
    while t > 0:
        if lcp(sorted_mcode, B, i, i + (l+t)*d) > lcp_min:
            l += t
        t = t // 2 
    j = i + l*d
    
    # Finding the split position
    lcp_node = lcp(sorted_mcode, B, i, j)
    s = 0
    l_div = 2
    while l * 2 > l_div:
        t = int(math.ceil(l / l_div))
        if lcp(sorted_mcode, B, i, i + (s+t)*d) > lcp_node:
            s = s + t
        l_div *= 2
    
    range_left = min(i,j)
    range_right = max(i,j)
    child_left = i + s*d + min(d,0)
    child_right = child_left + 1
    leaf_left = 0
    leaf_right = 0
    if range_left == child_left:
        leaf_left = 1
        bvh_aux[B,child_left,1] = i
    else:
        bvh_aux[B,child_left,0] = i
    if range_right == child_right:
        leaf_right = 1
        bvh_aux[B,child_right,1] = i
    else:
        bvh_aux[B,child_right,0] = i
        
    bvh_i[B,i,0] = range_left
    bvh_i[B,i,1] = range_right
    bvh_i[B,i,2] = child_left
    bvh_i[B,i,3] = child_right
    bvh_i[B,i,4] = leaf_left
    bvh_i[B,i,5] = leaf_right

def construct_brt(sorted_mcode, bvh_i, bvh_aux):
    THREADS = 32, 32
    
    blocks0 = (sorted_mcode.shape[0] - 1) // THREADS[0] + 1
    blocks1 = (sorted_mcode.shape[1] - 1) // THREADS[1] + 1
    construct_brt_kernel[(blocks0, blocks1), THREADS](sorted_mcode, bvh_i, bvh_aux)

@cuda.jit
def compute_bbox_kernel(bvh_i, bvh_f, bvh_aux): # binary radix tree
    B, i = cuda.grid(2)
    if B >= bvh_i.shape[0] or i >= bvh_i.shape[1]:
        return
    
    parent = bvh_aux[B,i,1] # leaf_parent
    if cuda.atomic.exch(bvh_aux, (B, parent, 2), 1) == 0: # first visit
        return
    # second visit
    child_left = bvh_i[B,parent,2] # leaf
    child_right = bvh_i[B,parent,3] # leaf
    bbox_left = cuda.local.array(6, np.float64)
    bbox_right = cuda.local.array(6, np.float64)
    for d in range(6):
        bbox_left[d] = bvh_f[B,child_left,0+d]
        bbox_right[d] = bvh_f[B,child_right,0+d]
    for d in range(3):
        bvh_f[B,parent,6+d] = min(bbox_left[0+d], bbox_right[0+d])
        bvh_f[B,parent,9+d] = max(bbox_left[3+d], bbox_right[3+d])
    
    while parent != 0:
        parent = bvh_aux[B,parent,0]
        if cuda.atomic.exch(bvh_aux, (B, parent, 2), 1) == 0: # first visit
            return
        child_left = bvh_i[B,parent,2]
        child_right = bvh_i[B,parent,3]
        left_is_leaf = bvh_i[B,parent,4]
        right_is_leaf = bvh_i[B,parent,5]
        for d in range(6):
            bbox_left[d] = bvh_f[B,child_left,(1-left_is_leaf)*6+d]
            bbox_right[d] = bvh_f[B,child_right,(1-right_is_leaf)*6+d]
        for d in range(3):
            bvh_f[B,parent,6+d] = min(bbox_left[0+d], bbox_right[0+d])
            bvh_f[B,parent,9+d] = max(bbox_left[3+d], bbox_right[3+d])

def compute_bbox(bvh_i, bvh_f, bvh_aux):
    THREADS = 32, 32
    
    blocks0 = (bvh_i.shape[0] - 1) // THREADS[0] + 1
    blocks1 = (bvh_i.shape[1] - 1) // THREADS[1] + 1
    compute_bbox_kernel[(blocks0, blocks1), THREADS](bvh_i, bvh_f, bvh_aux)

@cuda.jit(device=True)
def triangle2dipole(vf, result):
    A = cuda.local.array(3, np.float64)
    B = cuda.local.array(3, np.float64)
    for d in range(3):
        A[d] = vf[1,d] - vf[0,d]
        B[d] = vf[2,d] - vf[0,d]
    
    # dipole
    result[0] = A[1]*B[2] - A[2]*B[1]
    result[1] = A[2]*B[0] - A[0]*B[2]
    result[2] = A[0]*B[1] - A[1]*B[0]
    # center
    for d in range(3):
        result[0+d] *= 0.5
        result[3+d] = (A[d]+B[d]) / 3 + vf[0,d]

@cuda.jit(device=True)
def dot3(v1, v2):
    result = 0
    for i in range(3):
        result += v1[i] * v2[i]
    return result

@cuda.jit
def compute_dipole_kernel(vf, bvh_i, bvh_w, bvh_aux): # binary radix tree
    B, i = cuda.grid(2)
    if B >= bvh_i.shape[0] or i >= bvh_i.shape[1]:
        return
    
    # leaf node
    triangle2dipole(vf[B,i,:,:], bvh_w[B,i,0:6])
    
    # internal node
    parent = bvh_aux[B,i,1] # leaf_parent
    if cuda.atomic.exch(bvh_aux, (B, parent, 3), 1) == 0: # first visit
        return
    # second visit
    child_left = bvh_i[B,parent,2] # leaf
    child_right = bvh_i[B,parent,3] # leaf
    dipole_left = cuda.local.array(6, np.float64)
    dipole_right = cuda.local.array(6, np.float64)
    for d in range(6):
        dipole_left[d] = bvh_w[B,child_left,0+d] # leaf
        dipole_right[d] = bvh_w[B,child_right,0+d] # leaf
    area_left = dot3(dipole_left, dipole_left) ** 0.5
    area_right = dot3(dipole_right, dipole_right) ** 0.5
    node_area = area_left + area_right
    for d in range(3):
        bvh_w[B,parent,6+d] = dipole_left[0+d] + dipole_right[0+d]
        bvh_w[B,parent,9+d] = (area_left*dipole_left[3+d] + area_right*dipole_right[3+d]) / node_area
    bvh_w[B,parent,12] = node_area
    
    while parent != 0:
        parent = bvh_aux[B,parent,0]
        if cuda.atomic.exch(bvh_aux, (B, parent, 3), 1) == 0: # first visit
            return
        child_left = bvh_i[B,parent,2] # internal/leaf
        child_right = bvh_i[B,parent,3] # internal/leaf
        left_is_leaf = bvh_i[B,parent,4]
        right_is_leaf = bvh_i[B,parent,5]
        for d in range(6):
            dipole_left[d] = bvh_w[B,child_left,(1-left_is_leaf)*6+d] # interanl
            dipole_right[d] = bvh_w[B,child_right,(1-right_is_leaf)*6+d] # internal
        area_left = bvh_w[B,child_left,12]
        area_right = bvh_w[B,child_right,12]
        node_area = area_left + area_right
        for d in range(3):
            bvh_w[B,parent,6+d] = dipole_left[0+d] + dipole_right[0+d]
            bvh_w[B,parent,9+d] = (area_left*dipole_left[3+d] + area_right*dipole_right[3+d]) / node_area
        bvh_w[B,parent,12] = node_area

def compute_dipole(vf, bvh_i, bvh_w, bvh_aux):
    THREADS = 16, 32
    
    blocks0 = (bvh_i.shape[0] - 1) // THREADS[0] + 1
    blocks1 = (bvh_i.shape[1] - 1) // THREADS[1] + 1
    compute_dipole_kernel[(blocks0, blocks1), THREADS](vf, bvh_i, bvh_w, bvh_aux)
    
def construct_bvh(vf):
    """
    vf: f[B, f, vi=3, pos=3] / -1 <= pos <= 1
    bvh_i: i[B, f, (range=2, children=2, children_is_leaf=2)=6]
    bvh_f: f[B, f, (leaf_range=6, node_range=6)=12]
    bvh_w: f[B, f, (leaf_dipole=6, node_dipole=6, node_area=1)=13]
    bvh_aux: i[B, f, (node_parent=1, leaf_parent=1, bbox_done=1, dipole_done=1)=4]
    
    sort2og: i[B, f] - og index of triangle at [b, f] in the sorted list: sort2og[b, f]
    """
    # Get centroids
    c_raw = reduce(vf, 'B f vi p -> B f p', 'mean')  # centroid
    c = torch.clamp((c_raw + 1) * 0.5, min=0.0, max=1.0)  # normalized centroid
    
    # centroids to morton codes
    mcode = to_morton_codes(c) # f[B, f]
    sorted_mcode, sort2og = torch.sort(mcode, dim=-1)
    sorted_vf = sort_face_properties(vf, sort2og)
    
    # Construct binary radix tree
    bvh_i = torch.zeros((*sorted_mcode.shape, 6), dtype=torch.int32, device=vf.device) # i[B,f,6]
    bvh_aux = torch.zeros((*sorted_mcode.shape, 4), dtype=torch.int32, device=vf.device) # i[B,f,4]
    construct_brt(sorted_mcode, bvh_i, bvh_aux)
    
    # Compute bbox
    bvh_f = torch.zeros((*sorted_mcode.shape, 12), dtype=torch.float64, device=vf.device) # f[B,f,12]
    bvh_f[:,:,0:3] = reduce(sorted_vf, 'B f vi p -> B f p', 'min')
    bvh_f[:,:,3:6] = reduce(sorted_vf, 'B f vi p -> B f p', 'max')
    compute_bbox(bvh_i, bvh_f, bvh_aux)
    
    # Compute winding dipole
    bvh_w = torch.zeros((*sorted_mcode.shape, 13), dtype=torch.float64, device=vf.device) # f[B,f,12]
    compute_dipole(sorted_vf, bvh_i, bvh_w, bvh_aux)
    
    return bvh_i, bvh_f, bvh_w, bvh_aux, sort2og, sorted_vf

def sort_face_properties(pf, sort2og):
    """
    pf: [B, f, *]
    sort2og: i[B, f]
    """
    return batch_indexing(pf, sort2og, dim=1)