import os
import sys
import timeit

import trimesh
import torch

sys.path.append(os.path.join(os.path.abspath(''), "../src/"))  # diff-am src

from diffsd import fast_signed_distance_cuda as signed_distance
from commons import indexing

# Create icosphere mesh
ico = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
v, f = ico.vertices, ico.faces

v = torch.from_numpy(v).cuda() # f[V, 3: by dims (xyz)], vertex positions
f = torch.from_numpy(f).cuda() # i[F, 3: by verts (triangle faces)], face indices

print(f'#Faces: {f.shape[0]}')

# Array indexing to get vertex positions for each triangles
vf = v[f] # f[F, 3: by verts, 3: by dims]

N = 10_000

print(f'#Points: {N}')

print('* First run w/ compile time')

p = torch.randn((N, 3)).cuda()

start = timeit.default_timer()
dist = signed_distance(vf, p)[0]
end = timeit.default_timer()
elapsed_time = end - start

analytic_dist = torch.linalg.norm(p, dim=-1) - 1

error = dist - analytic_dist
rmse = ((error ** 2).mean() ** 0.5).item()

print(f'{elapsed_time = :.6f} s')
print(f'{rmse = :.6f}')


print('* Second run w/o compile time')

p = torch.randn((N, 3)).cuda()

start = timeit.default_timer()
dist = signed_distance(vf, p)[0]
end = timeit.default_timer()
elapsed_time = end - start

analytic_dist = torch.linalg.norm(p, dim=-1) - 1

error = dist - analytic_dist
rmse = ((error ** 2).mean() ** 0.5).item()

print(f'{elapsed_time = :.6f} s')
print(f'{rmse = :.6f}')


print('* Closest point')

p = torch.randn((N, 3)).cuda()

start = timeit.default_timer()
dist, w, cfi = signed_distance(vf, p)
# dist - f[N]: signed distance
# w - f[N,3: per verts]: barycentric weight of the closest point regarding to the face it lies on
# cfi - i[N]: face index which the closest point lies
cf = vf[cfi] # f[N,3,3]
cp = (w[...,None] * cf).sum(dim=-2) # f[N, 3]
end = timeit.default_timer()
elapsed_time = end - start

analytic_cp = p / torch.linalg.norm(p, dim=-1, keepdims=True)

error = torch.linalg.norm(cp - analytic_cp, dim=-1)
rmse = ((error ** 2).mean() ** 0.5).item()

print(f'{elapsed_time = :.6f} s')
print(f'{rmse = :.6f}')
