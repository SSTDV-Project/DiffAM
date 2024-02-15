import torch

from .indexing import *

def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian

    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html

    Parameters
    ----------
    verts : torch.Tensor [*B, V, 3]
        Vertex positions.
    faces : torch.Tensor [*B, F, 3]
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    batch_shape = verts.shape[:-2]
    V, F = verts.shape[-2], faces.shape[-2] 

    face_verts = batch_indexing(verts, faces, -2) # [*B, F, 3, 3]
    v0, v1, v2 = face_verts[...,0,:], face_verts[...,1,:], face_verts[...,2,:] # [*B, F, 3]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=-1) # [*B, F]
    B = (v0 - v2).norm(dim=-1)
    C = (v0 - v1).norm(dim=-1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt() # [*B, F]

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=-1) # [*B, F, 3]
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[..., [1, 2, 0]]
    jj = faces[..., [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, -1) #*batch_shape, F * 3)
    L = torch.sparse.DoubleTensor(idx, cot.view(*batch_shape, -1), (*batch_shape, V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.mT

#     # Add the diagonal indices
#     vals = torch.sparse.sum(L, dim=0).to_dense()
#     indices = torch.arange(V)
#     idx = torch.stack([indices, indices], dim=0)
#     L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    
    # For each vertex, compute the sum of areas for triangles containing it.
#     idx = faces.view(-1)
    idx = faces # [*B, F, 3]
    inv_areas = torch.zeros(*batch_shape, V, dtype=area.dtype, device=verts.device) # [*B, V]
#     val = torch.stack([area] * 3, dim=-1).view(-1)
    val = torch.stack([area] * 3, dim=-1) # [*B, F, 3]
    
#     inv_areas.scatter_add_(0, idx, val)
    batch_index_add_(inv_areas, idx, val, dim=-1)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(*batch_shape, V, 1)
    
    vals = torch.sparse.sum(L, dim=-2).to_dense()
    indices = torch.arange(V, device=verts.device)
    idx = torch.stack([indices, indices], dim=0)
    
    L = L.to_dense()
    L = torch.diag(torch.sum(L, dim=-1)) - L

    return L, inv_areas

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

def eigen_decomposition(L, iA):
#     Lsum = torch.sum(L, dim=-1)
#     M = (torch.diag(Lsum) - L)
    eigenvalues, eigenbases = torch.linalg.eigh(L)
    return eigenvalues, eigenbases

def cart2spec(x, eigenbases):
    return eigenbases.mT.matmul(x)

def spec2cart(u, eigenbases):
    return eigenbases.matmul(u)

def edge_length(verts, faces):
    """
    verts: f[*B, V, 3]
    faces: i[*B, F, 3]
    """
    face_verts = batch_indexing(verts, faces, dim=-2) # f[*B F 3 3]
    v0, v1, v2 = face_verts[..., 0, :], face_verts[..., 1, :], face_verts[..., 2, :]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = ((v1 - v2) ** 2).sum(dim=-1)
    B = ((v0 - v2) ** 2).sum(dim=-1)
    C = ((v0 - v1) ** 2).sum(dim=-1)

    return (A + B + C).mean() * 0.5

def edge_length_var(verts, faces):
    face_verts = verts[faces]
    edges = face_verts.roll(-1,1) - face_verts
    el = (edges ** 2).sum(dim=1) ** 0.5
    return el.var()

@torch.no_grad()
def cc_subdivide(verts, faces):
    """
    Catmull-Clark subdivision, but only uses OG vertices + (new) edge vertices, thus producing triangular faces
    """
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    edges = torch.stack([faces, faces.roll(-1, -1)], dim=-1).view(-1, 2) # E(=3F) x [v0, v1]
    edges[edges[:,0] > edges[:,1]] = edges[edges[:,0] > edges[:,1]][:,[1,0]] # sort edge vertices
    edges, edge_faces = edges.unique(dim=0, return_inverse=True)
    face_edges = edge_faces.view(-1,3) # face_edges[face_idx] = [*edge indices on the face]
    edge_faces = (torch.argsort(edge_faces) // 3).view(-1, 2) # edge_faces[edge_index] = [*adjacent face indices]

    verts = torch.cat([verts, torch.ones(num_verts)[:,None]], dim=-1)

    edge_mid = verts[edges].mean(dim=1)
    face_mid = verts[faces].mean(dim=1)

    edge_fmid = face_mid[edge_faces].mean(dim=1)
    edge_point = 0.5 * (edge_mid + edge_fmid)

    vert_favg = torch.index_add(torch.zeros_like(verts), 0, faces.view(-1), face_mid[:,None,:].expand(-1,3,4).reshape(-1,4))
    vert_favg /= vert_favg[:,[-1]]
    vert_eavg = torch.index_add(torch.zeros_like(verts), 0, edges.view(-1), edge_mid[:,None,:].expand(-1,2,4).reshape(-1,4))
    vert_num_edges = vert_eavg[:,[-1]]
    vert_eavg /= vert_num_edges
    vert_point = vert_favg + 2*vert_eavg + (vert_num_edges-3)*verts
    vert_point /= vert_num_edges

    new_verts = torch.cat([vert_point, edge_point], dim=0)
    new_faces = faces[:,None,:].repeat(1, 4, 1).reshape(-1, 4, 3)
    new_faces[:,0,0] = faces[:,0]
    new_faces[:,0,1] = face_edges[:,0] + num_verts
    new_faces[:,0,2] = face_edges[:,2] + num_verts
    new_faces[:,1,0] = faces[:,1]
    new_faces[:,1,1] = face_edges[:,1] + num_verts
    new_faces[:,1,2] = face_edges[:,0] + num_verts
    new_faces[:,2,0] = faces[:,2]
    new_faces[:,2,1] = face_edges[:,2] + num_verts
    new_faces[:,2,2] = face_edges[:,1] + num_verts
    new_faces[:,3,0] = face_edges[:,0] + num_verts
    new_faces[:,3,1] = face_edges[:,1] + num_verts
    new_faces[:,3,2] = face_edges[:,2] + num_verts
    new_faces = new_faces.reshape(-1,3)
    return new_verts[:,:3], new_faces

@torch.no_grad()
def loop_subdivide(verts, faces):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    edges = torch.stack([faces, faces.roll(-1, -1)], dim=-1).view(-1, 2) # E(=3F) x [v0, v1]
    edges[edges[:,0] > edges[:,1]] = edges[edges[:,0] > edges[:,1]][:,[1,0]] # sort edge vertices
    edges, edge_faces = edges.unique(dim=0, return_inverse=True)
    face_edges = edge_faces.view(-1,3) # face_edges[face_idx] = [*edge indices on the face]
    edge_faces = (torch.argsort(edge_faces) // 3).view(-1, 2) # edge_faces[edge_index] = [*adjacent face indices]

    verts = torch.cat([verts, torch.ones(num_verts)[:,None]], dim=-1)

    edge_face_verts = verts[faces[edge_faces]].view(-1,6,4)
    edge_verts = verts[edges]
    edge_point = (1/8)*(edge_face_verts.sum(dim=-2) + edge_verts.sum(dim=-2))

    vert_neighbors = torch.index_add(torch.zeros_like(verts), 0, edges[:,0], verts[edges[:,1]])
    vert_neighbors = torch.index_add(vert_neighbors, 0, edges[:,1], verts[edges[:,0]])

    vert_degrees = vert_neighbors[:,[-1]]
    vert_point = (5/8)*verts + (3 / (8 * vert_degrees))*vert_neighbors

    new_verts = torch.cat([vert_point, edge_point], dim=0)
    new_faces = faces[:,None,:].repeat(1, 4, 1).reshape(-1, 4, 3)
    new_faces[:,0,0] = faces[:,0]
    new_faces[:,0,1] = face_edges[:,0] + num_verts
    new_faces[:,0,2] = face_edges[:,2] + num_verts
    new_faces[:,1,0] = faces[:,1]
    new_faces[:,1,1] = face_edges[:,1] + num_verts
    new_faces[:,1,2] = face_edges[:,0] + num_verts
    new_faces[:,2,0] = faces[:,2]
    new_faces[:,2,1] = face_edges[:,2] + num_verts
    new_faces[:,2,2] = face_edges[:,1] + num_verts
    new_faces[:,3,0] = face_edges[:,0] + num_verts
    new_faces[:,3,1] = face_edges[:,1] + num_verts
    new_faces[:,3,2] = face_edges[:,2] + num_verts
    new_faces = new_faces.reshape(-1,3)

    return new_verts[:,:3], new_faces

@torch.no_grad()
def simple_subdivide(verts, faces):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    edges = torch.stack([faces, faces.roll(-1, -1)], dim=-1).view(-1, 2) # E(=3F) x [v0, v1]
    edges[edges[:,0] > edges[:,1]] = edges[edges[:,0] > edges[:,1]][:,[1,0]] # sort edge vertices
    edges, edge_faces = edges.unique(dim=0, return_inverse=True)
    face_edges = edge_faces.view(-1,3) # face_edges[face_idx] = [*edge indices on the face]
#     edge_faces = (torch.argsort(edge_faces) // 3).view(-1, 2) # edge_faces[edge_index] = [*adjacent face indices]

    verts = torch.cat([verts, torch.ones(num_verts)[:,None]], dim=-1)
    edge_point = verts[edges].mean(dim=1)

    new_verts = torch.cat([verts, edge_point], dim=0)
    new_faces = faces[:,None,:].repeat(1, 4, 1).reshape(-1, 4, 3)
    new_faces[:,0,0] = faces[:,0]
    new_faces[:,0,1] = face_edges[:,0] + num_verts
    new_faces[:,0,2] = face_edges[:,2] + num_verts
    new_faces[:,1,0] = faces[:,1]
    new_faces[:,1,1] = face_edges[:,1] + num_verts
    new_faces[:,1,2] = face_edges[:,0] + num_verts
    new_faces[:,2,0] = faces[:,2]
    new_faces[:,2,1] = face_edges[:,2] + num_verts
    new_faces[:,2,2] = face_edges[:,1] + num_verts
    new_faces[:,3,0] = face_edges[:,0] + num_verts
    new_faces[:,3,1] = face_edges[:,1] + num_verts
    new_faces[:,3,2] = face_edges[:,2] + num_verts
    new_faces = new_faces.reshape(-1,3)

    return new_verts[:,:3], new_faces

def _weighted_sample_grid(gi, gp, grid, grid_shape, offset, dim, value_dims, grid_center_offset):
    """
    gi: i[*B, p, 3], lower left grid corner
    gp: f[*B, p, 3], sample position (relative to cell size)
    grid: f[*B, gx, gy, gz, *n]
    grid_shape: i[3]
    offset: i[3]
    dim: int, grid.shape[dim] = gx
    value_dims: int, value_dims == |*n|
    grid_center_offset: float or f[3], offset from lower left corner to grid point, relative to cell size
    
    sample: f[*B, p, *n]
    """
    ogi = gi + offset # i[*B p 3], offset grid index 
    ogp = ogi + grid_center_offset # f[*B p 3], offset grid position (relative)
    w = torch.prod(1 - torch.abs(gp - ogp), axis=-1) # f[*B p], grid weight
    
    dim_start = dim
    dim_end = dim+3
    
    # clipped grid index
    cgi = torch.minimum(torch.maximum(ogi, torch.zeros_like(ogi, dtype=int)), grid_shape - 1).long()
    gv = batch_grid_indexing(grid, cgi, dim_start=dim_start, dim_end=dim_end) # f[*B p *n]
    
    return w[(...,) + ((None,) * value_dims)] * gv

def trilinear_sample(p, grid, gmin, gsize, dim=-3, grid_center_offset=0.5, cell_size=None):
    """
    p: f[*B, p, 3]
    grid: f[*B, gx, gy, gz, *n]
    gmin: f[3]
    gsize: f[3]
    dim: int, grid.shape[dim] == gx
    grid_center_offset: float or f[3], offset from lower left corner to grid point, relative to cell size
    cell_size: optional, f[3]
    
    sample: f[*B, p, *n]
    """
    if dim < 0:
        dim = len(grid.shape) + dim
    
    batch_shape, grid_shape, value_size = grid.shape[:dim], grid.shape[dim:dim+3], grid.shape[dim+3:]
    batch_shape_p, num_points = p.shape[:-2], p.shape[-2]
    assert batch_shape == batch_shape_p
    
    grid_shape = torch.tensor(grid_shape, dtype=int, device=p.device)
    if cell_size is None:
        cell_size = gsize / grid_shape
    
    gp = (p - gmin) / cell_size # f[*B p 3], grid position [0-N]
    gi = torch.floor(gp - grid_center_offset).long() # i[*B p 3], lower left grid corner index
    
    value_dims = len(value_size)
    
    samples = [
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([0,0,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([0,0,1], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([0,1,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([0,1,1], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([1,0,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([1,0,1], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([1,1,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid(gi, gp, grid, grid_shape, torch.tensor([1,1,1], device=p.device), dim, value_dims, grid_center_offset),
    ]
    sample = torch.stack(samples, dim=-1).sum(dim=-1)
    
    return sample

def grid_gradient(p, grid, gmin, gsize):
    """
    p: f[*B, p, 3]
    img: f[*B, gx, gy, gz]
    gmin: f[3]
    gsize: f[3]
    
    out: f[*B, p, 3] - image gradient at p
    """
    grid_shape = torch.tensor(grid.shape[-3:], device=p.device)
    cell_size = gsize / grid_shape
    
    grid_dx = (grid[...,1:,:,:] - grid[...,:-1,:,:]) / cell_size[0] # f[*B, gx-1, gy, gz]
    grid_dy = (grid[...,:,1:,:] - grid[...,:,:-1,:]) / cell_size[1] # f[*B, gx, gy-1, gz]
    grid_dz = (grid[...,:,:,1:] - grid[...,:,:,:-1]) / cell_size[2] # f[*B, gx, gy, gz-1]
    
    out_arr = [
        trilinear_sample(p, grid_dx, gmin, gsize, 
                        grid_center_offset=torch.tensor([1,0.5,0.5], device=p.device),
                        cell_size=cell_size),
        trilinear_sample(p, grid_dy, gmin, gsize, 
                        grid_center_offset=torch.tensor([0.5,1,0.5], device=p.device),
                        cell_size=cell_size),
        trilinear_sample(p, grid_dz, gmin, gsize, 
                        grid_center_offset=torch.tensor([0.5,0.5,1], device=p.device),
                        cell_size=cell_size),
    ]
    out = torch.stack(out_arr, dim=-1)
    return out

def _weighted_sample_grid2(gi, gp, grid, grid_shape, offset, dim, value_dims, grid_center_offset):
    """
    gi: i[*B, p, 2], lower left grid corner
    gp: f[*B, p, 2], sample position (relative to cell size)
    grid: f[*B, gx, gy, *n]
    grid_shape: i[2]
    offset: i[2]
    dim: int, grid.shape[dim] = gx
    value_dims: int, value_dims == |*n|
    grid_center_offset: float or f[2], offset from lower left corner to grid point, relative to cell size
    
    sample: f[*B, p, *n]
    """
    ogi = gi + offset # i[*B p 2], offset grid index 
    ogp = ogi + grid_center_offset # f[*B p 2], offset grid position (relative)
    w = torch.prod(1 - torch.abs(gp - ogp), axis=-1) # f[*B p], grid weight
    
    dim_start = dim
    dim_end = dim+2
    
    # clipped grid index
    cgi = torch.minimum(torch.maximum(ogi, torch.zeros_like(ogi, dtype=int)), grid_shape - 1).long()
    gv = batch_grid_indexing(grid, cgi, dim_start=dim_start, dim_end=dim_end) # f[*B p *n]
    
    return w[(...,) + ((None,) * value_dims)] * gv

def bilinear_sample(p, grid, gmin, gsize, dim=-2, grid_center_offset=0.5, cell_size=None):
    """
    p: f[*B, p, 2]
    grid: f[*B, gx, gy, *n]
    gmin: f[2]
    gsize: f[2]
    dim: int, grid.shape[dim] == gx
    grid_center_offset: float or f[2], offset from lower left corner to grid point, relative to cell size
    cell_size: optional, f[2]
    
    sample: f[*B, p, *n]
    """
    if dim < 0:
        dim = len(grid.shape) + dim
    
    batch_shape, grid_shape, value_size = grid.shape[:dim], grid.shape[dim:dim+2], grid.shape[dim+2:]
    batch_shape_p, num_points = p.shape[:-2], p.shape[-2]
    assert batch_shape == batch_shape_p
    
    grid_shape = torch.tensor(grid_shape, dtype=int, device=p.device)
    if cell_size is None:
        cell_size = gsize / grid_shape
    
    gp = (p - gmin) / cell_size # f[*B p 2], grid position [0-N]
    gi = torch.floor(gp - grid_center_offset).long() # i[*B p 2], lower left grid corner index
    
    value_dims = len(value_size)
    
    samples = [
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([0,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([0,1], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([1,0], device=p.device), dim, value_dims, grid_center_offset),
        _weighted_sample_grid2(gi, gp, grid, grid_shape, torch.tensor([1,1], device=p.device), dim, value_dims, grid_center_offset),
    ]
    sample = torch.stack(samples, dim=-1).sum(dim=-1)
    
    return sample


def bound_points(val, bmax=2, r=0.5):
    """
    val: f[*]
    Bound (-inf, inf) values into (-bmax, bmax) using the simple mapping:
    f(x) = x (|x| <= r * bmax)
        or sgn(x)*bmax - r*(1-r)*bmax**2 / x (otherwise)
    """
    mval = torch.sign(val)*bmax - r*(1-r)*bmax**2 / val
    cond = torch.abs(val) <= r * bmax
    return torch.where(cond, val, mval)

def unbound_sample_points(val, bmax=2, r=0.5):
    """
    inverse of `bound_points` function
    f(x) = x (|x| <= r * bmax)
        or r*(1-r)*bmax**2 / (sgn(x)*bmax - x) (otherwise)
    """
    mval = r*(1-r)*bmax**2 / (torch.sign(val)*bmax - val)
    cond = torch.abs(val) <= r * bmax
    return torch.where(cond, val, mval)

def grange(gshape, gmin, gsize, grid_offset=0.5):
    """
    gshape: i[N] = [gx, gy, ...]
    gmin: f[N]
    gsize: f[N]
    grid_offset: float or f[N]
    
    out: f[gx, gy, ..., N]
    """
    grid_a = [torch.arange(axis_shape, device=gmin.device) for axis_shape in gshape]
    grid_idx = torch.stack(torch.meshgrid(*grid_a), axis=-1).to(gmin.dtype)
    grid_pos = gmin + gsize * (grid_idx + grid_offset) / torch.tensor(gshape, device=gmin.device)
    return grid_pos
    