import trimesh
import numpy as np
import torch

def load_mesh_np(obj_path):
    """ .obj file loader
    normalizes shape into [-0.5, 0.5]^3 box
    """
    tm = trimesh.load(obj_path, process=False)
    tm.vertices -= tm.center_mass
    tm.vertices /= np.max(tm.extents)
    return tm.vertices, tm.faces

def load_mesh(obj_path, device=None, dtypes=None):
    vert_np, face_np = load_mesh_np(obj_path)
    vert, face = torch.from_numpy(vert_np), torch.from_numpy(face_np)
    if dtypes is not None:
        vert = vert.to(dtypes[0])
        face = face.to(dtypes[1])
    if device is not None:
        vert = vert.to(device)
        face = face.to(device)
    return vert, face
