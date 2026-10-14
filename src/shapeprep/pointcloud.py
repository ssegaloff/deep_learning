"""mesh loading, normalization, and surface sampling shared by training and the backend."""
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    # force="mesh" so a multi-part file comes back as one Trimesh instead of a Scene
    mesh = trimesh.load(path, file_type="off", force="mesh")
    # vertices no face uses aren't surface, but they would still stretch the bounding box
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError(f"no faces in {path}")
    return mesh


def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    lo, hi = mesh.bounds
    # bbox center depends on the extremes only, not on how densely each part was tessellated
    center = (lo + hi) / 2
    # the farthest surface point is always a vertex (distance is convex), so this radius is exact
    radius = np.linalg.norm(mesh.vertices - center, axis=1).max()
    if radius == 0:
        raise ValueError("degenerate mesh: all vertices coincide")
    # copy so callers never get their mesh mutated behind their back
    out = mesh.copy()
    out.apply_translation(-center)
    out.apply_scale(1.0 / radius)
    return out

def sample_points(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    areas = mesh.area_faces
    total = areas.sum()
    # a mesh can have faces yet no area (all degenerate); fail loudly instead of sampling garbage
    if total == 0:
        raise ValueError("mesh has zero surface area")
    # choose faces in proportion to area, so tessellation density can't bias where points land
    face_idx = rng.choice(len(areas), size=n, p=areas / total)
    tri = mesh.triangles[face_idx]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    # (u, v) is uniform on the unit square; reflecting the half past the diagonal makes it uniform on the triangle
    flip = (u + v) > 1
    u = np.where(flip, 1 - u, u)
    v = np.where(flip, 1 - v, v)
    # compute in float64 for accuracy, store float32 because that's torch's default and halves the file size
    return (a + u * (b - a) + v * (c - a)).astype(np.float32)