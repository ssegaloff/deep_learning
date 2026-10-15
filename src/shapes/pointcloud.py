"""mesh loading, normalization, surface sampling, and voxelization shared by training and the backend."""
from pathlib import Path

import numpy as np
import trimesh

# these define the model's input; training, baselines, and the backend must agree, so they live in one place
N_POINTS = 1024
GRID = 16


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


def voxelize(points: np.ndarray, grid: int = GRID) -> np.ndarray:
    # a point outside the unit sphere means normalize was skipped; fail loudly instead of clipping it into a border cell
    # the 1e-5 slack absorbs float32 rounding of points that sit exactly on the sphere
    if np.abs(points).max() > 1 + 1e-5:
        raise ValueError("points must be normalized to [-1, 1] before voxelizing")
    # map [-1, 1] onto cell indices 0..grid-1; clip because a coordinate of exactly 1.0 would land in cell `grid`
    idx = np.clip(((points + 1) / 2 * grid).astype(np.int64), 0, grid - 1)
    occ = np.zeros((grid, grid, grid), dtype=np.float32)
    # binary occupancy: a cell is 1 if any point hits it, so the feature doesn't depend on how many points were sampled
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    # flatten for logistic regression and the MLP; the 3D layout only matters to a convolutional model
    return occ.ravel()


def _load_prefix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # one place decides which stored points a model sees, so the voxel models and pointnet get the identical 1,024 points
    # NpzFile holds the file open; the with block closes it as soon as the arrays are read
    with np.load(path) as d:
        # slicing past the end returns fewer points without complaint, so check instead of trusting it
        if d["n_points"] < N_POINTS:
            raise ValueError(f"{path} stores {d['n_points']} points per object; N_POINTS is {N_POINTS}")
        # the first N_POINTS form a fixed sample; stored points are independent draws, so any fixed prefix is a valid uniform sample
        # the copy matters: a plain slice is a view that keeps the full 2,048-point array alive in memory
        points = np.ascontiguousarray(d["points"][:, :N_POINTS])
        labels = d["labels"]
    return points, labels


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points, labels = _load_prefix(path)
    return np.stack([voxelize(p, GRID) for p in points]), labels


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # pointnet reads coordinates directly: float32 (n, N_POINTS, 3), no voxelization
    # kept in the natural (object, point, xyz) layout; if the model wants channels first for Conv1d, it transposes inside forward
    return _load_prefix(path)