# ModelNet10 3D Object Classifier

(This README and much of the code were written collaboratively with Claude; see [AI and borrowed code](#ai-and-borrowed-code).)

> **Status: HW2 in progress.** Sections marked **TODO** describe parts not built yet. The HW1 version of this app (the Albemarle County building-footprint viewer) is preserved at the git tag `hw1-submission`.

## Description

A web application that shows 3D CAD models of household objects and classifies them with a small neural network. The user picks an object, sees the 3D mesh, and clicks **Classify** to see the model's predicted class and its probability for each class.

This is the first step toward a longer project on 3D shape completion. Classification comes first because it builds the full pipeline: mesh loading, preprocessing, training, a backend that serves meshes and predictions, and a 3D viewer.

## Data

- **Type:** 3D triangle meshes (OFF format), each labeled with one of 10 object classes.
- **Dataset:** ModelNet10. It has 4,899 meshes in 10 classes: bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet.
- **Citation (required by the dataset's README):** Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, J. Xiao. *3D ShapeNets: A Deep Representation for Volumetric Shapes.* CVPR 2015. (arXiv:1406.5670)
- **Not included in this repo.** The meshes are kept outside the repository and are never committed (`*.off` is in `.gitignore`). The frontend never holds data; meshes reach the viewer only through the backend.
- **Label provenance:** TODO. Describe how ModelNet's categories and labels were produced: categories derived from SUN, models collected through search engines, and labels verified by Amazon Mechanical Turk workers. Confirm against the paper before submitting.

### Class balance

ModelNet's own training set is imbalanced: chair has 889 models and bathtub has 106, about 8.4×. The test set is close to balanced. This is why we report macro-F1 alongside accuracy.

## Pipeline

### 1. Manifest: `src/shapeprep/manifest.py`

Writes `data/manifest.csv` (one row per mesh: `id, class_name, label, split, source_split, relpath`) and `data/meta.json` (class list, validation fraction, seed). Both are committed.

- **Validation split:** 15% of ModelNet's training set, stratified by class (scikit-learn `train_test_split`, seed 0). ModelNet's test set is left untouched, so the test set is never used for model selection.
- **Split sizes:** train 3,392 / val 599 / test 908.
- **The class list is stored, never re-derived**, so label numbers can't shift under a trained model.
- **Deterministic:** re-running produces a byte-identical CSV.

### 2. Mesh preprocessing: `src/shapeprep/pointcloud.py`

Shared by training and the backend, so the model sees the same preprocessing at prediction time as in training.

- **`load_mesh`:** loads an OFF file with trimesh and drops vertices no face uses. All 4,899 files load without errors.
- **`normalize`:** centers the mesh on its bounding-box center and scales so the farthest vertex is at distance 1.
  - The bounding-box center depends only on the object's extremes. A vertex average would be pulled toward densely modeled parts, such as a curved chair back next to legs made of a few vertices.
  - The farthest vertex gives the exact radius of the surface. Distance from a point is convex, so its maximum over a triangle is at a corner.
- **`sample_points`:** draws points uniformly from the surface.
  - It picks each triangle with probability proportional to its area, so how finely a part was modeled doesn't affect where points land.
  - It then picks a uniform point inside that triangle.
  - The sampler is written by hand (about 10 lines) instead of calling trimesh, so the area weighting can be checked by reading the code.
- **Order of operations:** normalizing, then sampling gives the same distribution as sampling, then normalizing with the same center and scale. Scaling multiplies every triangle's area by the same factor, so each triangle's share of the area is unchanged.

### 3. Point-cloud dataset: `training/build_dataset.py`

Samples 2,048 points from every normalized mesh, once, and writes `data/points/{train,val,test}.npz`. Each file contains:
- `points`: float32, shape (N, 2048, 3)
- `labels`: int64
- `ids`: the manifest ids, so any prediction can be traced back to its mesh
- `n_points` and `seed`: so each file records how it was made

The files are not committed; rebuild them with the steps below.

- **Why 2,048 points when the model uses fewer:** training draws a fresh random subset of each object's points every epoch, as augmentation. Validation, test, and the backend use a fixed subset. Meshes are loaded only once.
- **Reproducibility:** one random generator, consumed in manifest order, so a single seed reproduces every file.

### 4. Baselines: TODO

- Majority class: always predicts "chair". Expected test accuracy 100/908 ≈ 11.0% and macro-F1 ≈ 0.02 (arithmetic, not yet run).
- Logistic regression: the "shallow benchmark." Feature choice TODO.

### 5. Model and training: TODO

Architecture, training loop, loss curves, metrics, test performance, saved weights.

### 6. Web application: TODO

Backend routes (`GET /objects`, `GET /objects/{id}/mesh`, `POST /predict`) and the React Three Fiber viewer.

## Results: TODO

Training/validation loss vs. epoch, training accuracy, test accuracy and macro-F1, confusion matrix, comparison against the baselines.

## Features implemented

- [x] Dataset manifest with stratified train/val split
- [x] Mesh normalization and area-weighted surface sampling
- [x] Preprocessed point-cloud dataset
- [ ] Baselines
- [ ] Neural network and training
- [ ] 3D viewer
- [ ] Classify button with class probabilities

**Annotation:** TODO (the HW1 annotation feature has not been carried over yet).
**Backend/database:** TODO (FastAPI backend in progress).

## Libraries and frameworks

- **Python:** PyTorch, NumPy, scikit-learn, trimesh, FastAPI, Uvicorn
- **Frontend:** React (Vite); React Three Fiber + drei planned
- **Environment:** uv; Python ≥ 3.14 (see `pyproject.toml`)

## Running

### Setup

```bash
uv sync
export MODELNET_ROOT=/path/to/ModelNet10
```

`MODELNET_ROOT` is the folder that contains `bathtub/`, `bed/`, and so on, each with `train/` and `test/` inside.

### Rebuild the data

```bash
uv run python -m shapeprep.manifest --out-dir data
uv run python training/build_dataset.py
```

The first command is only needed if `data/manifest.csv` is missing; the committed one is the canonical split. The second command takes about a minute.

### Train, backend, frontend: TODO

## AI and borrowed code

- Code in this repository was written collaboratively with Claude (Anthropic). Claude drafted code section by section with explanations; I reviewed it, ran it, and checked each section's output before moving on. TODO: refine this statement before submitting.
- TODO: list any code adapted from my earlier coursework (DS6050) or other sources.