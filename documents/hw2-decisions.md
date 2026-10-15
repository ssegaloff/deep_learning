# HW2 decisions and reasoning

Why each choice was made, so it can be explained without notes (HW2 §2.1 and §4: you must be able to explain your code yourself). Labels: **verified** (with source), **general knowledge**, **guess**. Status: **decided**, **planned**, or **open**.

## What HW2 requires vs. what we chose to do
- **Required** (verified, Homework 2.pdf §2.2, §5, §6):
  - a network with at least one hidden layer, trained with backprop and gradient descent, with evidence that training happens
  - train/val loss curves, a training metric, and test performance
  - a lower error rate than a shallow model benchmark
  - a video covering which model was selected and why, and how well it performs and why
- **A model beyond the Part 1 MLP is expected (Sabine, Oct 15).** Part 1 builds the MLP "before integrating the model into your application" (§2.1), and §4 asks you to research and select an architecture. It can be simple. (Claude wrongly said on Oct 15 that a second network wasn't required; corrected.)
- **Not required:** ablations, sweeps, or equal tuning budgets. HW2 also says to keep the model small ("don't be a hero, yet," §2.1) and simple (§7).
- An earlier plan for a 6-run weight-decay × dropout sweep was Claude's scope expansion. Dropped on Oct 15.
- **Approach (Sabine, Oct 15): "level 1."** Train each model once with conventional, cited defaults, then change things only if needed.
  - Rule: judge every adjustment on val, never on test, and keep a log of what was tried.
  - Design choices can rest on sources and convention. Only empirical claims ("X helped") need a test.
- **Two networks (Sabine, Oct 15):** the voxel MLP (Part 1's required MLP) and PointNet (vanilla) as the second model.
  - **Reopened (Sabine, Oct 15, ~12:20), then decided (~12:28): keep PointNet, option (a).** Options considered:
    - (a) VoxelMLP + PointNet in HW2, as planned.
    - (b) VoxelMLP + a voxel CNN in HW2 (same input, controlled step).
    - (c) VoxelMLP only in HW2, PointNet in HW3.
  - Tradeoffs are in the next section.

## Second-model options (decided Oct 15: (a) PointNet)
- **(a) PointNet now.**
  - Justifiable if framed as a parallel approach, not as a controlled comparison. The reason to choose it is the forward path (segmentation in HW3, completion later) and order invariance by construction.
  - What it can't support: "PointNet did better *because* it uses points instead of voxels." Representation and architecture change together.
  - Benefit: HW3 (probably segmentation; guess, from HW1's suggested trajectory) would then extend a working, validated backbone rather than build a new architecture, a new task, and a new dataset at once.
  - Costs: more code, more to explain in a 3-minute video, more to answer questions about later (HW2 §4).
- **(b) Voxel CNN on the same 16³ input.**
  - Benefit: a controlled step. Same input, and the change is the architecture: local filters with shared weights instead of one fully connected layer. That supports a clean "why" story. The VoxelMLP overfits (about 1M parameters vs. 3,392 objects; train loss → 0.01 while val loss rises), and weight sharing is the textbook answer to that (general knowledge; whether it helps here is an empirical question).
  - Costs: likely a dead end for the segmentation and completion path. 16³ leaves little room for conv + pool; VoxNet was designed for 32³ (verified, VoxNet paper). Rejected earlier on those grounds.
- **(c) MLP only now, PointNet in HW3.** Ruled out: HW2 expects a model beyond the Part 1 MLP.
- **Claude's lean (judgment, not evidence):** (a) mildly over (c). Both over (b), unless the controlled story matters more to Sabine than the forward path.

## Data and splits
- **Validation split: 15% of ModelNet's train set, stratified, seed 0.** Decided. ModelNet's test set is never used for selection.
- **Class list stored in `meta.json`, never re-derived.** Decided. Re-deriving could shift label numbers under a trained model without any error.
- **Macro-F1 alongside accuracy.** Decided. Train is imbalanced (chair about 8.4× bathtub; verified from the manifest).
- **Balanced accuracy (mean per-class recall).** Decided (Sabine, Oct 15): added to `evaluate` and `baselines.py` before the real runs, so each run's single test pass records it. It's the metric VoxNet and PointNet's "avg. class" column report, so it allows like-for-like comparison with the papers. Cost: `baselines.json` gains a key (existing numbers unchanged).
- **Val accuracy (91%) above test (87%) while macro-F1 holds steady.** Explanation (guess, to confirm with the confusion matrix): class mix differs; val keeps train's imbalance, test is near-balanced.

## Preprocessing
- **Normalize, then sample.** Decided. Center on the bounding-box center; scale by the farthest vertex (exact radius, because distance is convex).
- **Area-weighted sampling.** Decided.
  - PointNet's paper uses the same preparation: 1,024 points sampled uniformly on mesh faces by area, normalized to the unit sphere (verified, PointNet §5.1).
- **No rotation normalization.** Decided. ModelNet10 is consistently aligned (general knowledge). Uploads would be out of distribution, which is one reason uploads are deferred.
- **2,048 points stored, first 1,024 used.** Decided. Stored points are independent draws, so any fixed prefix is a valid uniform sample.
- **`load_features` checks that the file stores at least `N_POINTS` points.** Done (commit `001a60b`). Slicing past the end returns fewer points without any error.

## Features (logistic regression and MLP)
- **16³ binary voxel occupancy (4,096 features).** Decided.
  - Logistic regression and the MLP need a fixed-length vector where each position means the same thing for every object. Raw points don't give that, because point order is arbitrary.
  - Chosen for clean comparison, simplicity, and interpretability, not because voxels are state of the art.
  - Voxel CNNs aren't the leading approach on ModelNet-style classification (general knowledge, hedged). Voxels remain common in LiDAR detection and medical volumes.
  - Costs: memory grows with the cube of the resolution; quantization blurs detail at 16³; most cells are empty.
- **Same features for logistic regression and the MLP.** Decided. Isolates one change, the hidden layer.
- **Why the shallow baseline uses voxels even though PointNet uses points:** logistic regression on raw points would be meaningless, for the same ordering reason. Say this in the README.

## Model 1: VoxelMLP
- **One hidden layer (256), ReLU, dropout 0.5, logits out.** Decided.
  - Without a nonlinearity, two linear layers collapse into one, which is logistic regression again.
  - No softmax in `forward`: `CrossEntropyLoss` applies log-softmax itself.
  - 1,051,402 parameters (verified on Legion: `torch.Size([5, 10]) 1051402`). Hidden size 256 is a guess, not tuned.

## Model 2: PointNet (vanilla), the second model
- **Source:** Qi, Su, Mo & Guibas, "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation," CVPR 2017, arXiv:1612.00593 (verified; PDF read Oct 15).
  - Not to be confused with PointNet++ (arXiv:1706.02413), or with Garcia-Garcia et al.'s 2016 "PointNet," which is a voxel CNN.
- **Why PointNet** (Sabine, Oct 15: "what sets me up best for future assignments"):
  - HW1's suggested 3D trajectory goes classification → point/mesh segmentation → generative → final app (verified, HW1 §2; HW1 says plans can change).
  - The PointNet paper includes a segmentation version: per-point features concatenated with the global feature (verified, Appendix C). The HW2 classifier would become the HW3 backbone.
  - Point-cloud completion and implicit-surface models use PointNet-style encoders, e.g. PCN and Occupancy Networks (general knowledge).
  - The per-vertex "observed vs. inferred" label is a per-point labeling problem, which is segmentation-shaped.
  - The core idea, a symmetric function (max over points) giving order invariance by construction, answers the "point clouds have no order" problem directly.
- **How to frame it:** PointNet changes both the representation and the architecture, so it is not a controlled step after the MLP. A difference from the voxel models is suggestive about voxelization but doesn't isolate it. (Claude's earlier phrasing, "asks whether voxelizing throws away information," was an overstatement.) A cleaner test of voxelization loss would vary the resolution with the model fixed.
- **Architecture (verified, Fig. 2 and Table 6):**
  - points n×3 → shared MLP (64, 64) → shared MLP (64, 128, 1024) → max pool over points → FC 512 → FC 256 → FC k
  - batch norm and ReLU on every layer except the last; dropout in the final MLP
  - "Vanilla" means without the input and feature T-Nets
- **Parameter count:**
  - The paper reports vanilla at 0.8M (Table 6). Claude's arithmetic gives 815,400 for k = 40 (819,624 with batch-norm parameters), which matches.
  - For k = 10: **807,690**, or **811,914** with batch norm (arithmetic). Check this when the class is written.
- **Paper results (ModelNet40 only; no ModelNet10 result reported):** vanilla 87.1%, full PointNet 89.2% (overall accuracy, Table 5). Trained with rotation around the up-axis plus Gaussian jitter σ = 0.02.
- **Paper's training recipe (verified, Appendix C):** Adam, lr 0.001, batch size 32, lr halved every 20 epochs, dropout keep probability 0.7 on the 256-unit layer (`p=0.3` in PyTorch), batch-norm momentum scheduled from 0.5 to 0.99.
- **Our choices at level 1:**
  - **Vanilla, no T-Nets.** Decided (Sabine, Oct 15).
    - What a T-Net is: a mini-PointNet that predicts a 3×3 matrix (aligning the input points) or a 64×64 matrix (aligning features), initialized to identity. The 64×64 matrix needs an orthogonality loss, weight 0.001; without it, the feature transform scored below no transform (86.9 vs. 87.1, Table 5).
    - Why vanilla first: the gain is uncertain on aligned ModelNet10 without rotation augmentation (guess). Vanilla is 0.8M parameters vs. 3.5M for the full model (Table 6). And "with T-Nets" becomes a clean later experiment that mirrors the paper's Table 5.
    - Cost: T-Nets would need an extra loss term in the training loop.
  - **Batch norm included.** Planned (as in the paper). Consequence: `model.eval()` matters twice over. Batch norm uses batch statistics in training and running averages in eval, so the backend must call `eval()`.
  - **Dropout 0.3** (the paper's value) for PointNet. Planned.
  - **Training recipe: the shared loop** (AdamW, lr 1e-3, weight decay 1e-2, early stopping on val loss, best weights restored), not the paper's step schedule. **Decided (Sabine, Oct 15, ~12:20).**
    - Whether the paper's recipe would score higher here is unknown. The paper doesn't compare recipes, and its recipe was used on ModelNet40 with augmentation, so it isn't validated for our setting either.
    - Tradeoff (general knowledge): lowering the lr late in training reduces step noise and often gains a little at the end. Early stopping with a constant lr instead picks the best point along a noisy path. A schedule adds a hyperparameter and assumes training runs long enough to reach the later decays, which fits awkwardly with early stopping.
    - Adjust only if val shows a need (e.g., val loss plateaus while noisy, or PointNet lands below the voxel models). Then add `StepLR` plus `scheduler.step()` once per epoch, and log it as a tried adjustment.
    - Since PointNet isn't a controlled comparison either way, one loop also means less code to explain.
  - **Batch size 32 for PointNet** (the paper's value, the one its lr 1e-3 was used with). **Decided (Sabine, Oct 15).** VoxelMLP stays at 64. Batch size is a per-model config value saved in the checkpoint.
    - Tradeoffs (general knowledge): smaller batches give noisier gradients and twice as many updates per epoch (106 vs. 53); some of that noise acts as regularization (magnitude debated). "Scale the lr with the batch" rules come from SGD (Goyal et al. 2017) and hold only loosely for Adam.
    - Batch norm concern is mild: in the shared-MLP layers, `BatchNorm1d` on (B, C, N) computes statistics over batch and points (verified, PyTorch docs), so 32 × 1,024 ≈ 33K values per channel. Only the FC 512 and FC 256 batch norms see just 32 values.
    - Batch norm in train mode raises an error on a batch of size 1. Not a risk here: 3,392 = 32 × 106 = 64 × 53 (arithmetic).
    - Hardware affects speed and memory, not which batch size is statistically better. The widest activation is about 128 MB at batch 32 (B × 1,024 × 1,024 float32; arithmetic), far below 12 GB. Different GPUs can give slightly different numbers even with the same seed (nondeterministic CUDA kernels; general knowledge).
  - **No augmentation at level 1.** Rotation and jitter can be added later as a labeled experiment.
- **Implementation (Sabine, Oct 15: "mirror the paper if we can unless it's a real cost"):**
  - Shared MLP as `nn.Conv1d(kernel_size=1)` + `BatchNorm1d` on (B, C, N). Reason: `BatchNorm1d` wants channels in dim 1. With `Linear` on (B, N, C) every batch norm needs a transpose, and at the 1,024-channel layer (N = C = 1,024) a forgotten transpose would normalize per point across channels without any error.
  - Kernel size must be 1: a wider kernel would mix each point with whichever points are stored next to it, making the output depend on point order.
  - Split into `point_mlp1` (3 → 64 → 64) and `point_mlp2` (64 → 64 → 128 → 1024), mirroring Fig. 2's two blocks. Same computation and parameter count; the segmentation version (HW3) needs the 64-dim per-point features from between them. Decided before any checkpoint is saved, since it fixes the `state_dict` keys.
  - Biases kept before batch norm (redundant, since batch norm subtracts the mean) so the count matches the paper's arithmetic.
- **Implementation note (general knowledge):** a "shared MLP" is the same weights applied to every point. It can be written as `nn.Conv1d(kernel_size=1)` over (batch, channels, points), or as `nn.Linear` over the last dimension of (batch, points, channels). Both compute the same function.
- **Robustness result relevant to shape completion (verified, Appendix B):** on ModelNet40 with half the input points removed, VoxNet fell from 86.3% to 46.0%, while PointNet dropped 3.7 points.

## Training (shared loop)
- **AdamW.** Decided (general knowledge):
  - Adam scales each weight's step by running averages of its gradients and squared gradients, so it is less sensitive to the learning rate than SGD.
  - AdamW (Loshchilov & Hutter, 2019) applies weight decay directly to the weights. Under plain Adam, an L2 term added to the gradient would be divided by the adaptive scaling, so weights with large gradient history would barely be regularized.
  - Tradeoff: tuned SGD with momentum sometimes generalizes slightly better, but needs a tuned learning rate and usually a schedule.
  - Adam is still gradient descent (same backprop gradients, scaled step), which satisfies HW2.
- **Learning rate 1e-3:** the Adam paper's default (general knowledge). This is also PointNet's value (verified). Diagnostic: a jagged or rising train loss means it's too high; a slow creep means it's too low.
- **Weight decay 1e-2:** PyTorch's AdamW default, untuned. It's the same *kind* of penalty as logistic regression's L2, but the strengths aren't on the same scale; no equivalence is claimed.
- **Batch size: per model.** 64 for VoxelMLP (a common middle value; guess-level); 32 for PointNet (see above).
- **Up to 200 epochs, patience 15 on val loss, best weights restored.** Decided. Stop on loss because it changes smoothly, while accuracy and F1 jump.
  - Strict `<` with no `min_delta`: a tiny improvement resets patience. Bounded by the 200-epoch cap, so a threshold would only be one more knob.
- **Implementation (done, commit `b9db5fb`):** `train_one_epoch` (zero_grad → forward → loss → backward → step) and `fit` (val after every epoch, snapshot on improvement, early stop, restore). Model-agnostic: `fit` only needs a model that maps a batch to logits.
  - The snapshot uses `{k: v.detach().clone() ...}` on the state dict. Without the clone it would hold references to the live tensors and keep training, a silent bug.
  - Train loss and accuracy are running averages over the epoch (dropout on, weights changing), the usual convention for curves. They read slightly worse than an eval-mode pass, and early train loss can sit above val loss for that reason. Plan: evaluate train once in eval mode at the end, in part 3.
- **Whole splits on the GPU, converted once.** Done in `to_tensors`. `DataLoader` must use `num_workers=0` with GPU tensors.
- **Fresh seed and fresh model for every run.** Decided.
- **Sanity anchor (verified Oct 15):** an untrained VoxelMLP has val loss 2.3030 against ln 10 = 2.3026. Its accuracy at initialization (5.7%) reflects which classes its near-random argmax happens to favor, so it isn't a check.
- **Part 2 check (verified on Legion, Oct 15; a loop check, not the reported result):** VoxelMLP, seed 0, batch 64. Best epoch 12 (val loss 0.2095, val accuracy 0.947); stopped at 27 = 12 + 15; the restored model reproduced the logged val loss exactly (0.209451).
  - Curve: train loss falls to about 0.01 while val loss bottoms out around epochs 7–12, then climbs to about 0.30. Classic overfitting, handled by early stopping.
  - Val accuracy stays flat (0.92–0.93) while val loss rises: the model grows more confident on the objects it gets wrong. Cross-entropy charges for confidence; accuracy only checks the argmax (interpretation, consistent with the numbers).
  - 0.947 is optimistic: it's the best of a noisy curve, and one SE on val is about 1.2 points. Test is measured once in part 3.

## Level-1 results (verified on Legion, Oct 15, ~13:00; seed 0; test evaluated once per model)
| Model | Params | Best epoch | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|---|
| Majority | - | - | 0.222 | 0.110 | 0.020 |
| Logistic regression | - | - | 0.910 | 0.870 | 0.866 |
| VoxelMLP | 1,051,402 | 12 (stopped 27) | 0.947 | 0.870 | 0.867 |
| PointNet (vanilla) | 811,914 | 18 (stopped 33) | 0.955 | **0.903** | **0.902** |
- Baseline balanced accuracy: majority 0.100 on both (recall 1 for chair, 0 for the other nine: 1/10, a sanity check); logistic regression val 0.847, test 0.863. Network balanced accuracies are in `results/<model>.json`.
- **The MLP reproduced the part 2 check epoch for epoch** (same losses to 4 decimals, best epoch 12): seeding works end to end on the GPU.
- **VoxelMLP ties logistic regression on test** (0.870 vs. 0.870). Its val advantage (0.947 vs. 0.910) didn't carry over. Likely contributors (interpretation): selecting the best epoch of a noisy val curve inflates val; one SE on val is about 1.2 points; the MLP overfits (train loss → 0.01). It does not clear the "lower error than a shallow benchmark" bar on its own.
- **PointNet clears the bar:** test error 9.7% vs. 13.0%, a 3.3-point gain, roughly 2–3 standard errors (rough; the two models are scored on the same test objects, so a paired comparison would be the precise version).
- **PointNet's val loss is jagged** (0.14 to 0.48 between neighboring epochs; epoch 25 dropped to 0.840 val accuracy). Likely contributors (general knowledge): batch norm's running averages lagging the changing weights, batch 32, constant lr. Train loss stays around 0.06–0.08, so it overfits much less than the MLP.
- **PointNet test confusions (verified from `results/pointnet.json`):** 88 errors of 908; two confusions make up 54 (61%).
  - **Table → desk 29, desk → table 6.** Table has the lowest recall (0.71). The asymmetry is unexplained: training-set imbalance would push the other way (more tables than desks). Possibilities (guesses, unverified): ambiguous ModelNet labels, or desks being more varied.
  - **Dresser ↔ night stand 12 / 13, symmetric. Likely why (Sabine, Oct 15): size normalization.** Both are boxes with drawers, mainly distinguished by absolute size, which unit-sphere normalization deliberately removes; only proportions remain. Use this in the video's "why."
  - Every other class has recall ≥ 0.94.
  - Possible later experiment (not now): keep scale as an extra input.
- **No adjustments after seeing test.** Any change now would be informed by test, even if judged on val. Possible later experiments (lr schedule, seeds) are labeled as such.

## Checkpoint and backend
- **Checkpoint contents:** `state_dict`, `config` (including batch size), `arch` (which class), `grid`, `n_points`, `classes`. Planned. The backend compares these with what it loads and raises an error on any mismatch. `load_state_dict` catches size mismatches but not preprocessing drift or a reordered class list.
- **`/predict` for dataset objects uses the stored `.npz` points, looked up by id.** Planned. Re-sampling a mesh draws different points, so borderline objects (such as the misclassified one in the video) could flip. This matches how PointNet consumes points anyway.
- **Uploads:** deferred (Sabine, Oct 15). Possible later: preprocessing works on any mesh. The real work would be an upload endpoint (needs `python-multipart`), non-OFF formats, frontend UI, and out-of-distribution poses.

## Frontend (React 19 + React Three Fiber)
- **Backend access: Vite dev proxy, not CORS.** Decided (Sabine, Oct 15, 15:31). The page calls `/api/...`; Vite forwards to `http://127.0.0.1:8000` and strips `/api`.
  - Why: the browser sees one origin, so CORS never comes up; one SSH tunnel (5173); no backend change; uvicorn stays on 127.0.0.1.
  - Cost: the proxy exists only under `npm run dev`. A built deployment would need FastAPI to serve the files, or CORS. Fine for a class demo.
  - Target is `127.0.0.1`, not `localhost`: since Node 17, Node uses the OS's DNS order, so `localhost` can resolve to `::1` first while uvicorn listens on IPv4 only (general knowledge).
  - Backend down → the proxy answers **502 Bad Gateway** (verified, Oct 15: the app showed "GET /objects: 502"). 502 means the proxy got nothing from the server behind it, not an error inside the backend.
- **Versions (verified on Legion with `npm view` and `npm ls`, Oct 15):** react / react-dom 19.3.0, @react-three/fiber 9.8.2 (peer `react >=19 <19.4`), @react-three/drei 10.7.9 (peer fiber `^9`), three 0.186.1.
  - `package.json` says `^19.2.8`, which resolved to 19.3.0 (Claude predicted 19.2.x; wrong) and would allow 19.4 once it ships. The committed `package-lock.json` is what holds React inside R3F's range. Don't `npm update` casually.
  - `stats-gl` (a drei dependency, used only by drei's `StatsGl`) pulls its own `three@0.170.0`. We don't import it, so it doesn't reach the bundle: no "Multiple instances of Three.js" warning in the console (verified, Oct 15).
  - The console shows `THREE.Clock: This module has been deprecated. Please use THREE.Timer instead.` R3F 9.8.2 still uses `Clock` internally; it's library-side and harmless. Leave it (silencing library warnings hides the next real one).
  - R3F's Expo / React Native peers are optional (general knowledge); npm didn't install or flag them.
- **npm audit:** one high advisory in `source-map-js` (GHSA-68fv-2mgg-jv7q, event-loop denial of service from a crafted source map), present before R3F was added. Fixed with plain `npm audit fix` (one package changed, within existing ranges). Never `--force`: it can jump major versions.
- **`src/api.js`: every backend call goes through one `request` helper.** `fetch` rejects only on network failure; a 404 or 500 resolves with `res.ok === false`, so the status is checked once there. It's also the only place that knows the `/api` prefix. `meshUrl(id)` returns a URL, because the loader fetches and caches by URL.
- **Object picker: class dropdown + split filter, default split test.** Decided (Sabine, Oct 15, 15:31). Test is the default because those are the objects the models never trained on. (Option rejected: a collapsible group per class showing all 4,899.)
  - Test counts are ModelNet10's official test split, passed through unchanged (e.g., bathtub 50, desk 86; seen in the app, Oct 15). Train/val counts come from our 15% stratified split.
  - **Derived, not synced, state:** the class list, the visible objects, and the default class (`selectedClass ?? classes[0]`) are computed during render, so they can't drift out of sync.
  - **Changing class or split clears the selection in the event handler**, not an effect, so no render ever shows a selection outside the list.
  - **`ignore` flag in the fetch effect:** StrictMode runs effects twice in dev; the cleanup marks the first response stale. The same pattern is needed for `/predict` (a slow response can arrive after the user picked another object).
  - The class list is built from class-name strings and is display-only. The "never re-derive the class list" rule protects the label-number mapping, which the frontend never sees.
- **Mesh viewer.**
  - `useLoader(PLYLoader, url)` inside `<Suspense>`: suspends until the file arrives and caches by URL.
  - **Fixed camera, no auto-framing:** meshes arrive normalized to the unit sphere (the training normalization), so one camera about 2.8 away fits every object (a 45° fov needs at least 1/sin 22.5° ≈ 2.61; arithmetic).
  - **Flat shading.** Decided (Sabine, Oct 15, 15:57). One normal per triangle, computed in the shader (general knowledge), so CAD edges stay crisp and the PLY needs no normals. Smooth shading can go blobby where CAD meshes share vertices across sharp edges (guess). Revisit if curved objects look bad.
  - **ModelNet10 is Z-up** (verified visually, Oct 15, 16:05: with three.js's Y-up, chairs lay on their backs; the blue Z axis ran along the chair's up). Fix: display-only rotation of −90° about X, which maps (x, y, z) → (x, z, −y), so +Z → +Y. The stored points and both models are untouched.
  - **The Canvas is mounted once, for the life of the page; the error boundary sits inside it, keyed by object id.** Done (commit `10a8fe9`), verified Oct 15: console clean while clicking and changing class/split; the camera angle carries across objects and classes.
    - How we got there (two Claude mistakes, both caught by the console):
      1. First version: the boundary outside the Canvas with `key={selectedId}`. That remounted the whole Canvas on every click (new WebGL context, camera reset) and, on a failed mesh, tore it down mid-render.
      2. Second version: the boundary moved inside, but `App` still swapped `<MeshViewer>` for a `<p>` when the selection cleared (class or split change), which unmounted the Canvas. Sabine's observation that the camera carried over only within one class and split pinned this down.
      - Symptom both times: "Attempted to synchronously unmount a root while React was already rendering" and a `removeChild` NotFoundError. Likely mechanism (guess): R3F runs its own React root inside the Canvas, and unmounting it during the outer root's commit triggers both.
    - Fix: `MeshViewer` is always rendered; the "Pick an object" hint and the error message are drei `Html` overlays inside the Canvas. The boundary takes a `fallback` function because only three objects or `Html` can render inside a Canvas.
    - A caught mesh error still logs in dev (the 502 request, plus R3F's report of the loader error, which Chrome labels "Uncaught" even though our boundary handled it; why it's labeled that way: unknown).
- **Classify + side-by-side probabilities.** Done (commit `7d77f58`).
  - The result is stored with its object id (`{id, status, data|message}`) and shown only when `result.id === selectedId`. A late response can land in state but never appears under another mesh (derived, not cleared). Updates are functional (`setResult(r => r.id === id ? … : r)`), so a slow earlier response can't overwrite a newer click's loading state.
  - The frontend checks that the backend's echoed `id` matches the request; a mismatch throws.
  - One `<table>`: rows are classes in the backend's fixed order (the confusion-matrix order), columns are models. Rows align across models; identity is the column header, not color; the HTML table is the accessible table view.
  - Percentages printed only from 1% up; exact values on hover. Correct/wrong is an icon plus the class name (✓ chair / ✗ desk); green/red only reinforce it.
  - Softmax outputs are confidence, not calibrated probabilities; overfit networks tend to be overconfident (general knowledge; Guo et al. 2017).
  - Verified Oct 15: the panel matches `/predict` JSON for table_0398 (MLP desk 0.797, table 0.093, bed 0.107; PointNet desk 0.912, table 0.088); backend down → "Prediction failed. POST /predict: 502".
- **Misclassified filter.** Agreed (Sabine, Oct 15, 15:31). `results/*.json` store only metrics, the confusion matrix, and history, not per-object predictions (verified), so the backend computes them.
  - **Backend `GET /predictions?split=&class_name=`** (commit `80b730f`): labels only. One forward function, `run_models(state, ids)`, now serves both `/predict` and `/predictions`, so the list's label and the Classify label can't disagree. Batches of 64: PointNet's widest activation is about 268 MB at 64 vs. about 3 GB for 755 train chairs in one pass (arithmetic). Batching is safe because eval-mode batch norm uses running averages, so an object's output doesn't depend on its batch-mates (last-bit float differences are possible; general knowledge). `split` is a `Literal`, so FastAPI returns 422 on anything else; an unknown `class_name` returns 422 by hand, because an empty list would look like "no errors."
    - Verified Oct 15: all 908 test objects through the batched path give mlp 790/908 = 0.87 and pointnet 820/908 = 0.9031; test tables wrong: mlp 42, pointnet 29 (matching the confusion matrices); bad split or class → 422.
  - **Frontend filter: All objects / Any model wrong / PointNet wrong / Voxel MLP wrong** (Sabine chose the select over a single checkbox; she renamed "either" to "any" so a third model fits). "Any" checks every model's label (`Object.values(...).some(...)`), not two named ones.
    - Fetched in the event handlers that make the labels needed (filter on, or class/split change while on), not in an effect. Cached per `split/class`; the list reads only the current key, so late responses can't show under another class. Errors aren't cached, so revisiting retries. Cost: after retraining and restarting the backend, an open page keeps old labels until reload.
    - `MODEL_LABELS` and `FILTERS` live in `src/models.js`, shared by the menu and the results table (and separate from component files, for the fast-refresh lint rule).
    - Verified Oct 15: table/test shows 29 (PointNet wrong) and 42 (Voxel MLP wrong) of 100; desk/test PointNet wrong 10 of 86; table_0396 under MLP wrong and Any only, table_0398 under all three; Classify on a filtered object agrees with the filter; backend down → error in the count line, then retry works.
  - **Test tables: any model wrong = 48.** So 42 + 29 − 48 = 23 wrong for both, 19 MLP only, 6 PointNet only (arithmetic). Most of PointNet's table errors are shared with the MLP despite different inputs and architectures, which suggests ambiguous objects rather than a model-specific weakness (interpretation; consistent with, not proof of, the ambiguous-label guess above).
  - **Video candidates:** table_0398 (both wrong, confidently: desk 0.80 / 0.91); table_0396 (only the MLP wrong: the models disagree); table_0393 (both wrong).

## Literature context (for the README and video; not targets)
- **VoxNet** (Maturana & Scherer, IROS 2015; PDF read Oct 15, verified):
  - C(32,5,2)–C(32,3,1)–P(2)–FC(128)–FC(K) on 32³, no padding, Leaky ReLU 0.1; 921,736 parameters (Claude's arithmetic reproduces it for K = 40).
  - ModelNet10 92% average per-class accuracy, with 12-rotation augmentation and voting.
  - In Table I's no-voting rows, test objects were randomly rotated. That's part of why rotation handling mattered so much there.
- **Consensus sweep (Oct 15; secondhand AI summary, unverified unless noted):**
  - Small voxel CNNs reach high 80s to low 90s (e.g., a beam-searched CNN, about 80K parameters, 30³, 88.0%).
  - Koba et al. 2025: 16³ non-uniform voxels, CNN–Transformer, 92.15%, abstract only.
  - Brock et al.'s 97.14% is an ensemble trained on ModelNet40.
  - Takeaway: published numbers differ in metric, resolution, and rotation protocol, so compare conditions, not just numbers.
- **Implicit neural representations** (DeepSDF, Occupancy Networks, IF-Net; general knowledge): relevant to shape completion and a later generative homework; not a fit for the HW2 classifier.

## Deferred experiments
- **Keep object scale as an input** (the scale factor from normalization), to test the dresser/night-stand explanation.
- **PointNet with T-Nets** (mirrors the paper's Table 5).
- **The paper's lr step schedule for PointNet**, only if val shows a need (see the recipe decision).
- **Augmentation:** rotation about the up-axis plus jitter (PointNet's recipe), or per-epoch resampling of 1,024 of the 2,048 stored points.
- **Seeds:** rerun the chosen models with a few seeds to show run-to-run variation.

## Housekeeping
- Start the ShapeNet-Part / PartNet access requests early if HW3 will be part segmentation, since approval times are out of our control.
- **Code layout (Sabine's call, deferred):** `train_one_epoch`, `fit`, `evaluate` could move into the package (e.g., `shapes/training.py`) so other scripts import them. For now `train.py` stays a script; ad hoc checks import it with `PYTHONPATH=training`.