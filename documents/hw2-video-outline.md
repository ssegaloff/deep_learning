# HW2 video outline (max 3:00, live demo)

The video must answer three questions (HW2 §6):
1. Which model was selected, and why.
2. How well or poorly it performs, and why.
3. What new features of the web app the model enables.

Budget: about 150 spoken words per minute, so about 450 words in total, with the demo taking a large share.

## 0:00–0:20 · Setup
- ModelNet10: 4,899 CAD meshes in 10 classes (bathtub, bed, chair, desk, dresser, monitor, night stand, sofa, table, toilet).
- Classification first because it builds the whole pipeline the shape-completion project will need: data → preprocessing → model → backend → viewer.

## 0:20–1:10 · The model and why (question 1)
- Preprocessing, in one breath:
  - Normalize each mesh to the unit sphere (bounding-box center, farthest vertex).
  - Sample 1,024 points in proportion to surface area, so densely modeled parts don't get extra points.
- Two networks, one selected:
  - Part 1 MLP: 16³ voxel occupancy (4,096 features) → 256 → 10. Same features as the logistic-regression baseline, so the hidden layer is the only change.
  - Selected: PointNet (vanilla, no T-Nets; Qi et al., CVPR 2017).
- Why PointNet:
  - Point clouds have no order. PointNet applies the same small MLP to every point, then takes the max over points. Max doesn't care about order, so the network is order-invariant by construction. (Checked: shuffling the points gives identical logits; max diff 0.0 on 8 test objects, Oct 15.)
  - It sets up what comes next: the segmentation version reuses its per-point features, and shape-completion models use PointNet-style encoders.
- Training, shared by both: cross-entropy, AdamW (lr 1e-3, weight decay 1e-2), early stopping on val loss with the best weights restored.
- Honest framing: PointNet changes both the representation (points vs. voxels) and the architecture, so the comparison doesn't say which one caused the gain.

## 1:10–1:50 · Performance and why (question 2)
- Loss curves: the MLP overfits (train loss → 0.01 while val loss rises after epoch 12); PointNet overfits much less, though its val loss is jagged.
- Test accuracy: majority 11.0% → logistic regression 87.0% → MLP 87.0% (a tie) → PointNet 90.3% (macro-F1 0.902).
- Is the gain real? One standard error is about 1.1 points on 908 test objects; PointNet's 3.3-point gain is roughly 2–3 SE, from one seed.
- MLP: val 94.7% didn't carry to test; picking the best epoch of a noisy val curve inflates val.
- Confusion matrix, PointNet: two confusions make up 61% of errors.
  - Table → desk (all 29 of table's errors; table recall 0.71).
  - Dresser ↔ night stand (12 / 13): both are boxes with drawers that differ mainly in size, and unit-sphere normalization removes size on purpose.

## 1:50–2:50 · The app (question 3)
Spine: the model turns the dataset into something you can question. Classify answers "what is this?"; the filter answers "where is the model wrong?"; the label review records what a person concludes.
- (~10 s) Picker: class, split (test by default, the objects the models never trained on). Click a table; the mesh renders; rotate and zoom. Info strip: id, class, split, file, vertex and face counts.
- (~15 s) Classify on a correct object: both models' probabilities side by side, true class marked.
- (~10 s) Show → "Any model wrong": 48 of 100 test tables. "PointNet wrong": 29; "Voxel MLP wrong": 42. 23 are wrong for both.
- (~10 s) table_0396: only the MLP is wrong. The models disagree.
- (~15 s) table_0398: both say desk, 80% and 91%. Rotate it: is it a table? Confident is not the same as calibrated.
- (~10 s) Label review: mark table_0398 "ambiguous" with a note; the history row is tagged "saw prediction" because Classify was on screen. Reload to show it persisted (SQLite). The models' errors become a review queue for the labels; nothing feeds back into training.
- Not enabled by the model, so mention only in passing: the info strip and the label review database (HW1-style features rebuilt for this viewer).

## 2:50–3:00 · Next
- HW3: segmentation, reusing PointNet's per-point features. Long term: shape completion.

## Notes
- No dropout experiment at level 1; that line was removed.
- The 0:20–1:10 and 1:10–1:50 sections are overfull. Cut after a timed read-through. The app section as listed is about 70 s; trim the correct-object Classify step first if needed.
- Pick the "correct object" for the Classify step ahead of time (a chair or toilet, recall ≥ 0.94).
- Before recording, delete `backend/reviews.db` (or leave earlier test reviews out of frame) so table_0398's history shows only the demo review.
- Rehearse explaining the training loop and PointNet's max-pool without notes; the course says you must be able to explain your code yourself.