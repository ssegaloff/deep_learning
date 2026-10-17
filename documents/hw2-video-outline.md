# HW2 video outline (max 3:00, live demo)

The video must answer three questions (HW2 §6), and demonstrate the extra-credit features if claimed (Interactive Training, Training Monitor):
1. Which model was selected, and why.
2. How well or poorly it performs, and why.
3. What new features of the web app the model enables.

Budget: about 150 spoken words per minute, about 450 words. Revised Oct 16 ~23:45: one live PointNet run covers the form, the live curves, and the final metrics + confusion matrix, so the confusion matrix is shown only once (in the app).

## Before recording
- Delete `backend/reviews.db` (or keep old reviews out of frame).
- uvicorn + `npm run dev` running; page on **Explore**, PointNet test tables listed.
- Pick the "correct object" for Classify ahead of time (a chair or toilet, recall ≥ 0.94).

## 0:00–0:15 · Setup
- ModelNet10: 4,899 CAD meshes, 10 classes. Classification first because it builds the whole pipeline shape completion will need: data → preprocessing → model → backend → viewer.

## 0:15–0:35 · Start a training run (Train view; shows Interactive Training)
- Click **Train**. "Every setting can be changed; each default says where it comes from: batch 32 and dropout from the PointNet paper." Tick the lr schedule ("the paper halves the learning rate every 20 epochs"), set it to **10** so a halving shows within the run. **Start**. The run trains in its own process on the GPU, about 45 s; the reported weights are never touched.
- Switch to **Explore** while it trains.

## 0:35–1:10 · The model and why (question 1), over the Explore view
- Preprocessing: normalize to the unit sphere, sample 1,024 points by surface area.
- Part 1 MLP: 16³ voxel occupancy → 256 → 10; same features as logistic regression, so the hidden layer is the only change.
- **Selected: PointNet** (vanilla). Point clouds have no order; PointNet applies one MLP to every point, then a max over points, so it's order-invariant by construction. Its per-point features are what segmentation and shape completion reuse.
- Honest framing: it changes both the representation and the architecture, so the comparison doesn't say which caused the gain.

## 1:10–1:50 · Explore demo (question 3, part 1)
- Click a table; mesh renders; rotate. Classify a correct object: both models' probabilities side by side.
- Show → "Any model wrong": 48 of 100 test tables (PointNet 29, MLP 42, 23 both).
- table_0398: both say desk, 80% / 91%. Rotate: is it a table? Confident is not calibrated.
- Label review: mark it "ambiguous"; the row records "saw prediction". Errors become a review queue for labels; nothing feeds back into training.

## 1:50–2:45 · Training Monitor + performance (questions 2 and 3)
- Click **Train**: the run's curves (done or nearly). Point at the "lr ½" marker and the "best" marker (the epoch whose weights are kept).
- Performance (reported level-1 numbers): majority 11.0% → logistic regression 87.0% → MLP 87.0% (tie; it overfits: train loss → 0.01, val rises after epoch 12) → **PointNet 90.3%**. One standard error ≈ 1.1 points on 908 objects; the 3.3-point gain is about 2–3 SE, one seed.
- On the finished run: train/val/test table, test labeled "not for choosing" (runs are compared on val). Confusion matrix: table → desk is most of table's errors (recall 0.71 at level 1); dresser ↔ night stand, boxes with drawers that differ mainly in size, which unit-sphere normalization removes on purpose.
- Note: the live run's numbers will differ a little from level 1 because of the schedule; say "this run" vs. "the reported model".

## 2:45–3:00 · Next
- HW3: segmentation, reusing PointNet's per-point features. Long term: shape completion.

## Notes
- If the run isn't done at 1:50, show the curves still growing (that is the "real-time" part) and read the reported numbers from the README table instead of the run's final table.
- Cut first if long: the Classify-on-a-correct-object step, then the label review.
- Rehearse explaining the training loop and PointNet's max-pool without notes.