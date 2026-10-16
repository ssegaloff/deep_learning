# HW2 decisions, part 3: Interactive Training + Training Monitor (optional challenges)

Continues `claude/hw2-decisions.md` (models, training, results, frontend through section 5) and `claude/hw2-decisions-viewer.md` (HW1-parity features, sections 6–9). Same labels: **verified**, **general knowledge**, **guess**. Started Oct 16, 2026; updated Oct 16 ~15:35 (sections 10c, 11, 12, 13).

## What the rubric asks (verified, Homework 2.pdf §5)
- **Interactive Training (+10):** the user can change one or more of: learning rate, number of epochs, hidden units, hidden layers, activation function, or other hyperparameters.
- **Training Monitor (+10):** a GUI showing "(real-time)" train and validation loss, accuracy and other metrics on train, val, and test, and prediction results.
- HW2 does not ask for a better model; §7 says keep the model simple. "Trying to improve on level 1" is experimentation, not a requirement.
- Whether "(real-time)" requires push updates is unclear; Sabine may ask the TA. Working answer: 1-second polling (an epoch takes longer than that).

## Decisions (Sabine, Oct 15 18:09 – Oct 16 11:50)
- **Order:** HW1 parity first (done), then Interactive Training, then the Training Monitor. Monitor-only for finished runs first; loading a finished run into Classify as a third column is a **later option**.
- **Navigation: hash routes** (option b): `#/explore`, `#/train` (view names not final). Reload keeps the view; back button works; no dependency. Rejected: tabs in state only (reload and back lose the view); React Router (a dependency for two views).
  - **Camera reset on switching views is acceptable** (Sabine): Explore may unmount. Expected to be safe because a tab click is an ordinary unmount from an event, not an unmount during a render like the section-3 bug (guess, to confirm in the browser). **Superseded in section 12:** the guess was wrong; Explore is now hidden, not unmounted, and the camera angle survives.
  - Polling state for a running job must live in `App` (or the Train view kept mounted), so switching views mid-run doesn't lose it. **Section 13 chose the second: `TrainView` kept mounted, state inside it.**
- **Both models are trainable.** Each form is prefilled with that model's level-1 config, labeled by source: PointNet's batch 32 / dropout 0.3 / lr 1e-3 are the paper's (Appendix C); the MLP's hidden 256 / dropout 0.5 / batch 64 are conventional, not cited.
  - **VoxelMLP knobs:** lr, max epochs, patience, hidden layer sizes (units and number of layers), activation (relu / tanh / gelu / sigmoid), dropout, weight decay, batch size, seed.
  - **PointNet knobs:** lr, max epochs, patience, batch size, dropout, weight decay, seed. Architecture fixed (T-Nets, augmentation are code-level experiments, deferred). **Section 13 added an lr schedule knob (`lr_step`) for both models.**
- **Training runs in a subprocess running `train.py`, not a thread in the server** (Sabine, 18:15). Why: real cancel (`terminate()`; the OS frees GPU memory when the process exits; Python can't kill a thread, general knowledge); a crash or CUDA OOM can't take the server down; the server stays CPU-only (a training thread would create a CUDA context held until the server exits, general knowledge); each UI run is a literal command line that can be rerun in a terminal; `fit` stays in the script (open item on code layout stays deferred). Costs: CLI flags, a progress file, startup per run (seconds), and orphan handling if uvicorn restarts mid-run.
- **Progress reaches the page by polling `GET /runs/{id}` about once a second** (Sabine). Alternatives: Server-Sent Events / WebSocket push (a streaming endpoint; behavior through the Vite proxy unchecked). Per-batch updates rejected (noisy; the rubric names per-epoch metrics).
- **Outputs: `runs/<run_id>/`, gitignored** (in `.gitignore` since `8b5b02d`). `backend/model/` and `results/` are never written by the UI. **One run at a time** (a second start → 409). This replaces the old "run-name argument" open item.
- **Test numbers:** the rubric requires them, so a finished run shows them, labeled "not for choosing"; runs are compared on val; the README's level-1 numbers remain the reported result (tuning by watching test would make test no longer an unbiased estimate). The backend passes the results JSON through unchanged; labeling and placement are section 14's job.

## Measured timings (verified on Legion, Oct 15 ~18:22; seed 0, default settings)
| | Wall time | Epochs | Per epoch (rough; wall time includes a few seconds of startup) |
|---|---|---|---|
| VoxelMLP | 6.8 s | 27 | under 0.25 s |
| PointNet | 44.8 s | 33 | about 1.2 s |
- Both reproduced their level-1 best epochs (12, 18) and test scores (0.870, 0.903) exactly: seeding still holds.
- Consequence: both models are practical to train interactively. The MLP finishes faster than the polling interval (its curve appears in jumps); PointNet's ~1.2 s/epoch suits the video's "real-time" demo.
- Measured epoch time per run is now recorded (`"seconds"` per epoch in `progress.jsonl`; 0.31 s for epoch 1 of a 2.2M-parameter MLP, verified Oct 16).

## Section 10a: `VoxelMLP` generalized (commit `2f89903`, verified Oct 15 ~18:28)
- `hidden: int | list[int]`; an int is normalized to `[int]`. Reason: the committed level-1 `mlp.pt` stores `hidden=256`, and re-saving it just to match new code would rewrite a committed result. Cost: two input forms.
- `activation` is a string looked up in `ACTIVATIONS = {"relu", "tanh", "gelu", "sigmoid"}` (Sabine asked to add sigmoid: its gradient is at most 0.25, from σ' = σ(1 − σ), so deeper stacks learn slowly; AdamW's per-weight step partly offsets small gradients, general knowledge, so the slowdown may be milder than under SGD). A string, not a class, because `torch.load`'s default `weights_only=True` only unpickles tensors and plain containers (general knowledge), and the backend relies on that default.
- Dropout after every hidden layer, one rate. Zero hidden layers → `ValueError` (that would be logistic regression, which HW2 §2.1 rules out). Unknown activation → `ValueError`.
- One hidden layer still builds Linear → act → Dropout → Linear, so the state_dict keys stay `net.0.*`, `net.3.*`.
- PyTorch's default init is tuned for ReLU-family activations, so tanh/sigmoid runs start somewhat off-scale (general knowledge); one possible reason they train differently.
- Verified: level-1 `mlp.pt` loads strictly through the backend's own `load_model` (1,051,402 params); `[512, 128]` + gelu gives keys `net.0`, `net.3`, `net.6`; `[]`, `[0]` rejected; `256.0` raises a bare `TypeError` ('float' object is not iterable) — loud but unhelpful, so **the backend must validate integer sizes before building a command** (done in section 11).

## Section 10b: `train.py` flags, run directory, progress file (commit `8b5b02d`, verified Oct 16 ~11:47)
- **New module `src/shapes/defaults.py`** (`TRAINING`, `MODEL_DEFAULTS`): one source of truth for the CLI defaults and the backend's request defaults (section 11). Copying numbers into the backend/frontend would drift without error. A narrow slice of the deferred code-layout item (constants only).
- **Flags:** `--lr` (finite > 0), `--weight-decay` (finite ≥ 0), `--max-epochs`, `--patience`, `--batch-size` (ints ≥ 1), `--dropout` ([0, 1)), `--hidden N [N ...]`, `--activation`, `--seed`, `--out-dir`; `--weights-dir` / `--results-dir` kept for the CLI. Validation is in argparse `type=` functions, so bad values fail before data loads.
- `--hidden` / `--activation` with PointNet → usage error (silently ignoring would look applied). `--out-dir` with `--weights-dir`/`--results-dir` → error.
- **Model-specific defaults use `is None`, not `or`:** `args.dropout or 0.5` would turn a requested 0.0 into 0.5.
- **`--out-dir DIR`** puts `<model>.pt`, `<model>.json`, and `progress.jsonl` in DIR. An existing `progress.jsonl` → error (each run gets a fresh directory).
- **Folders are created before training** (Sabine chose this, Oct 16): an unwritable path fails in a second instead of after the run. Cost: a run that fails early leaves an empty folder (harmless; the backend makes a new folder per run). The later `progress_path.parent.mkdir` is now redundant but harmless.
- **`progress.jsonl`:** one JSON object per epoch (`epoch, train_loss, train_accuracy, val_loss, val_accuracy, val_macro_f1, val_balanced_accuracy, seconds`), written by a callback (`fit(on_epoch=...)`), opened and closed per write. The backend treats a last line without its newline as not yet written.
- **NaN guard in `fit`:** with a user-chosen lr the loss can diverge from epoch 1; NaN compares false, so no epoch becomes "best". If `best_state is None` at the end → `RuntimeError("validation loss was never finite ...")` instead of a confusing `load_state_dict(None)` crash.
- `config` in the checkpoint and results JSON now records the actual lr, weight decay, batch size, epochs, patience used.
- **A Claude mistake caught by the checks:** the instructions said to rename `args.weights_dir` in "three places"; there were two `weights_dir` and two `results_dir` lines. One was missed, and `--out-dir` crashed after training with `NoneType has no attribute mkdir`. Check 1 had passed only because it passed `--weights-dir` explicitly. Fixed; a grep confirmed no stale uses.
- **Verified Oct 16:** defaults reproduce level 1 (best epoch 12, test 0.870); `--out-dir` run writes all three files, 5 progress lines with `"seconds"`; default folders created from nothing (run in a scratch folder with `uv run --project`); `--out-dir /proc/nope` fails at the `mkdir` before data loads; four rejections give usage errors (pointnet + `--hidden`, `--lr 0`, `--dropout 1`, reused `--out-dir`).
- Plumbing-run numbers are not experiments: a 5-epoch sigmoid [512, 128] run (best epoch 5 = last, still improving, test 0.726) and a 2-epoch default run (test 0.841) say nothing about sigmoid or about the model.
- **Ruff:** Claude added it as an optional check without explaining it; Sabine hadn't heard of it. Explained (Python linter by Astral; rule prefixes E/W pycodestyle, F pyflakes, I isort, RUF ruff's own; `# noqa: CODE` skips a rule on one line). **Not used going forward** unless Sabine asks; she removed the unused `# noqa: E731` and sorted the imports by hand. Unexplained: ruff ran I and RUF rules though the repo has no ruff config (a user-level config on Legion is a guess; irrelevant now).

## Section 10c: seed range in `train.py` (commit `7bff94f`, verified Oct 16 ~12:08)
- `seed_value` argparse type: 0 ≤ seed < 2³². `np.random.seed` accepts only that range (general knowledge); before this, `--seed -1` crashed inside `set_seed`. Found while writing the backend's request models.
- **Where each check lives (Sabine agreed):** `train.py` rejects what would crash or be meaningless (`--lr 0`, `--dropout 1`, an out-of-range seed). The backend's caps are app policy: a command-line user may reasonably try 8,000 units, so caps don't go in `train.py`. numpy's limit is now written in both files; it's a fact about numpy, not a setting, so the copies won't drift.
- Verified: grep shows one `--seed` line, `type=seed_value`; `-1` and `4294967296` → usage errors; `4294967295` accepted.
- Offered, not done: a `train.py` check for PointNet batch sizes that leave a last batch of one object (it would need the train split's size, so it would sit after data loads). The backend's range prevents it for UI runs.

## Section 11: backend run routes (commits `ce2da75`, `02e6b49`, `429d8f6`, `64fd991`; verified Oct 16 ~12:27)

### Decisions (Sabine, Oct 16 ~11:59–12:21)
- **Run records on disk:** `runs/<id>/run.json` holds the request, the exact command, status, `created_at`, `ended_at`, `returncode`. Written atomically: a temporary file, then `os.replace` (atomic, so a reader or a crash sees the old file or the new one, never half; general knowledge).
  - Tradeoff vs. memory only: survives restarts (the Monitor can list past runs), keeps the exact command next to the outputs, and makes orphan marking possible. Costs: a startup scan and two records to keep in step (the live `Popen` handle and the file).
  - Middle option rejected: inferring status from which files exist can't tell failed, cancelled, and interrupted apart after a restart.
- **Orphans:** mark at startup; don't try to find or kill a surviving process. A stored PID may have been reused by an unrelated process (general knowledge). Cost: after a hard crash, a surviving `train.py` could hold the GPU while the app shows nothing running (`nvidia-smi` and kill it by hand).
- **Caps (Sabine):** hidden units ≤ 5,000; hidden layers ≤ 4; max epochs ≤ 500; patience ≤ 500; PointNet batch 2–512; MLP batch 1–4,096; seed 0 to 2³² − 1.
  - Memory (arithmetic): a 4 × 5,000 MLP has about 96M parameters; weights, gradients, and AdamW's two running averages ≈ 1.5 GB, well within 12 GB. Each such run saves a checkpoint of about 384 MB in `runs/`. Overfitting at that size is something to watch in the Monitor, not a reason to forbid it.
  - **PointNet batch floor 2:** batch norm in train mode errors on a batch of one object. The last batch holds 3,392 mod b objects, and only b = 1 and b = 3,391 (prime) leave 1 (arithmetic). The 512 ceiling also bounds activation memory (widest activation ≈ 2 GB at 512 vs. ≈ 128 MB at 32; the full backward pass at 512 is a guess, so 512 is a cautious ceiling, not a measured one).
  - **Two changes from the first proposal, accepted:** no "patience ≤ max epochs" rule (a larger patience just turns early stopping off, a valid choice); MLP batch cap 4,096 instead of 3,392 (anything ≥ 3,392 is already full-batch gradient descent, and a cap independent of the dataset size can't go stale).
- **A run that finished while nothing polled is `done`, not `interrupted`** (Sabine, ~12:21; see the bug below).

### `backend/runs.py` (new; `main.py` imports from it, never the reverse)
- **Request models:** `MlpRun`, `PointNetRun`, a union discriminated on `model` (`RunRequest`), `REQUEST_MODELS`.
  - `strict=True`: `256.0`, `"256"`, `true` are not accepted as integers (verified in the sandbox; and on Legion over HTTP, `[256.0]` → 422 at `body.mlp.hidden.0`). A JSON integer is still accepted for a float field (`"lr": 1` → 1.0).
  - `extra="forbid"`: `hidden` sent with `pointnet` → 422 ("Extra inputs are not permitted"), the same reasoning as the argparse error.
  - `allow_inf_nan=False`; ranges mirror argparse (lr > 0, weight decay ≥ 0, dropout in [0, 1)).
  - Defaults are read from `shapes.defaults`, so an empty request `{"model": "pointnet"}` is the full level-1 config. `default_factory` gives each request its own `hidden` list.
  - `activation` is checked against `ACTIVATIONS` in a validator, so a new activation in `model.py` needs no second list.
- **`build_command(req, out_dir)`:** an argument list (never a shell string, so no injection), **every flag spelled out** (the saved command reproduces the run even if `shapes.defaults` changes later), `sys.executable` (the server's own Python and venv), absolute `TRAIN_SCRIPT`.
- **`RunManager`:**
  - A `threading.Lock` around check-then-start: FastAPI runs plain `def` routes in a thread pool (general knowledge), so two `POST /runs` could otherwise both see "nothing running."
  - **No background watcher:** `_refresh()` calls `poll()` at the start of `start`, `get`, `list`, `cancel`, and `shutdown`. `poll()` also reaps the exited child (no zombie; general knowledge). With 1-second polling, status is at most a second stale.
  - Ids are server-made, `YYYYMMDDTHHMMSSZ-<model>`, and checked against `RUN_ID` (a regex) before any id becomes a path, so `../x` can't escape `runs/` (path traversal; general knowledge). Ids sort chronologically. `mkdir(exist_ok=False)` → 409 "a run started this second" on a same-second collision (cancel then restart).
  - `run.json` is written before `Popen`, so even a run that fails to launch has a record (`OSError` → `failed`, 500).
  - stdout + stderr go to `runs/<id>/log.txt`, not `PIPE`: an unread pipe fills (about 64 KiB on Linux; general knowledge) and the child blocks on its next `print`.
  - `cwd` = repo root, because `train.py`'s `--points-dir` / `--meta` defaults are relative paths (verified, reading `train.py`).
  - Statuses: `running`; `done` (exit 0 **and** the results file exists); `failed` (with `log_tail`, the last 20 lines of `log.txt`); `cancelled` (a person chose to stop it); `interrupted` (the server went away).
  - `get` adds `progress` (complete lines of `progress.jsonl` only) and, when done, `results` (the run's `<model>.json`). Files are read outside the lock (safe: `progress.jsonl` only grows; `run.json` is swapped atomically).
  - Cancel: `terminate()` (SIGTERM; Python's default handling exits at once, general knowledge), wait 10 s, then `kill()`. Cancelling a run that isn't active → 409.
  - `shutdown()` (lifespan, after `yield`): `_refresh()`, then stops any active run and marks it `interrupted`.
  - `mark_interrupted()` (startup): a record still `running` becomes `done` if its results file exists (`train.py` writes it last; return code unknown, `null`), else `interrupted`.
  - `HTTPException` is raised directly in `runs.py`. Alternative: the manager raises its own errors and `main.py` maps them to HTTP codes (keeps the manager FastAPI-free; more code for a backend-only module).

### `backend/main.py`
- Startup check `REQUEST_MODELS.keys() == MODELS.keys()` (same pattern as `PREPARE`): a registered model without a request class fails at startup.
- `app.state.runs = RunManager(ROOT / "runs")`, `mark_interrupted()` before `yield`; `shutdown()` after `yield`.
- Routes: `GET /runs/defaults` (each model's full request with defaults, built as `cls(model=arch).model_dump()`, so the form's prefill is exactly what an empty request runs; **declared before `/runs/{run_id}`**, because routes match in declaration order and `"defaults"` would otherwise be read as an id); `POST /runs` (201; 409 if active; 422 on bad input); `GET /runs` (newest first); `GET /runs/{id}` (404 for unknown or malformed ids); `POST /runs/{id}/cancel`.

### Verified (Legion, Oct 16 ~12:06–12:27)
- Request models: both level-1 commands built; 8 bad requests rejected, each naming its field.
- Manager called directly: second start → 409; 2-epoch MLP `done`, test 0.841 (= section 10b's 2-epoch CLI run); PointNet cancelled after 5 s → return code −15; `../backend` and `nope` → 404; folders hold `log.txt`, `progress.jsonl`, `run.json` (plus `mlp.json`, `mlp.pt` when done).
- Over HTTP: defaults route; `[256.0]` → 422; 201 on start; 409 naming the active run.
- **A full default PointNet run started with `POST /runs` reproduced level 1: best epoch 18, test 0.9031.** The subprocess path changes nothing about training.
- Ctrl-C mid-run → after restart the run reads `interrupted`, return code −15; `nvidia-smi` shows no GPU processes.
  - **Finding:** −15 (SIGTERM) means the child was still alive when the shutdown step stopped it, so Ctrl-C did not reach it. Claude had said the terminal's process-group signal would kill it; that didn't happen here. Why: unknown (guess: `uv run` or the VS Code terminal delivers SIGINT to uvicorn only). The shutdown step is what covers this.
- **A Claude bug, caught by Sabine's question:** `shutdown()` didn't `_refresh()` first, so a run that finished while nothing polled was recorded `interrupted` when the server stopped (Claude had first described this as only a startup issue). Evidence: `20261016T162432Z-mlp interrupted 0` (exit 0, mislabeled; Sabine may delete the folder). Fixed in `64fd991`; verified: `20261016T162657Z-mlp done 0`.
- **Not tested:** the cancel route over HTTP (it only calls the tested `cancel`); the startup branch that turns a leftover `running` record with results into `done` (needs uvicorn killed without a clean shutdown, e.g. `kill -9`).
- Cosmetic: the pushed `runs.py` ends with an extra blank line.

## Section 12: Explore / Train navigation (commit pending at time of writing; verified Oct 16 ~14:00)

### Decisions (Sabine, Oct 16 ~13:11)
- **Explore's state stays in `App`** (selection, filter, label cache, Classify result). Rejected: moving it into an `ExploreView` component (cleaner `App`, but the selection is lost on every switch). Cost: `App` keeps growing; extraction later is a mechanical move. Run polling state will live in `App` too.
- **The view is read from the URL hash with `useSyncExternalStore`** (React's hook for state that lives outside React, general knowledge), in `src/route.js` (`VIEWS`, `useView`). The URL is the only copy, so nothing can drift. Rejected: `useState` + a `hashchange` listener in an effect (two copies of one fact).
  - `currentView` returns a string, so equal snapshots compare equal under `Object.is` and there's no re-render loop (general knowledge).
  - `Object.hasOwn(VIEWS, name)`, not `name in VIEWS`: `'toString' in VIEWS` is true through the prototype. Verified in the browser: `#/toString` and `#/nope` show Explore.
  - An unknown hash shows Explore without rewriting the URL.
- **Plain `<a href="#/train">` links, no click handlers:** the browser updates the hash, adds a history entry and fires `hashchange`, so back works for free.
- **A top bar across both views** (`NavBar.jsx`) holds the page's one `<h1>` (moved out of the sidebar) and the links; the active link is styled off `aria-current="page"`. Cost: about 40 px of viewer height.
- **`/objects` loading and error messages render inside the Explore branch only**, so Train works even if `/objects` fails (Claude flagged; Sabine agreed).
- `TrainView.jsx` is a placeholder until section 13.

### Explore is hidden, not unmounted (reverses "Explore may unmount")
- **Claude's guess was wrong:** the decision above (part 3, "Decisions") expected unmounting Explore to be safe because a link click unmounts from an event, not during a render. The first version did exactly that, and leaving Explore logged "Attempted to synchronously unmount a root while React was already rendering" plus a `removeChild` NotFoundError, the section 3 symptoms. Claude should have caught this from section 3's own history: its second bug was also triggered from an event handler (clearing the selection swapped out `MeshViewer`). The lesson from section 3 is "never unmount the Canvas", not "never unmount it during a render."
- **Fix:** Explore is always rendered inside `<div hidden={view !== 'explore'}>`; Train mounts only when open (it has no Canvas).
  - The `hidden` attribute is the browser stylesheet's `display: none`, which any author `display` rule overrides. `hidden` on `.app` (`display: grid`) would do nothing, so the attribute sits on a plain wrapper with no `display` rule.
  - Gain: the Canvas is created once per page load, and **the camera angle now survives a view switch** (the opposite of the reset Sabine had accepted). Costs: the WebGL context stays alive on Train (a few MB); R3F keeps drawing frames into a 0×0 canvas (the browser doesn't pause `requestAnimationFrame` for hidden elements, only background tabs; general knowledge). No warnings about a zero-size canvas appeared, and resizing the window on Train then returning gave correct proportions (verified).

### The `removeChild` error on first selection: a section 3 bug, not section 12
- **Symptom:** on a fresh load, "Attempted to synchronously unmount a root…" with nothing clicked; then `removeChild` NotFoundError (stack: `removeChildFromContainer` → `commitDeletionEffectsOnFiber` → `commitRoot`) when the first object is selected, pointing at `MeshViewer.jsx:40`, the "Pick an object." hint (a drei `Html` overlay inside the Canvas). The app kept working.
- **Not caused by today's changes (verified):** with local changes stashed, the pushed code showed both messages. `MeshViewer.jsx`, `ErrorBoundary.jsx`, `main.jsx`, `package.json` and `package-lock.json` were unchanged since `10a8fe9` apart from one moved comment (git log). So **section 3's "console clean" result was wrong for this path**; whether the error scrolled by or wasn't watched at that moment is unknown.
- **Mechanism.** Two independent readings agreed: Claude read drei's `src/web/Html.tsx` at main `bf6f4ad` (Sept 30, 2026), and Sabine's school Claude account read the drei 10.7.9 npm tarball and react-dom 19.3.0 without seeing Claude's hypothesis.
  - **Verified (source):** each `Html` creates its own React root (`ReactDOM.createRoot(el)`) in a layout effect, on an element created once; its cleanup calls `target.removeChild(el)` then `currentRoot.unmount()`.
  - **Verified (the other account's reading of R3F 9.8.2):** R3F forwards `StrictMode` into the Canvas, so StrictMode's dev-only mount → cleanup → mount of layout effects reaches `Html`. Roots get created twice on one element, and the cleanup's `unmount()` runs during R3F's commit, which is the call the "synchronously unmount" warning names.
  - **Verified (the other account's reading of react-dom):** `removeChildFromContainer` is a bare `container.removeChild(child)`, which throws if the node was already removed; an empty root's commit clears its container.
  - **Guess:** the exact order in which one root removes the other's node; why only the hint throws and not "Loading mesh…" (the other account's guess: the hint is the only `Html` mounted during Canvas startup, when `target` can change). The expected duplicate-`createRoot` warning didn't appear; Claude's guess is that `unmount()` clears the "has a root" mark before the second `createRoot` (React source not checked).
  - Corroborating, not identical: a pmndrs/react-three-a11y issue reports a similar `Html`-style root crashing under StrictMode (from the other account; not checked by Claude).
- **Fix:** the hint is plain DOM laid over the Canvas (`<p className="viewer-hint muted">` after `</Canvas>`; `.viewer { position: relative }`, `.viewer-hint` absolute, centered, `pointer-events: none` so orbit, zoom and pan pass through). No `Html` mounts at load. The Canvas still never unmounts, which was the point of putting the hint inside it in section 3.
  - Cost: the hint is the only status message outside the Canvas. The loading and error messages stay as `Html`, because they must live inside `Suspense` and the error boundary, which are inside the Canvas; they use the same root pattern and haven't thrown so far.
  - If one does: patch drei's `Html.js` to create its root once (`patch-package`; a patch to carry across upgrades). **Don't drop `StrictMode`:** it would hide this class of bug rather than fix it (Claude and the other account agree).
- **Verified (Legion, Oct 16 ~14:00):** lint clean; no unmount warning on load; select → Classify → Train → back (link and back button) → change class → select again: console shows only `THREE.Clock`; selection, Classify result and camera angle survive the switch; hint centered; orbit, zoom and pan work with it showing.

### Noted, not done
- **Recenter button** (Sabine): after panning, the mesh can drift off-center with no way back but a reload. A button resetting the camera and the OrbitControls target would fix it.
- DevTools Issues tab: "A form field element should have an id or name attribute" (4). Chrome uses these for autofill; fields wrapped in a `<label>` are still labeled for screen readers. Hygiene, not a bug.
- Claude's grep prediction for `MeshViewer.jsx` ("3 lines") missed two comments containing "Html"; the code was right. Predict grep counts against comments too.

## Section 13: Interactive Training form (verified Oct 16 ~14:44–15:31; commits pending at time of writing: `train.py` + `defaults.py`, then backend `lr_step` + sources + startup check, then frontend + `CHOICES`)

### Decisions (Sabine, Oct 16 ~14:09–14:37)
- **`TrainView` always mounted inside `<div hidden={view !== 'train'}>`, with all Train state inside it** (option a). **Deviation from the earlier "polling state lives in `App`", flagged and accepted.** Gain: same mechanism as Explore; form drafts and polling survive a view switch; `App` stops growing. Cost: `/runs/defaults` and `/runs` are fetched at page load even if Train is never opened.
- **No limits in the browser** (option a): the backend is the only authority; a 422 shows under the field it names. Rejected: `min`/`max` read from a JSON-schema route (no drift, but schema-walking code; the `hidden` list's limits sit inside `items`).
- **Text inputs, not `type="number"`** (Sabine, ~14:31). Number inputs report `""` for partial input like `1e-`, apply a `step` rule (default 1) that blocks 0.001 under native validation, and change value on scroll (general knowledge). Cost: no spinner arrows.
- **Hidden layers: one box per layer** with "+ layer" / "− layer" (option b); each layer gets its own 422 message (`loc` carries the index).
- **Section 13 includes a status line and Cancel** (option a); curves and metrics stay in 14.
- **Source labels served from `defaults.py`** (`SOURCES`), beside the values in `/runs/defaults`, plus a **startup check** (Sabine) that every request field except `model` has exactly one label. It catches a missing or orphaned label, not a wrong one.
- **lr schedule knob** (Sabine, ~14:37; scope addition she chose). See below.

### The PointNet paper's lr schedule (reread Oct 16)
- **Verified (PointNet App. C):** Adam, initial lr 0.001, momentum 0.9 (= Adam's β₁, PyTorch's default; general knowledge), batch 32, lr divided by 2 every 20 epochs, batch-norm decay 0.5 → 0.99, dropout keep 0.7, 3–6 hours on ModelNet with a GTX 1080. **No weight decay mentioned**, so PointNet's weight-decay label says only "PyTorch AdamW default".
- **Not the level-1 default** (the Oct 15 decision holds): our PointNet's best epoch was 18 and it stopped at 33, so the first halving (epoch 20) comes after the best epoch and early stopping likely ends the run before a second; the paper's hours imply far more epochs (count unknown). Copying the halving alone isn't the paper's recipe (no BN schedule, no augmentation, ModelNet10). Test has been seen, so any change is an experiment judged on val.
- **The case for trying it:** PointNet's val loss is jagged (0.14 to 0.48 between neighboring epochs), which the Oct 15 decision named as the trigger; a lower late lr reduces step noise (general knowledge). Whether it helps here is unknown.

### `train.py` and `defaults.py`
- `--lr-step N` (`positive_int`; absent = constant lr, level 1; `0` → usage error, not a hidden "off"). Factor fixed at `LR_FACTOR = 0.5` (the paper's); only the step is a knob (Claude's choice, flagged). Applies to both models (`fit` is shared); the MLP's label says "off at level 1", no paper citation.
- `StepLR` built only when the flag is given, so a level-1 run takes exactly the old code path. `scheduler.step()` once per epoch after training (PyTorch warns if the scheduler steps first; general knowledge).
- Each epoch record gets `"lr"`, read before the epoch trains (the rate its updates used); the CLI epoch line prints it; `config` records `lr_step`.
- `TRAINING["lr_step"] = None`; `SOURCES = {mlp: {...}, pointnet: {...}}` built from `_SHARED_SOURCES`.
- Whitespace-only line after the `mkdir`s fixed.
- **Verified (Legion):** grep 15 lines; no flag → best epoch 12, test 0.870, every epoch `lr 1.00e-03` (level 1 reproduced); `--lr-step 2 --max-epochs 5` → `1e-3, 1e-3, 5e-4, 5e-4, 2.5e-4`, epochs 1–2 identical to level 1 to the last digit, divergence from epoch 3; `--lr-step 0` → usage error. A paste lost a space (`halvesit`) in a label; caught by the grep, fixed before commit.

### Backend
- `_RunBase.lr_step: int | None = Field(None, ge=1, le=MAX_EPOCHS)`; `build_command` adds `--lr-step N` only when set.
  - **Deviation, flagged and accepted:** "every flag spelled out" can't hold for "off" (it's the flag's absence), so a saved command without `--lr-step` relies on `train.py`'s default staying off. The record's `request` keeps `lr_step: null` as ground truth. Closing the gap would need an explicit `--lr-step none`.
- `CHOICES` ClassVar on the request models (`MlpRun`: `{"activation": sorted(ACTIVATIONS)}`; others `{}`); pydantic doesn't treat a ClassVar as a field (verified in the sandbox: not in `model_fields`). Added so a new activation in `model.py` can't silently miss the menu (Claude's addition, flagged).
- `/runs/defaults` → `{arch: {"values", "sources", "choices"}}`. Sources and choices sit beside the values so the form can send `values` back unchanged (`extra="forbid"` would 422 on extra keys).
- Startup check in `lifespan`: `SOURCES[arch].keys() == cls.model_fields.keys() - {"model"}`, else `RuntimeError` naming missing and extra.
- Whitespace-only line in `mark_interrupted` fixed.
- **Verified (Legion):** defaults route shows values, sources, choices; `lr_step: 0` → 422 at `["body","mlp","lr_step"]`; a 3-epoch run with `lr_step: 2` → command has `--lr-step 2`, progress lr `[0.001, 0.001, 0.0005]`, done; commenting out the `seed` label → startup refuses with `SOURCES['mlp']: missing ['seed'], extra []`.

### Frontend
- `api.js`: `ApiError` (`status`, parsed `detail`; `.message` unchanged so Explore's error lines look the same); parsing tried, not assumed (the proxy's 502 is plain text); `postJson`; `getRunDefaults`, `startRun`, `listRuns`, `getRun`, `cancelRun`.
- `TrainView.jsx`:
  - Drafts are strings as typed, one per model. `parseNumber` treats empty and non-numeric as form errors (`Number("")` is 0). Non-integers in integer fields go to the backend, whose 422 names them.
  - Every field sent explicitly, including `lr_step: null`.
  - Client errors and 422s share keys (`lr`, `hidden.0`, `hidden`); an error without a field goes to the top. Errors clear on model switch.
  - `LABELS` / `ORDER` are layout only; a field the backend adds later still shows under its own name.
  - Schedule: checkbox, then "every N epochs", prefilled 20 (paper). "− layer" disabled at one layer (zero hidden layers would be logistic regression; a definition, not a cap). "Reset to defaults" button. PointNet selected first.
  - Polling: a `setTimeout` chain started after each response (no stacked requests), only while `running`; cleanup cancels (covers StrictMode). On mount and after a 409, `GET /runs` picks up a running job.
  - Every input has a `name` (autofill; DevTools Issues).
  - Start disabled while a run is active. Status line: model, id, status, epochs finished; failed runs say only "failed" (log tail is section 14).
  - Known wrinkle (guess, untested): a poll in flight as Cancel returns could show "running" for up to a second.
- CSS: `.train` fills its grid row (`height: 100%`) so the form scrolls inside it.
- **Verified (Legion, ~15:20–15:31):** lint clean; empty seed + `abc` lr → two field messages and no `POST /runs`; 5 layers → "at most 4 items"; `256.5` → "valid integer" under that box; PointNet + schedule → status advances once a second, Start disabled; Explore and back keeps status and the ticked box; reload picks the run up; Cancel → `cancelled` (**closes section 11's untested cancel-over-HTTP item**); a run started by curl → page's Start gets 409, message at top, the curl run's status line appears, Cancel stops it. Console: only `THREE.Clock` plus Chrome's own red log line for the 409 (Chrome logs every non-2xx fetch; our code catches it). Issues: the form-field notice dropped from 10 to 3; the train form's fields are off the list (the remaining 3 are probably Explore's menus; guess).

## Then
- **12, 13:** done (above).
- **14:** Monitor: live curves (train/val loss and accuracy per epoch; `lr` per epoch is now in progress, so halvings can be marked), then the finished run's train/val/test metrics, confusion matrix, per-class recall; test labeled "not for choosing"; failed runs show `log_tail`.
- **Later option:** load a finished run into Classify as a third column (runtime model loading needs an atomic swap of `state.models`; the Show filter's label cache and hard-coded "PointNet wrong" / "Voxel MLP wrong" would need work).