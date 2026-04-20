# Hard-Constraint PINN — Session Notes and Resume Plan

> Snapshot of decisions and open items from the hard-constraint exploration
> session. Written as a handoff doc for tomorrow, not a finished chapter.
> Cross-references: `docs/l2_loss_experiment.md`,
> `docs/experiment_1_ablation_design.md`.

---

## 1. What we built today

### 1.1 L2-norm loss (committed)

- `experiments/experiment_1/train_nondim_l2.py` + matching config.
- Per-term loss is now `sqrt(mean*N + eps)` = true L2 norm over the
  residual vector (not RMSE). Motivation: MSE gradients go quadratically
  flat near zero; L2 keeps them linear and breaks the epoch-~270 plateau.
- Initial result under L2 + 10k points: NSE ≈ 0.76.
- Initial result under L2 + **100k** points: NSE ≈ 0.98. **Confounded**
  with the point-count change — see the decoupling run in the ablation.
- Documented in `docs/l2_loss_experiment.md`.

### 1.2 Hard-constraint variant (committed pending re-test)

- `experiments/experiment_1/train_nondim_l2_hard.py` — wraps the base MLP
  in `HardConstrainedMLP` (Flax module) and replaces `compute_losses`
  with a 3-term variant (`pde`, `neg_h`, right-wall Neumann only).
- `configs/experiment_1/experiment_1_nondim_l2_hard.yaml` — defaults to
  **PDE-only** (`ic=0`, `bc=0`, `neg_h=0`).
- Ansatz in non-dim coordinates:
  ```
  g_h  = (1 - x*/Lx*) * h_left_nd(t*)
  g_hu = (1 - x*/Lx*) * hu_left_nd(t*)
  phi_tx = (t*/Tf*) * (x*/Lx*)
  phi_y  = 4 * (y*/Ly*) * (1 - y*/Ly*)

  h_out  = g_h  + phi_tx * N_h
  hu_out = g_hu + phi_tx * N_hu
  hv_out = phi_tx * phi_y * N_hv
  ```
- **Hard-enforced**: IC (`U=0 at t=0`), left Dirichlet (`h_left(t)` from
  Hunter), top/bottom walls (`hv=0`).
- **Still soft**: right-wall Neumann outflow (gradient BC, can't be
  ansatz-hardened without coordinate reflection tricks). Can be turned
  off via `bc_weight=0` for pure PDE-only.

### 1.3 Hunter singularity fix (committed)

- Observed NaN in training around epoch ~120. Root cause: Hunter solution
  `h_left(t) ∝ t^(3/7)` → `d h_left/dt ∝ t^(-4/7) → ∞` at `t=0`, which
  leaks into the PDE residual via the ansatz.
- Fix: new config knob `sampling.t_pde_min_dim` (default **1.0 s**) that
  shifts the PDE sampling range to `t* ∈ [t_min, Tf*]`. IC coverage is
  unaffected because the ansatz is already exact at `t=0`.
- **This fix is Exp-1-specific.** Exp 8 and others without analytical
  singularities should set `t_pde_min_dim = 0` and sample the full time
  range.

---

## 2. What we decided about the method

### 2.1 Hard constraints: what the network actually learns

The constraint is **structurally deterministic**, not learned:

- At `x*=0`, the ansatz multiplier `phi_tx = 0` identically, so the
  network's contribution is annihilated at that surface no matter what
  its weights are. The BC value is delivered by `g`, not by `N`.
- **The network cannot violate the constraint, and it cannot learn it
  either** — there is no gradient signal from constraint-surface points
  (they're multiplied by zero).
- What the network *does* learn is `N` at interior points where
  `phi > 0`. It implicitly learns to "approach" the prescribed
  boundary value smoothly, but the boundary value itself is hand-coded.

Clean academic framing to use in the write-up:

> *"Under soft constraints, IC/BC satisfaction is a learning problem
> competing with the PDE objective. Under hard constraints, it becomes
> a structural guarantee of the function class — the network's
> hypothesis space is restricted to functions that already satisfy the
> constraints. Optimization then operates entirely within the
> constrained function space, and the PDE residual is the only signal
> needed to select a solution within it."*

### 2.2 PDE sampling is **not** removed at the constraint surfaces

An important clarification for the thesis: when we hard-constrain IC/BC,
we do **not** remove `t=0`, `x=0`, `x=Lx`, `y=0`, `y=Ly` from the PDE
sampling domain. The network still takes those points as inputs and
the PDE residual is still enforced on them (except where the analytical
solution itself is singular — see Exp 1's `t_pde_min`). The only thing
that changes is the loss: `L_IC` and `L_BC` are dropped because the
ansatz makes them identically zero.

The `t_pde_min` offset in Exp 1 is a **problem-specific** workaround for
Hunter's non-smooth IC, not a structural property of hard constraints.

### 2.3 Ansatz math for reference

| Symbol      | Meaning                                    |
|-------------|---------------------------------------------|
| `g(x,y,t)`  | Particular solution satisfying BCs and IC exactly on their surfaces. Hand-coded. |
| `phi(x,y,t)`| Distance-to-constraint function; vanishes on constraint surfaces, positive elsewhere. Hand-coded or SDF-based. |
| `N(x,y,t)`  | Raw neural network output. Free parameters. |
| `U_out`     | `= g + phi * N`. Guaranteed to match `g` on constraint surfaces, free to differ from `g` in the interior. |

The full SWE solution family representable by this ansatz is
`{g + phi * N : N is any MLP output}`. It is a strictly smaller set
than the unconstrained MLP family, but it is still dense enough to
represent the true solution (given a reasonable `g`).

---

## 3. Scaling to other experiments — feasibility table

| Exp | IC       | Outer BC              | Interior obstacles | Feasibility         | Action |
|-----|----------|-----------------------|--------------------|---------------------|--------|
| 1   | trivial  | axis-aligned, analyt. | none               | **Done**            | Verify with ablation A0–A6 |
| 2   | trivial  | axis-aligned          | axis-aligned bldg  | Medium (SDF at corners non-smooth) | Try hard BC on outer, soft on building |
| 4   | non-zero (9.7 m) | axis-aligned  | none               | Medium (non-zero `g_IC`) | Write `g_IC(x,y)` for filled initial state |
| 6   | trivial  | axis-aligned          | none               | High                | Same recipe as Exp 1 |
| 7   | trivial  | axis-aligned          | none (terrain)     | High                | Same recipe as Exp 1 |
| 8   | trivial (assumed dry-bed) | axis-aligned | none (terrain is source term) | **High** | **Candidate for next test.** Same recipe as Exp 1. No `t_pde_min` needed unless analytical gives another singularity. |
| 9   | trivial  | axis-aligned          | none               | High                | Same recipe as Exp 1 |
| 10  | trivial  | **irregular outer**   | none               | Medium              | Need SDF-based `phi` for outer boundary (Tier 2, see §4) |
| 11  | inflow hydrograph | irregular outer | **irregular buildings** | Low (hybrid only)   | Learned SDF + partition-of-unity `g` (Tier 3, see §4) |

**Big takeaway:** for Exp 1, 2 (outer), 4, 6, 7, 8, 9 the method drops
from 5 loss terms (pde, ic, bc_left, bc_right, bc_walls) to **1–2**
(pde, optional Neumann). That is a massive structural simplification
and is a result on its own, independent of final NSE numbers.

---

## 4. Scaling `phi` and `g` to complex boundaries

### Tier 1 — axis-aligned (Exp 1, 2 outer, 4, 6, 7, 8, 9)
- **`phi`**: closed-form product of coordinate-distance factors.
- **`g`**: closed-form analytical expression.
- **Tools needed**: none beyond what we have today.

### Tier 2 — irregular static outer boundary (Exp 10)
- **`phi`**: triangle-mesh signed distance function (SDF). Our existing
  triangle-based sampling machinery already supports the per-point
  distance query needed for SDF.
- **Smoothness issue**: SDF has a *ridge* along the medial axis
  (non-smooth where multiple boundary edges are equidistant). Second
  derivatives in the SWE residual would contaminate points on the ridge.
  **Fix**: soft-min smoothing
  `phi ≈ -tau * log(sum(exp(-dist_e / tau)))`
  with `tau ≈ 1%` of the domain characteristic length. Smooth everywhere,
  exactly zero on the boundary in the limit.
- **`g`**: closed-form if the outer BC is a single condition (e.g.
  uniform inflow). Otherwise use a partition-of-unity blend.
- **Tools needed**: adapter in the sampling code to expose the
  distance-to-nearest-edge query as a differentiable function; a
  soft-min wrapper; nothing new for `g` in simple cases.

### Tier 3 — irregular + interior obstacles (Exp 11)
- **`phi`**: learned SDF network, pretrained offline on the triangle
  mesh and **frozen** at physics-training time. One forward pass
  replaces the triangle query at each PDE sample.
- **Trade-off**: constraint is no longer bit-exact, it is satisfied to
  within the SDF network's approximation error. Must report
  `max |U - prescribed_BC|` on the boundary as a diagnostic.
- **`g`**: probably needs a learned boundary-extension network too, or
  a hand-written partition-of-unity for the distinct boundary segments
  (inflow / outflow / wall).
- **Tools needed**: auxiliary SDF training script, boundary-extension
  module, constraint-violation metric.
- **Reality check**: this is a substantial chunk of infrastructure. For
  Exp 11 it is probably better to keep building BCs as soft losses,
  hard-constrain only IC and the simplest outer pieces, and use the SDF
  as a *loss-weighting* function (upweight samples near buildings)
  rather than as an ansatz multiplier. That is a hybrid and should be
  explicitly labeled as such in the write-up.

---

## 5. Open questions to resolve tomorrow

1. **Re-run the hard-constraint Exp 1 training** with the `t_pde_min`
   fix in place, full 5000 epochs, and check whether NSE surpasses the
   0.7084 observed before the NaN. Hypothesis: it should reach or
   exceed the L2 variant (0.76–0.98 depending on point count) because
   the loss landscape is simpler.
2. **Decide if A2 in the ablation design (pure L2 without eps) should
   be run before or after the hard-constraint variant.** The ablation
   chain is still the right academic framing; the hard-constraint work
   is either a separate chapter or an appendix ablation.
3. **Confirm Exp 8's IC is actually dry-bed** (I assumed so today, but
   EA Test 5 specs should be double-checked before writing the `g_IC`).
   If the two depressions are pre-filled, `g_IC` needs to be a sum of
   smooth bump functions over the depression footprints.
4. **Verify the hard-constraint NSE is not a hollow improvement.** If
   the network simply learns `N ≈ 0` everywhere, the ansatz would
   produce `U = g`, and `g` alone is a trivial linear interpolation
   between BCs — *not* the true PDE solution. Diagnostic: check the
   norm of `N_h` on a validation grid. If it is close to zero, the
   network is cheating and the PDE residual loss is misleading.
5. **Right-wall Neumann: soft or hard?** Currently soft. Two options:
   (a) leave it soft and accept the 2-term objective, (b) hard-constrain
   via a reflection trick: feed the network `x_eff = sin(pi * x* / (2
   Lx*))` whose `d/dx* = 0` at `x* = Lx*`, making every network output
   automatically Neumann-compliant at the right wall. Worth trying
   after the basic variant is stable.

---

## 6. Files touched this session

Committed:
- `configs/experiment_1/experiment_1_nondim.yaml` — HP tuning
- `configs/experiment_1/experiment_1_nondim_cosine.yaml`
- `configs/experiment_1/experiment_1_nondim_l2.yaml`
- `experiments/experiment_1/train_nondim_cosine.py`
- `experiments/experiment_1/train_nondim_l2.py`
- `docs/l2_loss_experiment.md`
- `docs/experiment_1_ablation_design.md`

Uncommitted:
- `experiments/experiment_1/train_nondim_l2_hard.py` — hard-constraint
  variant with `HardConstrainedMLP` and the `t_pde_min` fix.
- `configs/experiment_1/experiment_1_nondim_l2_hard.yaml` — PDE-only
  defaults.
- `docs/hard_constraint_session_notes.md` — this file.

---

## 7. Quick-start commands for tomorrow

Soft L2 baseline:
```bash
WANDB_MODE=disabled python -m experiments.experiment_1.train_nondim_l2 \
    --config configs/experiment_1/experiment_1_nondim_l2.yaml
```

Hard-constraint PDE-only:
```bash
WANDB_MODE=disabled python -m experiments.experiment_1.train_nondim_l2_hard \
    --config configs/experiment_1/experiment_1_nondim_l2_hard.yaml
```

Diagnostic for question 4 in §5 (check `N_h` norm on the validation
grid) — to be added to `validation_fn` in `train_nondim_l2_hard.py`.
