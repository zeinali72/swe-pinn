# Importance Sampling: Dynamic Pool Deviation from Wu et al. (2021)

## Reference

Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2021).
*A comprehensive study of non-adaptive and residual-based adaptive sampling for
physics-informed neural networks.*
arXiv:2104.12325.

## Original Deviation (now fixed)

The initial implementation maintained a **static pool**: 2 M candidate PDE
collocation points were generated once at startup (via LHS) and never
replaced. Every `resample_freq` epochs only the *active subset* (~100 K points)
was redrawn from this fixed pool using residual-proportional probabilities.

### Why this deviated from the paper

Algorithm 2 in Wu et al. implies the candidate set is refreshed at each
adaptive step so that the sampling distribution can cover regions of the domain
that develop high residuals *after* the initial draw. A fixed pool cannot adapt
to residual concentrations in areas that were sparsely represented in the
original LHS sample, capping the benefit of importance sampling.

## Fix Applied

Both `experiments/experiment_1/train_imp_samp.py` and
`experiments/experiment_8/train_imp_samp.py` now **resample the full pool from
the domain at every IS update step** before evaluating residuals. The active
subset is then drawn from the freshly sampled pool as before.

```
Each IS update (every resample_freq epochs):
  1. Resample 2M pool points from the full domain (LHS / domain sampler)
  2. Evaluate PDE residuals on the new pool with current params
  3. Compute sampling probabilities: p_j = α·(r_j/Σr) + (1-α)/N
  4. Draw n_pde active points proportional to p_j
  5. Compute importance-correction weights w_i = 1/(N·p_i), normalised to mean=1
```

## Cost

The only additional cost is the pool generation itself. For experiment_1 this
is a cheap LHS call on GPU. For experiment_8 the CPU-side chunked generation
adds a small overhead proportional to `pool_size / chunk_size` iterations, each
involving a `domain_sampler.sample_interior` call and a numpy array copy.
Residual evaluation (the dominant cost) is unchanged.
