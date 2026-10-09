# S1 — Step size and stability

Q1 has c = 106, so H = diag(2, 212), λ_max = 212 and 2/λ_max = 9.434e-3. On a quadratic, GD multiplies each
gradient component (one per eigen-direction) by (1 − αλ_i) per step, so the decay of each mode is known in advance.

**Stability threshold.** GD on Q1 with α = t · 2/λ_max (`results/s1_stability.csv`, `results/s1_modes.csv`):

| t    | α         | converged | iters | ρ(α) = max\|1−αλ_i\| | measured decay, last 50 steps | predicted iters |
|------|-----------|-----------|-------|----------------------|-------------------------------|-----------------|
| 0.5  | 4.717e-3  | yes       | 1031  | 0.99057              | 0.99057                       | 1031            |
| 0.9  | 8.491e-3  | yes       | 571   | 0.98302              | 0.98302                       | 571             |
| 0.99 | 9.340e-3  | yes       | 684   | 0.98132              | 0.98000                       | 684             |
| 1.01 | 9.528e-3  | no        | —     | 1.02000              | 1.02000                       | —               |
| 1.1  | 1.038e-2  | no        | —     | 1.20000              | 1.20000                       | —               |

The boundary 2/λ_max is sharp: 1 % below it GD converges, 1 % above it ‖∇f‖ grows by exactly ×1.02 per step until
the blow-up guard stops it. The measured asymptotic decay equals |1 − αλ_i| of the dominant mode to five digits. (The
`decay_measured` column of `s1_stability.csv` is a mean over the first 200 steps; at t = 0.5 it is 0.9706 rather than
0.9906 because the first step removes the stiff component completely, ‖∇f‖ drops by 0.0174 in one step.)

**ρ is a worst-case rate, not an iteration count.** At t = 0.99, ρ is the smallest of the five, yet GD needs more
iterations than at t = 0.9 (684 vs 571). At x⁰ = (1.3, 0.7) the gradient is (2.6, 148.4): the stiff component is
57 times larger, and at t = 0.99 it shrinks by a factor 0.98 with alternating sign, almost as slowly as the flat
one, so it is the last to reach the tolerance: ln(10⁻⁶)/ln 0.98 = 684 steps. The model
k = max_i ln(tol·‖g⁰‖ / |g⁰_i|) / ln|1 − αλ_i| reproduces all three counts exactly. Hence the "optimal" step
α* = 1/(1+c) (t = 0.9907), which minimises ρ to (c−1)/(c+1) = 0.98131, costs 733 iterations — the GD entry of
Table 1 — while a fine sweep of 500 values of t ∈ [0.5, 0.999] gives 522 at t = 0.983 (`s1_sweep.csv`). α* is best
for the worst start, not for ours.

**Backtracking** (α₀ = 1, c = 10⁻⁴; `results/s1_backtrack.csv`):

| problem | iters | mean accepted α | mean halvings | 1/λ_max(H) at x* | f-evals / iter |
|---------|-------|-----------------|---------------|------------------|----------------|
| R1      | 13756 | 2.10e-3         | 8.94          | 9.98e-4          | 10.9           |
| Q1      | 494   | 1.02e-2         | 6.69          | 4.72e-3          | 8.7            |
| project | 988   | 1.000           | 0.00          | 51.1             | 2.0            |

On R1 and Q1 the accepted step is about 2.1/λ_max, at the stability edge rather than at 1/λ_max: with c = 10⁻⁴
Armijo accepts any step up to about 2(1−c)/λ_max along the stiff direction. That is why GD-BT zig-zags in F2,
and why its mean on Q1 (1.02e-2) even exceeds 2/λ_max — some accepted steps briefly amplify the stiff component
while the decrease along the flat one still satisfies Armijo. The number of halvings is log₂(1/α): the medians are
2⁻⁹ on R1 and 2⁻⁷ on Q1. So **α₀ = 1 is harmless when the good step is much smaller than 1**: it costs about
log₂(1/α_good) ≈ 7–9 extra function evaluations per iteration, but no extra iterations.

On the project block H(x*) has eigenvalues 0.01405 and 0.01956 (F is an average distance, its curvature is about
1/distance), so 1/λ_max = 51 and the best fixed step is 2/(λ_min+λ_max) = 59.5 (`s1_project.csv`). Here α = 1 passes
Armijo at once (0 halvings in all 988 steps), and backtracking can only shrink α, never enlarge it. With α = 1 the
flat mode shrinks by 1 − λ_min = 0.98595 per step, which predicts ln(10⁻⁶)/ln 0.98595 ≈ 976 iterations; measured
988. Tuned GD reaches the tolerance in 24 iterations at α = 31.6 (Table 2, extended grid) and in 12 at α = 50 (extra
run), so GD-BT is 41 times slower than tuned GD (F3). **α₀ = 1 is useless when the good step is much larger than 1.**
Remedies: rescale the variables (e.g. coordinates in units of 10), start each line search at twice the previously
accepted α, or take α₀ from local curvature (Barzilai–Borwein step).

*Amir Amanbay (M1 — GD)*
