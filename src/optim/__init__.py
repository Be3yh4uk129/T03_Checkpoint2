"""Optimizers written from scratch (numpy + standard library only).

Common interface (Checkpoint 2, Section 3). Every solver has the form

    solver(prob, <tuned params>, max_iter=...) -> (x, hist, k)

* prob  -- a problems.base.Problem (f, grad, hess, x0, rule, tol)
* x     -- final point
* hist  -- np.ndarray of ALL iterates, shape (k+1, n), hist[0] = x0
* k     -- number of updates performed (stopping rule is checked at the TOP of
           the loop, before the update)

Failure: ||x|| > BLOWUP or iteration cap reached. Use common.reached_tol(prob, x)
to decide whether a run counts as converged ("-" in the tables otherwise).
"""
