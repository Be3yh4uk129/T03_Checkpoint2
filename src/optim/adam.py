
"""Adam optimizer. Owner: M4 (Adam).

Returns:
    x: final point
    hist: all iterates, including the starting point
    k: number of updates performed
"""
import numpy as np
from .common import MAX_ITER_GRAD, BLOWUP, converged


def adam(
    prob,
    alpha,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    max_iter=MAX_ITER_GRAD,
):
    """Run Adam using NumPy only. Returns (x, hist, k)."""

    with np.errstate(all="ignore"):
        x = np.array(prob.x0, dtype=float)
        g = np.asarray(prob.grad(x), dtype=float)
        g0_norm = np.linalg.norm(g)

        # First and second moving averages start at zero.
        m = np.zeros_like(x)
        s = np.zeros_like(x)

        # Save x0 as the first item in the history.
        hist = [x.copy()]
        k = 0

        while k < max_iter:
            # Check convergence BEFORE performing the next update.
            if converged(
                np.linalg.norm(g),
                g0_norm,
                prob.rule,
                prob.tol,
            ):
                break

            # Stop if the current point has blown up.
            norm_x = np.linalg.norm(x)
            if not np.isfinite(norm_x) or norm_x > BLOWUP:
                break

            # Update the moving averages.
            m = beta1 * m + (1.0 - beta1) * g
            s = beta2 * s + (1.0 - beta2) * (g ** 2)

            # Correct the initial zero bias.
            m_hat = m / (1.0 - beta1 ** (k + 1))
            s_hat = s / (1.0 - beta2 ** (k + 1))

            # Adam update: scaling is coordinate-wise.
            x = x - alpha * m_hat / (np.sqrt(s_hat) + eps)

            # Prepare the next iteration.
            g = np.asarray(prob.grad(x), dtype=float)
            hist.append(x.copy())
            k += 1

    return x, np.array(hist), k
