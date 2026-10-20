"""Matrix form of the reference model (storyline A, variant 1).

    minimise   sum_i sum_k d_ik x_ik + 2 sum_k D_k y_k
    (A1)   sum_k x_ik = 1              for every point i          (N rows,  "E")
    (A2)   x_ik - y_k <= 0             for every pair (i, k)      (N*N rows, "L")
    (A3)   sum_i q_i x_ik - Cap y_k <= 0   for every k            (N rows,  "L")
    (A4)   x_kk - y_k = 0              for every k                (N rows,  "E")
    (A5)   sum_k y_k = V                                          (1 row,   "E")
    bounds 0 <= x_ik <= 1, 0 <= y_k <= 1   (bounds are not rows)

Column order (n = N*N + N columns):
    x_ik  ->  column i*N + k        for i, k in 0 .. N-1
    y_k   ->  column N*N + k        for k   in 0 .. N-1

Row order (m = 2N + 1 equality rows first, then N*N + N inequality rows):
    (A1) i      ->  row 0 + i
    (A4) k      ->  row N + k
    (A5)        ->  row 2N
    (A2) (i, k) ->  row 2N + 1 + i*N + k
    (A3) k      ->  row 2N + 1 + N*N + k

Equality rows come first so that the designated row of Task 4, the single row
(A5), keeps the fixed index 2N and is easy to point at.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

# name -> (first row, number of rows), filled for the concrete N in build_matrix_form
ROW_FAMILIES = ("A1", "A2", "A3", "A4", "A5")

# the designated constraint of the team task file: (A5), right-hand side b = V
DESIGNATED_ROW = "A5"


@dataclass
class MatrixForm:
    """min c^T v  subject to  A v (sense) b,  lo <= v <= up.

    sense[i] is "E" for an equality row and "L" for a "<=" row. The model has no
    ">=" row, but the checker in src/kkt also understands "G".
    """

    c: np.ndarray
    A: sparse.csr_matrix
    b: np.ndarray
    sense: np.ndarray
    lo: np.ndarray
    up: np.ndarray
    N: int
    V: int
    Cap: int
    row_span: dict = field(default_factory=dict)   # family -> (start, count)
    col_span: dict = field(default_factory=dict)   # "x" / "y" -> (start, count)

    @property
    def n_variables(self):
        return int(self.c.size)

    @property
    def n_rows(self):
        return int(self.b.size)

    def row_of(self, family, index=0):
        """Global row number of entry `index` inside a constraint family."""
        start, count = self.row_span[family]
        if not 0 <= index < count:
            raise IndexError(f"{family} has {count} rows, asked for {index}")
        return start + index

    def family_of_row(self, row):
        """Which family a global row number belongs to."""
        for name, (start, count) in self.row_span.items():
            if start <= row < start + count:
                return name
        raise IndexError(row)

    def split_solution(self, v):
        """Split a full solution vector into the x matrix (N by N) and the y vector."""
        v = np.asarray(v, dtype=float)
        N = self.N
        return v[: N * N].reshape(N, N), v[N * N:]

    def scipy_blocks(self):
        """The (A_eq, b_eq, A_ub, b_ub) blocks scipy.optimize.linprog expects."""
        eq = self.sense == "E"
        ub = self.sense == "L"
        return (self.A[eq], self.b[eq], self.A[ub], self.b[ub])


def build_matrix_form(d) -> MatrixForm:
    """Assemble the reference model of instance `d` in matrix form."""
    N = int(d["N"])
    V = int(d["V"])
    Cap = int(d["Cap"])
    dist = np.asarray(d["dist"], dtype=float)
    D = np.asarray(d["D"], dtype=float)
    q = np.asarray(d["demand"], dtype=float)

    n_x = N * N
    n = n_x + N

    # objective: sum_i sum_k d_ik x_ik + 2 sum_k D_k y_k
    c = np.concatenate([dist.reshape(-1), 2.0 * D])

    rows, cols, vals = [], [], []
    b = []
    sense = []
    row_span = {}
    r = 0

    def add(row_entries, rhs, kind):
        nonlocal r
        for col, val in row_entries:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        b.append(rhs)
        sense.append(kind)
        r += 1

    # (A1) every point is served exactly once
    row_span["A1"] = (r, N)
    for i in range(N):
        add([(i * N + k, 1.0) for k in range(N)], 1.0, "E")

    # (A4) a center serves itself
    row_span["A4"] = (r, N)
    for k in range(N):
        add([(k * N + k, 1.0), (n_x + k, -1.0)], 0.0, "E")

    # (A5) exactly V centers
    row_span["A5"] = (r, 1)
    add([(n_x + k, 1.0) for k in range(N)], float(V), "E")

    # (A2) a point may only be served from a center
    row_span["A2"] = (r, N * N)
    for i in range(N):
        for k in range(N):
            add([(i * N + k, 1.0), (n_x + k, -1.0)], 0.0, "L")

    # (A3) the load of a center must not exceed the capacity
    row_span["A3"] = (r, N)
    for k in range(N):
        entries = [(i * N + k, float(q[i])) for i in range(N)]
        entries.append((n_x + k, -float(Cap)))
        add(entries, 0.0, "L")

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    mf = MatrixForm(
        c=c,
        A=A,
        b=np.asarray(b, dtype=float),
        sense=np.asarray(sense, dtype="<U1"),
        lo=np.zeros(n),
        up=np.ones(n),
        N=N,
        V=V,
        Cap=Cap,
        row_span=row_span,
        col_span={"x": (0, n_x), "y": (n_x, N)},
    )
    return mf
