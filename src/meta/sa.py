"""Simulated annealing for storyline A, variant 1 (capacitated choice of V centers).

Role SA-I (Amir Amanbay), configuration "SA, transfer+swap".

State
    centers : list of V distinct point indices, every center serves itself
    assign  : list of length N, assign[i] = the center (a point index) that serves point i
    load[k] : total demand served by center k, never above Cap for an accepted state

Cost (the reference model of Checkpoint 3)
    z = sum_i dist[i][assign[i]] + 2 * sum_{k in centers} D[k]

Moves (only moves that keep every load <= Cap are proposed; the capacity check reads
data only and is not an evaluation; every proposal whose cost is computed is one evaluation)
    transfer  one non-center point goes to another center        delta = 1 distance change
    swap      two non-center points of different centers swap      delta = 2 distance changes
    recenter  ADDITIONAL MOVE (listed in the README): a non-center point j takes over the role
              of its own center k; the whole cluster of k is now served by j, the loads do
              not change, delta = sum over the cluster of the distance changes + 2 (D_j - D_k)

Cooling: geometric, T_k = T0 * alpha**k with alpha = (Tend / T0) ** (1 / K), K = number of
proposals after the starting solution. The starting solution counts as one evaluation.

Only numpy and the standard library; the only random source is random.Random(seed).
z_ref and z_LP are never read here.
"""
import importlib.util
import math
import os
import random

import numpy as np

# move shares, reported in the README: 60 % transfer, 30 % swap, 10 % recenter
P_SWAP = 0.30
P_RECENTER = 0.10
MAX_TRIES = 60            # data-only resampling of an infeasible draw before falling back
N_BUCKETS = 100           # acceptance statistics are collected per 1 % of the proposals
TRACE_EVERY = 1000

TRANSFER, SWAP, RECENTER = 0, 1, 2

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GEN_PATH = os.path.join(_ROOT, "checkpoint4_generator_PUBLIC.py")
_GEN = None


# --------------------------------------------------------------------------- the checker
def _generator():
    global _GEN
    if _GEN is None:
        spec = importlib.util.spec_from_file_location("checkpoint4_generator_PUBLIC", _GEN_PATH)
        if spec is None or spec.loader is None:
            raise ImportError("checkpoint4_generator_PUBLIC.py not found at " + _GEN_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GEN = mod
    return _GEN


def check(d, solution):
    """Run the checker of the generator on solution = (centers, assign, vehicle).

    evaluate_A(d, variant, centers, assign, vehicle=None, tol=1e-9) returns
    (objective, list of violated conditions); feasible means the list is empty.
    Returns (cost, feasible, problems).
    """
    centers, assign, vehicle = solution
    cost, problems = _generator().evaluate_A(d, int(d.get("variant", 1)), list(centers),
                                             list(assign), vehicle)
    return float(cost), len(problems) == 0, problems


def full_cost(d, centers, assign):
    """The objective recomputed from scratch (used only to cross-check, never in the loop)."""
    dist, D = d["dist"], d["D"]
    return float(sum(dist[i][assign[i]] for i in range(len(assign))) + 2.0 * sum(D[k] for k in centers))


# --------------------------------------------------------------------------- the state
class State:
    """Current solution with the bookkeeping needed for O(1) deltas of transfer and swap."""

    def __init__(self, d, centers, assign):
        self.N = int(d["N"])
        self.V = int(d["V"])
        self.Cap = float(d["Cap"])
        self.q = [float(x) for x in d["demand"]]
        self.dist = [[float(x) for x in row] for row in d["dist"]]
        self.D = [float(x) for x in d["D"]]
        self.centers = [int(k) for k in centers]
        self.assign = [int(a) for a in assign]
        if len(set(self.centers)) != self.V or len(self.assign) != self.N:
            raise ValueError("start must have V distinct centers and N assignments")
        self.is_center = [False] * self.N
        for k in self.centers:
            self.is_center[k] = True
        for k in self.centers:
            if self.assign[k] != k:
                raise ValueError("every center must serve itself")
        for i in range(self.N):
            if not self.is_center[self.assign[i]]:
                raise ValueError("point %d is assigned to a non-center" % i)
        self.cpos = {k: p for p, k in enumerate(self.centers)}          # center -> slot
        self.nc = [i for i in range(self.N) if not self.is_center[i]]   # non-centers
        self.ncpos = [-1] * self.N
        for p, i in enumerate(self.nc):
            self.ncpos[i] = p
        self.load = [0.0] * self.N
        for i in range(self.N):
            self.load[self.assign[i]] += self.q[i]
        self.n_over = sum(1 for k in self.centers if self.load[k] > self.Cap + 1e-9)
        self.cost = full_cost(d, self.centers, self.assign)

    @property
    def feasible(self):
        return self.n_over == 0

    def solution(self):
        return (sorted(self.centers), list(self.assign), None)

    # ---- proposals: return (type, a, b, delta); only the capacity is checked, from data
    def propose(self, rng, p_swap=P_SWAP, p_recenter=P_RECENTER):
        u = rng.random()
        if u < p_recenter:
            return self._propose_recenter(rng)
        if u < p_recenter + p_swap:
            mv = self._propose_swap(rng)
        else:
            mv = self._propose_transfer(rng)
        return mv if mv is not None else self._propose_recenter(rng)

    def _propose_transfer(self, rng):
        nc, centers, Cap, q, load, assign, dist = self.nc, self.centers, self.Cap, self.q, self.load, self.assign, self.dist
        for _ in range(MAX_TRIES):
            i = nc[rng.randrange(len(nc))]
            c = centers[rng.randrange(self.V)]
            a = assign[i]
            if c == a or load[c] + q[i] > Cap:
                continue
            return (TRANSFER, i, c, dist[i][c] - dist[i][a])
        return None

    def _propose_swap(self, rng):
        nc, Cap, q, load, assign, dist = self.nc, self.Cap, self.q, self.load, self.assign, self.dist
        n = len(nc)
        for _ in range(MAX_TRIES):
            i = nc[rng.randrange(n)]
            j = nc[rng.randrange(n)]
            a, b = assign[i], assign[j]
            if a == b:
                continue
            if load[a] - q[i] + q[j] > Cap or load[b] - q[j] + q[i] > Cap:
                continue
            return (SWAP, i, j, dist[i][b] + dist[j][a] - dist[i][a] - dist[j][b])
        return None

    def _propose_recenter(self, rng):
        j = self.nc[rng.randrange(len(self.nc))]
        k = self.assign[j]
        dist, assign = self.dist, self.assign
        delta = 2.0 * (self.D[j] - self.D[k])
        dj, dk = dist[j], dist[k]
        for m in range(self.N):
            if assign[m] == k:
                delta += dj[m] - dk[m]          # dist is symmetric: dist[m][j] = dist[j][m]
        return (RECENTER, j, k, delta)

    # ---- apply an accepted move, cost updated by the delta (no recomputation)
    def apply(self, move):
        typ, a, b, delta = move
        q, load, assign, Cap = self.q, self.load, self.assign, self.Cap
        if typ == TRANSFER:
            i, c = a, b
            old = assign[i]
            before = (load[old] > Cap + 1e-9) + (load[c] > Cap + 1e-9)
            load[old] -= q[i]
            load[c] += q[i]
            assign[i] = c
            self.n_over += (load[old] > Cap + 1e-9) + (load[c] > Cap + 1e-9) - before
        elif typ == SWAP:
            i, j = a, b
            ca, cb = assign[i], assign[j]
            before = (load[ca] > Cap + 1e-9) + (load[cb] > Cap + 1e-9)
            load[ca] += q[j] - q[i]
            load[cb] += q[i] - q[j]
            assign[i], assign[j] = cb, ca
            self.n_over += (load[ca] > Cap + 1e-9) + (load[cb] > Cap + 1e-9) - before
        else:
            j, k = a, b
            for m in range(self.N):
                if assign[m] == k:
                    assign[m] = j
            load[j], load[k] = load[k], 0.0
            slot = self.cpos.pop(k)
            self.centers[slot] = j
            self.cpos[j] = slot
            self.is_center[j], self.is_center[k] = True, False
            p = self.ncpos[j]                      # k takes the place of j among the non-centers
            self.nc[p] = k
            self.ncpos[k], self.ncpos[j] = p, -1
        self.cost += delta


# --------------------------------------------------------------------------- start and rules
def random_start(d, rng):
    """V random distinct centers, then every other point (random order) to the nearest center
    that still has room. Reads data only; the caller charges one evaluation for it."""
    N, V, Cap = int(d["N"]), int(d["V"]), float(d["Cap"])
    q, dist = d["demand"], d["dist"]
    for _ in range(200):
        centers = rng.sample(range(N), V)
        assign = list(range(N))
        load = {k: float(q[k]) for k in centers}
        others = [i for i in range(N) if i not in load]
        rng.shuffle(others)
        ok = True
        for i in others:
            best, best_d = None, math.inf
            for k in centers:
                if load[k] + q[i] <= Cap and dist[i][k] < best_d:
                    best, best_d = k, dist[i][k]
            if best is None:
                ok = False
                break
            assign[i] = best
            load[best] += q[i]
        if ok:
            return centers, assign
    # practically unreachable on a 55 % full instance: nearest center, capacity ignored
    centers = rng.sample(range(N), V)
    assign = [i if i in centers else min(centers, key=lambda k: dist[i][k]) for i in range(N)]
    return centers, assign


def random_walk_deltas(d, rng, n_walk, state=None):
    """Random walk of n_walk proposals that accepts everything; returns the deltas and the state."""
    if state is None:
        state = State(d, *random_start(d, rng))
    deltas = []
    for _ in range(n_walk):
        mv = state.propose(rng)
        deltas.append(mv[3])
        state.apply(mv)
    return deltas, state


def temperatures_from_deltas(deltas):
    """Rule of Section 4: T0 = 4.5 * median, Tend = 0.22 * 5th percentile of the positive deltas."""
    pos = np.array([x for x in deltas if x > 1e-12])
    if len(pos) < 5:
        return None
    return 4.5 * float(np.median(pos)), 0.22 * float(np.percentile(pos, 5))


def rule_params(d, seed=100, n_walk=3000):
    """The rule-based T0 and Tend (computed once, outside the final runs, on tuning seed 100)."""
    rng = random.Random(seed)
    deltas, _ = random_walk_deltas(d, rng, n_walk)
    pos = [x for x in deltas if x > 1e-12]
    T0, Tend = temperatures_from_deltas(deltas)
    return {"T0": T0, "Tend": Tend, "Delta": float(np.median(pos)),
            "Delta_min": float(np.percentile(pos, 5)), "n_walk": n_walk, "n_positive": len(pos),
            "walk_seed": seed}


# --------------------------------------------------------------------------- the solver
def solve(d, variant, seed, budget, params=None, start=None):
    """Simulated annealing. params = {"T0": ..., "Tend": ...} (optional "p_swap", "p_recenter").

    Without T0/Tend in params, the rule of Section 4 is applied inside the run: a random walk
    of min(3000, budget // 10) proposals that accepts everything, charged to the budget.
    """
    if variant != 1:
        raise NotImplementedError("this solver implements storyline A, variant 1")
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be at least 1")
    params = dict(params or {})
    p_swap = float(params.get("p_swap", P_SWAP))
    p_rec = float(params.get("p_recenter", P_RECENTER))
    rng = random.Random(seed)

    # starting solution: one evaluation
    if start is not None:
        state = State(d, start[0], start[1])
    else:
        state = State(d, *random_start(d, rng))
    evals = 1
    start_cost = state.cost

    best_cost = state.cost if state.feasible else math.inf
    best_sol = state.solution() if state.feasible else None
    trace = [(evals, best_cost)]

    def keep_best():
        nonlocal best_cost, best_sol
        if state.feasible and state.cost < best_cost - 1e-12:
            best_cost = state.cost
            best_sol = state.solution()

    # temperatures
    if "T0" in params and "Tend" in params:
        T0, Tend = float(params["T0"]), float(params["Tend"])
        source = "params"
    else:
        n_walk = min(3000, (budget - 1) // 10)
        deltas = []
        for _ in range(n_walk):                      # charged: each proposal is one evaluation
            mv = state.propose(rng, p_swap, p_rec)
            deltas.append(mv[3])
            state.apply(mv)
            evals += 1
            keep_best()
            if evals % TRACE_EVERY == 0:
                trace.append((evals, best_cost))
        temps = temperatures_from_deltas(deltas)
        if temps is None:                           # tiny budget: a scale read from the data
            T0 = 0.1 * float(np.median(d["dist"]))
            Tend = 1e-3 * T0
        else:
            T0, Tend = temps
        if best_sol is not None:                    # anneal from the best point of the walk
            state = State(d, best_sol[0], best_sol[1])
        source = "rule (inside run, walk of %d)" % n_walk
    if not (T0 > 0 and Tend > 0):
        raise ValueError("T0 and Tend must be positive")

    K = budget - evals                              # proposals left for the annealing
    alpha = (Tend / T0) ** (1.0 / K) if K > 0 else 1.0
    T = T0

    # acceptance statistics of worsening moves (delta > 0), per 1 % of the proposals
    nb = N_BUCKETS if K >= N_BUCKETS else max(K, 1)
    b_worse = [0] * nb
    b_acc = [0] * nb
    b_T = [0.0] * nb
    b_n = [0] * nb
    w5 = max(1, int(round(0.05 * K)))
    first = [0, 0]                                  # [worsening, accepted] in the first 5 %
    last = [0, 0]                                   # same in the last 5 %
    n_acc = 0
    exp, rand = math.exp, rng.random

    for k in range(K):
        mv = state.propose(rng, p_swap, p_rec)
        evals += 1
        delta = mv[3]
        bi = k * nb // K
        b_n[bi] += 1
        b_T[bi] += T
        if delta <= 0.0:
            accept = True
        else:
            accept = rand() < exp(-delta / T)
            b_worse[bi] += 1
            b_acc[bi] += accept
            if k < w5:
                first[0] += 1
                first[1] += accept
            if k >= K - w5:
                last[0] += 1
                last[1] += accept
        if accept:
            state.apply(mv)
            n_acc += 1
            if delta < 0.0 or best_sol is None:
                keep_best()
        T *= alpha
        if evals % TRACE_EVERY == 0:
            trace.append((evals, best_cost))

    if trace[-1][0] != evals:
        trace.append((evals, best_cost))

    # the reported numbers come from the checker
    if best_sol is None:                            # never feasible: report the current one
        best_sol = state.solution()
    cost, feasible, problems = check(d, best_sol)

    acceptance = [
        {"bucket": i, "T_mean": (b_T[i] / b_n[i]) if b_n[i] else None, "proposals": b_n[i],
         "worsening": b_worse[i], "accepted": b_acc[i],
         "share": (b_acc[i] / b_worse[i]) if b_worse[i] else None}
        for i in range(nb)
    ]
    return {
        "solution": best_sol,
        "cost": cost,
        "evals": evals,
        "feasible": feasible,
        "trace": trace,
        # extras for the analysis (not part of the interface)
        "problems": problems,
        "start_cost": start_cost,
        "T0": T0, "Tend": Tend, "alpha": alpha, "temperature_source": source,
        "p_swap": p_swap, "p_recenter": p_rec,
        "accepted_share_all": n_acc / K if K > 0 else None,
        "acceptance": acceptance,
        "accept_first5": (first[1] / first[0]) if first[0] else None,
        "accept_last5": (last[1] / last[0]) if last[0] else None,
        "worsening_first5": first[0], "worsening_last5": last[0],
    }


# --------------------------------------------------------------------------- Table 2 helper
def count_improving(d, solution, tol=1e-9):
    """Improving transfer and swap moves from a solution (feasible and delta < -tol).

    Recenter moves change the set of centers and are not counted, as the handout says.
    """
    st = State(d, solution[0], solution[1])
    n_tr = n_sw = 0
    nc = st.nc
    for i in nc:
        a = st.assign[i]
        for c in st.centers:
            if c != a and st.load[c] + st.q[i] <= st.Cap and st.dist[i][c] - st.dist[i][a] < -tol:
                n_tr += 1
    for x in range(len(nc)):
        i = nc[x]
        for y in range(x + 1, len(nc)):
            j = nc[y]
            a, b = st.assign[i], st.assign[j]
            if a == b:
                continue
            if st.load[a] - st.q[i] + st.q[j] > st.Cap or st.load[b] - st.q[j] + st.q[i] > st.Cap:
                continue
            if st.dist[i][b] + st.dist[j][a] - st.dist[i][a] - st.dist[j][b] < -tol:
                n_sw += 1
    return n_tr, n_sw