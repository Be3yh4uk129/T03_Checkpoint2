"""Checkpoint 4, role GA-P: representation and genetic operators for storyline A, variant 1.

Author: Abdrashit Yersultan (GA-P).  Configuration: GA, penalty.

Representation
--------------
An individual is one integer array `a` of length N:

    a[i] = the point index of the center that serves point i.

A point c is a center exactly when it serves itself, a[c] == c, so the centers are
the fixed points of the array and no second structure has to be kept in step with it.
An individual is structurally valid when

    1. it has exactly V fixed points (exactly V centers), and
    2. every entry points at a fixed point (nobody is served by a non-center).

Crossover and mutation keep both properties, so the only constraint an offspring may
break is the capacity, which is what the penalty configuration asks for.

Penalty
-------
    violation(a) = sum over centers c of max(0, load(c) - Cap)      [demand units]
    fitness(a)   = cost(a) + rho * violation(a)

`cost` is the objective of the reference model of Checkpoint 3,
sum_i d[i, a[i]] + 2 * sum_c D[c], and agrees with evaluate_A of the generator on
every structurally valid individual.
"""
import numpy as np

# Default penalty weight.
#
# The a priori argument gives 10: the largest distance in the instance is about 131 and the
# smallest demand is 15, so a reassignment that repairs an overload can save at most about
# 131 / 15 = 8.7 cost units per unit of overload, and a weight above that makes repairing an
# overload always worthwhile. The tuning experiment of experiments/exp_rho.py, run on the
# tuning seeds 100 to 104 only, says the opposite: rho = 1 reaches a mean cost of 1005.4
# against 1015.8 at rho = 10, while every run still ends feasible. A weight below the
# break-even value lets the population cross the infeasible region instead of being held
# out of it, and on an instance filled to only 55.3 percent that costs nothing in
# feasibility. The final runs therefore use rho = 1.
RHO_DEFAULT = 1.0


# ----------------------------------------------------------------- representation

def centers_of(a):
    """The centers of an individual: the points that serve themselves."""
    a = np.asarray(a)
    return np.where(a == np.arange(a.size))[0]


def is_valid(d, a):
    """Structural validity: exactly V centers, and everybody served by a center."""
    a = np.asarray(a)
    if a.size != d['N']:
        return False
    if a.min() < 0 or a.max() >= d['N']:
        return False
    centers = centers_of(a)
    if centers.size != d['V']:
        return False
    return bool(np.all(np.isin(a, centers)))


def load_vector(d, a):
    """Load per point index: the total demand served, zero for every non-center."""
    a = np.asarray(a)
    return np.bincount(a, weights=d['demand'], minlength=a.size)


def loads(d, a):
    """Total demand served by each center, as a dictionary center -> load."""
    lv = load_vector(d, a)
    return {int(c): float(lv[c]) for c in centers_of(a)}


def _nearest_center(d, i, centers):
    """The center of `centers` closest to point i."""
    return int(centers[int(np.argmin(d['dist'][i, centers]))])


def random_individual(d, rng):
    """V centers drawn uniformly, every other point served by a uniformly random center.

    Deliberately not greedy: the initial population has to be spread out, and with a
    fill ratio of 755 / (13 * 105) = 55.3 percent a uniform assignment is feasible more
    often than not, so the penalty has something to work with from the first generation.
    """
    N, V = d['N'], d['V']
    centers = rng.choice(N, size=V, replace=False)
    a = centers[rng.integers(0, V, size=N)]
    a[centers] = centers
    return a.astype(np.int64)


# ----------------------------------------------------------------- evaluation

def cost_of(d, a):
    """Objective of the reference model: assignment distances plus the depot round trips."""
    a = np.asarray(a)
    is_center = a == np.arange(a.size)
    return float(d['dist'][np.arange(a.size), a].sum() + 2.0 * d['D'][is_center].sum())


def violation_of(d, a):
    """Total overload in demand units, summed over the centers.

    A non-center slot of the load vector is zero, so it can never be overloaded and the
    sum over all slots is the same as the sum over the centers.
    """
    lv = load_vector(d, a)
    return float(np.maximum(lv - float(d['Cap']), 0.0).sum())


def evaluate(d, a, rho=RHO_DEFAULT):
    """Return (fitness, cost, violation, feasible) for one individual.

    One pass over the array: this is the function the whole budget is spent on.
    """
    a = np.asarray(a)
    n = a.size
    is_center = a == np.arange(n)
    cost = float(d['dist'][np.arange(n), a].sum() + 2.0 * d['D'][is_center].sum())
    lv = np.bincount(a, weights=d['demand'], minlength=n)
    viol = float(np.maximum(lv - float(d['Cap']), 0.0).sum())
    return cost + rho * viol, cost, viol, viol <= 0.0


def to_solution(d, a):
    """The individual in the form evaluate_A of the generator expects."""
    a = np.asarray(a)
    return [int(c) for c in centers_of(a)], [int(x) for x in a], None


# ----------------------------------------------------------------- operators

def crossover(d, a1, a2, rng):
    """Uniform crossover on the assignment, with the center sets merged first.

    The centers common to both parents are inherited; the remaining places are filled
    from the symmetric difference, so the child's centers are always a subset of the
    union of the parents' centers. Every other point then follows one of its parents
    when that parent's center survived into the child, and otherwise moves to its
    nearest surviving center. The result always has exactly V centers.
    """
    V = d['V']
    c1, c2 = set(int(x) for x in centers_of(a1)), set(int(x) for x in centers_of(a2))
    common = sorted(c1 & c2)
    rest = sorted(c1 ^ c2)
    rng.shuffle(rest)
    child_centers = common + rest[: V - len(common)]
    child_centers = np.array(sorted(child_centers), dtype=np.int64)

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    take_first = rng.random(a1.size) < 0.5
    child = np.where(take_first, a1, a2)

    in_child = np.isin(child, child_centers)
    for i in np.where(~in_child)[0]:
        child[i] = _nearest_center(d, int(i), child_centers)
    child[child_centers] = child_centers
    return child.astype(np.int64)


def mutate(d, a, rng, n_components=1, p_center=0.2):
    """Change `n_components` components of the individual, in place on a copy.

    Each component is one of two moves:

      transfer      a non-center point is reassigned to another center (probability 1 - p_center)
      center move   a center is replaced by a non-center point (probability p_center)

    The center move is the additional move of the README: it changes the set of centers,
    which transfer and swap never do, and without it the center sets of the population
    could only ever shrink towards the intersection of the initial ones.
    """
    a = np.array(a, dtype=np.int64, copy=True)
    N, V = d['N'], d['V']
    for _ in range(int(n_components)):
        centers = centers_of(a)
        if rng.random() < p_center and centers.size == V:
            old = int(centers[rng.integers(0, centers.size)])
            outside = np.setdiff1d(np.arange(N), centers, assume_unique=False)
            if outside.size == 0:
                continue
            new = int(outside[rng.integers(0, outside.size)])
            served = np.where(a == old)[0]
            a[new] = new
            remaining = np.append(np.setdiff1d(centers, [old]), new)
            for i in served:
                if i == old:
                    continue
                a[i] = _nearest_center(d, int(i), remaining)
            a[old] = _nearest_center(d, old, remaining)
            a[new] = new
        else:
            non_centers = np.setdiff1d(np.arange(N), centers, assume_unique=False)
            if non_centers.size == 0 or centers.size < 2:
                continue
            i = int(non_centers[rng.integers(0, non_centers.size)])
            choices = centers[centers != a[i]]
            a[i] = int(choices[rng.integers(0, choices.size)])
    return a


def swap_mutate(d, a, rng):
    """Exchange the centers of two points served by different centers.

    Not used by the GA loop by default; it exists because question 4 of the analysis
    has to count improving swaps as well as improving transfers.
    """
    a = np.array(a, dtype=np.int64, copy=True)
    centers = set(int(c) for c in centers_of(a))
    movable = [i for i in range(d['N']) if i not in centers]
    if len(movable) < 2:
        return a
    for _ in range(20):
        i, j = (int(movable[k]) for k in rng.choice(len(movable), size=2, replace=False))
        if a[i] != a[j]:
            a[i], a[j] = a[j], a[i]
            return a
    return a
