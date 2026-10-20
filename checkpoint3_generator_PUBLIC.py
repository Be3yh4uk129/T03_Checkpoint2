"""Checkpoint 3 -- public instance generator (Introduction to Optimization, ItO2025).

Uses the SAME team seed as Checkpoints 1 and 2. Your team's call (story, variant and, if any, an
extra argument) is written in your team's task file. Example:

    d = cp34_instance(SEED, 'A', 2)          # storyline A, variant 2
    d = cp34_instance(SEED, 'B', 3)          # storyline B, variant 3

The function returns only DATA. You write the model of the Checkpoint 3 handout yourselves, and
the same call gives the same instance in Checkpoint 4 (that generator version adds solution checkers).
Use PuLP below version 4: pip install "pulp<4".
"""
import numpy as np

# tangent points of g(u) = u / (1 - u) for storyline B (fixed for everyone)
TANGENT_POINTS = (0.0, 0.4, 0.7, 0.9)


def _logistics(seed, variant):
    rng = np.random.default_rng(seed)
    V = int(rng.integers(8, 15))
    N = int(rng.integers(35, 46))
    H_max = int(rng.integers(7, 11))                 # drawn only to keep the random stream identical to Checkpoint 1
    warehouse = rng.uniform(40, 60, size=2)
    points = rng.uniform(0, 100, size=(N, 2))
    demand = rng.integers(15, 30, size=N)
    data = dict(V=V, N=N, warehouse=warehouse, points=points, demand=demand)
    if variant == 1:
        data['Cap'] = int(rng.integers(80, 121))
    elif variant == 2:
        data['Cap'] = int(rng.integers(80, 121))
        data['is_premium'] = rng.choice([0, 1], size=N, p=[0.75, 0.25])
        data['penalty'] = rng.uniform(200, 500, size=N) * data['is_premium']
    elif variant == 3:
        data['vehicle_cap'] = rng.integers(60, 150, size=V)
        data['vehicle_cost_per_km'] = rng.uniform(0.8, 1.6, size=V)
    return data


def _cloud(seed, variant):
    rng = np.random.default_rng(seed)
    S = int(rng.integers(6, 13))
    M = int(rng.integers(30, 51))
    cpu_cap = rng.integers(32, 65, size=S)
    mem_cap = rng.integers(64, 257, size=S)
    cpu_req = rng.integers(1, 9, size=M)
    mem_req = rng.integers(2, 33, size=M)
    data = dict(S=S, M=M, cpu_cap=cpu_cap, mem_cap=mem_cap, cpu_req=cpu_req, mem_req=mem_req)
    if variant == 2:
        data['zone'] = rng.integers(0, 3, size=S)
        data['is_premium_vm'] = rng.choice([0, 1], size=M, p=[0.75, 0.25])
        present_zones = np.unique(data['zone'])
        vm_zone = rng.choice(present_zones, size=M)
        data['vm_required_zone'] = np.where(data['is_premium_vm'] == 1, vm_zone, -1)
    elif variant == 3:
        data['is_gpu_server'] = rng.choice([0, 1], size=S, p=[0.6, 0.4])
        data['needs_gpu'] = rng.choice([0, 1], size=M, p=[0.8, 0.2])
    return data


def cp34_instance(seed, story, variant, cap=None):
    """Instance data for Checkpoints 3 and 4 (the same call in both).
    story 'A' (logistics) or 'B' (cloud); variant 1, 2 or 3.
    cap: only if your task file says so -- overrides the vehicle capacity Cap (variants A1, A2)."""
    if story == 'A':
        d = _logistics(seed, variant)
        if cap is not None:
            d['Cap'] = int(cap)
        P, W = d['points'], d['warehouse']
        d['dist'] = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)   # d_ik, point i -> point k
        d['D'] = np.linalg.norm(P - W, axis=1)                               # D_k, warehouse -> point k
    elif story == 'B':
        d = _cloud(seed, variant)
        if cap is not None:
            raise ValueError("argument cap is defined for storyline A only")
    else:
        raise ValueError("story must be 'A' or 'B'")
    d['story'], d['variant'] = story, variant
    return d
