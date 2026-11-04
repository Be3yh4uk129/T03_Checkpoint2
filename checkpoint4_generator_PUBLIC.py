"""Checkpoint 4 -- public instance generator and solution checkers (includes the Checkpoint 3 data; Introduction to Optimization, ItO2025).

Uses the SAME team seed as Checkpoints 1 and 2. Your team's call (story, variant and, if any, an
extra argument) is written in your team's task file. Example:

    d = cp34_instance(SEED, 'A', 2)          # storyline A, variant 2
    d = cp34_instance(SEED, 'B', 3)          # storyline B, variant 3

The function returns only DATA. You write the model (Checkpoint 3) and the metaheuristic (Checkpoint 4)
yourselves. Two checkers are provided so that everyone measures the objective in the same way:

    cost, problems = evaluate_A(d, variant, centers, assign, vehicle=None)
    cost, problems = evaluate_B(d, assign)

`problems` is an empty list for a feasible solution, otherwise it lists what is violated.
Use the checkers to verify your solutions; the reference model in the task handout is what is graded.
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
    """Instance data for Checkpoints 3 and 4.
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


# ------------------------------------------------------------------ checkers
def latency_B(u):
    """Piecewise-linear latency of one server at CPU utilisation u in [0, 1]:
    max over the tangents of u/(1-u) at the points TANGENT_POINTS."""
    return max(u * (1.0 / (1.0 - um) ** 2) - um ** 2 / (1.0 - um) ** 2 for um in TANGENT_POINTS)


def evaluate_B(d, assign, tol=1e-9):
    """assign[k] = server (0..S-1) of virtual machine k. Returns (objective, list of violated conditions)."""
    S, M = d['S'], d['M']
    problems = []
    if len(assign) != M:
        return float('nan'), ['assign must have one entry per virtual machine']
    cpu = np.zeros(S); mem = np.zeros(S)
    for k in range(M):
        l = int(assign[k])
        if not 0 <= l < S:
            problems.append(f'VM {k}: server {l} does not exist'); continue
        cpu[l] += d['cpu_req'][k]; mem[l] += d['mem_req'][k]
        if d['variant'] == 2 and d['vm_required_zone'][k] != -1 and d['zone'][l] != d['vm_required_zone'][k]:
            problems.append(f'VM {k}: wrong zone')
        if d['variant'] == 3 and d['needs_gpu'][k] == 1 and d['is_gpu_server'][l] == 0:
            problems.append(f'VM {k}: needs a GPU server')
    for l in range(S):
        if cpu[l] > d['cpu_cap'][l] + tol: problems.append(f'server {l}: CPU {cpu[l]:.0f} > {d["cpu_cap"][l]}')
        if mem[l] > d['mem_cap'][l] + tol: problems.append(f'server {l}: memory {mem[l]:.0f} > {d["mem_cap"][l]}')
    cost = sum(latency_B(min(cpu[l] / d['cpu_cap'][l], 1.0)) for l in range(S))
    return float(cost), problems


def evaluate_A(d, variant, centers, assign, vehicle=None, tol=1e-9):
    """Storyline A.
    centers : list of the V distinct point indices chosen as centers
    assign  : list of N entries; assign[i] is the point index of the center that serves point i
              (it must be one of `centers`); -1 means 'not served' and is allowed only for premium points in variant 2
    vehicle : variant 3 only; vehicle[c] is the vehicle (0..V-1) placed at centers[c]; a permutation of range(V)
    Returns (objective, list of violated conditions)."""
    N, V = d['N'], d['V']
    problems = []
    centers = [int(c) for c in centers]
    if len(set(centers)) != V:
        problems.append(f'need exactly {V} distinct centers, got {len(set(centers))}')
    if len(assign) != N:
        return float('nan'), ['assign must have one entry per point']
    cset = set(centers)
    load = {c: 0.0 for c in cset}
    cost = 0.0
    for i in range(N):
        a = int(assign[i])
        if a == -1:
            if variant == 2 and d['is_premium'][i] == 1 and i not in cset:
                cost += d['penalty'][i]
            else:
                problems.append(f'point {i} is not served and may not be dropped')
            continue
        if a not in cset:
            problems.append(f'point {i}: assigned to {a}, which is not a center'); continue
        load[a] += d['demand'][i]
        cost += d['dist'][i, a]
    for c in cset:
        if int(assign[c]) != c:
            problems.append(f'center {c} must serve itself')
    if variant in (1, 2):
        cost += 2.0 * sum(d['D'][c] for c in cset)
        for c in cset:
            if load[c] > d['Cap'] + tol: problems.append(f'center {c}: load {load[c]:.0f} > Cap {d["Cap"]}')
    else:
        if vehicle is None or sorted(int(v) for v in vehicle) != list(range(V)) or len(vehicle) != len(centers):
            problems.append('vehicle must be a permutation of 0..V-1, one vehicle per center (same order as centers)')
        else:
            for c, v in zip(centers, vehicle):
                cost += 2.0 * d['vehicle_cost_per_km'][int(v)] * d['D'][c]
                if load[c] > d['vehicle_cap'][int(v)] + tol:
                    problems.append(f'center {c}: load {load[c]:.0f} > capacity {d["vehicle_cap"][int(v)]} of vehicle {int(v)}')
    return float(cost), problems
