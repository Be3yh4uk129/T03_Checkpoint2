"""Interface of Checkpoint 4 and the thin wrapper around the checker.

Your pair writes ONE function with this signature (SA-I or GA-A owns it; the single fifth member writes it alone).
Put it into src/meta/sa.py (SA) or src/meta/ga.py (GA) and keep the name `solve`.
"""
import checkpoint4_generator_PUBLIC as gen


def solve(d, variant, seed, budget, params=None, start=None):
    """Return a dict with the entries
        'solution' : (centers, assign, vehicle) for storyline A (vehicle=None in variants 1 and 2), assign for storyline B
        'cost'     : cost returned by the checker for 'solution'
        'evals'    : number of evaluations spent; it must equal `budget`
        'feasible' : True exactly when the checker reports no problems
        'trace'    : list of pairs (evaluations, best feasible cost so far), at least every 1000 evaluations
    Rules: use only random.Random(seed) (or numpy.random.default_rng(seed)); never read z_ref or z_LP;
    `start` is optional and unused in Checkpoint 4, but the argument must be accepted."""
    raise NotImplementedError('write your method here')


def check(d, variant, solution):
    """Run the checker of the generator. Returns (cost, list_of_problems)."""
    if d['story'] == 'A':
        centers, assign, vehicle = solution
        return gen.evaluate_A(d, variant, centers, assign, vehicle)
    return gen.evaluate_B(d, solution)
