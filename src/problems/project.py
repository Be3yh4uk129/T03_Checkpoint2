"""Project block (Part C): unconstrained relaxation of the team's storyline.

Owner: M3 (team of four).  Everything here comes from the team's task file.

Storyline A - placing ONE hub.  Checkpoint 1 assigned delivery points to vehicles
under capacity.  Here the capacities and the assignment are dropped and we ask a
simpler question: where would a single hub serve all points best?

    F(x) = (1 / sum_i d_i) * sum_i d_i * sqrt(||x - p_i||^2 + delta^2),   delta = 1

F is the average distance travelled per unit of demand; the smoothing delta removes
the kink at the data points.  Start x0 = warehouse, relative stopping rule.
Check values from the task file:  F(x0) = 36.2588,  ||grad F(x0)|| = 0.088782.

Derivatives (by hand).  Write r_i = sqrt(||x - p_i||^2 + delta^2) and W = sum_i d_i.

    F(x)      = (1/W) sum_i d_i r_i
    grad F(x) = (1/W) sum_i d_i (x - p_i) / r_i
    H(x)      = (1/W) sum_i d_i [ I / r_i - (x - p_i)(x - p_i)^T / r_i^3 ]

H is positive definite everywhere: for any v,

    v^T H_i v = (1/r_i^3) ( ||v||^2 r_i^2 - ((x - p_i).v)^2 )
              >= (1/r_i^3) ( ||v||^2 (||x-p_i||^2 + delta^2) - ||v||^2 ||x-p_i||^2 )
              =  delta^2 ||v||^2 / r_i^3  >  0        (Cauchy-Schwarz)

so F is strictly convex and has a unique minimiser - which is why Newton is safe
here (see S5).

Data: logistics_variant(3, 1) from the Checkpoint 1 generator, embedded verbatim so
that this module is reproducible on its own.  The variant extras (premium penalties,
vehicle capacities and costs) are not used at this checkpoint.
"""
import numpy as np
from .base import Problem

DELTA = 1.0

POINTS = np.array([
    [9.412864224039918, 43.31269402364738],
    [47.9051298140834, 15.973891463707856],
    [73.45771514092145, 11.367201992140341],
    [39.1228190495662, 51.674018262136364],
    [43.06280204141778, 58.679857143814075],
    [73.78377872921602, 95.62672548360986],
    [28.420116374879147, 64.85472070798251],
    [69.62159966701554, 29.27207490124871],
    [0.14900835088361708, 97.34602747664127],
    [29.840122301687565, 31.39860020343368],
    [89.17110704451572, 58.516293989090805],
    [47.130966518183136, 77.32770096488164],
    [3.0346007662471197, 70.69650956556235],
    [37.424383347847076, 9.085271350425783],
    [66.05000674278948, 93.14638547413546],
    [20.719116808100125, 63.0090199785343],
    [29.816309065742473, 74.17566800693304],
    [72.21648081421175, 21.871542456880455],
    [82.98868742743123, 65.76522108732432],
    [68.27989078603503, 82.0075750170535],
    [42.857290429846195, 75.8705461154919],
    [87.84801846662539, 10.23199219220744],
    [84.97683374661537, 39.39273326323352],
    [47.96839235122749, 14.633456975819847],
    [69.84263449470936, 29.197861598785103],
    [87.11391497935891, 27.53743769480771],
    [56.18097187308899, 39.96562211304527],
    [61.290949190243914, 19.66392397721237],
    [18.02875408430952, 74.68603856498379],
    [75.22234183692774, 56.697787445291226],
    [92.10796772829299, 20.577504822901883],
    [85.09011245916514, 16.898731132708622],
    [96.43577208942796, 62.36927281061517],
    [60.68837880100148, 97.05587631326237],
    [78.7032713788771, 78.99174760885543],
])
DEMAND = np.array([18, 29, 15, 23, 20, 15, 16, 19, 17, 25, 18, 24, 27, 22, 16, 15, 19, 21, 22, 19, 27, 23, 29, 29, 25, 15, 18, 26, 23, 29, 25, 27, 15, 19, 25], dtype=float)
WAREHOUSE = np.array([56.02548930412794, 51.643240721287356])

def project_f(x, delta=DELTA, points=None, demand=None):
    P = POINTS if points is None else points
    d = DEMAND if demand is None else demand
    r = np.sqrt(np.sum((x - P) ** 2, axis=1) + delta ** 2)
    return float(np.dot(d, r) / d.sum())


def project_grad(x, delta=DELTA, points=None, demand=None):
    P = POINTS if points is None else points
    d = DEMAND if demand is None else demand
    diff = x - P                                     # (N, 2)
    r = np.sqrt(np.sum(diff ** 2, axis=1) + delta ** 2)
    return (d / r) @ diff / d.sum()


def project_hess(x, delta=DELTA, points=None, demand=None):
    P = POINTS if points is None else points
    d = DEMAND if demand is None else demand
    diff = x - P
    r = np.sqrt(np.sum(diff ** 2, axis=1) + delta ** 2)
    w = d / r                                        # d_i / r_i
    H = np.eye(2) * w.sum()
    H -= np.einsum("i,ij,ik->jk", d / r ** 3, diff, diff)
    return H / d.sum()


def weighted_centroid(points=None, demand=None):
    """Minimiser of the demand-weighted SQUARED-distance cost (Checkpoint 1 stand-in).

    Closed form; used in S5 to compare with the true minimiser of F."""
    P = POINTS if points is None else points
    d = DEMAND if demand is None else demand
    return (d @ P) / d.sum()


def make_project(delta=DELTA) -> Problem:
    """The project block: start at the warehouse, relative stopping rule."""
    return Problem(
        name="project" if delta == DELTA else f"project(delta={delta:g})",
        f=lambda x: project_f(x, delta),
        grad=lambda x: project_grad(x, delta),
        hess=lambda x: project_hess(x, delta),
        x0=WAREHOUSE.copy(),
        rule="rel",
        tol=1e-6,
        x_star=None,
    )
