"""Checkpoint 3: regenerate Tables 1, 2 and 3 and every number quoted in the report.

    PYTHONPATH=src python -m experiments.run_all_cp3

Writes into results/:
    cp3_table1.csv          model sizes, check values, z_LP, solver, time, fractional entries
    cp3_table2.csv          the five optimality conditions (a) to (e)
    cp3_table3.csv          left slope, dual price, right slope of the designated row
    cp3_active_rows.csv     tight rows per constraint family (Task 5)
    cp3_fractional.csv      the fractional entries one by one (Task 2)

The whole run takes a few seconds. The name differs from the Checkpoint 2
runner so that both checkpoints stay reproducible from the same repository.
"""
import csv
import os
import platform
import sys

import numpy as np

from model import build_matrix_form, instance_summary, load_instance
from model.instance import CHECK_VALUES
from lp.solve import fractional_report, solve_highs, solve_pulp_cbc
from kkt.checker import active_rows, check_conditions
from kkt.convention import sign_convention_test
from sens.perturb import interval_check, perturb_row

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")
TOL = 1e-6


def _write(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote results/{name}")


def main():
    print("Checkpoint 3, team T03, storyline A variant 1, seed 3")

    # ---------------------------------------------------------------- Task 1
    d = load_instance()
    mf = build_matrix_form(d)
    got = instance_summary(d)

    table1 = []
    for key, label in (
        ("N", "N (points)"),
        ("V", "V (centers)"),
        ("Cap", "Cap"),
        ("sum_demand", "sum of demand"),
        ("sum_dist", "sum_i sum_k d_ik"),
        ("n_variables", "number of variables"),
        ("n_rows", "number of constraint rows"),
    ):
        value = got[key]
        expected = CHECK_VALUES[key]
        if key == "sum_dist":
            value, expected = round(value, 3), round(expected, 3)
        table1.append([label, value, expected, "match" if value == expected else "MISMATCH"])

    print("\nTask 1, check values")
    for label, value, expected, verdict in table1:
        print(f"  {label:<28} {value!s:>12}   task file {expected!s:>12}   {verdict}")
    assert all(r[3] == "match" for r in table1), "a check value does not match the task file"

    # ---------------------------------------------------------------- Task 2
    sol = solve_highs(mf)
    z_cbc, cbc_seconds, pulp_version = solve_pulp_cbc(mf)
    fr = fractional_report(mf, sol.x, tol=TOL)

    print("\nTask 2, LP relaxation")
    print(f"  z_LP                 = {sol.z:.6f}")
    print(f"  HiGHS                  {sol.seconds:.3f} s, {sol.n_iterations} iterations")
    print(f"  PuLP {pulp_version} + CBC    {z_cbc:.6f}  ({cbc_seconds:.3f} s), "
          f"relative difference {abs(z_cbc - sol.z) / abs(sol.z):.2e}")
    print(f"  fractional variables = {fr['n_fractional_total']} "
          f"({fr['n_fractional_y']} centers y, {fr['n_fractional_x']} assignments x)")
    print(f"  points split over several centers = {fr['n_points_split']}")

    table1 += [
        ["z_LP (HiGHS)", f"{sol.z:.6f}", "", ""],
        ["z_LP (PuLP " + pulp_version + " + CBC)", f"{z_cbc:.6f}", "", ""],
        ["relative difference of the two solvers", f"{abs(z_cbc - sol.z) / abs(sol.z):.2e}", "", ""],
        ["solver of record", "HiGHS (scipy.optimize.linprog)", "", ""],
        ["solve time, s", f"{sol.seconds:.3f}", "", ""],
        ["simplex iterations", sol.n_iterations, "", ""],
        ["fractional variables, total", fr["n_fractional_total"], "", ""],
        ["fractional centers y_k", fr["n_fractional_y"], "", ""],
        ["fractional assignments x_ik", fr["n_fractional_x"], "", ""],
        ["points served from several centers", fr["n_points_split"], "", ""],
        ["centers open integrally (y_k = 1)", fr["n_open_centers_integral"], "", ""],
        ["python", platform.python_version(), "", ""],
    ]
    _write("cp3_table1.csv", ["quantity", "value", "task file", "verdict"], table1)

    frac_rows = [["y", k, "", f"{v:.9f}"] for k, v in fr["fractional_y"]]
    for i, parts in sorted(fr["split_points"].items()):
        for k, v in parts:
            frac_rows.append(["x", i, k, f"{v:.9f}"])
    _write("cp3_fractional.csv", ["variable", "point i / center k", "center k", "value"], frac_rows)

    # ---------------------------------------------------------------- Task 3
    conv = sign_convention_test()
    out = check_conditions(mf.c, mf.A, mf.b, mf.sense, mf.lo, mf.up, sol.x, sol.pi, tol=TOL)

    print("\nTask 3, optimality conditions (tolerance 1e-6)")
    print(f"  sign convention test: pi = {conv['pi']:+.1f} on a '<=' row, "
          f"{'handout convention, used unchanged' if conv['pi_matches_handout'] else 'needs negation'}")
    labels = {
        "a": "primal feasibility, largest violation",
        "b": "dual feasibility (signs), largest violation",
        "c": "complementary slackness on rows, max |pi_i * slack_i|",
        "d": "reduced costs, largest violation",
        "e": "strong duality, |c^T x - dual objective|",
    }
    table2 = []
    for key in "abcde":
        print(f"  ({key}) {labels[key]:<52} {out[key]:.3e}")
        table2.append([f"({key})", labels[key], f"{out[key]:.3e}", "ok" if out[key] <= TOL else "FAIL"])
    table2.append(["", "primal objective c^T x", f"{out['primal_objective']:.6f}", ""])
    table2.append(["", "dual objective b^T pi + sum min(0, r_j)", f"{out['dual_objective']:.6f}", ""])
    _write("cp3_table2.csv", ["condition", "what is checked", "value", "verdict"], table2)
    assert out["all_satisfied"], "an optimality condition failed"

    # ---------------------------------------------------------------- Task 4
    pert = perturb_row(mf, mf.row_of("A5"))
    chk = interval_check(pert, tol=TOL)

    print("\nTask 4, sensitivity of (A5), b = 13")
    print(f"  z(b - 0.01) = {pert['z_minus']:.6f}")
    print(f"  z(b)        = {pert['z_base']:.6f}")
    print(f"  z(b + 0.01) = {pert['z_plus']:.6f}")
    print(f"  left slope  = {pert['left_slope']:.6f}")
    print(f"  pi          = {pert['pi']:.6f}")
    print(f"  right slope = {pert['right_slope']:.6f}")
    print(f"  left <= pi <= right: {chk['inside']}; degenerate: {chk['degenerate']}")

    table3 = [
        ["designated row", "(A5)  sum_k y_k = V"],
        ["right-hand side b", f"{pert['b']:.2f}"],
        ["perturbation step", f"{pert['step']:.2f}"],
        ["z(b - 0.01)", f"{pert['z_minus']:.6f}"],
        ["z(b)", f"{pert['z_base']:.6f}"],
        ["z(b + 0.01)", f"{pert['z_plus']:.6f}"],
        ["left slope", f"{pert['left_slope']:.6f}"],
        ["dual price pi", f"{pert['pi']:.6f}"],
        ["right slope", f"{pert['right_slope']:.6f}"],
        ["left <= pi <= right", "yes" if chk["inside"] else "NO"],
        ["degenerate at b", "yes" if chk["degenerate"] else "no"],
        ["right slope - left slope", f"{chk['slope_difference']:.3e}"],
    ]
    _write("cp3_table3.csv", ["quantity", "value"], table3)
    assert chk["inside"], "the dual price is outside the slope interval"

    # ---------------------------------------------------------------- Task 5
    mask = active_rows(mf.A, mf.b, mf.sense, sol.x, tol=TOL)
    print("\nTask 5, active rows per family")
    rows = []
    for fam in ("A1", "A2", "A3", "A4", "A5"):
        start, count = mf.row_span[fam]
        n = int(mask[start:start + count].sum())
        share = n / count
        print(f"  {fam}  {n:5d} of {count:5d}   ({share:6.1%})")
        rows.append([fam, n, count, f"{share:.4f}"])
    total = int(mask.sum())
    rows.append(["total", total, mf.n_rows, f"{total / mf.n_rows:.4f}"])
    print(f"  total {total:5d} of {mf.n_rows:5d}")
    _write("cp3_active_rows.csv", ["family", "active rows", "rows", "share"], rows)

    print("\nall tables written to results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
