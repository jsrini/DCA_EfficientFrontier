"""Compute the convex-hull efficient frontier for tax-free (56 strategies + 3 winddowns).

The report's plot_frontier.py uses Pareto dominance to define efficiency; this is defensible for
discrete-strategy choice. The stricter finance-textbook definition is the upper-left convex hull of
the (pain, wealth) point cloud, which excludes Pareto-optimal points that lie below the line
between two other Pareto points.

Reports both Pareto set and convex-hull set for the within-cap (|termpain| <= 30%) region.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_tf_points():
    """56 strategies from tpb + 3 winddowns in tax-free."""
    d = np.load("results/tpb_1980_taxfree.npy", allow_pickle=True).item()
    pts = []
    for nm in d["names"]:
        M = d["data"][nm]["M"]
        medW = float(np.median(M[:, 0])) / 1e6
        tp_mag = abs(float(np.percentile(M[:, 8], 10)) * 100)   # positive magnitude
        pts.append((nm, tp_mag, medW))
    # Winddowns from committed summary rows (results/winddown_summary_rows.tex):
    pts.append(("Wind-down sell 40%",  27.1, 7.90))
    pts.append(("Wind-down sell 50%",  25.1, 7.77))
    pts.append(("Wind-down sell 60%",  23.6, 7.51))
    return pts


def pareto(pts):
    """Points not dominated on both axes (higher wealth better, lower pain better)."""
    return [p for p in pts if not any(
        q[2] >= p[2] and q[1] <= p[1] and (q[2] > p[2] or q[1] < p[1])
        for q in pts if q[0] != p[0])]


def upper_convex_hull(pts):
    """Points on the upper-left convex hull of the pain-wealth cloud (concave frontier).
    Take Pareto set, sort by pain ascending, then walk removing any point that lies below the line
    joining its neighbors (i.e., not a vertex of the concave envelope)."""
    p = sorted(pareto(pts), key=lambda x: (x[1], -x[2]))
    hull = []
    for pt in p:
        while len(hull) >= 2:
            # Check whether removing hull[-1] would leave hull[-2] -> pt above hull[-2] -> hull[-1] -> pt
            a, b, c = hull[-2], hull[-1], pt
            # b is on the hull only if it's above the line a->c at pain=b[1]
            if a[1] == c[1]:
                # degenerate, keep b if wealth is higher
                if b[2] >= max(a[2], c[2]):
                    break
                hull.pop()
                continue
            y_line = a[2] + (c[2] - a[2]) * (b[1] - a[1]) / (c[1] - a[1])
            if b[2] > y_line + 1e-9:
                break
            hull.pop()
        hull.append(pt)
    return hull


def main():
    pts = load_tf_points()
    within = [p for p in pts if p[1] <= 30.0]
    print(f"Tax-free strategies within -30% cap: {len(within)} of {len(pts)}\n")

    par = pareto(within)
    par_sorted = sorted(par, key=lambda x: x[1])
    print(f"Pareto-optimal within cap ({len(par)}):")
    for p in par_sorted:
        print(f"  {p[0]:35s} pain=-{p[1]:5.2f}%  W=${p[2]:.2f}M  W/TP={p[2]/p[1]:.3f}")

    hull = upper_convex_hull(within)
    hull_names = {p[0] for p in hull}
    print(f"\nConvex-hull members ({len(hull)}):")
    for p in hull:
        print(f"  {p[0]:35s} pain=-{p[1]:5.2f}%  W=${p[2]:.2f}M  W/TP={p[2]/p[1]:.3f}")

    dropped = [p for p in par_sorted if p[0] not in hull_names]
    if dropped:
        print(f"\nPareto-optimal but NOT on convex hull ({len(dropped)}):")
        for p in dropped:
            print(f"  {p[0]:35s} pain=-{p[1]:5.2f}%  W=${p[2]:.2f}M")


if __name__ == "__main__":
    main()
