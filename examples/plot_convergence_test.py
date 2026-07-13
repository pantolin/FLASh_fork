"""
Sweeps test_convergence.py over degrees/mesh sizes and plots the result.

Each configuration is solved in its own subprocess with a fixed
PYTHONHASHSEED (see test_convergence.py's docstring for why).

Reproduces Figure 9 (Section 5.1.1, "Accuracy of the p-FEM discretization").
"""

import json
import os
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from _paths import FIGS_DIR

cutfem_degrees = [1, 2, 3]
pfem_degrees = [6, 8, 10]
all_degrees = cutfem_degrees + pfem_degrees
h_mesh_sizes = [2, 4, 8, 16, 24]

USETEX = shutil.which("latex") is not None

plt.rcParams.update({
    "text.usetex": USETEX,
    "text.latex.preamble": r"\usepackage{amsfonts}",
    "font.family": "serif",
    "font.size": 24,
    "legend.fontsize": 12,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "lines.markersize": 10,
})
if USETEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"

markers = ['o', 's', 'd', 'P', 'X', '*']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
FIGS_DIR.mkdir(exist_ok=True, parents=True)


def solve_point(degree, n_cells):
    env = dict(os.environ, PYTHONHASHSEED="0")
    out = subprocess.run(
        [sys.executable, "test_convergence.py", "--degree", str(degree), "--n-cells", str(n_cells)],
        capture_output=True, text=True, check=True, env=env,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def plot_panel(results, x_values, xlabel, filename):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for i, degree in enumerate(all_degrees):
        errors = np.array(results[degree]["L2_error"])
        linestyle = '-' if degree in cutfem_degrees else 'None'
        ax.loglog(x_values(degree), errors, marker=markers[i], color=colors[i], linestyle=linestyle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$L^2$ error")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / filename), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    print("\n" + "=" * 80)
    print("CONVERGENCE STUDY: CutFEM h-refinement (p=1,2,3) vs. p-FEM single element (p=6,8,10)")
    print("=" * 80)

    results = {degree: {"h": [], "n_dofs": [], "nnz": [], "L2_error": []} for degree in all_degrees}

    for degree in all_degrees:
        kind = "p-FEM" if degree in pfem_degrees else "CutFEM"
        n_cells_list = [1] if degree in pfem_degrees else h_mesh_sizes

        for n_cells in n_cells_list:
            point = solve_point(degree, n_cells)

            results[degree]["h"].append(1.0 / n_cells)
            results[degree]["n_dofs"].append(point["n_dofs"])
            results[degree]["nnz"].append(point["nnz"])
            results[degree]["L2_error"].append(point["L2_error"])

            print(f"  {kind} p={degree:2d} | n_cells={n_cells:3d} | DoFs={point['n_dofs']:7d} | "
                  f"nnz={point['nnz']:8d} | L2 error={point['L2_error']:.3e}")

    plot_panel(
        results,
        lambda d: np.sqrt(np.array(results[d]["n_dofs"])),
        r"$\sqrt{\mathrm{DoFs}}$",
        "convergence_test_result_a.pdf",
    )
    plot_panel(
        results,
        lambda d: np.array(results[d]["h"]),
        r"mesh size $h$",
        "convergence_test_result_b.pdf",
    )
    plot_panel(
        results,
        lambda d: np.array(results[d]["nnz"]),
        "non-zero entries",
        "convergence_test_result_c.pdf",
    )

    fig, ax = plt.subplots(figsize=(9.0, 1.0))
    for i, degree in enumerate(all_degrees):
        label = f"CutFEM $p={degree}$" if degree in cutfem_degrees else f"p-FEM $p={degree}$"
        linestyle = '-' if degree in cutfem_degrees else 'None'
        ax.plot([], [], marker=markers[i], color=colors[i], linestyle=linestyle, label=label)
    ax.axis('off')
    fig.legend(loc='center', frameon=False, ncol=len(all_degrees))
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "convergence_test_legend.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
