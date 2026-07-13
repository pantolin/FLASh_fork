"""
Reproduces Figure 21 (Section 5.2.2, "Lattice wrench"): the h-refinement
study and the proposed method's error against it.
"""

import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from _paths import FIGS_DIR, RESULTS_DIR

PAPER_DIR = RESULTS_DIR / "wrench_paper"

USETEX = shutil.which("latex") is not None

plt.rcParams.update({
"text.usetex": USETEX,
"text.latex.preamble": r"\usepackage{amsfonts}",
"font.family": "serif",
"font.size": 24,
"legend.fontsize": 12,
"xtick.labelsize": 18,
"ytick.labelsize": 18,
"lines.markersize": 10
})
if USETEX:
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
else:
    plt.rcParams["mathtext.fontset"] = "cm"

AMP = r"\&" if USETEX else "&"
markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h', 'x']
colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
FIGS_DIR.mkdir(exist_ok=True, parents=True)


def load(name):
    return json.loads((PAPER_DIR / f"{name}.json").read_text())


def normalized(d):
    fy = d["F_applied"][1]
    return {
        "C": d["compliance"] / fy**2,
        "u": np.array(d["u_neu_mean"]) / abs(fy),
        "n_dofs": d["n_dofs"],
    }


def plot_wrench_h_study():

    ks = [1, 2, 3, 4]
    runs = {k: normalized(load(f"run_refined_window_k{k}_p3_cholesky")) for k in ks}
    ref = normalized(load("run_refined_window_k6_p3_cholesky"))
    rom = normalized(load("run_p8_bddc_rom_stab0.0005"))

    dofs = np.array([runs[k]["n_dofs"] for k in ks])

    err_C = np.array([abs(runs[k]["C"] - ref["C"]) / abs(ref["C"]) for k in ks])
    err_u = np.array([np.linalg.norm(runs[k]["u"] - ref["u"]) / np.linalg.norm(ref["u"]) for k in ks])

    rom_err_C = abs(rom["C"] - ref["C"]) / abs(ref["C"])
    rom_err_u = float(np.linalg.norm(rom["u"] - ref["u"]) / np.linalg.norm(ref["u"]))

    cases = [
        ("a", err_C, rom_err_C, "Compliance relative error"),
        ("b", err_u, rom_err_u, "Displacement relative error"),
    ]

    for tag, err, rom_err, ylabel in cases:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dofs, err, marker=markers[0], mfc='none', color=colors[0])
        ax.plot([rom["n_dofs"]], [rom_err], marker=markers[6], mfc=colors[2], color=colors[2], linestyle='None', markersize=18)
        ax.set_xlabel("Number of DoFs")
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([2e3, 5e3, 1e4, 2e4])
        ax.set_xticklabels([r"$2{\times}10^{3}$", r"$5{\times}10^{3}$", r"$10^{4}$", r"$2{\times}10^{4}$"])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(str(FIGS_DIR / f"wrench_h_study_{tag}_results.pdf"), bbox_inches='tight', pad_inches=0.05)
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.plot([], [], marker=markers[0], mfc='none', color=colors[0], label="$h$-refinement ($p=3$)")
    ax.plot([], [], marker=markers[6], mfc=colors[2], color=colors[2], linestyle='None', markersize=18, label=f"ROM {AMP} stabilization ($p=8$)")
    ax.axis('off')
    fig.legend(loc='center', frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "wrench_h_study_legend.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    plot_wrench_h_study()
