"""
Second-stage plotting for the test_*.py / test_rom_basis.py data files.

Maps each plotting function to the figure it reproduces:
plot_test_solver_comparison -> Figure 16 (Sec. 5.1.3),
plot_test_rom_accuracy -> Figure 15 (Sec. 5.1.2),
plot_test_acceleration_efficiency -> Figure 17 (Sec. 5.1.3),
plot_test_scalability -> Figure 18 (Sec. 5.1.3),
plot_test_fast_assembly_accuracy -> Figure 13 (Sec. 5.1.2),
plot_test_rom_basis(_bis) -> Figure 14 (Sec. 5.1.2).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import h5py
from _paths import FIGS_DIR, RESULTS_DIR, ROM_DATA_DIR


# Use LaTeX text rendering when a TeX installation is available; otherwise
# fall back to matplotlib's mathtext with the Computer Modern font set.
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

# LaTeX needs the ampersand escaped; mathtext must NOT escape it.
AMP = r"\&" if USETEX else "&"
markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h', 'x']
colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
FIGS_DIR.mkdir(exist_ok=True, parents=True)




# --- Plot results from test_solver_comparison.py ---
def plot_test_solver_comparison():

    file_path = RESULTS_DIR / "test_solver_comparison" / "data.h5"
    with h5py.File(file_path, "r") as f:
        bddc_iters = f["bddc_iters"][:]
        pcg_iters = f["pcg_iters"][:]
        bddc_setup_time = f["bddc_setup_time"][:]
        pcg_setup_time = f["pcg_setup_time"][:]
        cholesky_setup_time = f["cholesky_setup_time"][:]
        bddc_solve_time = f["bddc_solve_time"][:]
        pcg_solve_time = f["pcg_solve_time"][:]
        cholesky_solve_time = f["cholesky_solve_time"][:]
        number_of_subdomains = f["number_of_subdomains"][:]

    # Figure 1: Iterations
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(number_of_subdomains,bddc_iters,marker='o',label="BDDC")
    ax.plot(number_of_subdomains,pcg_iters,marker='s',label="PCG")
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Iterations")
    ax.set_xlim(0,700)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_a_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 2: Setup time
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(number_of_subdomains,bddc_setup_time,marker='o')
    ax.plot(number_of_subdomains,pcg_setup_time,marker='s')
    ax.plot(number_of_subdomains,cholesky_setup_time,marker='^')
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Setup Time $(s)$")
    ax.set_xlim(0,700)
    ax.set_ylim(1e-3,1e2)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_b_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 3: Solve time
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(number_of_subdomains,bddc_solve_time,marker='o')
    ax.plot(number_of_subdomains,pcg_solve_time,marker='s')
    ax.plot(number_of_subdomains,cholesky_solve_time,marker='^')
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Solve Time $(s)$")
    ax.set_xlim(0,700)
    ax.set_ylim(1e-3,1e2)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_c_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 4: Total time
    bddc_total=bddc_setup_time+bddc_solve_time
    pcg_total=pcg_setup_time+pcg_solve_time
    cholesky_total=cholesky_setup_time+cholesky_solve_time
    fig,ax=plt.subplots(figsize=(8,6))
    l1,=ax.plot(number_of_subdomains,bddc_total,marker='o')
    l2,=ax.plot(number_of_subdomains,pcg_total,marker='s')
    l3,=ax.plot(number_of_subdomains,cholesky_total,marker='^')
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Total Time $(s)$")
    ax.set_xlim(0,700)
    ax.set_ylim(1e-3,1e2)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_d_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Single horizontal legend
    fig_legend=plt.figure(figsize=(6,0.5))
    fig_legend.legend([l1,l2,l3],["BDDC","PCG","Cholesky"],loc="center",ncol=3,frameon=False)
    plt.savefig(str(FIGS_DIR / "test_1_legend.pdf"),bbox_inches="tight", pad_inches=0)
    plt.close()

# --- Plot results from test_solver_comparison_bis.py (adds algebraic multigrid) ---
def plot_test_solver_comparison_bis():

    file_path = RESULTS_DIR / "test_solver_comparison_bis" / "data.h5"
    with h5py.File(file_path, "r") as f:
        bddc_iters = f["bddc_iters"][:]
        amg_iters = f["amg_iters"][:]
        gamg_iters = f["gamg_iters"][:]
        bddc_setup_time = f["bddc_setup_time"][:]
        amg_setup_time = f["amg_setup_time"][:]
        gamg_setup_time = f["gamg_setup_time"][:]
        cholesky_setup_time = f["cholesky_setup_time"][:]
        bddc_solve_time = f["bddc_solve_time"][:]
        amg_solve_time = f["amg_solve_time"][:]
        gamg_solve_time = f["gamg_solve_time"][:]
        cholesky_solve_time = f["cholesky_solve_time"][:]
        number_of_subdomains = f["number_of_subdomains"][:]

    # Figure 1: Iterations
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(number_of_subdomains,bddc_iters,marker='o',label="BDDC")
    ax.plot(number_of_subdomains,amg_iters,marker='s',label="SOR")
    ax.plot(number_of_subdomains,gamg_iters,marker='v',label="AMG")
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Iterations")
    ax.set_xlim(0,700)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_a_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 2: Setup time
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(number_of_subdomains,bddc_setup_time,marker='o')
    ax.plot(number_of_subdomains,amg_setup_time,marker='s')
    ax.plot(number_of_subdomains,gamg_setup_time,marker='v')
    ax.plot(number_of_subdomains,cholesky_setup_time,marker='^')
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Setup Time $(s)$")
    ax.set_xlim(0,700)
    ax.set_ylim(1e-3,1e2)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_b_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 3: Solve time
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(number_of_subdomains,bddc_solve_time,marker='o')
    ax.plot(number_of_subdomains,amg_solve_time,marker='s')
    ax.plot(number_of_subdomains,gamg_solve_time,marker='v')
    ax.plot(number_of_subdomains,cholesky_solve_time,marker='^')
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Solve Time $(s)$")
    ax.set_xlim(0,700)
    ax.set_ylim(1e-3,1e2)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_c_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 4: Total time
    bddc_total=bddc_setup_time+bddc_solve_time
    amg_total=amg_setup_time+amg_solve_time
    gamg_total=gamg_setup_time+gamg_solve_time
    cholesky_total=cholesky_setup_time+cholesky_solve_time
    fig,ax=plt.subplots(figsize=(8,6))
    l1,=ax.plot(number_of_subdomains,bddc_total,marker='o')
    l2,=ax.plot(number_of_subdomains,amg_total,marker='s')
    l4,=ax.plot(number_of_subdomains,gamg_total,marker='v')
    l3,=ax.plot(number_of_subdomains,cholesky_total,marker='^')
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Total Time $(s)$")
    ax.set_xlim(0,700)
    ax.set_ylim(1e-3,1e2)
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_1_d_results.pdf"),bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Single horizontal legend
    fig_legend=plt.figure(figsize=(8,0.5))
    fig_legend.legend([l1,l2,l4,l3],["BDDC","SOR","AMG","Cholesky"],loc="center",ncol=4,frameon=False)
    plt.savefig(str(FIGS_DIR / "test_1_legend.pdf"),bbox_inches="tight", pad_inches=0)
    plt.close()

# --- Plot results from test_rom_accuracy.py ---
def plot_test_rom_accuracy():

    file_path = RESULTS_DIR / "test_rom_accuracy" / "data.h5"
    with h5py.File(file_path, "r") as f:
        iterations = f["iterations"][:]
        rom_iterations = f["rom_iterations"][:]
        stab_errors = f["stab_errors"][:]
        rom_errors = f["rom_errors"][:]
        total_errors = f["total_errors"][:]
        number_of_subdomains = f["number_of_subdomains"][:]
        stabilizations = f["stabilizations"][:]

    def format_stab_label(stab_value, name="stab"):
        if stab_value == 0:
            return r"$\rho=0$"
        exponent = int(f"{stab_value:e}".split('e')[1])
        mantissa = stab_value / (10**exponent)
        return fr"$\rho={mantissa:.0f} \cdot 10^{{{exponent}}}$"

    # Figure 1: Iterations
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, stab in enumerate(stabilizations):
        ax.plot(number_of_subdomains,     iterations[:, idx], color=colors[idx], marker=markers[idx], mfc='none', label=format_stab_label(stab), linestyle='-')
        ax.plot(number_of_subdomains, rom_iterations[:, idx], color=colors[idx], marker=markers[idx], mfc='none', label=format_stab_label(stab) + f" ROM ", linestyle='--')
    ax.set_xlabel("Number of subdomains")
    ax.set_xlim(0,700)
    ax.set_ylabel("Iterations")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_2_a_results.pdf"), bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 2: Stabilization Error
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, stab in enumerate(stabilizations[1:], start=1):
        ax.plot(number_of_subdomains, stab_errors[:, idx], color=colors[idx], marker=markers[idx], mfc='none', label=format_stab_label(stab), linestyle='-')
    ax.set_xlabel("Number of subdomains")
    ax.set_xlim(0,700)
    ax.set_ylabel("$L^2$ error")
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_2_b_results.pdf"), bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 3: ROM error
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, stab in enumerate(stabilizations):
        ax.plot(number_of_subdomains, rom_errors[:, idx], color=colors[idx], marker=markers[idx], mfc='none', label=format_stab_label(stab)+ f" ROM ", linestyle='--')
    ax.set_xlabel("Number of subdomains")
    ax.set_xlim(0,700)
    ax.set_ylabel("$L^2$ error")
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_2_c_results.pdf"), bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 4: Total Error
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, stab in enumerate(stabilizations):
        ax.plot(number_of_subdomains, total_errors[:, idx], color=colors[idx], marker=markers[idx], mfc='none', label=format_stab_label(stab)+ f" ROM ", linestyle='--')
    ax.set_xlabel("Number of subdomains")
    ax.set_xlim(0,700)
    ax.set_ylabel("$L^2$ error")
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_2_d_results.pdf"), bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure: legend a
    fig, ax = plt.subplots(figsize=(8, 1))
    for idx, stab in enumerate(stabilizations):
        ax.plot([], [], color=colors[idx], marker=markers[idx], mfc='none', label=format_stab_label(stab))
    legend = fig.legend(loc='center', frameon=False, ncol=len(stabilizations))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_2_legend_a.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Figure: legend b
    fig, ax  = plt.subplots(figsize=(6, 1))
    ax.plot([], [], color='none',  linestyle='None', label='Linestyle:')
    ax.plot([], [], color='black', linewidth=1, linestyle='-',  label='only stabilization')
    ax.plot([], [], color='black', linewidth=1, linestyle='--', label='stabilization and ROM')
    legend = fig.legend(loc='center', frameon=False, ncol=5)
    ax.axis('off')
    plt.tight_layout()
    #plt.savefig("figs/test_2_legend_b.pdf", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# --- Plot results from test_3.py ---
def plot_test_3():
    file_path = RESULTS_DIR / "test_3" / "data.h5"
    with h5py.File(file_path, "r") as f:
        number_of_subdomains = f["number_of_subdomains"][:]
        errors_bas_vs_basws = f["errors_bas_vs_basws"][:]
        errors_fa_vs_faws = f["errors_fa_vs_faws"][:]
        errors_rom_vs_romws = f["errors_rom_vs_romws"][:]
        errors_bas_vs_fa = f["errors_bas_vs_fa"][:]
        errors_fa_vs_rom = f["errors_fa_vs_rom"][:]
        errors_bas_vs_rom = f["errors_bas_vs_rom"][:]
        errors_bas_vs_romws = f["errors_bas_vs_romws"][:]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Baseline+WS, FA+WS, ROM+WS
    axs[0].plot(number_of_subdomains, errors_bas_vs_basws, marker='o', label="Baseline vs Baseline+WS")
    axs[0].plot(number_of_subdomains, errors_fa_vs_faws, marker='s', label="FA vs FA+WS")
    axs[0].plot(number_of_subdomains, errors_rom_vs_romws, marker='^', label="ROM vs ROM+WS")
    axs[0].set_xlabel("Number of subdomains")
    axs[0].set_ylabel("Error")
    axs[0].set_title("WS Variants")
    axs[0].set_yscale('log')
    axs[0].grid(True)
    axs[0].legend()

    # Subplot 2: Baseline vs FA, FA vs ROM, Baseline vs ROM
    axs[1].plot(number_of_subdomains, errors_bas_vs_fa, marker='o', label="Baseline vs FA")
    axs[1].plot(number_of_subdomains, errors_fa_vs_rom, marker='s', label="FA vs ROM")
    axs[1].plot(number_of_subdomains, errors_bas_vs_rom, marker='^', label="Baseline vs ROM")
    axs[1].set_xlabel("Number of subdomains")
    axs[1].set_ylabel("Error")
    axs[1].set_title("No WS")
    axs[1].set_yscale('log')
    axs[1].grid(True)
    axs[1].legend()

    # Subplot 3: Baseline vs ROM+WS
    axs[2].plot(number_of_subdomains, errors_bas_vs_romws, marker='d', label="Baseline vs ROM+WS")
    axs[2].set_xlabel("Number of subdomains")
    axs[2].set_ylabel("Error")
    axs[2].set_title("Baseline vs ROM+WS")
    axs[2].set_yscale('log')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.suptitle("Test 3: Error Analysis", y=1.04)
    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(FIGS_DIR / "test_3_results.pdf"), bbox_inches='tight')
    plt.close()

# --- Plot results from test_acceleration_efficiency.py ---
def plot_test_acceleration_efficiency():
    file_path = RESULTS_DIR / "test_acceleration_efficiency" / "data.h5"
    with h5py.File(file_path, "r") as f:
        number_of_subdomains = f["number_of_subdomains"][:]
        errors_bas_vs_basws = f["errors_bas_vs_basws"][:]
        errors_baws_vs_faws = f["errors_baws_vs_faws"][:]
        errors_faws_vs_romws = f["errors_faws_vs_romws"][:]
        errors_bas_vs_romws = f["errors_bas_vs_romws"][:]
        bas_iters = f["bas_iters"][:]
        basws_iters = f["basws_iters"][:]
        faws_iters = f["faws_iters"][:]
        romws_iters = f["romws_iters"][:]
        bas_setup_time = f["bas_setup_time"][:]
        romws_setup_time = f["romws_setup_time"][:]
        bas_assemble_time = f["bas_assemble_time"][:]
        romws_assemble_time = f["romws_assemble_time"][:]
        bas_solve_time = f["bas_solve_time"][:]
        romws_solve_time = f["romws_solve_time"][:]

    # Figure 1: Iterations
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(number_of_subdomains, bas_iters  , color=colors[0], marker=markers[0], mfc='none', label="Baseline")
    ax.plot(number_of_subdomains, romws_iters, color=colors[2], marker=markers[2], mfc='none', label=f"ROM {AMP} stabilization")
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Iterations")
    ax.set_xlim(0,700)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_4_a_results.pdf"), bbox_inches='tight', pad_inches=0.00)
    plt.close()

    # Figure 2: Time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(number_of_subdomains, bas_setup_time + bas_assemble_time, marker=markers[0], mfc=colors[0], color=colors[0], label="Baseline, setup and assemble")
    ax.plot(number_of_subdomains, bas_setup_time + bas_assemble_time + bas_solve_time, marker=markers[0], mfc='none', color=colors[0], linestyle='--', label="Baseline, total")
    ax.plot(number_of_subdomains, romws_setup_time + romws_assemble_time, marker=markers[2], mfc=colors[2], color=colors[2], label=f"ROM {AMP} stabilization, setup and assemble")
    ax.plot(number_of_subdomains, romws_setup_time + romws_assemble_time + romws_solve_time, marker=markers[2], mfc='none', color=colors[2], linestyle='--', label=f"ROM {AMP} stabilization, total")
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Time $(s)$")
    ax.set_xlim(0,700)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_4_b_results.pdf"), bbox_inches='tight', pad_inches=0.00)
    plt.close()

# --- Plot results from test_scalability.py ---
def plot_test_scalability():
    file_path = RESULTS_DIR / "test_scalability" / "data.h5"
    with h5py.File(file_path, "r") as f:
        number_of_subdomains = f["number_of_subdomains"][:]
        iters = f["iters"][:]
        setup_time = f["setup_time"][:]
        assemble_time = f["assemble_time"][:]
        solve_time = f["solve_time"][:]

    # Figure 1: Iterations
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(number_of_subdomains, iters, marker='o', label="Iterations")
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Iterations")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_5_a_results.pdf"), bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Figure 2: Setup, Assemble, Solve, and Total Time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(number_of_subdomains, setup_time+assemble_time+solve_time, marker=markers[0], label="Total time")
    ax.plot(number_of_subdomains, setup_time, marker=markers[1], label="Assemble time")
    ax.plot(number_of_subdomains, assemble_time, marker=markers[2], label="Setup time")
    ax.plot(number_of_subdomains, solve_time, marker=markers[3], label="Solve time")
    ax.set_xlabel("Number of subdomains")
    ax.set_ylabel("Time $(s)$")
    #ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_5_b_results.pdf"), bbox_inches='tight', pad_inches=0.05)
    plt.close()

# --- Plot results from test_fast_assembly_accuracy.py ---
def plot_test_fast_assembly_accuracy():
    file_path = RESULTS_DIR / "test_fast_assembly_accuracy" / "data.h5"
    with h5py.File(file_path, "r") as f:
        number_of_subdomains = f["number_of_subdomains"][:]
        fa_degrees = f["fa_degrees"][:]
        errors = f["errors"][:]
        iterations = f["iterations"][:]
        fa_iterations = f["fa_iterations"][:]

    fa_degrees = fa_degrees[0:-1]

    # Subplot 1: Error vs. Number of subdomains for each fa_degree
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, degree in enumerate(fa_degrees):
        ax.plot(number_of_subdomains, errors[:, idx], marker=markers[idx], mfc='none', color=colors[idx], label=f"$p={int(degree)}$")
    ax.set_xlabel("Number of subdomains")
    ax.set_xlim(0,700)
    ax.set_ylabel("$L^2$ Error")
    ax.set_yscale('log')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_6_a_results.pdf"), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Subplot 2: Iterations vs. Number of subdomains for baseline and each fa_degree
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, degree in enumerate(fa_degrees):
        ax.plot(number_of_subdomains, fa_iterations[:, idx],  marker=markers[idx], mfc='none',  color=colors[idx], label=f"$p={int(degree)}$")
    ax.plot(number_of_subdomains, iterations, linewidth=3, marker='', linestyle='--', color=colors[-1], label="Baseline")
    ax.set_xlabel("Number of subdomains")
    ax.set_xlim(0,700)
    ax.set_ylabel("Number of iterations")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_6_b_results.pdf"), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Legend
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.plot([], [], linewidth=3, marker='', linestyle='--', color=colors[-1], label="Baseline")
    for idx, degree in enumerate(fa_degrees):
        ax.plot([], [],  marker=markers[idx], mfc='none',  color=colors[idx], label=f"$p={int(degree)}$")
    legend = fig.legend(loc='center', frameon=False, ncol=len(fa_degrees)+1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "test_6_legend.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# --- Plot results from test_rom_basis.py ---
def plot_test_rom_basis():
    """
    Plot the error decay as a function of basis size for each n in the ROM basis test
    for both schoen_iwp_3 (parameter space ℙ¹) and schoen_iwp_4 (parameter space ℙ²).
    """
    import os

    # Subplot 1: Schwarz Diamond
    paths = {"P1 (Schwarz Diamond)": ROM_DATA_DIR / "schwarz_diamond_3" / "K_core" / "error_data.h5",
             "P2 (Schwarz Diamond)": ROM_DATA_DIR / "schwarz_diamond_4" / "K_core" / "error_data.h5",}
    fig, ax = plt.subplots(figsize=(8, 6))
    (label, file_path) = list(paths.items())[0]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
        indices = np.linspace(0, len(errors[0]) - 1, 10, dtype=int)
        indices = np.unique(np.append(indices, 10 - 1))  # guarantee last point
        ax.plot(basis_number[0], errors[0], marker=markers[0], mfc=colors[0], color=colors[0], linestyle='-', markevery=indices)
        ax.plot(basis_number[1], errors[1], marker=markers[1], mfc=colors[1], color=colors[1], linestyle='-', markevery=indices)
    (label, file_path) = list(paths.items())[1]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
    ax.plot(basis_number[0], errors[0], marker=markers[0], mfc='white'   , color=colors[0], linestyle='-', markevery=indices)
    ax.plot(basis_number[1], errors[1], marker=markers[1], mfc='white'   , color=colors[1], linestyle='-', markevery=indices)
    ax.set_xlabel("Basis size")
    ax.set_ylabel(r"L$_\infty$ error")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e0)
    ax.grid(True)
    plt.tight_layout()
    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(FIGS_DIR / "rom_basis_test_results_a.pdf"), bbox_inches="tight")
    plt.close()



    # Subplot 2: Schoen IWP
    paths = {"P1 (Schoen IWP)": ROM_DATA_DIR / "schoen_iwp_3" / "K_core" / "error_data.h5",
             "P2 (Schoen IWP)": ROM_DATA_DIR / "schoen_iwp_4" / "K_core" / "error_data.h5",}
    fig, ax = plt.subplots(figsize=(8, 6))
    (label, file_path) = list(paths.items())[0]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
        indices = np.linspace(0, len(errors[0]) - 1, 10, dtype=int)
        indices = np.unique(np.append(indices, 10 - 1))  # guarantee last point
        ax.plot(basis_number[0], errors[0], marker=markers[0], mfc=colors[0], color=colors[0], linestyle='-', markevery=indices)
        ax.plot(basis_number[1], errors[1], marker=markers[1], mfc=colors[1], color=colors[1], linestyle='-', markevery=indices)
    (label, file_path) = list(paths.items())[1]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
    ax.plot(basis_number[0], errors[0], marker=markers[0], mfc='white'   , color=colors[0], linestyle='-', markevery=indices)
    ax.plot(basis_number[1], errors[1], marker=markers[1], mfc='white'   , color=colors[1], linestyle='-', markevery=indices)
    ax.set_xlabel("Basis size")
    ax.set_ylabel(r"L$_\infty$ error")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e0)
    ax.grid(True)
    plt.tight_layout()
    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(FIGS_DIR / "rom_basis_test_results_b.pdf"), bbox_inches="tight")
    plt.close()

    # Subplot 3: Legend
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.plot([], [], marker=markers[0], mfc=colors[0], color=colors[0], linestyle='-',label=r"$\mathbb{P}_1 \quad n_k=1$")
    ax.plot([], [], marker=markers[0], mfc='white',   color=colors[0], linestyle='-',label=r"$\mathbb{P}_2 \quad n_k=1$")
    ax.plot([], [], marker=markers[1], mfc=colors[1], color=colors[1], linestyle='-',label=r"$\mathbb{P}_1 \quad n_k=2$")
    ax.plot([], [], marker=markers[1], mfc='white',   color=colors[1], linestyle='-',label=r"$\mathbb{P}_2 \quad n_k=2$")
    ax.axis('off')
    legend = fig.legend(loc='center', frameon=False, ncol=4)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "rom_basis_test_legend.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# --- Plot results from test_rom_basis_bis.py ---
def plot_test_rom_basis_bis():
    """
    Plot the error decay as a function of basis size for each n in the ROM basis test
    for both schoen_frd (subplot c) and schwarz_primitive (subplot d), reading the
    error_data_bis.h5 files produced by test_rom_basis_bis.py.
    """

    # Subplot 3: Schoen FRD
    paths = {"P1 (Schoen FRD)": ROM_DATA_DIR / "schoen_frd_3" / "K_core" / "error_data_bis.h5",
             "P2 (Schoen FRD)": ROM_DATA_DIR / "schoen_frd_4" / "K_core" / "error_data_bis.h5",}
    fig, ax = plt.subplots(figsize=(8, 6))
    (label, file_path) = list(paths.items())[0]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
        indices = np.linspace(0, len(errors[0]) - 1, 10, dtype=int)
        indices = np.unique(np.append(indices, 10 - 1))  # guarantee last point
        ax.plot(basis_number[0], errors[0], marker=markers[0], mfc=colors[0], color=colors[0], linestyle='-', markevery=indices)
        ax.plot(basis_number[1], errors[1], marker=markers[1], mfc=colors[1], color=colors[1], linestyle='-', markevery=indices)
    (label, file_path) = list(paths.items())[1]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
    ax.plot(basis_number[0], errors[0], marker=markers[0], mfc='white'   , color=colors[0], linestyle='-', markevery=indices)
    ax.plot(basis_number[1], errors[1], marker=markers[1], mfc='white'   , color=colors[1], linestyle='-', markevery=indices)
    ax.set_xlabel("Basis size")
    ax.set_ylabel(r"L$_\infty$ error")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e0)
    ax.grid(True)
    plt.tight_layout()
    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(FIGS_DIR / "rom_basis_test_results_c.pdf"), bbox_inches="tight")
    plt.close()



    # Subplot 4: Schwarz Primitive
    paths = {"P1 (Schwarz Primitive)": ROM_DATA_DIR / "schwarz_primitive_3" / "K_core" / "error_data_bis.h5",
             "P2 (Schwarz Primitive)": ROM_DATA_DIR / "schwarz_primitive_4" / "K_core" / "error_data_bis.h5",}
    fig, ax = plt.subplots(figsize=(8, 6))
    (label, file_path) = list(paths.items())[0]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
        indices = np.linspace(0, len(errors[0]) - 1, 10, dtype=int)
        indices = np.unique(np.append(indices, 10 - 1))  # guarantee last point
        ax.plot(basis_number[0], errors[0], marker=markers[0], mfc=colors[0], color=colors[0], linestyle='-', markevery=indices)
        ax.plot(basis_number[1], errors[1], marker=markers[1], mfc=colors[1], color=colors[1], linestyle='-', markevery=indices)
    (label, file_path) = list(paths.items())[1]
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]
    ax.plot(basis_number[0], errors[0], marker=markers[0], mfc='white'   , color=colors[0], linestyle='-', markevery=indices)
    ax.plot(basis_number[1], errors[1], marker=markers[1], mfc='white'   , color=colors[1], linestyle='-', markevery=indices)
    ax.set_xlabel("Basis size")
    ax.set_ylabel(r"L$_\infty$ error")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e0)
    ax.grid(True)
    plt.tight_layout()
    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(FIGS_DIR / "rom_basis_test_results_d.pdf"), bbox_inches="tight")
    plt.close()

    # Legend (same as plot_test_rom_basis, regenerated so this function is standalone)
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.plot([], [], marker=markers[0], mfc=colors[0], color=colors[0], linestyle='-',label=r"$\mathbb{P}_1 \quad n_k=1$")
    ax.plot([], [], marker=markers[0], mfc='white',   color=colors[0], linestyle='-',label=r"$\mathbb{P}_2 \quad n_k=1$")
    ax.plot([], [], marker=markers[1], mfc=colors[1], color=colors[1], linestyle='-',label=r"$\mathbb{P}_1 \quad n_k=2$")
    ax.plot([], [], marker=markers[1], mfc='white',   color=colors[1], linestyle='-',label=r"$\mathbb{P}_2 \quad n_k=2$")
    ax.axis('off')
    legend = fig.legend(loc='center', frameon=False, ncol=4)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "rom_basis_test_legend.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":

    # plot_test_solver_comparison()
    # plot_test_solver_comparison_bis()
    # plot_test_rom_accuracy()
    # plot_test_3()
    # plot_test_acceleration_efficiency()
    # plot_test_scalability()
    # plot_test_fast_assembly_accuracy()
    plot_test_rom_basis()
    # plot_test_rom_basis_bis()
