import numpy as np
import matplotlib.pyplot as plt

import os


import h5py

# --- Plot results from test_1.py ---
def plot_test_1():

    file_path = "results/test_1/data.h5"
    with h5py.File(file_path, "r") as f:
        bddc_iters = f["bddc_iters"][:]
        amg_iters = f["amg_iters"][:]
        bddc_setup_time = f["bddc_setup_time"][:]
        amg_setup_time = f["amg_setup_time"][:]
        cholesky_setup_time = f["cholesky_setup_time"][:]
        bddc_solve_time = f["bddc_solve_time"][:]
        amg_solve_time = f["amg_solve_time"][:]
        cholesky_solve_time = f["cholesky_solve_time"][:]
        number_of_subdomains = f["number_of_subdomains"][:]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # Iterations
    axs[0, 0].plot(number_of_subdomains, bddc_iters, label="BDDC", marker='o')
    axs[0, 0].plot(number_of_subdomains, amg_iters, label="AMG", marker='s')
    axs[0, 0].set_xlabel("Number of subdomains")
    axs[0, 0].set_ylabel("Iterations")
    axs[0, 0].set_title("Solver Iterations vs Subdomains")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_yscale('log')

    # Setup time
    axs[0, 1].plot(number_of_subdomains, bddc_setup_time, label="BDDC", marker='o')
    axs[0, 1].plot(number_of_subdomains, amg_setup_time, label="AMG", marker='s')
    axs[0, 1].plot(number_of_subdomains, cholesky_setup_time, label="Cholesky", marker='^')
    axs[0, 1].set_xlabel("Number of subdomains")
    axs[0, 1].set_ylabel("Setup Time (s)")
    axs[0, 1].set_title("Setup Time vs Subdomains")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_yscale('log')

    # Solve time
    axs[1, 0].plot(number_of_subdomains, bddc_solve_time, label="BDDC", marker='o')
    axs[1, 0].plot(number_of_subdomains, amg_solve_time, label="AMG", marker='s')
    axs[1, 0].plot(number_of_subdomains, cholesky_solve_time, label="Cholesky", marker='^')
    axs[1, 0].set_xlabel("Number of subdomains")
    axs[1, 0].set_ylabel("Solve Time (s)")
    axs[1, 0].set_title("Solve Time vs Subdomains")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].set_yscale('log')

    # Total time (setup + solve)
    bddc_total = bddc_setup_time + bddc_solve_time
    amg_total = amg_setup_time + amg_solve_time
    cholesky_total = cholesky_setup_time + cholesky_solve_time
    axs[1, 1].plot(number_of_subdomains, bddc_total, label="BDDC", marker='o')
    axs[1, 1].plot(number_of_subdomains, amg_total, label="AMG", marker='s')
    axs[1, 1].plot(number_of_subdomains, cholesky_total, label="Cholesky", marker='^')
    axs[1, 1].set_xlabel("Number of subdomains")
    axs[1, 1].set_ylabel("Total Time (s)")
    axs[1, 1].set_title("Total Time vs Subdomains")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.suptitle("Test 1: Solver Performance vs Subdomains", y=1.04)
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/test_1_results.pdf", bbox_inches='tight')
    plt.close()

# --- Plot results from test_2.py ---
def plot_test_2():
	
    file_path = "results/test_2/data.h5"
    with h5py.File(file_path, "r") as f:
        iterations = f["iterations"][:]
        rom_iterations = f["rom_iterations"][:]
        stab_errors = f["stab_errors"][:]
        rom_errors = f["rom_errors"][:]
        total_errors = f["total_errors"][:]
        number_of_subdomains = f["number_of_subdomains"][:]
        stabilizations = f["stabilizations"][:]
        
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

	# Iterations subplot
    for idx, stab in enumerate(stabilizations):
        axs[0, 0].plot(number_of_subdomains, iterations[:, idx], label=f"stab={stab:.1e}", marker='o')
        axs[0, 0].plot(number_of_subdomains, rom_iterations[:, idx], label=f"ROM stab={stab:.1e}", marker='s', linestyle='--')
        
    axs[0, 0].set_xlabel("Number of subdomains")
    axs[0, 0].set_ylabel("Iterations")
    axs[0, 0].set_title("Iterations vs Subdomains")
    axs[0, 0].legend(fontsize=8, ncol=2)
    axs[0, 0].grid(True)
    axs[0, 0].set_yscale('log')

    # ROM error subplot
    for idx, stab in enumerate(stabilizations):
        axs[0, 1].plot(number_of_subdomains, rom_errors[:, idx], label=f"stab={stab:.1e}", marker='s')
        
    axs[0, 1].set_xlabel("Number of subdomains")
    axs[0, 1].set_ylabel("ROM Error")
    axs[0, 1].set_title("ROM Error vs Subdomains")
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(True)
    axs[0, 1].set_yscale('log')

    # Stab error subplot
    for idx, stab in enumerate(stabilizations):
        axs[1, 0].plot(number_of_subdomains, stab_errors[:, idx], label=f"stab={stab:.1e}", marker='o')
        
    axs[1, 0].set_xlabel("Number of subdomains")
    axs[1, 0].set_ylabel("Stabilization Error")
    axs[1, 0].set_title("Stabilization Error vs Subdomains")
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True)
    axs[1, 0].set_yscale('log')

    # Total error subplot
    for idx, stab in enumerate(stabilizations):
        axs[1, 1].plot(number_of_subdomains, total_errors[:, idx], label=f"stab={stab:.1e}", marker='^')
        
    axs[1, 1].set_xlabel("Number of subdomains")
    axs[1, 1].set_ylabel("Total Error")
    axs[1, 1].set_title("Total Error vs Subdomains")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(True)
    axs[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.suptitle("Test 2: Error and Iteration Analysis (lines: stabilization values)", y=1.04)
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/test_2_results.pdf", bbox_inches='tight')
    plt.close()
	
# --- Plot results from test_3.py ---
def plot_test_3():
    file_path = "results/test_3/data.h5"
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
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/test_3_results.pdf", bbox_inches='tight')
    plt.close()

# --- Plot results from test_4.py ---
def plot_test_4():
    file_path = "results/test_4/data.h5"
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

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: All error curves
    axs[0, 0].plot(number_of_subdomains, errors_bas_vs_basws, marker='o', label="Baseline vs Baseline+WS")
    axs[0, 0].plot(number_of_subdomains, errors_baws_vs_faws, marker='s', label="BAWS vs FAWS")
    axs[0, 0].plot(number_of_subdomains, errors_faws_vs_romws, marker='^', label="FAWS vs ROMWS")
    axs[0, 0].plot(number_of_subdomains, errors_bas_vs_romws, marker='d', label="Baseline vs ROM+WS")
    axs[0, 0].set_xlabel("Number of subdomains")
    axs[0, 0].set_ylabel("Error")
    axs[0, 0].set_title("Error Curves")
    axs[0, 0].set_yscale('log')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Subplot 2: Iterations
    axs[0, 1].plot(number_of_subdomains, bas_iters, marker='o', label="Baseline")
    axs[0, 1].plot(number_of_subdomains, basws_iters, marker='s', label="Baseline+WS")
    axs[0, 1].plot(number_of_subdomains, faws_iters, marker='^', label="FA+WS")
    axs[0, 1].plot(number_of_subdomains, romws_iters, marker='d', label="ROM+WS")
    axs[0, 1].set_xlabel("Number of subdomains")
    axs[0, 1].set_ylabel("Iterations")
    axs[0, 1].set_title("Iterations")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Subplot 3: Total solve time (setup + solve)
    axs[1, 0].plot(number_of_subdomains, bas_setup_time + bas_solve_time, marker='o', label="Baseline")
    axs[1, 0].plot(number_of_subdomains, romws_setup_time + romws_solve_time, marker='d', label="ROM+WS")
    axs[1, 0].set_xlabel("Number of subdomains")
    axs[1, 0].set_ylabel("Setup Solve Time (s)")
    axs[1, 0].set_title("Setup Solve Time")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Subplot 4: Total solve time (setup + solve)
    axs[1, 1].plot(number_of_subdomains, bas_setup_time + bas_assemble_time + bas_solve_time, marker='o', label="Baseline")
    axs[1, 1].plot(number_of_subdomains, romws_setup_time + romws_assemble_time + romws_solve_time, marker='d', label="ROM+WS")
    axs[1, 1].set_xlabel("Number of subdomains")
    axs[1, 1].set_ylabel("Total Solve Time (s)")
    axs[1, 1].set_title("Total Solve Time")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.suptitle("Test 4: Error, Iteration, and Solve Time Analysis", y=1.04)
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/test_4_results.pdf", bbox_inches='tight')
    plt.close()

# --- Plot results from test_5.py ---
def plot_test_5():
    file_path = "results/test_5/data.h5"
    with h5py.File(file_path, "r") as f:
        number_of_subdomains = f["number_of_subdomains"][:]
        iters = f["iters"][:]
        setup_time = f["setup_time"][:]
        assemble_time = f["assemble_time"][:]
        solve_time = f["solve_time"][:]

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Subplot 1: Iterations
    axs[0].plot(number_of_subdomains, iters, marker='o', label="Iterations")
    axs[0].set_xlabel("Number of subdomains")
    axs[0].set_ylabel("Iterations")
    axs[0].set_title("Iterations")
    axs[0].grid(True)
    axs[0].legend()

    # Subplot 2: Setup and Assemble Time
    axs[1].plot(number_of_subdomains, setup_time, marker='s', label="Setup Time")
    axs[1].plot(number_of_subdomains, assemble_time, marker='^', label="Assemble Time")
    axs[1].plot(number_of_subdomains, solve_time, marker='d', label="Solve Time")
    axs[1].plot(number_of_subdomains, setup_time+assemble_time+solve_time, marker='o', label="Total Time")
    axs[1].set_xlabel("Number of subdomains")
    axs[1].set_ylabel("Time (s)")
    axs[1].set_title("Solve Time")
    axs[1].set_yscale('log')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.suptitle("Test 5: Iteration and Timing Analysis", y=1.04)
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/test_5_results.pdf", bbox_inches='tight')
    plt.close()

# --- Plot results from test_6.py ---
def plot_test_6():
    file_path = "results/test_6/data.h5"
    with h5py.File(file_path, "r") as f:
        number_of_subdomains = f["number_of_subdomains"][:]
        fa_degrees = f["fa_degrees"][:]
        errors = f["errors"][:]
        iterations = f["iterations"][:]
        fa_iterations = f["fa_iterations"][:]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Error vs. Number of subdomains for each fa_degree
    for idx, degree in enumerate(fa_degrees):
        axs[0].plot(number_of_subdomains, errors[:, idx], marker='o', label=f"FA degree={int(degree)}")
    axs[0].set_xlabel("Number of subdomains")
    axs[0].set_ylabel("Error")
    axs[0].set_title("Error vs Subdomains (by FA degree)")
    axs[0].set_yscale('log')
    axs[0].grid(True)
    axs[0].legend()

    # Subplot 2: Iterations vs. Number of subdomains for baseline and each fa_degree
    axs[1].plot(number_of_subdomains, iterations, marker='o', label="Baseline")
    for idx, degree in enumerate(fa_degrees):
        axs[1].plot(number_of_subdomains, fa_iterations[:, idx], marker='s', label=f"FA degree={int(degree)}")
    axs[1].set_xlabel("Number of subdomains")
    axs[1].set_ylabel("Iterations")
    axs[1].set_title("Iterations vs Subdomains (by FA degree)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.suptitle("Test 6: Error and Iteration Analysis", y=1.04)
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/test_6_results.pdf", bbox_inches='tight')
    plt.close()

# --- Plot results from rom_basis_test.py ---
def plot_rom_basis_test():
    """
    Plot the error decay as a function of basis size for each n in the ROM basis test.
    """
    import os
    file_path = os.path.join("results", "rom_data", "schoen_iwp_4", "error_data.h5")
    with h5py.File(file_path, "r") as f:
        basis_number = f["basis_number"][:]
        errors = f["errors"][:]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(errors.shape[0]):
        ax.plot(basis_number[i], errors[i], label=f"n={i+1}")
    ax.set_xlabel("Basis size")
    ax.set_ylabel("Relative error (mean, $L^\infty$)")
    ax.set_yscale('log')
    ax.set_title("ROM Basis Test: Error Decay vs Basis Size")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/rom_basis_test_results.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    plot_test_1()
    plot_test_2()
    plot_test_3()
    plot_test_4()
    plot_test_5()
    plot_test_6()

    plot_rom_basis_test()



