"""Aggregate the wrench_paper runs: h-convergence table/plots and ROM comparison.

Reads the JSONs produced by wrench_validation.py --paper-field and
wrench_refined.py; writes summary.txt / summary.csv / convergence figures.
Load-dependent QoIs are also reported normalized by the applied load (the
boundary-mass Neumann assembly makes the discrete load discretization-
dependent).
"""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from _paths import FIGS_DIR, RESULTS_DIR

PAPER_DIR = RESULTS_DIR / "wrench_paper"


def load(name: str) -> dict:
    return json.loads((PAPER_DIR / f"{name}.json").read_text())


def rel(a, b) -> float:
    return float(np.linalg.norm(np.atleast_1d(np.asarray(a) - np.asarray(b)))
                 / np.linalg.norm(np.atleast_1d(b)))


def norm_qois(d: dict) -> dict:
    fy = d["F_applied"][1]
    return {
        "C_norm": d["compliance"] / fy**2,
        "u_corner_norm": np.array(d["u_corner"]) / abs(fy),
        "u_neu_norm": np.array(d["u_neu_mean"]) / abs(fy),
        "F_stress_norm": np.array(d["F_stress"]) / abs(fy),
        "Mz_stress_norm": d["Mz_stress"] / abs(fy),
        "fy": fy,
    }


def main() -> None:
    ks = sorted(
        int(p.stem.split("_k")[1].split("_")[0])
        for p in PAPER_DIR.glob("run_refined_window_k*_p3_cholesky.json")
    )
    runs = {k: load(f"run_refined_window_k{k}_p3_cholesky") for k in ks}

    ref = norm_qois(runs[ks[-1]])

    lines = [
        "h-refinement study (window mode, p = 3, full quadrature, rho = 0)",
        "k  n_sub  n_dofs   area            compliance      C/Fy^2        u_corner (x, y)                 eq_check",
    ]
    errs_C, errs_u, errs_un, errs_F = {}, {}, {}, {}
    for k in ks:
        d = runs[k]
        n = norm_qois(d)
        eq = float(np.linalg.norm(np.array(d["F_consistent"]) + np.array(d["F_applied"]))
                   / np.linalg.norm(d["F_applied"]))
        errs_C[k] = abs(n["C_norm"] - ref["C_norm"]) / abs(ref["C_norm"])
        errs_u[k] = float(np.linalg.norm(n["u_corner_norm"] - ref["u_corner_norm"])
                          / np.linalg.norm(ref["u_corner_norm"]))
        errs_un[k] = float(np.linalg.norm(n["u_neu_norm"] - ref["u_neu_norm"])
                           / np.linalg.norm(ref["u_neu_norm"]))
        errs_F[k] = float(np.linalg.norm(n["F_stress_norm"] - ref["F_stress_norm"])
                          / np.linalg.norm(ref["F_stress_norm"]))
        corner_flag = "" if d["corner_info"].get("material", True) else " (void)"
        lines.append(
            f"{k}  {d['n_subdomains']:<5} {d['n_dofs']:<8} {d['material_area']:.8e} "
            f"{d['compliance']:.6e} {n['C_norm']:.6e} "
            f"({d['u_corner'][0]: .6e}, {d['u_corner'][1]: .6e}){corner_flag}  {eq:.2e}"
        )

    lines += [
        "",
        "errors vs finest k (load-normalized):",
        *(f"  k={k}: compliance {errs_C[k]:.3e}, u_corner {errs_u[k]:.3e}, "
          f"u_neu_mean {errs_un[k]:.3e}, F_stress {errs_F[k]:.3e}"
          for k in ks[:-1]),
    ]

    area_spread = max(runs[k]["material_area"] for k in ks) - min(runs[k]["material_area"] for k in ks)
    lines.append(f"\nmaterial area spread across k (geometry invariance): {area_spread:.3e}")

    rom = load("run_p8_bddc_rom_stab0.0005")
    full = load("run_p8_bddc_stab0.0005")
    plain = load("run_p8_cholesky")
    nrom, nfull, nref = norm_qois(rom), norm_qois(full), ref

    lines += [
        "",
        "paper configuration (p = 8) vs h-refined reference (finest k, p = 3), load-normalized:",
        f"  ROM+BDDC+stab:   C/Fy^2 {nrom['C_norm']:.6e}  err vs ref {abs(nrom['C_norm']-nref['C_norm'])/nref['C_norm']:.3e}",
        f"                   u_corner err {np.linalg.norm(nrom['u_corner_norm']-nref['u_corner_norm'])/np.linalg.norm(nref['u_corner_norm']):.3e}, "
        f"u_neu_mean err {np.linalg.norm(nrom['u_neu_norm']-nref['u_neu_norm'])/np.linalg.norm(nref['u_neu_norm']):.3e}",
        f"  full+BDDC+stab:  C/Fy^2 {nfull['C_norm']:.6e}  err vs ref {abs(nfull['C_norm']-nref['C_norm'])/nref['C_norm']:.3e}",
        "",
        "ROM effect at p = 8 (ROM+stab vs full+stab, raw):",
        f"  compliance {rel(rom['compliance'], full['compliance']):.3e}, "
        f"u_corner {rel(rom['u_corner'], full['u_corner']):.3e}, "
        f"Mz_stress {rel(rom['Mz_stress'], full['Mz_stress']):.3e}",
        "stabilization effect at p = 8 (full+stab vs full, raw):",
        f"  compliance {rel(full['compliance'], plain['compliance']):.3e}, "
        f"u_corner {rel(full['u_corner'], plain['u_corner']):.3e}",
        "",
        f"reaction on hexagon (ROM run, exact by equilibrium): F = {[-v for v in rom['F_applied']]}, "
        f"Mz = {-rom['Mz_applied']:.6e}",
        f"BDDC iterations (ROM run): {rom['stats'].get('iterations')}",
    ]

    table = "\n".join(lines)
    (PAPER_DIR / "summary.txt").write_text(table + "\n")
    print(table)

    with open(PAPER_DIR / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "n_subdomains", "n_dofs", "material_area", "Fy_applied",
                    "compliance", "compliance_norm", "u_corner_x", "u_corner_y",
                    "F_stress_x", "F_stress_y", "Mz_stress",
                    "err_C_norm_vs_kmax", "err_u_corner_vs_kmax"])
        for k in ks:
            d, n = runs[k], norm_qois(runs[k])
            w.writerow([k, d["n_subdomains"], d["n_dofs"], d["material_area"], n["fy"],
                        d["compliance"], n["C_norm"], *d["u_corner"],
                        *d["F_stress"], d["Mz_stress"],
                        errs_C.get(k, ""), errs_u.get(k, "")])

    hs = [1.0 / k for k in ks[:-1]]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.loglog(hs, [errs_C[k] for k in ks[:-1]], "o-", label="compliance (normalized)")
    ax.loglog(hs, [errs_un[k] for k in ks[:-1]], "^-", label="loaded-boundary mean displacement (normalized)")
    ax.loglog(hs, [errs_u[k] for k in ks[:-1]], "s-", label="corner displacement (normalized)")
    ax.loglog(hs, [errs_F[k] for k in ks[:-1]], "d-", label="stress reaction (normalized)")
    ax.set_xlabel("$h = 1/k$ (children per cell direction)")
    ax.set_ylabel("relative error vs finest $k$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGS_DIR / "wrench_h_convergence.pdf")
    fig.savefig(FIGS_DIR / "wrench_h_convergence.png", dpi=200)
    print(f"figures written to {FIGS_DIR}")


if __name__ == "__main__":
    main()
