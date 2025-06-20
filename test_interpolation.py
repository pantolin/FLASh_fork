import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import h5py
import os

from deim_int import create_RBF_interpolator, interpolate_coefficients

from plotter import Plotter


class Plots(Plotter):

    @classmethod
    def plot_1(
        cls,
        size,
        data,
        path: str,
        dir: str,
        title: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        legend_loc = (0.5, -0.15),
        fa = [0.05, 0.95, 0.9, 0.05, 0.25, 0.5],
        nc = 3
    ) -> None:

        cls._folder = os.path.join(os.path.join(os.getcwd(), "figs"), dir)
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(figsize=size)

        fig.suptitle(title, fontsize=16)

        x = data["x"]
        y = data["y"]
        label = data["label"]

        for ind in range(len(x)):
            ax.semilogy(x[ind], y[ind], "-", markersize=4, label=label[ind])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.grid()

        ax.legend(
            loc="upper center",
            bbox_to_anchor=legend_loc,
            ncol=nc,
            fontsize=12,
            edgecolor="black",  # Legend border color
            fancybox=False      # Rounded box edges
        )

        plt.savefig(cls.add_folder(path), bbox_inches="tight")


def interpolator_error(coefficients, test_coefficients, parameters, test_parameters):

    interpolator = create_RBF_interpolator(parameters, coefficients.T)
    coefs_i = interpolate_coefficients(interpolator, parameters)

    test_coefs_i = interpolate_coefficients(interpolator, test_parameters)

    error = np.linalg.norm(coefficients.T-coefs_i, np.inf)
    error = np.average(error/np.linalg.norm(coefficients.T, np.inf))

    test_error = np.linalg.norm(test_coefficients.T-test_coefs_i, np.inf)
    test_error = np.average(test_error/np.linalg.norm(test_coefficients.T, np.inf))

    return error, test_error

if __name__ == "__main__":      


    folder = os.path.join(os.getcwd(), "coefficient_data")
    name = f"merged_results"

    h5f = h5py.File(os.path.join(folder, f"{name}.h5"),'r')
    parameters = h5f["parameters"][:]
    coefficients = h5f["coefficients"][:]
    h5f.close()

    folder = os.path.join(os.getcwd(), "test_coefficient_data")
    name = f"merged_results"

    h5f = h5py.File(os.path.join(folder, f"{name}.h5"),'r')
    test_parameters = h5f["parameters"][:]
    test_coefficients = h5f["coefficients"][:]
    h5f.close()

    errors = []
    test_errors = []

    for i in range(6, parameters.shape[0]):

        error, test_error = interpolator_error(coefficients[:i], test_coefficients, parameters[:i], test_parameters)
        errors.append(error)
        test_errors.append(test_error)

    errors = np.hstack(errors)
    test_errors = np.hstack(test_errors)

    data = {"x": [np.power(np.arange(test_errors.size)+6,0.25)], "y": [test_errors], "label": [""]}

    Plots.plot_1(
        (5, 4),
        data,
        "interpolating_error.pdf",
        "K",
        f"Interpolating coefficient errors for $K$ ($4$ parameters)",
        f"$N_s^{{0.25}}$",
        f"$\\frac{{\|\\theta-\hat{{\\theta}}\|}}{{\|\\theta\|}}$"
    )

