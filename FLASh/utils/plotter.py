
import os
import matplotlib.pyplot as plt

class Plotter:

    _folder = os.path.join(os.getcwd(), "figs")
    args = ["-o"]
    kwargs = {"markevery": [0, -1], "markersize": 2}

    @staticmethod
    def __clear__() -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()

    @staticmethod
    def __setup_config__() -> None:
        """It sets up the matplotlib configuration."""

        plt.rc("text", usetex=True)
        plt.rcParams.update({"font.size": 11, "font.family": "serif"})
        plt.rcParams['text.usetex'] = False

    @classmethod
    def add_folder(cls, path: str) -> str:
        """It adds the default folder to the input path.

        Parameters
        ----------
        path: str
            A path in string.

        Returns
        -------
        str
            The path with the added folder.
        """

        if not os.path.exists(cls._folder):
            os.mkdir(cls._folder)

        return os.path.join(cls._folder, path)

    @classmethod
    def get_plot(
        cls,
        data,
        path: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
    ) -> None:

        cls.__clear__()
        cls.__setup_config__()

        x = data["x"]
        y = data["y"]
        labels = data["label"]

        for i in range(len(x)):

            plt.semilogy(x[i], y[i], "o-", markersize=4, label = labels[i])
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(
            loc="upper center", 
            bbox_to_anchor=(0.5, -0.1), 
            ncol=2, 
            fontsize=12, 
            edgecolor="black",  # Legend border color
            fancybox=False  # Rounded box edges
        )
        plt.savefig(cls.add_folder(path), bbox_inches="tight")


    @classmethod
    def get_subplot(
        cls,
        n_plots,
        size,
        data,
        path: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
    ) -> None:

        cls.__clear__()
        cls.__setup_config__()

        fig, axes = plt.subplots(n_plots[0], n_plots[1], figsize=size, sharey=False)

        for i in range(n_plots[0]):
            for j in range(n_plots[1]):

                plot_data = data[i * n_plots[1] + j]

                x = plot_data["x"]
                y = plot_data["y"]
                label = plot_data["label"]

                for ind in range(len(x)):

                    axes[i][j].plot(x[ind], y[ind], "-", markersize=4, label = label[ind])
                
                axes[i][j].set_xlabel(xlabel)
                axes[i][j].set_ylabel(ylabel)
                axes[i][j].grid()
                axes[i][j].legend(
                    loc="upper center", 
                    bbox_to_anchor=(0.5, -0.1), 
                    ncol=2, 
                    fontsize=12, 
                    edgecolor="black", 
                    fancybox=False 
                )

        plt.savefig(cls.add_folder(path), bbox_inches="tight")

 