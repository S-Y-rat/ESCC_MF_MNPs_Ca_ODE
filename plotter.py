from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd


def ybroken_axis(ax_high, ax_low, *, d=0.5, color="k"):
    def yaxis_breaks_marker(ax, position):
        ax.plot(
            (0, 1),
            position,
            transform=ax.transAxes,
            marker=[(-1, -d), (1, d)],
            markersize=12,
            linestyle="none",
            color=color,
            mec=color,
            mew=1,
            clip_on=False,
        )

    ax_high.spines.bottom.set_visible(False)
    ax_low.spines.top.set_visible(False)
    yaxis_breaks_marker(ax_high, (0, 0))
    yaxis_breaks_marker(ax_low, (1, 1))


class Plotter:
    def __init__(
        self,
        t0: int,
        t1: int,
        td: int = 10,
        figures_dir=Path("figures"),
        figkwargs: dict = dict(figsize=(10, 5)),
        dpi: int = 600,
    ):
        self.t0 = t0
        self.t1 = t1
        self.td = td
        self.figures_dir = figures_dir
        self.figkwargs = figkwargs
        self.dpi = dpi
        figures_dir.mkdir(exist_ok=True)

    @staticmethod
    def show() -> None:
        plt.show()

    def savefmts(self, fmts: list[str]):
        def internal(fig: Figure, name: str) -> None:
            def figure_path(fmt: str) -> Path:
                return self.figures_dir / f"{name}.{fmt}"

            for fmt in fmts:
                fig.savefig(figure_path(fmt), dpi=self.dpi)

        return internal

    def fig_1_ts_cyt_signals(self, df: pd.DataFrame, *, fignum=1, alpha=1.0):
        fig = plt.figure(fignum, **self.figkwargs)
        fig.subplots_adjust(wspace=0.35)
        ax_peak = fig.add_subplot(1, 4, 1)
        ax_oscillations = fig.add_subplot(1, 4, (2, 4))
        axs = (ax_peak, ax_oscillations)

        plot_kwargs = dict(data=df, x="t", y="c", hue="label", alpha=alpha)
        sns.lineplot(**plot_kwargs, ax=ax_peak, legend=False)  # type: ignore
        sns.lineplot(**plot_kwargs, ax=ax_oscillations)  # type: ignore

        for ax, title in zip(axs, ("A", "B")):
            ax.set_xlabel(None)  # type: ignore
            ax.set_ylabel(None)  # type: ignore
            ax.set_title(title)

        ax_oscillations.set_ylim(bottom=0.0, top=0.58)
        ax_peak.set_xlim(left=self.t0 - self.td // 3, right=25.0)
        ax_oscillations.set_xlim(left=100.0, right=self.t1 + self.td)
        ax_oscillations.legend(loc="upper right")
        ax_peak.set_ylabel("$c(t), \\mu M$")
        fig.supxlabel("$t, s$")
        fig.suptitle("$c(t)$ is $Ca^{{2+}}$ concentration in the cytosol")
        return fig, axs

    def fig_2_ts_2_er_signals(self, df: pd.DataFrame, *, fignum=2):
        fig = plt.figure(fignum, **self.figkwargs)
        ax_peak = fig.add_subplot(1, 3, 1)
        ax_oscillations = fig.add_subplot(1, 3, (2, 3))
        axs = (ax_peak, ax_oscillations)

        sns.lineplot(data=df, x="t", y="c_e", hue="label", ax=ax_peak, legend=False)
        sns.lineplot(data=df, x="t", y="c_e", hue="label", ax=ax_oscillations)

        for ax, title in zip(axs, ("A", "B")):
            ax.set_xlabel(None)  # type: ignore
            ax.set_ylabel(None)  # type: ignore
            ax.set_title(title)

        cutout_t = 400
        ax_peak.set_yscale("log")
        ax_peak.set_xlim(left=self.t0 - self.td, right=cutout_t)
        ax_peak.set_ylabel("$c_{{e}}(t), \\mu M$")
        ax_peak.set_title("A")

        ax_oscillations.set_ylim(bottom=15.0, top=22.0)
        ax_oscillations.set_xlim(left=cutout_t, right=self.t1 + self.td)
        ax_oscillations.set_title("B")
        ax_oscillations.legend(loc="upper right")

        fig.subplots_adjust(wspace=0.25)
        fig.supxlabel("$t, s$")
        fig.suptitle("$c_{{e}}(t)$ is $Ca^{{2+}}$ concentration in ER")
        return fig, axs

    def fig_3_ts_2_speed_signals(self, df: pd.DataFrame, *, fignum=3):
        fig = plt.figure(fignum, **self.figkwargs)
        ax = sns.lineplot(data=df, x="t", y="v", hue="label")
        ax.set_xlim(left=self.t0 - self.td, right=self.t1 + self.td)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("$t, s$")
        ax.set_ylabel("$v$, $\\mu M/s$")
        ax.legend(loc="lower right")
        fig.suptitle("$Ca^{{2+}}$ influx from outside of the cell")
        return fig, ax

    @staticmethod
    def fig_4_extrema_end_median_regr_2_models(
        no_mf_min_max: tuple[jax.Array, jax.Array],
        mf_min_max: tuple[jax.Array, jax.Array],
        *,
        d_jitter=0.25,
        scatter_alpha=0.2,
        scatter_marker=".",
        fignum=4,
        figkwargs=dict(figsize=(10, 6)),
    ):
        def group_jitter(key, ax, ys, group_idx: int, color=None):
            kwargs = dict(
                alpha=scatter_alpha,
                marker=scatter_marker,
                color=mf_max_color,
            )
            if color is not None:
                kwargs["color"] = color

            ax.scatter(
                jnp.zeros_like(ys)
                + group_idx
                + jax.random.uniform(key, ys.shape, minval=-d_jitter, maxval=d_jitter),
                ys,
                **kwargs,
            )

        def regression_plot(ax, coefs: jax.Array):
            ax.plot(
                x,
                jnp.polyval(coefs, x),
                marker="o",
                color=regression_color,
                label=f"{coefs[0]:.4f}x+{coefs[1]:.4f}",
            )

        def get_ax_xgridcolor(ax):
            return ax.get_xgridlines()[0].get_color()

        keys = jax.random.split(jax.random.key(54362), 6)

        no_mf_min, no_mf_max = no_mf_min_max
        mf_min, mf_max = mf_min_max

        x = jnp.array([0.0, 1.0])

        y_max = jnp.array([jnp.median(no_mf_max), jnp.median(mf_max)])
        y_min = jnp.array([jnp.median(no_mf_min), jnp.median(mf_min)])
        y_mf = jnp.array([y_max[1], y_min[1]])

        coefs_max = jnp.polyfit(x, y_max, 1)
        coefs_min = jnp.polyfit(x, y_min, 1)
        coefs_mf = jnp.polyfit(x, y_mf, 1)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        no_mf_max_color = colors[0]
        no_mf_min_color = colors[1]
        mf_max_color = colors[2]
        mf_min_color = colors[3]
        regression_color = colors[-1]
        legend_kwargs = dict(loc="lower left")

        row_high = ["max_high", "mf_high"]
        row_low = ["max_low", "mf_low"]
        pair_max, pair_mf = list(zip(row_high, row_low))

        fig = plt.figure(fignum, **figkwargs)
        rows, cols = 3, 3
        ax_dict = dict(
            max_high=fig.add_subplot(rows, cols, 1),
            max_low=fig.add_subplot(rows, cols, (4, 7)),
            min=fig.add_subplot(1, cols, 2),
            mf_high=fig.add_subplot(rows, cols, 3),
            mf_low=fig.add_subplot(rows, cols, (6, 9)),
        )
        fig.subplots_adjust(hspace=0.05, wspace=0.3)
        for high, low in zip(row_high, row_low):
            ax_dict[high].spines.bottom.set_visible(False)
            ax_dict[low].spines.top.set_visible(False)
            ax_dict[high].sharex(ax_dict[low])
            ax_dict[high].set_ylim(bottom=1.22, top=1.6)
            ax_dict[low].set_ylim(top=0.52, bottom=0.0)

        for part in pair_max:
            group_jitter(keys[0], ax_dict[part], no_mf_max, 0, no_mf_max_color)
            group_jitter(keys[1], ax_dict[part], mf_max, 1, mf_max_color)

        ybroken_axis(
            ax_high=ax_dict[pair_max[0]],
            ax_low=ax_dict[pair_max[1]],
            color=get_ax_xgridcolor(ax_dict["max_high"]),  # type: ignore
        )

        regression_plot(ax_dict["max_low"], coefs_max)  # type: ignore
        ax_dict["max_low"].set_xticks(x, ["No MF", "MF"])
        ax_dict["max_low"].set_xlabel("Local\nMaxima")
        ax_dict["max_high"].set_title("A")
        ax_dict["max_low"].legend(**legend_kwargs)

        group_jitter(keys[2], ax_dict["min"], no_mf_min, 0, no_mf_min_color)
        group_jitter(keys[3], ax_dict["min"], mf_min, 1, mf_min_color)
        ax_dict["min"].set_ylim(bottom=0.0)

        regression_plot(ax_dict["min"], coefs_min)  # type: ignore
        ax_dict["min"].set_xlabel("Local\nMinima")
        ax_dict["min"].set_xticks(x, ["No MF", "MF"])
        ax_dict["min"].set_title("B")
        ax_dict["min"].legend(**legend_kwargs)

        for part in pair_mf:
            group_jitter(keys[4], ax_dict[part], mf_max, 0, mf_max_color)
            group_jitter(keys[5], ax_dict[part], mf_min, 1, mf_min_color)

        ybroken_axis(
            ax_high=ax_dict[pair_mf[0]],
            ax_low=ax_dict[pair_mf[1]],
            color=get_ax_xgridcolor(ax_dict["mf_high"]),  # type: ignore
        )

        regression_plot(ax_dict["mf_low"], coefs_mf)  # type: ignore
        ax_dict["mf_low"].set_xticks(x, ["Local\nMaxima", "Local\nMinima"])
        ax_dict["mf_low"].set_xlabel("MF")
        ax_dict["mf_high"].set_title("C")
        ax_dict["mf_low"].legend(**legend_kwargs)

        fig.suptitle("Local extrema and median regressions")
        fig.supylabel("C, $\\mu M$")
        return fig, ax_dict
