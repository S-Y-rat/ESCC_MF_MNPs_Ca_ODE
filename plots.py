# %%
from collections.abc import Callable
from functools import partial

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, Solution
import jax
import jax.numpy as jnp
from scipy.stats import levene, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

from magnetic_field import MagneticFieldParameters
from calcium_model import CalciumModel

# %%
T0, T1 = 0, 1800
TD = 10

FIGKWARGS = dict(figsize=(10, 5))


# %%
def simulate(
    t0: float,
    t1: float,
    diff_fun_system: Callable[[jax.Array, jax.Array, None], jax.Array],
    initial_values: jax.Array,
):
    term = ODETerm(diff_fun_system)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 10000))
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    return diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=initial_values,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )


# %%
def batched_adapter(*diff_fun_systems: CalciumModel):
    def func(t: jax.Array, Ss: jax.Array, args=None):
        Ss = jnp.array_split(Ss, len(diff_fun_systems))
        return jnp.concat([f(t, S, args) for f, S in zip(diff_fun_systems, Ss)])

    return jax.jit(func)


def multisim(
    *diff_fun_systems: CalciumModel,
    eq_num=4,
) -> tuple[Solution, pd.DataFrame]:
    def J_in_fn(t: float, c_e: float, model: CalciumModel):
        return model.delta * model.J_in(c_e) + model.J_magn(model.mp, t)

    batched_model = simulate(
        t0=T0,
        t1=T1,
        diff_fun_system=batched_adapter(*diff_fun_systems),
        initial_values=jnp.concat([sys.initial_values for sys in diff_fun_systems]),
    )
    return batched_model, pd.concat(
        [
            pd.DataFrame(
                dict(
                    idx=idx,
                    B=model.mp.B,
                    t=batched_model.ts,
                    c=batched_model.ys[:, 0 + eq_num * idx],
                    c_e=batched_model.ys[:, 1 + eq_num * idx],
                    h=batched_model.ys[:, 2 + eq_num * idx],
                    p=batched_model.ys[:, 3 + eq_num * idx],
                    v=J_in_fn(
                        batched_model.ts, batched_model.ys[:, 1 + eq_num * idx], model
                    ),
                    label=model.mp.legend_MF,
                )
            )
            for idx, model in enumerate(diff_fun_systems)
        ]
    )


magnetic_params_default = MagneticFieldParameters(
    regime="uniform", time_dependence="rotating"
)
magmetic_model_no_field = magnetic_params_default._replace(B=0)
magnetic_params_25_mT = [
    magnetic_params_default,
    magnetic_params_default._replace(omega=(1.5e-3 * jnp.pi)),
    magnetic_params_default._replace(omega=(1e-3 * jnp.pi)),
    magnetic_params_default._replace(omega=(0.5e-3 * jnp.pi)),
]
magnetic_params_100_mT = [
    params._replace(B=100e-3) for params in magnetic_params_25_mT
]
models = [
    CalciumModel(mp=fn_obj)
    for fn_obj in [
        magmetic_model_no_field,
        *magnetic_params_25_mT,
        *magnetic_params_100_mT,
    ]
]
batched_model, df_models = multisim(*models)

IDX = "idx"
MODEL_IDX_DEFAULT_CHANG = 0
MODEL_IDX_DEFAULT_MAGN, MODEL_IDX_1_5, MODEL_IDX_1_0, MODEL_IDX_0_5 = list(range(1, 5))
df_defaults = df_models.loc[
    df_models[IDX].isin([MODEL_IDX_DEFAULT_CHANG, MODEL_IDX_DEFAULT_MAGN])
]


# %%
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


def Q2_Q3(ax, groups):
    def get_whiskers(xs: jax.Array):
        return jnp.abs(
            jnp.array([jnp.percentile(xs, 25), jnp.percentile(xs, 75)]) - jnp.median(xs)
        )

    ax.errorbar(
        jnp.arange(len(groups)),
        list(map(jnp.median, groups)),
        yerr=list(map(get_whiskers, groups)),
        linestyle="none",
        ecolor="black",
        capsize=7,
    )


# %%
def fig_1_ts_cyt_signals(df: pd.DataFrame, *, fignum=1, alpha=1.0):
    fig = plt.figure(fignum, **FIGKWARGS)
    fig.subplots_adjust(wspace=0.35)
    ax_peak = fig.add_subplot(1, 4, 1)
    ax_oscillations = fig.add_subplot(1, 4, (2, 4))
    axs = (ax_peak, ax_oscillations)

    plot_kwargs = dict(data=df, x="t", y="c", hue="label", alpha=alpha)
    sns.lineplot(**plot_kwargs, ax=ax_peak, legend=False)
    sns.lineplot(**plot_kwargs, ax=ax_oscillations)

    for ax, title in zip(axs, ("A", "B")):
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(title)

    ax_oscillations.set_ylim(bottom=0.0, top=0.58)
    ax_peak.set_xlim(left=T0 - TD // 3, right=25.0)
    ax_oscillations.set_xlim(left=100.0, right=T1 + TD)
    ax_oscillations.legend(loc="upper right")
    ax_peak.set_ylabel("$c(t), \\mu M$")
    fig.supxlabel("$t, s$")
    fig.suptitle("$c(t)$ is $Ca^{{+2}}$ concentration in the cytosol")
    return fig, axs


fig, _ = fig_1_ts_cyt_signals(df_defaults)
fig.savefig("fig_1_ts_2_cyt_signals.svg")


# %%
@jax.jit
def time_mask(t: jax.Array, min_time: float, max_time: float):
    return (t > min_time) & (t < max_time)


def find_local_fn_time(
    fn, t: jax.Array, c: jax.Array, min_time: float, max_time: float
) -> float:
    res = jnp.where(c == fn(c[time_mask(t, min_time, max_time)]))
    return t[res[0]][0]


def find_c_periods():
    find_local_min_time = partial(find_local_fn_time, jnp.min)
    magnetic_model_find_local_min_time = partial(
        find_local_min_time,
        df_models.loc[df_models[IDX] == MODEL_IDX_DEFAULT_MAGN]["t"].to_numpy(),
        df_models.loc[df_models[IDX] == MODEL_IDX_DEFAULT_MAGN]["c"].to_numpy(),
    )

    interval_1 = magnetic_model_find_local_min_time(
        800, 900
    ) - magnetic_model_find_local_min_time(200, 400)
    interval_2 = magnetic_model_find_local_min_time(
        1400, 1600
    ) - magnetic_model_find_local_min_time(800, 1000)

    print(
        f"First interval for c(t) magn: {interval_1:.2f} s\nSecond interval for c(t) magn: {interval_2:.2f} s"
    )


find_c_periods()


# %%
def fig_2_ts_2_er_signals(df: pd.DataFrame, *, fignum=2):
    fig = plt.figure(fignum, **FIGKWARGS)
    ax_peak = fig.add_subplot(1, 3, 1)
    ax_oscillations = fig.add_subplot(1, 3, (2, 3))
    axs = (ax_peak, ax_oscillations)

    sns.lineplot(data=df, x="t", y="c_e", hue="label", ax=ax_peak, legend=False)
    sns.lineplot(data=df, x="t", y="c_e", hue="label", ax=ax_oscillations)

    for ax, title in zip(axs, ("A", "B")):
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(title)

    cutout_t = 400
    ax_peak.set_yscale("log")
    ax_peak.set_xlim(left=T0 - TD, right=cutout_t)
    ax_peak.set_ylabel("$c_{{e}}(t), \\mu M$")
    ax_peak.set_title("A")

    ax_oscillations.set_ylim(bottom=15.0, top=22.0)
    ax_oscillations.set_xlim(left=cutout_t, right=T1 + TD)
    ax_oscillations.set_title("B")
    ax_oscillations.legend(loc="upper right")

    fig.subplots_adjust(wspace=0.25)
    fig.supxlabel("$t, s$")
    fig.suptitle("$c_{{e}}(t)$ is $Ca^{{+2}}$ concentration in ER")
    return fig, axs


fig, _ = fig_2_ts_2_er_signals(df_defaults)
fig.savefig("fig_2_ts_2_er_signals.svg")


# %%
def fig_3_ts_2_speed_signals(df: pd.DataFrame, *, fignum=1):
    fig = plt.figure(fignum, **FIGKWARGS)
    ax = sns.lineplot(data=df, x="t", y="v", hue="label")
    ax.set_xlim(left=T0 - TD, right=T1 + TD)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("$t, s$")
    ax.set_ylabel("$v$, $\\mu M/s$")
    ax.legend(loc="lower right")
    fig.suptitle("$Ca^{{+2}}$ influx from outside of the cell")
    return fig, ax


fig, _ = fig_3_ts_2_speed_signals(df_defaults)
fig.savefig("fig_3_ts_2_speed_signals.svg")


# %%
def find_v_periods():
    find_local_max_time = partial(find_local_fn_time, jnp.max)
    df_mm = df_defaults.loc[df_defaults[IDX] == MODEL_IDX_DEFAULT_MAGN]
    J_in_magn_local_max_time = partial(
        find_local_max_time, df_mm["t"].to_numpy(), df_mm["v"].to_numpy()
    )

    interval_J_in_magn_1 = J_in_magn_local_max_time(
        800, 1000
    ) - J_in_magn_local_max_time(200, 400)
    interval_J_in_magn_2 = J_in_magn_local_max_time(
        1400, 1500
    ) - J_in_magn_local_max_time(800, 1000)
    print(
        f"First interval for v(t) magn: {interval_J_in_magn_1:.2f} s\nSecond interval for v(t) magn: {interval_J_in_magn_2:.2f} s"
    )


find_v_periods()


# %%
def get_local_min_max_mask(xs: jax.Array):
    left = xs[:-2]
    center = xs[1:-1]
    right = xs[2:]
    false = jnp.zeros_like(xs).astype(jnp.bool)
    return (
        jax.vmap(false.at[1:-1].set)(
            jnp.array(
                [(left > center) & (center < right), (left < center) & (center > right)]
            )
        )
        .at[:, 0]
        .set([xs[0] < xs[1], xs[0] > xs[1]])
        .at[:, -1]
        .set([xs[-1] < xs[-2], xs[-1] > xs[-2]])
    )


def get_local_min_max(xs: jax.Array):
    min_mask, max_mask = get_local_min_max_mask(xs)
    return xs[min_mask], xs[max_mask]


# %%
def fig_4_extrema_end_median_regr_2_models(
    no_mf_min_max: tuple[jax.Array, jax.Array],
    mf_min_max: tuple[jax.Array, jax.Array],
    *,
    d_jitter=0.25,
    scatter_alpha=0.2,
    scatter_marker=".",
    fignum=1,
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
    pair_top_bound = (1.5, 0.52)
    pair_bottom_bound = (1.22, 0.0)

    fig = plt.figure(fignum, figsize=(10, 6))
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

    for part, top_bound, bottom_bound in zip(
        pair_max, pair_top_bound, pair_bottom_bound
    ):
        group_jitter(keys[0], ax_dict[part], no_mf_max, 0, no_mf_max_color)
        group_jitter(keys[1], ax_dict[part], mf_max, 1, mf_max_color)
        ax_dict[part].set_ylim(bottom=bottom_bound, top=top_bound)

    ybroken_axis(
        *(ax_dict[part] for part in pair_max),
        color=get_ax_xgridcolor(ax_dict["max_high"]),
    )

    Q2_Q3(ax_dict["max_low"], [no_mf_max, mf_max])
    regression_plot(ax_dict["max_low"], coefs_max)
    ax_dict["max_low"].set_xticks(x, ["No MF", "MF"])
    ax_dict["max_low"].set_xlabel("Local\nMaxima")
    ax_dict["max_high"].set_title("A")
    ax_dict["max_low"].legend(**legend_kwargs)

    group_jitter(keys[2], ax_dict["min"], no_mf_min, 0, no_mf_min_color)
    group_jitter(keys[3], ax_dict["min"], mf_min, 1, mf_min_color)
    ax_dict["min"].set_ylim(bottom=0.0)

    Q2_Q3(ax_dict["min"], [no_mf_min, mf_min])
    regression_plot(ax_dict["min"], coefs_min)
    ax_dict["min"].set_xlabel("Local\nMinima")
    ax_dict["min"].set_xticks(x, ["No MF", "MF"])
    ax_dict["min"].set_title("B")
    ax_dict["min"].legend(**legend_kwargs)

    for part, top_bound, bottom_bound in zip(
        pair_mf, pair_top_bound, pair_bottom_bound
    ):
        group_jitter(keys[4], ax_dict[part], mf_max, 0, mf_max_color)
        group_jitter(keys[5], ax_dict[part], mf_min, 1, mf_min_color)
        ax_dict[part].set_ylim(bottom=bottom_bound, top=top_bound)

    ybroken_axis(
        *(ax_dict[part] for part in pair_mf),
        color=get_ax_xgridcolor(ax_dict["mf_high"]),
    )

    Q2_Q3(ax_dict["mf_low"], [mf_max, mf_min])
    regression_plot(ax_dict["mf_low"], coefs_mf)
    ax_dict["mf_low"].set_xticks(x, ["Local\nMaxima", "Local\nMinima"])
    ax_dict["mf_low"].set_xlabel("MF")
    ax_dict["mf_high"].set_title("C")
    ax_dict["mf_low"].legend(**legend_kwargs)

    fig.suptitle("Local extrema and median regressions")
    fig.supylabel("C, $\\mu M$")
    return fig, ax_dict


fig, _ = fig_4_extrema_end_median_regr_2_models(
    get_local_min_max(
        df_models.loc[df_models[IDX] == MODEL_IDX_DEFAULT_CHANG]["c"].to_numpy()
    ),
    get_local_min_max(
        df_models.loc[df_models[IDX] == MODEL_IDX_DEFAULT_MAGN]["c"].to_numpy()
    ),
)
fig.savefig("fig_4_extrema_end_median_regr_2_models.svg")


# %%
def find_levene(df: pd.DataFrame) -> None:
    levene_all = levene(
        batched_model.ys[:, 0 : (4 - 1)],
        batched_model.ys[:, 4 : (8 - 1)],
        batched_model.ys[:, 8 : (12 - 1)],
        batched_model.ys[:, 12 : (16 - 1)],
    )
    levene_mf = levene(
        batched_model.ys[:, 4 : (8 - 1)],
        batched_model.ys[:, 8 : (12 - 1)],
        batched_model.ys[:, 12 : (16 - 1)],
    )
    for levene_test, label in zip(
        [levene_all, levene_mf],
        [
            "all models",
            "all magnetic models",
        ],
    ):
        print(
            f"Results of Levene test for {label}:\n\tstatistic={levene_test.statistic},\n\tp-value={levene_test.pvalue}"
        )


find_levene(df_models)

# %%
fig, _ = fig_1_ts_cyt_signals(
    df_models.loc[
        (df_models[IDX] != MODEL_IDX_DEFAULT_CHANG) & (df_models["B"] == 25e-3)
    ],
    fignum=5,
    alpha=0.7,
)
fig.savefig("fig_5_ts_cyt_signals.svg")

# %%
fig, _ = fig_1_ts_cyt_signals(
    df_models.loc[df_models["B"] == 100e-3],
    fignum=6,
    alpha=0.7,
)
fig.savefig("fig_6_ts_cyt_signals.svg")

# %%
plt.show()
