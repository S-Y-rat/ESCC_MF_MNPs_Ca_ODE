# %%
from functools import partial
from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.stats import levene, ttest_ind
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

from magnetic_field import MagneticFieldParameters
from calcium_model import CalciumModel
from plotter import Plotter
from solver import multisim, interpolate_data

# %%
T0, T1 = 0, 1800
plotter = Plotter(T0, T1)


# %%
def df_loc_fn(
    df: pd.DataFrame,
    *,
    Bs: Optional[list[float]] = None,
    omegas: Optional[list[float]] = None,
) -> pd.DataFrame:
    B_field, omega_field = "B", "omega"
    if (Bs is not None) & (omegas is not None):
        return df.loc[df[B_field].isin(Bs) & df[omega_field].isin(omegas)]  # type: ignore
    elif Bs is not None:
        return df.loc[df[B_field].isin(Bs)]
    elif omegas is not None:
        return df.loc[df[omega_field].isin(omegas)]
    raise ValueError("No parameters provided")


magnetic_params_default = MagneticFieldParameters(
    regime="uniform", time_dependence="rotating"
)
magnetic_params_25_mT = [
    magnetic_params_default,
    magnetic_params_default._replace(omega=(1.5e-3 * jnp.pi)),
    magnetic_params_default._replace(omega=(1e-3 * jnp.pi)),
    magnetic_params_default._replace(omega=(0.5e-3 * jnp.pi)),
]
magmetic_params_no_field = [params._replace(B=0) for params in magnetic_params_25_mT]
magnetic_params_10_mT = [params._replace(B=10e-3) for params in magnetic_params_25_mT]
magnetic_params_100_mT = [params._replace(B=100e-3) for params in magnetic_params_25_mT]
models = [
    CalciumModel(mp=fn_obj)
    for fn_obj in [
        *magmetic_params_no_field,  # idx = range(0, 4)
        *magnetic_params_10_mT,  # idx = range(4, 8)
        *magnetic_params_25_mT,  # idx = range(8, 12)
        *magnetic_params_100_mT,  # idx = range(12, 16)
    ]
]
batched_sol = multisim(t0=T0, t1=T1, *models)

# %%
df_models = interpolate_data(
    sol=batched_sol,
    models=models,
    ts=jnp.linspace(T0, T1, 10 * (T1 - T0)),
)
df_defaults = df_loc_fn(df_models, Bs=[0.0, 25e-3], omegas=[1.7e-3 * jnp.pi])


# %%
fig, _ = plotter.fig_1_ts_cyt_signals(df_defaults)
fig.savefig("fig_1_ts_2_cyt_signals.svg")


# %%
@jax.jit
def time_mask(t: jax.Array, min_time: float, max_time: float):
    return (t > min_time) & (t < max_time)


def find_local_fn_time(
    fn, t: jax.Array, c: jax.Array, min_time: float, max_time: float
) -> float:
    res = jnp.where(c == fn(c[time_mask(t, min_time, max_time)]))
    return t[res[0]].item()


def find_c_periods():
    df = df_loc_fn(df_defaults, Bs=[25e-3])
    find_local_min_time = partial(find_local_fn_time, jnp.min)
    magnetic_model_find_local_min_time = partial(
        find_local_min_time,
        df["t"].to_numpy(),
        df["c"].to_numpy(),
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
fig, _ = plotter.fig_2_ts_2_er_signals(df_defaults)
fig.savefig("fig_2_ts_2_er_signals.svg")


# %%
fig, _ = plotter.fig_3_ts_2_speed_signals(df_defaults)
fig.savefig("fig_3_ts_2_speed_signals.svg")


# %%
def find_v_periods():
    find_local_max_time = partial(find_local_fn_time, jnp.max)
    df_mm = df_loc_fn(df_defaults, Bs=[25e-3])
    J_in_magn_local_max_time = partial(
        find_local_max_time, df_mm["t"].to_numpy(), df_mm["v"].to_numpy()
    )

    interval_J_in_magn_1 = J_in_magn_local_max_time(
        800, 1000
    ) - J_in_magn_local_max_time(200, 400)
    interval_J_in_magn_2 = J_in_magn_local_max_time(
        1400, 1600
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
def plot_c():
    df_no_field = df_loc_fn(df_defaults, Bs=[0.0])
    df_field = df_loc_fn(df_defaults, Bs=[25e-3])
    fig, _ = plotter.fig_4_extrema_end_median_regr_2_models(
        get_local_min_max(jnp.asarray(df_no_field["c"].to_numpy())),
        get_local_min_max(jnp.asarray(df_field["c"].to_numpy())),
    )
    return fig


plot_c().savefig("fig_4_extrema_end_median_regr_2_models.svg")


# %%
def find_levene(df: pd.DataFrame) -> None:
    B = 25
    df_filtered = df_loc_fn(df, Bs=[B * 1e-3])
    cols = ["c", "c_e", "h"]

    def fmt_fn(vs: list[float]) -> str:
        return ", ".join([f"{key}: {value:.4f}" for key, value in zip(cols, vs)])

    gen_fn = lambda: (group[cols].to_numpy() for _, group in df_filtered.groupby("idx"))
    levene_all = levene(
        df_loc_fn(df_defaults, Bs=[0])[cols].to_numpy(),
        *gen_fn(),
    )
    levene_mf = levene(
        *gen_fn(),
    )
    for levene_test, label in zip(
        [levene_all, levene_mf],
        [
            f"all models ({B} mT)",
            f"all magnetic models (0.0 mT and {B} mT)",
        ],
    ):
        print(
            "Frequencies (pi mHz): {}\nResults of Levene test for {}:\n\tstatistic={{{}}},\n\tp-value={{{}}}".format(
                jnp.unique(df_filtered["omega"].to_numpy() / jnp.pi * 1e3),
                label,
                fmt_fn(levene_test.statistic),
                fmt_fn(levene_test.pvalue),
            )
        )


find_levene(df_models)
print("\n#########################\n")
find_levene(df_models.loc[df_models["omega"] != 0.5e-3 * jnp.pi])

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[(df_models["B"] == 25e-3)].sort_values(by="omega", ascending=False),
    fignum=5,
    alpha=0.7,
)
fig.savefig("fig_5_ts_cyt_signals.svg")

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[df_models["B"] == 100e-3].sort_values(by="omega", ascending=False),
    fignum=6,
    alpha=0.7,
)
fig.savefig("fig_6_ts_cyt_signals.svg")

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[df_models["B"] == 10e-3].sort_values(by="omega", ascending=False),
    fignum=7,
    alpha=0.7,
)
fig.savefig("fig_7_ts_cyt_signals.svg")

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[df_models["omega"] == 1.5e-3 * jnp.pi].sort_values(
        by="B", ascending=True
    ),
    fignum=8,
    alpha=0.7,
)
fig.savefig("fig_8_ts_cyt_signals.svg")

# %%
plotter.show()
