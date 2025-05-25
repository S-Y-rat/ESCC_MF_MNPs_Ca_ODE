# %%
from functools import partial

import jax
import jax.numpy as jnp
from scipy.stats import levene, ttest_ind
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

from magnetic_field import MagneticFieldParameters
from calcium_model import CalciumModel
from plotter import Plotter
from solver import multisim

# %%
T0, T1 = 0, 1800
plotter = Plotter(T0, T1)


# %%
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
magnetic_params_100_mT = [params._replace(B=100e-3) for params in magnetic_params_25_mT]
models = [
    CalciumModel(mp=fn_obj)
    for fn_obj in [
        magmetic_model_no_field,
        *magnetic_params_25_mT,
        *magnetic_params_100_mT,
    ]
]
batched_model, df_models = multisim(t0=T0, t1=T1, *models)

IDX = "idx"
MODEL_IDX_DEFAULT_CHANG = 0
MODEL_IDX_DEFAULT_MAGN, MODEL_IDX_1_5, MODEL_IDX_1_0, MODEL_IDX_0_5 = list(range(1, 5))
df_defaults = df_models.loc[
    df_models[IDX].isin([MODEL_IDX_DEFAULT_CHANG, MODEL_IDX_DEFAULT_MAGN])
]


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
fig, _ = plotter.fig_2_ts_2_er_signals(df_defaults)
fig.savefig("fig_2_ts_2_er_signals.svg")


# %%
fig, _ = plotter.fig_3_ts_2_speed_signals(df_defaults)
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
fig, _ = plotter.fig_4_extrema_end_median_regr_2_models(
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
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[
        (df_models[IDX] != MODEL_IDX_DEFAULT_CHANG) & (df_models["B"] == 25e-3)
    ],
    fignum=5,
    alpha=0.7,
)
fig.savefig("fig_5_ts_cyt_signals.svg")

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[df_models["B"] == 100e-3],
    fignum=6,
    alpha=0.7,
)
fig.savefig("fig_6_ts_cyt_signals.svg")

# %%
plotter.show()
