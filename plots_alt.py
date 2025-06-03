# %%
import time
from pathlib import Path

import jax

import jax.numpy as jnp
import seaborn as sns
import pandas as pd
from scipy.integrate import solve_ivp

from magnetic_field import MagneticFieldParameters
from calcium_model import CalciumModel
from plotter import Plotter
from solver import batched_adapter
from plots_impl import (
    df_loc_fn,
    find_c_periods,
    peak_max,
    find_v_periods,
    local_minima_periods_medians,
    scatter_c_effects,
)

jax.config.update("jax_enable_x64", True)
sns.set_theme(style="whitegrid")

# %%
start_time = time.perf_counter()
T0, T1 = 0, 1800
plotter = Plotter(t0=T0, t1=T1, figures_dir=Path("alt_figures"))
figsaver = plotter.savefmts(["pdf", "svg"])


# %%
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


# %%
def data_interpolator(*models: CalciumModel) -> pd.DataFrame:
    def J_in_fn(t: jax.Array, c_e: jax.Array, model: CalciumModel):
        return model.delta * model.J_in(c_e) + model.J_magn(model.mp, t)

    dense = solve_ivp(
        fun=jax.jit(batched_adapter(*models)),
        t_span=(T0, T1),
        y0=jnp.concat([m.initial_values for m in models]),
        dense_output=True,
        rtol=1e-8,
        atol=1e-8,
    )
    ts = jnp.linspace(T0, T1, 10 * (T1 - T0))
    interpolated = dense.sol(ts)
    eq_num = interpolated.shape[0] // len(models)

    return pd.concat(
        [
            pd.DataFrame(
                dict(
                    idx=idx,
                    B=model.mp.B,
                    omega=model.mp.omega,
                    t=ts,
                    c=interpolated[0 + eq_num * idx],
                    c_e=interpolated[1 + eq_num * idx],
                    h=interpolated[2 + eq_num * idx],
                    p=interpolated[3 + eq_num * idx],
                    v=J_in_fn(ts, interpolated[1 + eq_num * idx], model),
                    label=model.mp.legend_MF,
                )
            )
            for idx, model in enumerate(models)
        ]
    )


df_models = data_interpolator(*models)
df_defaults = df_loc_fn(df_models, Bs=[0.0, 25e-3], omegas=[1.7e-3 * jnp.pi])


# %%
fig, _ = plotter.fig_1_ts_cyt_signals(df_defaults)
figsaver(fig, "Figure_4")


# %%
find_c_periods(df_loc_fn(df_defaults, Bs=[25e-3]))


# %%
peak_max(df_loc_fn(df_defaults, Bs=[25e-3]))

# %%
fig, _ = plotter.fig_2_ts_2_er_signals(df_defaults)
figsaver(fig, "Figure_5")


# %%
fig, _ = plotter.fig_3_ts_2_speed_signals(df_defaults)
figsaver(fig, "Figure_6")


# %%
find_v_periods(df_loc_fn(df_defaults, Bs=[25e-3]))


# %%
print("Periods of local minima for [Ca2+]")
local_minima_periods_medians(df_defaults, "c")
print("\nPeriods of local minima for v")
local_minima_periods_medians(df_defaults, "v")


# %%
scatter_c_effects(df_defaults, plotter, figsaver)

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[(df_models["B"] == 25e-3)].sort_values(by="omega", ascending=False),
    fignum=5,
    alpha=0.7,
)
figsaver(fig, "Figure_8")

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[df_models["B"] == 100e-3].sort_values(by="omega", ascending=False),
    fignum=6,
    alpha=0.7,
)
figsaver(fig, "Figure_9")

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[df_models["B"] == 10e-3].sort_values(by="omega", ascending=False),
    fignum=7,
    alpha=0.7,
)
figsaver(fig, "Figure_10")

# %%
fig, _ = plotter.fig_1_ts_cyt_signals(
    df_models.loc[df_models["omega"] == 1.5e-3 * jnp.pi].sort_values(
        by="B", ascending=True
    ),
    fignum=8,
    alpha=0.7,
)
figsaver(fig, "Figure_11")

# %%
plotter.show()
end_time = time.perf_counter()
print(f"Overall execution time is {end_time - start_time:.2f} s")
