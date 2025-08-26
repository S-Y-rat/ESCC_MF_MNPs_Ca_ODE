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
sns.set_theme(style="whitegrid", palette=sns.color_palette("colorblind"))

# %%
start_time = time.perf_counter()
T0, T1 = 0, 1800
plotter = Plotter(t0=T0, t1=T1, figures_dir=Path("alt_figures"))
figsaver = plotter.savefmts(["pdf", "svg"])


# %%
magnetic_params_default = MagneticFieldParameters(
    regime="uniform", time_dependence="rotating"
)
models = [
    CalciumModel(mp=fn_obj)
    for fn_obj in [
        magnetic_params_default._replace(B=0),
        magnetic_params_default,
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
B, OMEGA = 25e-3, 1.7e-3 * jnp.pi
df_defaults = df_loc_fn(df_models, Bs=[0.0, B], omegas=[OMEGA])


# %%
fig, _ = plotter.fig_3(df_defaults)
figsaver(fig, "Figure_3")


# %%
find_c_periods(df_loc_fn(df_defaults, Bs=[B]))


# %%
peak_max(df_loc_fn(df_defaults, Bs=[B]))

# %%
fig, _ = plotter.fig_4(df_defaults)
figsaver(fig, "Figure_4")


# %%
fig, _ = plotter.fig_5(df_defaults)
figsaver(fig, "Figure_5")


# %%
find_v_periods(df_loc_fn(df_defaults, Bs=[B]))


# %%
print("Periods of local minima for [Ca2+]")
local_minima_periods_medians(df_defaults, "c", B=B)
print("\nPeriods of local minima for v")
local_minima_periods_medians(df_defaults, "v", B=B)


# %%
scatter_c_effects(df_defaults, plotter, figsaver, B=B)

# %%
plotter.show()
end_time = time.perf_counter()
print(f"Overall execution time is {end_time - start_time:.2f} s")
