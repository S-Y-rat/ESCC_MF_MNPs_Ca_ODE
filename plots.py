# %%
import time

import jax

import jax.numpy as jnp
import seaborn as sns

from magnetic_field import MagneticFieldParameters
from calcium_model import CalciumModel
from plotter import Plotter
from solver import multisim, interpolate_data
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
plotter = Plotter(t0=T0, t1=T1)
figsaver = plotter.savefmts(["pdf", "svg"])


# %%
B, OMEGA = 25e-3, 1.7e-3 * jnp.pi
magnetic_params_default = MagneticFieldParameters(
    regime="uniform",
    time_dependence="rotating",
    B=B,
    omega=OMEGA,
)
models = [
    CalciumModel(mp=fn_obj)
    for fn_obj in [
        magnetic_params_default._replace(B=0),
        magnetic_params_default,
    ]
]
batched_sol = multisim(t0=T0, t1=T1, *models)

# %%
df_models = interpolate_data(
    sol=batched_sol,
    models=models,
    ts=jnp.linspace(T0, T1, 10 * (T1 - T0)),
)
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
end_time = time.perf_counter()
print(f"Execution time before plotting is {end_time - start_time:.2f} s")
plotter.show()
