from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from scipy.stats import ttest_ind
import pandas as pd

from plotter import Plotter


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


@jax.jit
def time_mask(t: jax.Array, min_time: float, max_time: float) -> jax.Array:
    return (t > min_time) & (t < max_time)


def find_local_fn_time(
    fn, t: jax.Array, c: jax.Array, min_time: float, max_time: float
) -> float:
    res = jnp.where(c == fn(c[time_mask(t, min_time, max_time)]))
    return t[res[0]].item()


def find_c_periods(df: pd.DataFrame) -> None:
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


def peak_max(df: pd.DataFrame) -> None:
    local_max: float = jnp.max(jnp.asarray(df["c"][df["t"] <= 10])).item()
    print(f"Maximal [Ca2+] at first 200 s (IP3R-induced peak): {local_max:.2f} muM")


def find_v_periods(df_mm: pd.DataFrame) -> None:
    find_local_max_time = partial(find_local_fn_time, jnp.max)
    J_in_magn_local_max_time = partial(
        find_local_max_time, df_mm["t"].to_numpy(), df_mm["v"].to_numpy()
    )

    local_mins = jnp.array(
        [
            J_in_magn_local_max_time(200, 400),
            J_in_magn_local_max_time(800, 1000),
            J_in_magn_local_max_time(1400, 1600),
        ]
    )
    diffs = jnp.diff(local_mins)
    formated_local_means = [f"{m:.2f}" for m in local_mins]

    print(f"Negative peaks are at {formated_local_means} s")
    print(
        f"First interval for v(t) magn: {diffs[0]:.2f} s\nSecond interval for v(t) magn: {diffs[1]:.2f} s"
    )


def get_local_min_max_mask(xs: jax.Array) -> jax.Array:
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


def local_minima_periods_medians(df_defaults: pd.DataFrame, df_param: str) -> None:
    def internal(B: float):
        df = df_loc_fn(df_defaults, Bs=[B])
        return jnp.asarray(df["t"].to_numpy())[
            get_local_min_max_mask(jnp.asarray(df[df_param].to_numpy()))[0]
        ]

    no_mf, mf = 0.0, 25e-3
    no_mf_local_min_t = internal(no_mf)
    mf_local_min_t = internal(mf)

    no_mf_diffs = jnp.diff(no_mf_local_min_t)
    mf_diffs = jnp.diff(mf_local_min_t)

    no_mf_diffs_formated = [f"{t:.2f}" for t in no_mf_diffs]
    mf_diffs_formated = [f"{t:.2f}" for t in mf_diffs]

    print(f"Differenses when no MF: {no_mf_diffs_formated} s")
    print(f"Differenses when MF: {mf_diffs_formated} s")

    print("Corresponding shapes: {}, {}".format(no_mf_diffs.shape, mf_diffs.shape))

    print(
        "Corresponding medians: {:.2f} s, {:.2f} s".format(
            jnp.median(no_mf_diffs), jnp.median(mf_diffs)
        )
    )


def scatter_c_effects(df_defaults: pd.DataFrame, plotter: Plotter, figsaver) -> None:
    def plot_c():
        df_no_mf = df_loc_fn(df_defaults, Bs=[0.0])
        df_mf = df_loc_fn(df_defaults, Bs=[25e-3])

        no_mf_local_min_max = get_local_min_max(jnp.asarray(df_no_mf["c"].to_numpy()))
        mf_local_min_max = get_local_min_max(jnp.asarray(df_mf["c"].to_numpy()))

        fig, _ = plotter.fig_4_extrema_end_median_regr_2_models(
            no_mf_local_min_max,
            mf_local_min_max,
        )
        return fig, no_mf_local_min_max, mf_local_min_max

    fig, no_mf_loc, mf_loc = plot_c()
    figsaver(fig, "Figure_7")
    print("\nExtrema: min, max")
    print(
        f"Shapes of local extrema when no MF: {no_mf_loc[0].shape}, {no_mf_loc[1].shape}"
    )
    print(f"Shapes of local extrema when MF: {mf_loc[0].shape}, {mf_loc[1].shape}")
    print(
        "T-test for local minima and local maxima when MF: {}".format(
            ttest_ind(*mf_loc)
        )
    )
