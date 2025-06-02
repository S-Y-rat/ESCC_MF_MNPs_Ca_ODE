from collections.abc import Callable
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, Solution
import jax
import jax.numpy as jnp
import pandas as pd

from calcium_model import CalciumModel


def simulate(
    t0: float,
    t1: float,
    diff_fun_system: Callable[[jax.Array, jax.Array, None], jax.Array],
    initial_values: jax.Array,
):
    term = ODETerm(diff_fun_system)  # type: ignore
    solver = Dopri5()
    saveat = SaveAt(dense=True)
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)
    return diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=initial_values,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=16 * 4096,
    )


def batched_adapter(*diff_fun_systems: CalciumModel):
    def func(t: jax.Array, Ss: jax.Array, args=None):
        splited = jnp.array_split(Ss, len(diff_fun_systems))
        return jnp.concat([f(t, S, args) for f, S in zip(diff_fun_systems, splited)])

    return func


def multisim(
    *diff_fun_systems: CalciumModel,
    t0: float,
    t1: float,
) -> Solution:
    return simulate(
        t0=t0,
        t1=t1,
        diff_fun_system=jax.jit(batched_adapter(*diff_fun_systems)),
        initial_values=jnp.concat([sys.initial_values for sys in diff_fun_systems]),
    )


def interpolate_data(
    sol: Solution,
    models: list[CalciumModel],
    ts: jax.Array,
) -> pd.DataFrame:
    def J_in_fn(t: jax.Array, c_e: jax.Array, model: CalciumModel):
        return model.delta * model.J_in(c_e) + model.J_magn(model.mp, t)

    interpolated: jax.Array = jax.vmap(sol.evaluate)(ts)
    eq_num = interpolated[0].shape[0] // len(models)
    return pd.concat(
        [
            pd.DataFrame(
                dict(
                    idx=idx,
                    B=model.mp.B,
                    omega=model.mp.omega,
                    t=ts,
                    c=interpolated[:, 0 + eq_num * idx],
                    c_e=interpolated[:, 1 + eq_num * idx],
                    h=interpolated[:, 2 + eq_num * idx],
                    p=interpolated[:, 3 + eq_num * idx],
                    v=J_in_fn(ts, interpolated[:, 1 + eq_num * idx], model),
                    label=model.mp.legend_MF,
                )
            )
            for idx, model in enumerate(models)
        ]
    )
