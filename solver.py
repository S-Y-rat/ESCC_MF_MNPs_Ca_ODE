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


def batched_adapter(*diff_fun_systems: CalciumModel):
    def func(t: jax.Array, Ss: jax.Array, args=None):
        Ss = jnp.array_split(Ss, len(diff_fun_systems))
        return jnp.concat([f(t, S, args) for f, S in zip(diff_fun_systems, Ss)])

    return jax.jit(func)


def multisim(
    *diff_fun_systems: CalciumModel,
    t0: float,
    t1: float,
    eq_num=4,
) -> tuple[Solution, pd.DataFrame]:
    def J_in_fn(t: float, c_e: float, model: CalciumModel):
        return model.delta * model.J_in(c_e) + model.J_magn(model.mp, t)

    batched_model = simulate(
        t0=t0,
        t1=t1,
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
