from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from magnetic_field import MagneticFieldParameters


class CalciumModel(eqx.Module):
    """Model class that contains Chang model (dSdt) and it's version with respect of WSS (dWdt)"""

    # Set of dimensional parameters used for transformation
    # between dimensional and dimensionless parameters and variables
    # Набір розмірних параметрів, що використовуються для перетворення
    # між розмірними та безрозмірними параметрами та змінними

    # Table 1 (Addition 1. Magnetic shear stress in Chang's model).
    k_f: float = (
        3.9  # Scaling factor that controls [Ca2+] release through IP3R; IP3R density and channel activity
    )
    # k_f = 10
    # k_f = 3.9 s-1
    V_p: float = 0.9  # Maximum capacity of SERCA pump
    K: float = 1.9e-5  # Used to adjust the Ca 2+ concentration in ER
    k_p: float = 0.2  # Half-maximal [IP3] for IP3R Напівмаксимальний [IP3] для IP3R
    delta: float = (
        1.5  # Used to adjust ratio of [Ca2+] across plasma membrane to ER membrane
    )
    alpha_0: float = (
        0.0027  # Flow of calcium into the cell through an unspecified leak (невизначений витік)
    )
    alpha_1: float = (
        0.385  # Rate constant for SOC channels Константа швидкості для каналів SOC
    )
    # alpha_1 = 0.07
    K_e: float = (
        8.0  # Half-maximal [Ca2+] ER for SOC channels Напівмаксимальний [Ca2+] в ER для каналів SOC
    )
    V_pm: float = (
        0.11  # Maximum capacity of plasma membrane pump Максимальна потужність насоса плазматичної мембрани
    )
    K_pm: float = (
        0.3  # Half-maximal [Ca2+] for plasma membrane pump Напівмаксимальний [Ca2+] для насоса плазматичної мембрани
    )
    gamma: float = 5.5  # γ Ratio of cytoplasmic volume to ER volume
    K_h: float = 0.08  # The concentration of Ca2+ activated IP3R
    K_tau: float = 0.1  # The concentration of Ca 2+ in response to β
    tau_max: float = 1420.0  #
    # tau_max = 1000
    # tau_max = 10
    beta_p: float = (
        0.027  # Rate of decay of p to its steady state Швидкість розпаду p до стаціонарного стану
    )
    p_s: float = 0.1
    k_beta: float = 0.4
    K_c: float = 0.2  # Half-maximal [Ca2+] for IP3R
    T_Plank: float = 500
    K4_Plank: float = 0.32  # K4=0.32  # microM, Michaelis–Menten constant
    # qmax_Plank = 17.6  # qmax=17.6 microM/s
    qmax_Plank: float = 0.04  # Max. WSS-induced Ca2+ influx rate
    # alpha_Plank = 1_000
    alpha_Plank: float = (
        1_000_000  # alpha=2 No-load channel constant, (1 + alpha)**(-1) is the probability that
    )
    # a channel is in the open state in the no-load case W=0
    fe_Plank: float = 0.0134  # fe=0.0134  # is the fraction of the energy within
    # the membrane that gates the shear-sensitive Ca2+ channels
    # частка енергії всередині мембрани, яка пропускає чутливі до зсуву Ca2+ канали
    Te_Plank: float = 310
    N_Plank: float = 1e12
    epsilon_Plank: float = 0.1
    delta_Plank: float = 1e-5
    l_Plank: float = 3.5e-5  # довжина клітини у напрямку напруження зсуву
    c_0: float = 0.1  # The initial сoncentration of Ca2+ in cytosol
    c_e_0: float = 100.0  # The initial concentration of Ca2+ in ER
    h_0: float = 0.08  # The initial rate at which Ca2+ can activate IP3Rs
    p_0: float = 0.1  # The initial rate of Ca2+ activation IP3
    mp: Optional[MagneticFieldParameters] = None

    @property
    def k(self):
        return 1.3807e-23  # kg*m2/(s2*K), Boltzmann constant

    def __str__(self):
        return "\n".join(
            (
                "Set of dimensional parameters used "
                "for transformation between dimensional",
                "and dimensionless parameters and variables:",
                "Table 1 (Addition 1. Magnetic shear stress in Chang's model).",
                f"k = {self.k} kg*m2/(s2*K), Boltzmann constant",
                f"Te_Plank = {self.Te_Plank} room temperature, K; 310 K",
                f"N_Plank = {self.N_Plank:.3e} m**(-2), "
                "ion channel density per unit area",
                f"epsilon_Plank = {self.epsilon_Plank} is the fraction "
                "of the applied load borne by the plasma membrane",
                f"delta_Plank = {self.delta_Plank} is the membrane shear modulus",
                f"fe_Plank = {self.fe_Plank} is the fraction of the energy within "
                "the membrane that gates the shear-sensitive Ca2+ channels",
                f"k_f = {self.k_f} Scaling factor that controls [Ca2+] release "
                "through IP3R; IP3R density and channel activity",
                f"V_p = {self.V_p} Maximum capacity of SERCA pump",
                f"K = {self.K} Used to adjust the Ca 2+ concentration in ER",
                f"k_p = {self.k_p} " "Half-maximal [IP3] for IP3R",
                f"delta = {self.delta} Used to adjust ratio of [Ca2+] across "
                "plasma membrane to ER membrane",
                f"alpha_0 = {self.alpha_0} Flow of calcium into the cell "
                "through an unspecified leak",
                f"alpha_1 = {self.alpha_1} Rate constant for SOC channels",
                f"K_e = {self.K_e} Half-maximal [Ca2+] ER for SOC channels",
                f"V_pm = {self.V_pm} Maximum capacity of plasma membrane pump",
                f"K_pm = {self.K_pm} Half-maximal [Ca2+] for plasma membrane pump",
                f"gamma = {self.gamma} Ratio of cytoplasmic volume to ER volume",
                f"K_h = {self.K_h} The concentration of Ca2+ activated IP3R",
                f"K_tau = {self.K_tau} " "The concentration of Ca 2+ in response to β",
                f"tau_max = {self.tau_max} Maximum period for IP3R",
                f"beta_p = {self.beta_p} Rate of decay of p to its steady state",
                f"p_s = {self.p_s}",
                f"k_beta = {self.k_beta}",
                f"K_c = {self.K_c} Half-maximal [Ca2+] for IP3R",
                f"T_Plank = {self.T_Plank} T=500 s, Reference timescale",
                f"K4_Plank = {self.K4_Plank} microM, Michaelis-Menten constant",
                f"qmax_Plank = {self.qmax_Plank} Max. " "WSS-induced Ca2+ influx rate",
                f"alpha_Plank = {self.alpha_Plank} alpha=2 No-load channel "
                "constant, (1 + alpha)**(-1) is the probability that"
                "a channel is in the open state in the no-load case W=0",
                f"fe_Plank = {self.fe_Plank} "
                "fe=0.0134 is the fraction of the energy within the membrane that"
                "gates the shear-sensitive Ca2+ channels",
                "\nDefining the initial conditions",
                f"c_0 = {self.c_0} The initial concentration of Ca2+ in cytosol",
                f"c_e_0 = {self.c_e_0} The initial concentration of Ca2+ in ER",
                f"h_0 = {self.h_0} "
                "The initial rate at which Ca2+ can activate IP3Rs",
                f"p_0 = {self.p_0} The initial rate of Ca2+ activation IP3",
            )
        )

    @eqx.filter_jit
    def __call__(self, t: jax.Array, S: jax.Array, args=None) -> jax.Array:
        old_system = self.dSdt(t, S, args)
        if self.mp is not None:
            return old_system.at[0].set(old_system[0] + self.J_magn(self.mp, t))
        return old_system

    def dSdt(self, t: jax.Array, S: jax.Array, args=None) -> jax.Array:
        c, c_e, h, p = S

        tau_h = self.tau_max * self.K_tau**4 / (self.K_tau**4 + c**4)

        return jnp.array(
            [
                self.J_IP3R(c, c_e, h, p)
                - self.J_serca(c, c_e)
                + self.delta * (self.J_in(c_e) - self.J_pm(c)),
                self.gamma * (self.J_serca(c, c_e) - self.J_IP3R(c, c_e, h, p)),
                (self.h_infty(c) - h) / (tau_h),
                self.beta_p * (self.p_s - p),
            ]
        )

    @property
    def initial_values(self) -> jax.Array:
        return jnp.array([self.c_0, self.c_e_0, self.h_0, self.p_0], dtype=jnp.float32)

    # Dependence of dimensionless magnetic pressure on dimensionless time
    def tau_magn(self, mp: MagneticFieldParameters, t):
        # τ_magn – WSS (напруження зсуву), індуковане впливом МП на ланцюжок БМН, вбудованих у клітинну мембрану (1)
        return mp.P(t)  # equation (5 for gradient МF,6 for uniform МF) from reference 2

    # Dimensional function returning the dimensionless strain energy density function
    # (розмірна функція густини енергії деформації) W
    # for a two-dimensional membrane depending on the Wall Shear Stress (WSS - напруження зсуву стінки (мембрани))
    # tau_w in  Pa
    def w(self, tau_w):
        x = self.epsilon_Plank * tau_w * self.l_Plank + jnp.sqrt(
            16 * (self.delta_Plank**2)
            + (self.epsilon_Plank * tau_w * self.l_Plank) ** 2
        )
        # x is denominator (знаменник) in equation after (24)
        return (x - 4 * self.delta_Plank) ** 2 / (
            8 * x
        )  # equation after (24) from reference 3

    # Probability of channel opening
    def p_open(self, tau_w):
        return 1 / (
            1
            + self.alpha_Plank
            * jnp.exp(
                -self.fe_Plank * self.w(tau_w) / (self.k * self.Te_Plank * self.N_Plank)
            )
        )  # equation (2) from reference 2

    # Dimensionless q_in dependence on Wall Shear Stress
    def q_in(self, tau_w):
        return self.qmax_Plank / (
            1
            + self.alpha_Plank
            * jnp.exp(
                -self.fe_Plank * self.w(tau_w) / (self.k * self.Te_Plank * self.N_Plank)
            )
        )  # second term of equation (4) from reference 2

    def h_infty(self, c):
        return self.K_h**4 / (self.K_h**4 + c**4)

    def J_in(self, c_e):
        return self.alpha_0 + self.alpha_1 * self.K_e**4 / (self.K_e**4 + c_e**4)

    def J_pm(self, c):
        return self.V_pm * c**2 / (self.K_pm**2 + c**2)

    def J_serca(self, c, c_e):
        return self.V_p * (c**2 - self.K * c_e**2) / (c**2 + self.k_p**2)

    def J_IP3R(self, c, c_e, h, p):
        B = p**2 / (self.k_p**2 + p**2)
        m_hat_alpha = m_hat_beta = c**4 / (self.K_c**4 + c**4)
        h_hat_alpha = self.h_infty(c)
        alpha = (1 - B) * (1 - m_hat_alpha * h_hat_alpha)
        beta = B * m_hat_beta * h
        P_o = beta / (beta + self.k_beta * (beta + alpha))
        return self.k_f * P_o * (c_e - c)

    def J_magn(self, mp: MagneticFieldParameters, t):
        return self.q_in(self.tau_magn(mp, t))
