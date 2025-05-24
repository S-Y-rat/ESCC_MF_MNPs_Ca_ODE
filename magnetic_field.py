from typing import NamedTuple
import jax.numpy as jnp


class MagneticFieldParameters(NamedTuple):
    """The class MagneticFieldParameters() contains parameters of magnetic field flux density, frequency and gradient


    This class contains also parameters of magnetic nanoparticles.
    """

    regime: str = "gradient"
    time_dependence: str = "oscillating"
    Ms: float = (
        510e3  # Ms=510*(10**3)  # Ms is magnetization saturation of magnetic nanoparticle, A/m
    )

    r: float = 100e-9  # r=100*(10**(-9))  # r is magnetic nanoparticle radius, m
    # G0 = 300  # G0 is the maximum value of dBz/dz, B=i*const – j*G*y+ k*G*z
    # G0 = 10**5 # maximum achievable G0 максимально досяжний G0
    G0: float = (
        145.0  # G0=30  G0 is the maximum value of dBz/dz, B=i*const – j*G*y+ k*G*z   (4)
    )
    omega: float = (
        1.7 * jnp.pi * 0.001
    )  # omega=0.2  # omega is dimensionless frequency of magnetic field oscillation
    T1: float = (
        2 * jnp.pi / omega
    )  # T1=2*np.pi/omega  # T1 is dimensionless period of magnetic field oscillation
    N: int = 10
    B: float = 0.5e-4 * 500

    @property
    def __is_uni_rot(self):
        return self.regime == "uniform" and self.time_dependence == "rotating"

    @property
    def __is_grad_osc(self):
        return self.regime == "gradient" and self.time_dependence == "oscillating"

    @property
    def legend_MF(self) -> str:
        if self.__is_uni_rot:
            return f"$B={self.B*1e3}$ mT, $\\nu={self.omega*1e3/jnp.pi}\\cdot\\pi$ mHz"
        return f"G0={self.G0} T/m"

    @property
    def legend_0MF(self) -> str:
        if self.__is_uni_rot:
            return "B=0 mT"
        return "G0=0 T/m"

    def __str__(self) -> str:
        if self.__is_grad_osc:
            return "\n".join(
                (
                    "The set of magnetic parameters:",
                    f"r = {self.r} m, Magnetic nanoparticle radius",
                    f"Ms = {self.Ms} A/m, Magnetization saturation of magnetic nanoparticle",
                    f"G0 = {self.G0} T/m, Maximum value of dBz/dz, B=i*const - j*G*y+ k*G*z",
                    f"omega = {self.omega} Dimensionless frequency of magnetic field oscillation",
                    f"T1 = {self.T1} Dimensionless period of magnetic field oscillation",
                )
            )
        if self.__is_uni_rot:
            return "\n".join(
                (
                    "The set of magnetic parameters:",
                    f"r = {self.r} m, Magnetic nanoparticle radius",
                    f"Ms = {self.Ms} A/m, Magnetization saturation of magnetic nanoparticle",
                    f"omega = {self.omega} Dimensionless frequency of magnetic field oscillation",
                    f"T1 = {self.T1} Dimensionless period of magnetic field oscillation",
                    f"B = {self.B} T, uniform rotating magnetic field flux density",
                    f"N = {self.N}, number of magnetic nanoparticles in the chain",
                )
            )
        return "Choose correct regime and time dependence of MF influence"

    def P(self, t):
        """P - magnetic pressure магнітний тиск"""
        if self.__is_grad_osc:
            return 4 * self.r * self.Ms * self.G0 * jnp.cos(self.omega * t) / 3  # (8)
        if self.__is_uni_rot:
            return (
                jnp.pi * self.Ms * self.B * jnp.cos(self.omega * t) / (6 * self.N)
            )  # (?)
        raise AttributeError(
            f"No P for {self.regime} regime with {self.time_dependence} time dependence"
        )
