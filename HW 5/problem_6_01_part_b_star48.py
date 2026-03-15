from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Star48Inputs:
    grain_length_m: float = 1.2
    grain_diameter_m: float = 1.2
    cavity_diameter_m: float = 0.5
    throat_diameter_m: float = 0.1
    exit_diameter_m: float = 0.75
    rho_prop_kg_m3: float = 1800.0
    c_star_m_s: float = 1527.0
    gamma: float = 1.2
    a_cm_s_mpa_n: float = 0.4
    p_ambient_pa: float = 0.0  # upper-stage vacuum assumption


@dataclass
class RunResult:
    n: float
    t_s: np.ndarray
    p_pa: np.ndarray
    thrust_n: np.ndarray

    @property
    def burn_time_s(self) -> float:
        return float(self.t_s[-1]) if self.t_s.size else 0.0


def area_ratio_from_mach(M: float, gamma: float) -> float:
    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * term**exponent


def exit_mach_from_area_ratio(epsilon: float, gamma: float) -> float:
    """Supersonic root of A/A* = epsilon via bisection."""
    lo, hi = 1.0001, 50.0
    for _ in range(250):
        mid = 0.5 * (lo + hi)
        if area_ratio_from_mach(mid, gamma) < epsilon:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def thrust_coefficient(gamma: float, area_ratio: float, p_ambient_over_pc: float = 0.0) -> float:
    M_e = exit_mach_from_area_ratio(area_ratio, gamma)
    p_e_over_p_c = (1.0 + 0.5 * (gamma - 1.0) * M_e**2) ** (-gamma / (gamma - 1.0))

    momentum_term = np.sqrt(
        (2.0 * gamma**2 / (gamma - 1.0))
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        * (1.0 - p_e_over_p_c ** ((gamma - 1.0) / gamma))
    )
    pressure_term = (p_e_over_p_c - p_ambient_over_pc) * area_ratio
    return float(momentum_term + pressure_term)


def burn_rate_constant_si(a_cm_s_mpa_n: float, n: float) -> float:
    """Convert a from [cm/s-MPa^n] to [m/s-Pa^n]."""
    a_m_s_mpa_n = a_cm_s_mpa_n * 1e-2
    return a_m_s_mpa_n / (1e6**n)


def simulate_star48(n: float, dt: float, inputs: Star48Inputs) -> RunResult:
    r0 = 0.5 * inputs.cavity_diameter_m
    r_outer = 0.5 * inputs.grain_diameter_m
    A_t = 0.25 * np.pi * inputs.throat_diameter_m**2
    A_e = 0.25 * np.pi * inputs.exit_diameter_m**2
    epsilon = A_e / A_t

    a_si = burn_rate_constant_si(inputs.a_cm_s_mpa_n, n)
    C_F = thrust_coefficient(inputs.gamma, epsilon, p_ambient_over_pc=0.0)

    t_vals: list[float] = []
    p_vals: list[float] = []
    F_vals: list[float] = []

    t = 0.0
    r = r0

    while r < r_outer:
        A_b = 2.0 * np.pi * inputs.grain_length_m * r

        # Quasi-steady chamber pressure from mass generation = nozzle mass flow.
        p_c = (inputs.rho_prop_kg_m3 * a_si * inputs.c_star_m_s * A_b / A_t) ** (1.0 / (1.0 - n))

        r_dot = a_si * p_c**n
        thrust = C_F * p_c * A_t

        t_vals.append(t)
        p_vals.append(p_c)
        F_vals.append(thrust)

        # Forward Euler step for regression.
        r += r_dot * dt
        t += dt

    return RunResult(
        n=n,
        t_s=np.asarray(t_vals),
        p_pa=np.asarray(p_vals),
        thrust_n=np.asarray(F_vals),
    )


def make_plots(results: list[RunResult], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for res in results:
        axes[0].plot(res.t_s, res.p_pa / 1e6, linewidth=1.8, label=f"n = {res.n:.1f}")
        axes[1].plot(res.t_s, res.thrust_n / 1e3, linewidth=1.8, label=f"n = {res.n:.1f}")

    axes[0].set_ylabel("Chamber Pressure [MPa]")
    axes[0].set_title("Star 48 (Cylindrical Cavity): Pressure vs Time")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Vacuum Thrust [kN]")
    axes[1].set_title("Star 48 (Cylindrical Cavity): Thrust vs Time")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def print_summary(results: list[RunResult]) -> None:
    print("Forward-Euler summary (vacuum thrust)")
    print("-------------------------------------")
    for res in results:
        p0 = res.p_pa[0] / 1e6
        pf = res.p_pa[-1] / 1e6
        F0 = res.thrust_n[0] / 1e3
        Ff = res.thrust_n[-1] / 1e3
        print(
            f"n = {res.n:.1f}: burn = {res.burn_time_s:.3f} s, "
            f"p0 = {p0:.3f} MPa, pf = {pf:.3f} MPa, "
            f"F0 = {F0:.1f} kN, Ff = {Ff:.1f} kN"
        )


def main() -> None:
    inputs = Star48Inputs()
    n_values = [0.3, 0.4, 0.5]
    dt = 1e-3

    results = [simulate_star48(n=n, dt=dt, inputs=inputs) for n in n_values]

    out_dir = Path("plots_problem_6_01")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "star48_pressure_thrust.png"

    make_plots(results, out_png)
    print_summary(results)
    print(f"Saved plot: {out_png.resolve()}")


if __name__ == "__main__":
    main()
