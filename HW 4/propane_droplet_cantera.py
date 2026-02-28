#!/usr/bin/env python3
"""
Propane droplet combustion sweep using Python + Cantera.

Computes:
    - fuel mass burning rate, mdot_F [kg/s]
    - fuel vapor mass fraction at surface, Y_F,s [-]
    - droplet surface temperature, T_s [K]
    - flame temperature, T_f [K]
    - flame radius, r_f [m]

for ambient temperatures from 300 K to 1000 K at 1 atm and 10 atm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ModelParams:
    d0_m: float = 100e-6
    y_o_inf: float = 1.0
    mechanism: str = "gri30.yaml"
    # NIST Antoine ranges for propane, log10(P_bar)=A-B/(T+C), T[K]
    antoine_low: tuple[float, float, float] = (4.01158, 834.26, -22.763)   # 166.02-231.41 K
    antoine_mid: tuple[float, float, float] = (3.98292, 819.296, -24.417)  # 230.6-320.7 K
    antoine_high: tuple[float, float, float] = (4.53678, 1149.36, 24.906)  # 277.6-360.8 K
    # NIST correlation: Hvap(kJ/mol)=A*exp(-alpha*Tr)*(1-Tr)^beta, Tr=T/Tc
    tc_k: float = 369.8
    hvap_a_kj_per_mol: float = 27.9
    hvap_alpha: float = 0.0208
    hvap_beta: float = 0.3766


class CanteraProps:
    def __init__(self, mechanism: str) -> None:
        self.gas = ct.Solution(mechanism)
        self.eq = ct.Solution(mechanism)
        self.w_f = self.gas.molecular_weights[self.gas.species_index("C3H8")]  # kg/kmol
        self.w_o2 = self.gas.molecular_weights[self.gas.species_index("O2")]  # kg/kmol
        # Stoichiometric oxidizer/fuel mass ratio for C3H8 + 5 O2 -> ...
        self.sigma = (5.0 * self.w_o2) / self.w_f
        self.mix = "C3H8:1, O2:5"

    def cp_and_lambda(self, t_k: float, p_pa: float) -> tuple[float, float]:
        self.gas.TPX = t_k, p_pa, self.mix
        return self.gas.cp_mass, self.gas.thermal_conductivity

    def species_h_mole(self, species: str, t_k: float, p_pa: float) -> float:
        self.gas.TPX = t_k, p_pa, f"{species}:1"
        return self.gas.enthalpy_mole

    def heat_of_combustion(self, t_k: float, p_pa: float) -> float:
        h_r = self.species_h_mole("C3H8", t_k, p_pa) + 5.0 * self.species_h_mole("O2", t_k, p_pa)
        h_p = 3.0 * self.species_h_mole("CO2", t_k, p_pa) + 4.0 * self.species_h_mole("H2O", t_k, p_pa)
        return (h_r - h_p) / self.w_f  # J/kg_fuel

    def adiabatic_flame_temperature(self, t_k: float, p_pa: float) -> float:
        self.eq.TPX = t_k, p_pa, self.mix
        self.eq.equilibrate("HP")
        return self.eq.T


def propane_psat_pa(t_k: float, p: ModelParams) -> float:
    # Piecewise Antoine fit by temperature range.
    if t_k < 230.6:
        a, b, c = p.antoine_low
    elif t_k < 277.6:
        a, b, c = p.antoine_mid
    else:
        a, b, c = p.antoine_high
    p_bar = 10.0 ** (a - b / (t_k + c))
    return p_bar * 1.0e5


def y_f_surface_from_ts(t_s_k: float, p_pa: float, p: ModelParams, props: CanteraProps) -> float:
    x_f = np.clip(propane_psat_pa(t_s_k, p) / p_pa, 1.0e-12, 0.999999)
    y_f = (x_f * props.w_f) / (x_f * props.w_f + (1.0 - x_f) * props.w_o2)
    return float(np.clip(y_f, 1.0e-12, 0.999999))


def h_fg_effective(t_s_k: float, p: ModelParams, props: CanteraProps) -> float:
    tr = np.clip(t_s_k / p.tc_k, 0.0, 0.9999)
    hvap_kj_per_mol = p.hvap_a_kj_per_mol * math.exp(-p.hvap_alpha * tr) * (1.0 - tr) ** p.hvap_beta
    return hvap_kj_per_mol * 1.0e6 / props.w_f  # J/kg


def residuals(
    x: np.ndarray,
    t_inf_k: float,
    p_pa: float,
    t_f_ad_k: float,
    p: ModelParams,
    props: CanteraProps,
) -> np.ndarray:
    mdot, y_fs, t_s, t_f, r_f = x
    r_s = 0.5 * p.d0_m

    mdot = max(mdot, 1.0e-15)
    y_fs = float(np.clip(y_fs, 1.0e-12, 0.999999))
    t_s = float(np.clip(t_s, 140.0, 360.0))
    t_f = float(np.clip(t_f, 400.0, 5000.0))
    r_f = max(r_f, 1.001 * r_s)

    b_m = y_fs / (1.0 - y_fs)
    cp_g, lam_g = props.cp_and_lambda(0.5 * (t_inf_k + t_f), p_pa)
    q_c = props.heat_of_combustion(t_inf_k, p_pa)
    h_eff = h_fg_effective(t_s, p, props)

    e1 = b_m - (p.y_o_inf * q_c / props.sigma + cp_g * (t_inf_k - t_s)) / h_eff
    e2 = y_fs - y_f_surface_from_ts(t_s, p_pa, p, props)
    e3 = t_f - t_f_ad_k
    e4 = r_f - r_s * math.log(1.0 + b_m) / math.log(1.0 + p.y_o_inf / props.sigma)
    mdot_calc = math.pi * p.d0_m * lam_g / cp_g * math.log(1.0 + b_m)
    e5 = mdot - mdot_calc

    return np.array([e1, e2, e3, e4, e5], dtype=float)


def invert_psat_target(target_pa: float, p: ModelParams) -> float:
    lo, hi = 140.0, 360.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if propane_psat_pa(mid, p) > target_pa:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def initial_guess(
    t_inf_k: float,
    p_pa: float,
    t_f_ad_k: float,
    p: ModelParams,
    props: CanteraProps,
    prev: np.ndarray | None,
) -> np.ndarray:
    if prev is not None:
        x0 = prev.copy()
        x0[3] = t_f_ad_k
        return x0

    t_s0 = invert_psat_target(0.97 * p_pa, p)
    y0 = y_f_surface_from_ts(t_s0, p_pa, p, props)
    b0 = y0 / (1.0 - y0)
    cp_g, lam_g = props.cp_and_lambda(0.5 * (t_inf_k + t_f_ad_k), p_pa)
    r_s = 0.5 * p.d0_m
    r_f0 = r_s * math.log(1.0 + b0) / math.log(1.0 + p.y_o_inf / props.sigma)
    mdot0 = math.pi * p.d0_m * lam_g / cp_g * math.log(1.0 + b0)
    return np.array([mdot0, y0, t_s0, t_f_ad_k, r_f0], dtype=float)


def solve_state(
    t_inf_k: float,
    p_pa: float,
    p: ModelParams,
    props: CanteraProps,
    prev: np.ndarray | None,
) -> np.ndarray:
    t_f_ad = props.adiabatic_flame_temperature(t_inf_k, p_pa)
    x = initial_guess(t_inf_k, p_pa, t_f_ad, p, props, prev)

    for _ in range(40):
        f = residuals(x, t_inf_k, p_pa, t_f_ad, p, props)
        if np.linalg.norm(f, ord=2) < 1.0e-8:
            break

        j = np.zeros((5, 5), dtype=float)
        for k in range(5):
            h = 1.0e-6 * max(abs(x[k]), 1.0)
            xp = x.copy()
            xp[k] += h
            fp = residuals(xp, t_inf_k, p_pa, t_f_ad, p, props)
            j[:, k] = (fp - f) / h

        dx = np.linalg.lstsq(j, -f, rcond=None)[0]
        n0 = np.linalg.norm(f, ord=2)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            xn = x + alpha * dx
            xn[0] = max(xn[0], 1.0e-15)
            xn[1] = np.clip(xn[1], 1.0e-12, 0.999999)
            xn[2] = np.clip(xn[2], 140.0, 360.0)
            xn[3] = np.clip(xn[3], 400.0, 5000.0)
            xn[4] = max(xn[4], 1.001 * 0.5 * p.d0_m)
            fn = residuals(xn, t_inf_k, p_pa, t_f_ad, p, props)
            if np.linalg.norm(fn, ord=2) < n0:
                x = xn
                improved = True
                break
        if not improved:
            break

    return x


def run_sweep() -> list[dict[str, float]]:
    p = ModelParams()
    props = CanteraProps(p.mechanism)
    t_vals = np.linspace(300.0, 1000.0, 71)
    p_vals = [ct.one_atm, 10.0 * ct.one_atm]
    rows: list[dict[str, float]] = []

    for p_pa in p_vals:
        prev = None
        for t_inf in t_vals:
            x = solve_state(t_inf, p_pa, p, props, prev)
            prev = x.copy()
            mdot, y_fs, t_s, t_f, r_f = x
            rows.append(
                {
                    "pressure_atm": p_pa / ct.one_atm,
                    "t_inf_k": t_inf,
                    "mdot_f_kg_s": mdot,
                    "y_f_s": y_fs,
                    "t_s_k": t_s,
                    "t_f_k": t_f,
                    "r_f_m": r_f,
                    "r_f_over_r_s": r_f / (0.5 * p.d0_m),
                }
            )

    return rows


def save_csv(rows: list[dict[str, float]], path: str) -> None:
    header = [
        "pressure_atm",
        "t_inf_k",
        "mdot_f_kg_s",
        "y_f_s",
        "t_s_k",
        "t_f_k",
        "r_f_m",
        "r_f_over_r_s",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    f"{r[c]:.10e}" if c not in ("pressure_atm", "t_inf_k") else f"{r[c]:.6f}"
                    for c in header
                )
                + "\n"
            )


def save_plots(rows: list[dict[str, float]], path: str) -> None:
    pressures = sorted(set(r["pressure_atm"] for r in rows))
    fig, axs = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    ax = axs.ravel()

    for p_atm in pressures:
        data = [r for r in rows if abs(r["pressure_atm"] - p_atm) < 1.0e-12]
        t = np.array([r["t_inf_k"] for r in data])
        mdot = np.array([r["mdot_f_kg_s"] for r in data])
        y = np.array([r["y_f_s"] for r in data])
        ts = np.array([r["t_s_k"] for r in data])
        tf = np.array([r["t_f_k"] for r in data])
        rf_um = 1.0e6 * np.array([r["r_f_m"] for r in data])
        label = f"{p_atm:.0f} atm"

        ax[0].plot(t, mdot, label=label, linewidth=2)
        ax[1].plot(t, y, label=label, linewidth=2)
        ax[2].plot(t, ts, label=label, linewidth=2)
        ax[3].plot(t, tf, label=label, linewidth=2)
        ax[4].plot(t, rf_um, label=label, linewidth=2)

    ax[0].set_ylabel(r"$\dot{m}_F$ [kg/s]")
    ax[1].set_ylabel(r"$Y_{F,s}$ [-]")
    ax[2].set_ylabel(r"$T_s$ [K]")
    ax[3].set_ylabel(r"$T_f$ [K]")
    ax[4].set_ylabel(r"$r_f$ [$\mu$m]")

    for i in range(5):
        ax[i].set_xlabel(r"$T_\infty$ [K]")
        ax[i].grid(alpha=0.3)
        ax[i].legend()

    ax[5].axis("off")
    fig.suptitle("Propane Droplet Combustion (Cantera-based Property Model)", fontsize=14)
    fig.savefig(path, dpi=200)


def main() -> None:
    rows = run_sweep()
    save_csv(rows, "droplet_results_propane.csv")
    save_plots(rows, "droplet_results_propane.png")

    for p_atm in (1.0, 10.0):
        data = [r for r in rows if abs(r["pressure_atm"] - p_atm) < 1.0e-12]
        first = data[0]
        last = data[-1]
        print(
            f"{p_atm:.0f} atm: T_inf {first['t_inf_k']:.0f}->{last['t_inf_k']:.0f} K, "
            f"mdot {first['mdot_f_kg_s']:.3e}->{last['mdot_f_kg_s']:.3e} kg/s, "
            f"Ts {first['t_s_k']:.2f}->{last['t_s_k']:.2f} K, "
            f"Tf {first['t_f_k']:.1f}->{last['t_f_k']:.1f} K, "
            f"rf {1e6*first['r_f_m']:.1f}->{1e6*last['r_f_m']:.1f} um"
        )


if __name__ == "__main__":
    main()
