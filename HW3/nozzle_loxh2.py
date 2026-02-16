#!/usr/bin/env python3
"""
LOX/LH2 nozzle expansion analysis (equilibrium + frozen) using Cantera.

Implements the HW3 part (d) workflow in Python:
- sets chamber inlet state from LOX/LH2 at O/F=6, T=200 K, P=2e7 Pa
- computes chamber equilibrium stagnation state via HP equilibration
- steps through nozzle by decreasing pressure
- enforces isentropic expansion (SP)
- computes u from energy conservation, M from sound speed, and A/A* from continuity
- repeats for frozen chemistry (composition fixed at chamber-equilibrium composition)

Outputs:
- equilibrium_nozzle.csv
- frozen_nozzle.csv
- summary printed to terminal

Note:
- If running Cantera >= 3, .cti input files are deprecated. This script attempts to
  auto-convert .cti to .yaml when possible.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

try:
    import cantera as ct
except ImportError as exc:
    raise SystemExit(
        "Cantera is not installed. Install it first, e.g. `pip install cantera`."
    ) from exc


@dataclass
class FlowState:
    p: float
    T: float
    rho: float
    h: float
    s: float
    a: float
    u: float
    M: float
    Y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOX/LH2 nozzle solver with Cantera")
    parser.add_argument(
        "--mech",
        type=str,
        default="h2o2_highT.cti",
        help="Path to mechanism file (.cti or .yaml)",
    )
    parser.add_argument("--Tin", type=float, default=200.0, help="Inlet temperature [K]")
    parser.add_argument("--Pin", type=float, default=2.0e7, help="Inlet pressure [Pa]")
    parser.add_argument(
        "--of",
        type=float,
        default=6.0,
        help="Oxidizer-to-fuel mass ratio O/F (LOX/LH2)",
    )
    parser.add_argument(
        "--throat-diameter",
        type=float,
        default=0.2616,
        help="Nozzle throat diameter [m]",
    )
    parser.add_argument(
        "--expansion-ratio",
        type=float,
        default=77.52,
        help="Nozzle geometric expansion ratio A_e/A*",
    )
    parser.add_argument(
        "--nscan",
        type=int,
        default=300,
        help="Pressure scan points for throat/exit bracketing",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=350,
        help="Number of pressure points saved from chamber to exit",
    )
    parser.add_argument(
        "--pmin",
        type=float,
        default=1.0,
        help="Minimum pressure [Pa] used for bracketing",
    )
    parser.add_argument(
        "--pa-vac",
        type=float,
        default=0.0,
        help="Vacuum ambient pressure [Pa]",
    )
    parser.add_argument(
        "--pa-sl",
        type=float,
        default=101325.0,
        help="Sea-level ambient pressure [Pa]",
    )
    parser.add_argument(
        "--g0",
        type=float,
        default=9.80665,
        help="Standard gravity [m/s^2] for Isp",
    )
    return parser.parse_args()


def _cti_to_yaml_if_needed(mech_path: Path) -> Path:
    if mech_path.suffix.lower() != ".cti":
        return mech_path

    yaml_path = mech_path.with_suffix(".yaml")

    # Try conversion utility available in modern Cantera versions.
    try:
        from cantera import cti2yaml  # type: ignore

        if (not yaml_path.exists()) or (yaml_path.stat().st_mtime < mech_path.stat().st_mtime):
            cti2yaml.convert(
                filename=str(mech_path),
                output_name=str(yaml_path),
            )
        return yaml_path
    except Exception:
        # Fall back to .cti direct load for old Cantera builds.
        return mech_path


def load_gas(mech: str) -> Tuple[ct.Solution, Path]:
    mech_path = Path(mech).expanduser().resolve()
    if not mech_path.exists():
        raise FileNotFoundError(f"Mechanism file not found: {mech_path}")

    load_path = _cti_to_yaml_if_needed(mech_path)
    gas = ct.Solution(str(load_path))
    return gas, load_path


def lox_lh2_mole_composition(of_mass: float) -> Dict[str, float]:
    """
    Build H2/O2 reactant mole numbers corresponding to target mass O/F.

    Let n_O2 = 1 mol (mass = 32 g). Then m_H2 = 32 / (O/F) g,
    so n_H2 = m_H2 / 2 = 16 / (O/F) mol.
    """
    if of_mass <= 0.0:
        raise ValueError("O/F must be positive")

    n_o2 = 1.0
    n_h2 = 16.0 / of_mass
    return {"O2": n_o2, "H2": n_h2}


def eq_sound_speed(gas: ct.Solution, rel_dp: float = 1.0e-5) -> float:
    """
    Equilibrium sound speed from a^2 = (dp/drho)_s,eq via finite differences.

    This follows the formal definition for equilibrium flow.
    """
    saved = gas.state
    s0 = gas.entropy_mass
    p0 = gas.P

    dp = max(1.0, rel_dp * p0)
    p_minus = max(1.0, p0 - dp)
    p_plus = p0 + dp

    # Central difference at constant entropy with equilibrium re-established.
    gas.SP = s0, p_plus
    gas.equilibrate("SP")
    rho_plus = gas.density

    gas.SP = s0, p_minus
    gas.equilibrate("SP")
    rho_minus = gas.density

    gas.state = saved

    drho = rho_plus - rho_minus
    dp_total = p_plus - p_minus
    if drho <= 0.0 or dp_total <= 0.0:
        raise RuntimeError("Invalid derivative in eq_sound_speed; check state/step size")

    a2 = dp_total / drho
    return float(math.sqrt(max(a2, 0.0)))


def bisect_log(
    f: Callable[[float], float],
    p_lo: float,
    p_hi: float,
    tol_rel: float = 1.0e-8,
    max_iter: int = 120,
) -> float:
    """Bisection in pressure with geometric midpoint (good for wide dynamic ranges)."""
    if not (p_lo > 0.0 and p_hi > 0.0 and p_lo < p_hi):
        raise ValueError("Require 0 < p_lo < p_hi")

    f_lo = f(p_lo)
    f_hi = f(p_hi)
    if f_lo == 0.0:
        return p_lo
    if f_hi == 0.0:
        return p_hi
    if f_lo * f_hi > 0.0:
        raise ValueError("Root is not bracketed in bisect_log")

    for _ in range(max_iter):
        p_mid = math.sqrt(p_lo * p_hi)
        f_mid = f(p_mid)
        if f_mid == 0.0:
            return p_mid

        if f_lo * f_mid < 0.0:
            p_hi = p_mid
            f_hi = f_mid
        else:
            p_lo = p_mid
            f_lo = f_mid

        if abs(p_hi - p_lo) / p_mid < tol_rel:
            return p_mid

    return math.sqrt(p_lo * p_hi)


def bracket_crossing(
    values: Iterable[float],
    pressures: Iterable[float],
    target: float,
) -> Tuple[float, float]:
    """
    Find first bracket where sequence crosses target while pressure decreases.

    Returns (p_lo, p_hi) with p_lo < p_hi.
    """
    vals = list(values)
    ps = list(pressures)
    if len(vals) != len(ps):
        raise ValueError("values and pressures must have same length")

    for i in range(1, len(vals)):
        v0 = vals[i - 1] - target
        v1 = vals[i] - target
        if v0 == 0.0:
            p = ps[i - 1]
            return p, p
        if v1 == 0.0:
            p = ps[i]
            return p, p
        if v0 * v1 < 0.0:
            p_hi = max(ps[i - 1], ps[i])
            p_lo = min(ps[i - 1], ps[i])
            return p_lo, p_hi

    raise RuntimeError("No crossing found in provided scan range")


def make_equilibrium_evaluator(
    gas: ct.Solution,
    s0: float,
    h0: float,
) -> Callable[[float], FlowState]:
    def eval_state(p: float) -> FlowState:
        gas.SP = s0, p
        gas.equilibrate("SP")

        h = gas.enthalpy_mass
        u2 = max(0.0, 2.0 * (h0 - h))
        u = math.sqrt(u2)
        a = eq_sound_speed(gas)
        M = u / a if a > 0.0 else float("nan")

        return FlowState(
            p=float(gas.P),
            T=float(gas.T),
            rho=float(gas.density),
            h=float(h),
            s=float(gas.entropy_mass),
            a=float(a),
            u=float(u),
            M=float(M),
            Y=gas.Y.copy(),
        )

    return eval_state


def make_frozen_evaluator(
    gas: ct.Solution,
    s0: float,
    h0: float,
) -> Callable[[float], FlowState]:
    def eval_state(p: float) -> FlowState:
        gas.SP = s0, p  # composition remains fixed unless equilibrate() is called

        h = gas.enthalpy_mass
        u2 = max(0.0, 2.0 * (h0 - h))
        u = math.sqrt(u2)
        a = float(gas.sound_speed)
        M = u / a if a > 0.0 else float("nan")

        return FlowState(
            p=float(gas.P),
            T=float(gas.T),
            rho=float(gas.density),
            h=float(h),
            s=float(gas.entropy_mass),
            a=float(a),
            u=float(u),
            M=float(M),
            Y=gas.Y.copy(),
        )

    return eval_state


def find_throat_pressure(
    eval_state: Callable[[float], FlowState],
    p0: float,
    p_min: float,
    nscan: int,
) -> float:
    ps = np.geomspace(p0, p_min, nscan)
    p_prev = float(ps[0])
    f_prev = eval_state(p_prev).M - 1.0

    for p in ps[1:]:
        p_curr = float(p)
        f_curr = eval_state(p_curr).M - 1.0
        if f_prev == 0.0:
            return p_prev
        if f_curr == 0.0:
            return p_curr
        if f_prev * f_curr < 0.0:
            p_lo = min(p_prev, p_curr)
            p_hi = max(p_prev, p_curr)
            return bisect_log(lambda x: eval_state(x).M - 1.0, p_lo, p_hi)
        p_prev, f_prev = p_curr, f_curr

    raise RuntimeError(
        "Failed to bracket throat (M=1). Increase --nscan or lower --pmin."
    )


def find_exit_pressure_for_area_ratio(
    eval_state: Callable[[float], FlowState],
    p_throat: float,
    rho_u_star: float,
    area_ratio_target: float,
    p_min: float,
    nscan: int,
) -> float:
    ps = np.geomspace(p_throat, p_min, nscan)

    st_prev = eval_state(float(ps[0]))
    ar_prev = rho_u_star / (st_prev.rho * st_prev.u)
    f_prev = ar_prev - area_ratio_target
    p_prev = float(ps[0])

    for p in ps[1:]:
        p_curr = float(p)
        st_curr = eval_state(p_curr)
        ar_curr = rho_u_star / (st_curr.rho * st_curr.u)
        f_curr = ar_curr - area_ratio_target

        if f_prev == 0.0:
            return p_prev
        if f_curr == 0.0:
            return p_curr
        if f_prev * f_curr < 0.0:
            p_lo = min(p_prev, p_curr)
            p_hi = max(p_prev, p_curr)

            def f(x: float) -> float:
                st = eval_state(x)
                return rho_u_star / (st.rho * st.u) - area_ratio_target

            return bisect_log(f, p_lo, p_hi)

        p_prev, f_prev = p_curr, f_curr

    raise RuntimeError(
        "Failed to bracket exit pressure for target area ratio. "
        "Increase --nscan or lower --pmin."
    )


def build_profile(
    eval_state: Callable[[float], FlowState],
    species_names: List[str],
    p_chamber: float,
    p_exit: float,
    rho_u_star: float,
    mdot: float,
    A_star: float,
    npoints: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Include exact chamber state separately because u=0 at stagnation point.
    p_flow = np.geomspace(p_chamber * (1.0 - 1.0e-8), p_exit, npoints - 1)
    pressures = np.concatenate(([p_chamber], p_flow))

    T = np.zeros_like(pressures)
    rho = np.zeros_like(pressures)
    u = np.zeros_like(pressures)
    M = np.zeros_like(pressures)
    a = np.zeros_like(pressures)
    A = np.full_like(pressures, np.nan)
    A_ratio = np.full_like(pressures, np.nan)
    Y = np.zeros((pressures.size, len(species_names)))

    for i, p in enumerate(pressures):
        st = eval_state(float(p))
        T[i] = st.T
        rho[i] = st.rho
        u[i] = st.u
        M[i] = st.M
        a[i] = st.a
        Y[i, :] = st.Y

        # First row represents stagnation chamber state.
        if i == 0:
            u[i] = 0.0
            M[i] = 0.0
            continue

        if st.u > 1.0e-12:
            A[i] = mdot / (st.rho * st.u)
            A_ratio[i] = A[i] / A_star

    return pressures, T, rho, u, M, a, A, A_ratio, Y


def write_csv(
    path: Path,
    species_names: List[str],
    p: np.ndarray,
    T: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    M: np.ndarray,
    a: np.ndarray,
    A: np.ndarray,
    A_ratio: np.ndarray,
    Y: np.ndarray,
) -> None:
    header = [
        "P_Pa",
        "T_K",
        "rho_kg_m3",
        "u_m_s",
        "M",
        "a_m_s",
        "A_m2",
        "A_over_Astar",
    ] + [f"Y_{name}" for name in species_names]

    data_cols = [p, T, rho, u, M, a, A, A_ratio]
    data_cols.extend([Y[:, k] for k in range(Y.shape[1])])
    data = np.column_stack(data_cols)

    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")


def thrust_and_isp(
    mdot: float,
    u_e: float,
    p_e: float,
    A_e: float,
    p_amb: float,
    g0: float,
) -> Tuple[float, float]:
    thrust = mdot * u_e + (p_e - p_amb) * A_e
    isp = thrust / (mdot * g0)
    return thrust, isp


def main() -> None:
    args = parse_args()

    gas_eq, load_path = load_gas(args.mech)
    species_names = gas_eq.species_names

    # Chamber reactants at inlet conditions.
    X_reactants = lox_lh2_mole_composition(args.of)
    gas_eq.TPX = args.Tin, args.Pin, X_reactants

    # Chamber equilibrium stagnation state (adiabatic, constant pressure combustion).
    gas_eq.equilibrate("HP")
    p0 = gas_eq.P
    T0 = gas_eq.T
    h0_eq = gas_eq.enthalpy_mass
    s0_eq = gas_eq.entropy_mass

    # Evaluator for equilibrium nozzle expansion (isentropic + chemical equilibrium).
    eval_eq = make_equilibrium_evaluator(gas_eq, s0_eq, h0_eq)

    # Find throat and exit for equilibrium model.
    p_t_eq = find_throat_pressure(eval_eq, p0, args.pmin, args.nscan)
    st_t_eq = eval_eq(p_t_eq)

    rho_u_star_eq = st_t_eq.rho * st_t_eq.u
    p_e_eq = find_exit_pressure_for_area_ratio(
        eval_eq,
        p_t_eq,
        rho_u_star_eq,
        args.expansion_ratio,
        args.pmin,
        args.nscan,
    )

    A_star = math.pi * (args.throat_diameter ** 2) / 4.0
    mdot_eq = rho_u_star_eq * A_star
    A_e = A_star * args.expansion_ratio
    st_e_eq = eval_eq(p_e_eq)
    Fvac_eq, Ispvac_eq = thrust_and_isp(
        mdot_eq, st_e_eq.u, p_e_eq, A_e, args.pa_vac, args.g0
    )
    Fsl_eq, Ispsl_eq = thrust_and_isp(
        mdot_eq, st_e_eq.u, p_e_eq, A_e, args.pa_sl, args.g0
    )

    p_eq, T_eq, rho_eq, u_eq, M_eq, a_eq, A_eq, AR_eq, Y_eq = build_profile(
        eval_eq,
        species_names,
        p0,
        p_e_eq,
        rho_u_star_eq,
        mdot_eq,
        A_star,
        args.npoints,
    )

    # Frozen model: freeze composition at chamber-equilibrium composition.
    gas_fr, _ = load_gas(str(load_path))
    gas_fr.TPY = T0, p0, gas_eq.Y

    h0_fr = gas_fr.enthalpy_mass
    s0_fr = gas_fr.entropy_mass

    eval_fr = make_frozen_evaluator(gas_fr, s0_fr, h0_fr)

    p_t_fr = find_throat_pressure(eval_fr, p0, args.pmin, args.nscan)
    st_t_fr = eval_fr(p_t_fr)

    rho_u_star_fr = st_t_fr.rho * st_t_fr.u
    p_e_fr = find_exit_pressure_for_area_ratio(
        eval_fr,
        p_t_fr,
        rho_u_star_fr,
        args.expansion_ratio,
        args.pmin,
        args.nscan,
    )

    mdot_fr = rho_u_star_fr * A_star
    st_e_fr = eval_fr(p_e_fr)
    Fvac_fr, Ispvac_fr = thrust_and_isp(
        mdot_fr, st_e_fr.u, p_e_fr, A_e, args.pa_vac, args.g0
    )
    Fsl_fr, Ispsl_fr = thrust_and_isp(
        mdot_fr, st_e_fr.u, p_e_fr, A_e, args.pa_sl, args.g0
    )

    p_fr, T_fr, rho_fr, u_fr, M_fr, a_fr, A_fr, AR_fr, Y_fr = build_profile(
        eval_fr,
        species_names,
        p0,
        p_e_fr,
        rho_u_star_fr,
        mdot_fr,
        A_star,
        args.npoints,
    )

    out_eq = Path("equilibrium_nozzle.csv").resolve()
    out_fr = Path("frozen_nozzle.csv").resolve()

    write_csv(out_eq, species_names, p_eq, T_eq, rho_eq, u_eq, M_eq, a_eq, A_eq, AR_eq, Y_eq)
    write_csv(out_fr, species_names, p_fr, T_fr, rho_fr, u_fr, M_fr, a_fr, A_fr, AR_fr, Y_fr)

    print("Mechanism loaded from:", load_path)
    print("\n=== Chamber (equilibrium stagnation) ===")
    print(f"P0 = {p0:.6e} Pa")
    print(f"T0 = {T0:.2f} K")
    print(f"h0 = {h0_eq:.6e} J/kg")
    print(f"s0 = {s0_eq:.6e} J/(kg-K)")

    print("\n=== Equilibrium nozzle ===")
    print(f"Throat pressure p* = {p_t_eq:.6e} Pa")
    print(f"Exit pressure (A/A*={args.expansion_ratio}) p_e = {p_e_eq:.6e} Pa")
    print(f"Throat state: T* = {st_t_eq.T:.2f} K, M* = {st_t_eq.M:.6f}")
    print(f"Exit state: T_e = {st_e_eq.T:.2f} K, u_e = {st_e_eq.u:.3f} m/s")
    print(f"mdot_eq = {mdot_eq:.6f} kg/s")
    print(f"F_vac,eq = {Fvac_eq:.3f} N")
    print(f"F_sl,eq = {Fsl_eq:.3f} N")
    print(f"Isp_vac,eq = {Ispvac_eq:.6f} s")
    print(f"Isp_sl,eq = {Ispsl_eq:.6f} s")

    print("\n=== Frozen nozzle ===")
    print(f"Throat pressure p* = {p_t_fr:.6e} Pa")
    print(f"Exit pressure (A/A*={args.expansion_ratio}) p_e = {p_e_fr:.6e} Pa")
    print(f"Throat state: T* = {st_t_fr.T:.2f} K, M* = {st_t_fr.M:.6f}")
    print(f"Exit state: T_e = {st_e_fr.T:.2f} K, u_e = {st_e_fr.u:.3f} m/s")
    print(f"mdot_fr = {mdot_fr:.6f} kg/s")
    print(f"F_vac,fr = {Fvac_fr:.3f} N")
    print(f"F_sl,fr = {Fsl_fr:.3f} N")
    print(f"Isp_vac,fr = {Ispvac_fr:.6f} s")
    print(f"Isp_sl,fr = {Ispsl_fr:.6f} s")

    print("\nSaved:")
    print(out_eq)
    print(out_fr)


if __name__ == "__main__":
    main()
