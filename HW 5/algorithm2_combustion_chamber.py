"""Algorithm 2 (combustion chamber model) in Python.

State vector for ODE integration (with scipy.solve_ivp / RK45):
    y = [D2, vd, Tg]

where
    D2 : droplet diameter squared [m^2]
    vd : droplet velocity [m/s]
    Tg : gas temperature [K]

Required functions from the homework algorithm:
1) chamber_rhs(...)                  -> RHS for ODE system
2) evaporation_constant(Tg, phi)     -> K(Tg, phi)
3) dhg_dphi(Tg, P, phi)              -> numerical derivative dh_g / dphi
4) dhg_dT(Tg, P, phi)                -> numerical derivative dh_g / dT

Cantera mechanism used for gas/transport properties: gri30.yaml
"""

from __future__ import annotations

from dataclasses import dataclass

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ChamberContext:
    gas: ct.Solution
    P: float
    A_cs: float
    FOR_st: float
    phi_T: float
    D0: float
    ml_0: float
    mg_0: float
    mox_0: float
    mf_0: float
    rho_l: float
    h_l: float
    T_boil: float
    hfg: float
    io2: int
    ifuel: int


CTX: ChamberContext | None = None


def set_context(ctx: ChamberContext) -> None:
    global CTX
    CTX = ctx


def _require_ctx() -> ChamberContext:
    if CTX is None:
        raise RuntimeError("Context is not set. Call set_context(...) first.")
    return CTX


def _set_gas_state(T: float, P: float, phi: float) -> None:
    """Set CH4/O2 mixture state in GRI30 at (T, P, phi)."""
    ctx = _require_ctx()
    gas = ctx.gas

    phi = max(float(phi), 1e-12)
    Y = np.zeros(gas.n_species)
    Y[ctx.io2] = 1.0
    Y[ctx.ifuel] = phi * ctx.FOR_st
    gas.TPY = T, P, Y


def evaporation_constant(Tg: float, phi: float) -> float:
    """Compute evaporation constant K(Tg, phi) [m^2/s]."""
    del phi  # retained for API parity with assignment statement

    ctx = _require_ctx()
    gas = ctx.gas

    Tbar = 0.5 * (ctx.T_boil + Tg)

    # Pure-fuel transport properties using GRI30 composition vector.
    Y_f = np.zeros(gas.n_species)
    Y_f[ctx.ifuel] = 1.0
    gas.TPY = Tbar, ctx.P, Y_f
    k_f = gas.thermal_conductivity
    cp_f = gas.cp_mass

    # Pure-oxidizer transport properties using GRI30 composition vector.
    Y_o = np.zeros(gas.n_species)
    Y_o[ctx.io2] = 1.0
    gas.TPY = Tbar, ctx.P, Y_o
    k_o = gas.thermal_conductivity

    # Same weighted thermal conductivity model used in the MATLAB/Python port.
    k_g = 0.4 * k_f + 0.6 * k_o

    Bq = cp_f * (Tg - ctx.T_boil) / ctx.hfg
    Bq = max(Bq, -0.999999999)

    return 8.0 * k_g * np.log(1.0 + Bq) / (ctx.rho_l * cp_f)


def dhg_dphi(Tg: float, P: float, phi: float) -> float:
    """Numerical derivative dh_g/dphi via central difference."""
    ctx = _require_ctx()
    gas = ctx.gas

    dphi = max(abs(phi) * 1e-4, 1e-10)
    phi1 = max(phi - 0.5 * dphi, 1e-12)
    phi2 = phi + 0.5 * dphi

    _set_gas_state(Tg, P, phi1)
    gas.equilibrate("TP")
    h1 = gas.enthalpy_mass

    _set_gas_state(Tg, P, phi2)
    gas.equilibrate("TP")
    h2 = gas.enthalpy_mass

    return (h2 - h1) / (phi2 - phi1)


def dhg_dT(Tg: float, P: float, phi: float) -> float:
    """Numerical derivative dh_g/dT via central difference."""
    ctx = _require_ctx()
    gas = ctx.gas

    dT = max(abs(Tg) * 1e-4, 1e-6)
    T1 = max(Tg - 0.5 * dT, 10.0)
    T2 = Tg + 0.5 * dT

    _set_gas_state(T1, P, phi)
    gas.equilibrate("TP")
    h1 = gas.enthalpy_mass

    _set_gas_state(T2, P, phi)
    gas.equilibrate("TP")
    h2 = gas.enthalpy_mass

    return (h2 - h1) / (T2 - T1)


def chamber_rhs(_x: float, y: np.ndarray) -> np.ndarray:
    """RHS function for solve_ivp (ODE45 analog)."""
    rhs, _ = chamber_rhs_with_aux(_x, y)
    return rhs


def chamber_rhs_with_aux(_x: float, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """RHS + useful auxiliary values."""
    ctx = _require_ctx()
    gas = ctx.gas

    D2 = float(y[0])
    vd = max(float(y[1]), 1e-9)
    Tg = max(float(y[2]), 10.0)

    if D2 <= 0.0:
        return np.zeros(3), {
            "phi": ctx.phi_T,
            "vg": 0.0,
            "Re": 0.0,
            "Cd": 0.0,
            "K": 0.0,
            "dmldx": 0.0,
            "dphidx": 0.0,
        }

    D = np.sqrt(D2)

    # 1) Compute flow rates.
    ml = ctx.ml_0 * (D / ctx.D0) ** 3
    mg = ctx.mg_0 + ctx.ml_0 - ml
    mf = (ctx.phi_T * ctx.FOR_st * mg - ml) / (1.0 + ctx.phi_T * ctx.FOR_st)
    phi = mf / (ctx.FOR_st * (mg - mf))
    mox = mg - mf

    # 2) Gas properties from Cantera.
    _set_gas_state(Tg, ctx.P, phi)
    gas.equilibrate("TP")
    h_g = gas.enthalpy_mass
    rho_g = gas.density
    mu_g = max(gas.viscosity, 1e-18)

    # 3) vg, Re, Cd.
    vg = mg / (rho_g * ctx.A_cs)
    Re = max(rho_g * abs(vg - vd) * D / mu_g, 1e-12)
    Cd = 24.0 / Re + 6.0 / (1.0 + np.sqrt(Re)) + 0.4

    # 4) Evaporation constant K(Tg, phi).
    K = evaporation_constant(Tg, phi)

    # 5) Algebraic dml/dx and dphi/dx.
    dmldx = -1.5 * ctx.ml_0 * D * K / (ctx.D0**3 * vd)
    dphidx = -dmldx / (ctx.FOR_st * ctx.mox_0)

    # 6) ODE RHS terms: dD2/dx, dvd/dx, dTg/dx.
    dD2dx = -K / vd
    dvddx = 3.0 * Cd * rho_g * (vg - vd) * abs(vg - vd) / (4.0 * ctx.rho_l * D * vd)
    dTgdx = ((dmldx * (h_g - ctx.h_l) / mg) - dhg_dphi(Tg, ctx.P, phi) * dphidx) / dhg_dT(Tg, ctx.P, phi)

    rhs = np.array([dD2dx, dvddx, dTgdx], dtype=float)
    aux = {"phi": phi, "vg": vg, "Re": Re, "Cd": Cd, "K": K, "dmldx": dmldx, "dphidx": dphidx}
    return rhs, aux


def build_default_context(D0: float = 70e-6) -> ChamberContext:
    """Build context values matching the class CH4/O2 chamber model setup."""
    # Inputs used in the course model.
    vd_0 = 10.0
    phi_0 = 0.45
    phi_T = 1.139
    A_cs = 0.157
    A_injector = 0.0157
    Tg_inlet = 600.0
    P = 3.4474e6

    ch4 = ct.Methane()
    o2 = ct.Oxygen()
    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"

    io2 = gas.species_index("O2")
    ifuel = gas.species_index("CH4")

    # Find CH4 boiling temperature at chamber pressure from density jump.
    T_grid = np.linspace(91.0, 200.0, 1000)
    rho_grid = np.zeros_like(T_grid)
    for i, T in enumerate(T_grid):
        ch4.TP = T, P
        rho_grid[i] = ch4.density
    i_jump = int(np.argmax(np.abs(np.diff(rho_grid))))
    T_boil = float(T_grid[i_jump])

    ch4.TP = T_boil, P
    rho_l = ch4.density
    h_l = ch4.enthalpy_mass

    MW_f = ch4.mean_molecular_weight
    MW_o = o2.mean_molecular_weight
    FOR_st = MW_f / (2.0 * MW_o)

    hfg = 509272.25

    ml_0 = rho_l * A_injector * vd_0
    mox_0 = ml_0 / (FOR_st * (phi_T - phi_0))
    mf_0 = phi_0 * ml_0 / (phi_T - phi_0)
    mg_0 = mox_0 + mf_0

    # (Optional) set inlet equilibrium state once.
    Y = np.zeros(gas.n_species)
    Y[io2] = 1.0
    Y[ifuel] = phi_0 * FOR_st
    gas.TPY = Tg_inlet, P, Y
    gas.equilibrate("HP")

    return ChamberContext(
        gas=gas,
        P=P,
        A_cs=A_cs,
        FOR_st=FOR_st,
        phi_T=phi_T,
        D0=D0,
        ml_0=ml_0,
        mg_0=mg_0,
        mox_0=mox_0,
        mf_0=mf_0,
        rho_l=rho_l,
        h_l=h_l,
        T_boil=T_boil,
        hfg=hfg,
        io2=io2,
        ifuel=ifuel,
    )


def example_solve(x_end: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Small runnable example using solve_ivp (ODE45 equivalent)."""
    ctx = build_default_context(D0=70e-6)
    set_context(ctx)

    # Initial state: D2, vd, Tg.  Tg here from class model equilibrium inlet.
    y0 = np.array([ctx.D0**2, 10.0, 3196.0], dtype=float)

    sol = solve_ivp(
        chamber_rhs,
        t_span=(0.0, x_end),
        y0=y0,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    return sol.t, sol.y


if __name__ == "__main__":
    x, y = example_solve()
    print(f"Solved {y.shape[1]} axial points. Final D = {np.sqrt(max(y[0, -1], 0.0))*1e6:.2f} um")
