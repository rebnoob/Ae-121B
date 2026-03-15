"""
Dependencies:
    pip install cantera numpy scipy matplotlib
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp


@dataclass
class CaseResult:
    D0: float
    x: np.ndarray
    D: np.ndarray
    vd: np.ndarray
    Tg: np.ndarray
    phi: np.ndarray
    vg: np.ndarray
    ml: np.ndarray
    mg: np.ndarray
    mox: np.ndarray
    mf: np.ndarray
    dTgdx: np.ndarray


params: dict[str, object] = {}


def set_gas_state(T: float, phi: float) -> None:
    gas = params["gas"]
    nsp = params["nsp"]
    io2 = params["io2"]
    ich4 = params["ich4"]
    FOR_st = params["FOR_st"]
    P = params["P"]

    Y = np.zeros(nsp)
    Y[io2] = 1.0
    Y[ich4] = phi * FOR_st
    gas.TPY = T, P, Y


def evaporation_constant(Tg: float, phi: float) -> float:
    del phi  # kept for API parity with the MATLAB/Python model

    gas = params["gas"]
    nsp = params["nsp"]
    ich4 = params["ich4"]
    io2 = params["io2"]
    T_boil = params["T_boil_droplets"]
    hfg = params["hfg"]
    rho_l = params["rho_ch4_l"]
    P = params["P"]

    Tbar = 0.5 * (T_boil + Tg)

    Y_ch4 = np.zeros(nsp)
    Y_ch4[ich4] = 1.0
    gas.TPY = Tbar, P, Y_ch4
    k_ch4 = gas.thermal_conductivity
    cp_ch4 = gas.cp_mass

    Y_o2 = np.zeros(nsp)
    Y_o2[io2] = 1.0
    gas.TPY = Tbar, P, Y_o2
    k_free_stream = gas.thermal_conductivity

    kg = 0.4 * k_ch4 + 0.6 * k_free_stream
    Bq = cp_ch4 * (Tg - T_boil) / hfg
    Bq = max(Bq, -0.999999999)

    return 8.0 * kg * np.log(1.0 + Bq) / (rho_l * cp_ch4)


def dhdphi(Tg: float, phi: float) -> float:
    gas = params["gas"]
    d = 1e-4

    set_gas_state(Tg, (1.0 - d / 2.0) * phi)
    gas.equilibrate("TP")
    h1 = gas.enthalpy_mass

    set_gas_state(Tg, (1.0 + d / 2.0) * phi)
    gas.equilibrate("TP")
    h2 = gas.enthalpy_mass

    return (h2 - h1) / (d * phi)


def dhdT(Tg: float, phi: float) -> float:
    gas = params["gas"]
    d = 1e-4

    set_gas_state((1.0 - d / 2.0) * Tg, phi)
    gas.equilibrate("TP")
    h1 = gas.enthalpy_mass

    set_gas_state((1.0 + d / 2.0) * Tg, phi)
    gas.equilibrate("TP")
    h2 = gas.enthalpy_mass

    return (h2 - h1) / (d * Tg)


def model_core(sol: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    gas = params["gas"]
    FOR_st = params["FOR_st"]
    D0 = params["D0"]
    ml_0 = params["ml_0"]
    mg_0 = params["mg_0"]
    mox_0 = params["mox_0"]
    mf_0 = params["mf_0"]
    phi_T = params["phi_T"]
    A_cs = params["A_cs"]
    h_ch4_l = params["h_ch4_l"]
    rho_ch4_l = params["rho_ch4_l"]
    P = params["P"]
    R = params["R"]

    D = float(sol[0])
    vd = float(sol[1])
    Tg = float(sol[2])

    if D > 0.0:
        ml = ml_0 * (D / D0) ** 3
        mg = mg_0 + ml_0 - ml
        mf = (phi_T * FOR_st * mg - ml) / (1.0 + phi_T * FOR_st)
        phi = mf / (FOR_st * (mg - mf))
        mox = mg - mf

        set_gas_state(Tg, phi)
        gas.equilibrate("TP")
        h_gas = gas.enthalpy_mass
        rho_g = gas.density
        mu_g = gas.viscosity

        vg = mg / (rho_g * A_cs)

        Re = rho_g * abs(vg - vd) * D / max(mu_g, 1e-18)
        Re = max(Re, 1e-12)
        Cd = 24.0 / Re + 6.0 / (1.0 + np.sqrt(Re)) + 0.4

        Kgas = evaporation_constant(Tg, phi)

        dmldx = -1.5 * ml_0 * D * Kgas / (D0**3 * vd)
        dphidx = -dmldx / (FOR_st * mox_0)

        dDdx = -Kgas / (2.0 * D * vd)
        dvddx = 3.0 * Cd * rho_g * (vg - vd) * abs(vg - vd) / (4.0 * rho_ch4_l * D * vd)

        dhdphi_v = dhdphi(Tg, phi)
        dhdT_v = dhdT(Tg, phi)
        dTgdx = ((dmldx * (h_gas - h_ch4_l) / mg) - dhdphi_v * dphidx) / dhdT_v

    else:
        phi = phi_T
        ml = 0.0
        mg = mg_0 + ml_0
        mf = mf_0 + ml_0
        mox = mg - mf

        set_gas_state(Tg, phi)
        gas.equilibrate("TP")
        MWgas = gas.mean_molecular_weight / 1000.0
        h_gas = gas.enthalpy_mass
        Rgas = R / MWgas
        rho_g = gas.density
        vg = mg / (rho_g * A_cs)

        dDdx = 0.0
        dvddx = 0.0
        dTgdx = 0.0
        Kgas = evaporation_constant(Tg, phi)
        dphidx = 0.0
        dmldx = 0.0

    derivs = np.array([dDdx, dvddx, dTgdx], dtype=float)
    aux = {
        "phi": phi,
        "vg": vg,
        "ml": ml,
        "mg": mg,
        "mox": mox,
        "mf": mf,
        "dTgdx": dTgdx,
        "Kgas": Kgas,
        "dmldx": dmldx,
        "dphidx": dphidx,
        "h_gas": h_gas,
        "rho_g": rho_g,
    }
    return derivs, aux


def model_rhs(_x: float, sol: np.ndarray) -> np.ndarray:
    derivs, _ = model_core(sol)
    return derivs


def setup_parameters() -> tuple[dict[str, float], np.ndarray]:
    vd_0 = 10.0
    phi_0 = 0.45
    phi_T = 1.139
    A_cs = 0.157
    A_injector = 0.0157
    L = 0.75
    Tg_inlet = 600.0
    P = 3.4474e6
    R = 8.314
    D_range = np.array([30, 50, 70, 90, 110], dtype=float) * 1e-6

    CH4_data = ct.Methane()
    O2_data = ct.Oxygen()
    gas = ct.Solution("gri30.yaml")
    gas.transport_model = "mixture-averaged"

    nsp = gas.n_species
    ich4 = gas.species_index("CH4")
    io2 = gas.species_index("O2")

    T_sweep = np.linspace(91.0, 200.0, 1000)
    ch4_density = np.zeros_like(T_sweep)
    for i, T in enumerate(T_sweep):
        CH4_data.TP = T, P
        ch4_density[i] = CH4_data.density
    idx = int(np.argmax(np.abs(np.diff(ch4_density))))
    T_boil_droplets = float(T_sweep[idx])

    CH4_data.TP = T_boil_droplets, P
    rho_ch4_l = CH4_data.density
    h_ch4_l = CH4_data.enthalpy_mass
    MWCH4 = CH4_data.mean_molecular_weight
    MWO2 = O2_data.mean_molecular_weight

    FOR_st = MWCH4 / (2.0 * MWO2)
    MWCH4 = MWCH4 / 1000.0
    MWO2 = MWO2 / 1000.0

    Dhc = 50.0115e6
    hfg = 509272.25

    ml_0 = rho_ch4_l * A_injector * vd_0
    mox_0 = ml_0 / (FOR_st * (phi_T - phi_0))
    mf_0 = phi_0 * ml_0 / (phi_T - phi_0)
    mg_0 = mox_0 + mf_0

    Y = np.zeros(nsp)
    Y[io2] = 1.0
    Y[ich4] = phi_0 * FOR_st
    gas.TPY = Tg_inlet, P, Y
    gas.equilibrate("HP")
    Tg0 = gas.T

    params.update(
        {
            "gas": gas,
            "nsp": nsp,
            "ich4": ich4,
            "io2": io2,
            "FOR_st": FOR_st,
            "phi_T": phi_T,
            "P": P,
            "A_cs": A_cs,
            "R": R,
            "MWCH4": MWCH4,
            "MWO2": MWO2,
            "h_ch4_l": h_ch4_l,
            "T_boil_droplets": T_boil_droplets,
            "Dhc": Dhc,
            "hfg": hfg,
            "rho_ch4_l": rho_ch4_l,
            "ml_0": ml_0,
            "mg_0": mg_0,
            "mox_0": mox_0,
            "mf_0": mf_0,
        }
    )

    initial = {
        "vd_0": vd_0,
        "phi_0": phi_0,
        "phi_T": phi_T,
        "A_cs": A_cs,
        "A_injector": A_injector,
        "L": L,
        "Tg_inlet": Tg_inlet,
        "P": P,
        "R": R,
        "Tg0": Tg0,
    }
    return initial, D_range


def run_case(D0: float, L: float, vd_0: float, Tg0: float) -> CaseResult:
    params["D0"] = D0

    def event_d_zero(_x: float, y: np.ndarray) -> float:
        return y[0]

    event_d_zero.terminal = True
    event_d_zero.direction = -1

    sol = solve_ivp(
        fun=model_rhs,
        t_span=(0.0, L),
        y0=np.array([D0, vd_0, Tg0], dtype=float),
        method="RK45",
        rtol=1e-5,
        atol=1e-8,
        events=event_d_zero,
    )

    x = sol.t
    Y = sol.y.copy()
    Y[0, Y[0, :] < 0.0] = 0.0

    n = x.size
    phi = np.zeros(n)
    vg = np.zeros(n)
    ml = np.zeros(n)
    mg = np.zeros(n)
    mox = np.zeros(n)
    mf = np.zeros(n)
    dTgdx = np.zeros(n)

    for i in range(n):
        _, aux = model_core(Y[:, i])
        phi[i] = aux["phi"]
        vg[i] = aux["vg"]
        ml[i] = aux["ml"]
        mg[i] = aux["mg"]
        mox[i] = aux["mox"]
        mf[i] = aux["mf"]
        dTgdx[i] = aux["dTgdx"]

    return CaseResult(
        D0=D0,
        x=x,
        D=Y[0, :],
        vd=Y[1, :],
        Tg=Y[2, :],
        phi=phi,
        vg=vg,
        ml=ml,
        mg=mg,
        mox=mox,
        mf=mf,
        dTgdx=dTgdx,
    )


def plot_all(results: list[CaseResult], out_dir: str) -> None:
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    labels = [f"D0 = {int(round(r.D0 * 1e6))} um" for r in results]

    plt.figure(figsize=(8, 5))
    for i, r in enumerate(results):
        plt.plot(r.x, r.D / r.D0, color=colors[i], label=labels[i], linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("D(x)/D0 [-]")
    plt.title("Problem 5.07(a): Normalized Droplet Diameter")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 0.75)
    plt.ylim(0.0, 1.05)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_D_over_D0.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, r in enumerate(results):
        plt.plot(r.x, r.Tg, color=colors[i], label=labels[i], linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Tg(x) [K]")
    plt.title("Problem 5.07(a): Gas Temperature")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 0.75)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_Tg.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, r in enumerate(results):
        plt.loglog(r.x, r.dTgdx, color=colors[i], label=labels[i], linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("dTg/dx [K/m]")
    plt.title("Problem 5.07(a): Gas Temperature Gradient")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_dTgdx.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, r in enumerate(results):
        plt.plot(r.x, r.phi, color=colors[i], label=labels[i], linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("phi(x) [-]")
    plt.title("Problem 5.07(a): Equivalence Ratio")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 0.75)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "4_phi.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for i, r in enumerate(results):
        plt.plot(r.x, r.vd, color=colors[i], linestyle="-", linewidth=2)
        plt.plot(r.x, r.vg, color=colors[i], linestyle="--", linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Problem 5.07(a): Gas and Droplet Velocities")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 0.75)

    color_handles = [Line2D([0], [0], color=colors[i], lw=2, label=labels[i]) for i in range(len(results))]
    style_handles = [
        Line2D([0], [0], color="k", lw=2, linestyle="-", label="vd(x) droplet"),
        Line2D([0], [0], color="k", lw=2, linestyle="--", label="vg(x) gas"),
    ]
    leg1 = plt.legend(handles=color_handles, loc="upper right", fontsize=8, title="Injector D0")
    plt.gca().add_artist(leg1)
    plt.legend(handles=style_handles, loc="upper left", fontsize=9, title="Variable")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "5_vg_vd.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9.5, 6))
    style_map = {
        "ml": "-",
        "mg": "--",
        "mox": "-.",
        "mf": ":",
    }
    for i, r in enumerate(results):
        plt.plot(r.x, r.ml, color=colors[i], linestyle=style_map["ml"], linewidth=2)
        plt.plot(r.x, r.mg, color=colors[i], linestyle=style_map["mg"], linewidth=2)
        plt.plot(r.x, r.mox, color=colors[i], linestyle=style_map["mox"], linewidth=2)
        plt.plot(r.x, r.mf, color=colors[i], linestyle=style_map["mf"], linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Mass flow rate [kg/s]")
    plt.title("Problem 5.07(a): Liquid/Vapor/Oxidizer/Fuel Mass Flow Rates")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 0.75)

    color_handles = [Line2D([0], [0], color=colors[i], lw=2, label=labels[i]) for i in range(len(results))]
    style_handles = [
        Line2D([0], [0], color="k", lw=2, linestyle=style_map["ml"], label="ml(x) liquid"),
        Line2D([0], [0], color="k", lw=2, linestyle=style_map["mg"], label="mg(x) total gas"),
        Line2D([0], [0], color="k", lw=2, linestyle=style_map["mox"], label="mox(x) oxidizer gas"),
        Line2D([0], [0], color="k", lw=2, linestyle=style_map["mf"], label="mf(x) fuel gas"),
    ]
    leg1 = plt.legend(handles=color_handles, loc="upper right", fontsize=8, title="Injector D0")
    plt.gca().add_artist(leg1)
    plt.legend(handles=style_handles, loc="center right", fontsize=8, title="Variable")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "6_mass_flow_rates.png"), dpi=200)
    plt.close()


def main() -> None:
    initial, D_range = setup_parameters()
    L = initial["L"]
    vd_0 = initial["vd_0"]
    Tg0 = initial["Tg0"]

    results = []
    for D0 in D_range:
        case = run_case(D0=D0, L=L, vd_0=vd_0, Tg0=Tg0)
        results.append(case)

    out_dir = os.path.join(os.getcwd(), "plots_problem_5_07a")
    os.makedirs(out_dir, exist_ok=True)
    plot_all(results, out_dir)

    print("Completed Problem 5.07(a) run for D0 = [30, 50, 70, 90, 110] um")
    print(f"Saved six required plots to: {out_dir}")


if __name__ == "__main__":
    main()
