"""
Ae121b - Combustion Chamber Model
Pablo Guerrero - 02/2017
Python conversion

Dependencies: cantera, numpy, scipy, matplotlib
  pip install cantera numpy scipy matplotlib

Cantera note:
  - GRI30 gas:          ct.Solution('gri30.yaml')
  - Pure-fluid methane: ct.Methane()   (PureFluid, replaces MATLAB Methane())
  - Pure-fluid oxygen:  ct.Oxygen()    (PureFluid, replaces MATLAB Oxygen())
"""

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Global parameter dictionary – replaces MATLAB 'global' variables.
# Populated in main() and read by all helper functions.
# ─────────────────────────────────────────────────────────────────────────────
params = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: set gas mixture state given T and equivalence ratio phi
# ─────────────────────────────────────────────────────────────────────────────
def set_gas_state(T, phi):
    """
    Set the GRI30 gas mixture to temperature T [K], pressure P [Pa],
    and the CH4/O2 mass-fraction composition corresponding to
    equivalence ratio phi.
    """
    gas    = params['gas']
    nsp    = params['nsp']
    io2    = params['io2']
    ich4   = params['ich4']
    FOR_st = params['FOR_st']
    P      = params['P']

    Y = np.zeros(nsp)
    Y[io2]  = 1.0
    Y[ich4] = phi * FOR_st          # mass fraction of CH4 relative to O2
    gas.TPY = T, P, Y


# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaporation constant K [m^2/s]   (D^2-law model)
# ─────────────────────────────────────────────────────────────────────────────
def evaporation_constant(Tg, phi):
    """
    Compute the evaporation constant K [m^2/s] at axial gas temperature
    Tg [K] and equivalence ratio phi.

    Uses a weighted average of pure-CH4 and pure-O2 thermal conductivities
    evaluated at the film temperature Tbar = (T_boil + Tg) / 2,
    and the Spalding thermal transfer number Bq.
    """
    gas      = params['gas']
    nsp      = params['nsp']
    ich4     = params['ich4']
    io2      = params['io2']
    T_boil   = params['T_boil_droplets']
    hfg      = params['hfg']
    rho_l    = params['rho_ch4_l']
    P        = params['P']

    Tg  = np.atleast_1d(np.asarray(Tg, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    N   = len(phi)
    K_evap = np.zeros(N)

    for k in range(N):
        Tbar = (T_boil + Tg[k]) / 2.0      # film temperature [K]

        # Thermal conductivity of pure CH4 at film temperature
        Y_ch4 = np.zeros(nsp);  Y_ch4[ich4] = 1.0
        gas.TPY = Tbar, P, Y_ch4
        k_ch4  = gas.thermal_conductivity   # [W/(m.K)]
        cp_ch4 = gas.cp_mass                # [J/(kg.K)]

        # Thermal conductivity of pure O2 at film temperature
        # (free-stream approximation gives slower evaporation ~0.12 m^2/s;
        #  using the mixture free-stream would give ~0.30 m^2/s)
        Y_o2 = np.zeros(nsp);  Y_o2[io2] = 1.0
        gas.TPY = Tbar, P, Y_o2
        k_free_stream = gas.thermal_conductivity  # [W/(m.K)]

        kg = 0.4 * k_ch4 + 0.6 * k_free_stream   # weighted average [W/(m.K)]

        # Spalding thermal transfer number (thermal only, no combustion term)
        Bq = cp_ch4 * (Tg[k] - T_boil) / hfg     # [-]
        K_evap[k] = 8.0 * kg * np.log(1.0 + Bq) / (rho_l * cp_ch4)  # [m^2/s]

    return float(K_evap[0]) if N == 1 else K_evap


# ─────────────────────────────────────────────────────────────────────────────
# Helper: dh/dphi  –  numerical derivative of gas enthalpy w.r.t. phi
# ─────────────────────────────────────────────────────────────────────────────
def dhdphi(Tg, phi):
    """
    Central-difference derivative of equilibrium enthalpy_mass with respect
    to equivalence ratio phi.  d = 1e-4 step.
    Returns [J/kg].
    """
    gas = params['gas']
    d   = 1e-4

    Tg  = np.atleast_1d(np.asarray(Tg,  dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    N   = len(phi)
    dhg_dfg = np.zeros(N)

    for k in range(N):
        set_gas_state(Tg[k], (1 - d / 2) * phi[k]);  gas.equilibrate('TP')
        h1 = gas.enthalpy_mass  # [J/kg]

        set_gas_state(Tg[k], (1 + d / 2) * phi[k]);  gas.equilibrate('TP')
        h2 = gas.enthalpy_mass  # [J/kg]

        dhg_dfg[k] = (h2 - h1) / (d * phi[k])  # [J/kg]

    return float(dhg_dfg[0]) if N == 1 else dhg_dfg


# ─────────────────────────────────────────────────────────────────────────────
# Helper: dh/dT  –  numerical derivative of gas enthalpy w.r.t. temperature
# ─────────────────────────────────────────────────────────────────────────────
def dhdT(Tg, phi):
    """
    Central-difference derivative of equilibrium enthalpy_mass with respect
    to temperature Tg.  d = 1e-4 step.
    Returns [J/(kg.K)].
    """
    gas = params['gas']
    d   = 1e-4

    Tg  = np.atleast_1d(np.asarray(Tg,  dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    N   = len(phi)
    dhg_dTg = np.zeros(N)

    for k in range(N):
        set_gas_state((1 - d / 2) * Tg[k], phi[k]);  gas.equilibrate('TP')
        h1 = gas.enthalpy_mass  # [J/kg]

        set_gas_state((1 + d / 2) * Tg[k], phi[k]);  gas.equilibrate('TP')
        h2 = gas.enthalpy_mass  # [J/kg]

        dhg_dTg[k] = (h2 - h1) / (d * Tg[k])  # [J/(kg.K)]

    return float(dhg_dTg[0]) if N == 1 else dhg_dTg


# ─────────────────────────────────────────────────────────────────────────────
# Core model: evaluated at a single axial position x (scalar inputs)
# ─────────────────────────────────────────────────────────────────────────────
def model_core(x, sol_1d):
    """
    One-dimensional combustion chamber model at a single axial position x.

    State vector  sol_1d = [D, vd, Tg]
      D   – droplet diameter            [m]
      vd  – droplet velocity            [m/s]
      Tg  – gas-phase temperature       [K]

    Returns
    -------
    dSoldx : ndarray, shape (3,)
        Spatial derivatives [dD/dx, dvd/dx, dTg/dx].
    Var_out : ndarray, shape (16,)
        Auxiliary variables (see index map at bottom of function).
    """
    gas      = params['gas']
    FOR_st   = params['FOR_st']
    D0       = params['D0']
    ml_0     = params['ml_0']
    mg_0     = params['mg_0']
    mox_0    = params['mox_0']
    mf_0     = params['mf_0']
    phi_T    = params['phi_T']
    P        = params['P']
    A_cs     = params['A_cs']
    R        = params['R']
    h_ch4_l  = params['h_ch4_l']
    rho_ch4_l = params['rho_ch4_l']

    D  = float(sol_1d[0])
    vd = float(sol_1d[1])
    Tg = float(sol_1d[2])

    dSoldx = np.zeros(3)

    if D > 0:
        # ── Mass flow rates ───────────────────────────────────────────
        ml  = ml_0 * (D / D0)**3                                  # liquid mass flow rate [kg/s]
        mg  = mg_0 + ml_0 - ml                                    # total gas mass flow rate [kg/s]
        mf  = (phi_T * FOR_st * mg - ml) / (1.0 + phi_T * FOR_st)  # fuel (gas-phase) [kg/s]
        phi = mf / (FOR_st * (mg - mf))                           # local equivalence ratio [-]
        mox = mg - mf                                             # oxidizer mass flow rate [kg/s]

        # ── Thermodynamic properties from Cantera (equilibrium at T,P) ──
        set_gas_state(Tg, phi)
        gas.equilibrate('TP')
        h_gas       = gas.enthalpy_mass                 # [J/kg]
        rhogas      = gas.density                       # [kg/m^3]
        viscosity_g = gas.viscosity                     # [Pa.s]

        # ── Gas velocity ──────────────────────────────────────────────
        vg = mg / (rhogas * A_cs)                       # [m/s]

        # ── Droplet drag (Schiller-Naumann type) ──────────────────────
        Re = rhogas * abs(vg - vd) * D / viscosity_g   # Reynolds number [-]
        Cd = 24.0 / Re + 6.0 / (1.0 + np.sqrt(Re)) + 0.4  # drag coefficient [-]

        # ── Evaporation constant ──────────────────────────────────────
        Kgas = evaporation_constant(Tg, phi)            # [m^2/s]

        # ── Source terms / axial derivatives ─────────────────────────
        dmldx  = -1.5 * ml_0 * D * Kgas / (D0**3 * vd)          # [kg/(s.m)]
        dphidx = -dmldx / (FOR_st * mox_0)                       # [1/m]

        dSoldx[0] = -Kgas / (2.0 * D * vd)                       # dD/dx  (non-dim)
        dSoldx[1] = (3.0 * Cd * rhogas * (vg - vd) * abs(vg - vd)
                     / (4.0 * rho_ch4_l * D * vd))               # dvd/dx [Hz]
        dSoldx[2] = ((dmldx * (h_gas - h_ch4_l) / mg
                      - dhdphi(Tg, phi) * dphidx)
                     / dhdT(Tg, phi))                             # dTg/dx [K/m]

    else:
        # ── Post-evaporation: droplet fully consumed ──────────────────
        phi    = phi_T
        ml     = 0.0
        mg     = mg_0 + ml_0
        mf     = mf_0 + ml_0
        mox    = mg - mf
        dphidx = 0.0
        dmldx  = 0.0

        set_gas_state(Tg, phi)                              # no equilibrate call here
        MWgas  = gas.mean_molecular_weight / 1000.0         # [kg/mol]
        h_gas  = gas.enthalpy_mass                          # [J/kg]
        Rgas   = R / MWgas                                  # [J/(kg.K)]
        rhogas = P / Rgas / Tg                              # [kg/m^3]
        vg     = mg / (rhogas * A_cs)                       # [m/s]
        Kgas   = evaporation_constant(Tg, phi)              # [m^2/s]

    # ── Pack auxiliary variables ──────────────────────────────────────
    # Index map (matches MATLAB Var_out rows 1-16):
    #   0  phi       equivalence ratio        [-]
    #   1  vg        gas velocity             [m/s]
    #   2  ml        liquid mass flow rate    [kg/s]
    #   3  mg        gas mass flow rate       [kg/s]
    #   4  mox       oxidizer mass flow rate  [kg/s]
    #   5  mf        fuel mass flow rate      [kg/s]
    #   6  dhdphi    dh/dphi                  [J/kg]
    #   7  dhdT      dh/dT                    [J/(kg.K)]
    #   8  dphidx    dphi/dx                  [1/m]
    #   9  dmldx     dml/dx                   [kg/(s.m)]
    #  10  dD/dx     droplet diameter deriv.  (non-dim)
    #  11  dvd/dx    droplet velocity deriv.  [Hz]
    #  12  dTg/dx    gas temperature deriv.   [K/m]
    #  13  h_gas     gas enthalpy             [J/kg]
    #  14  K_gas     evaporation constant     [m^2/s]
    #  15  rhogas    gas density              [kg/m^3]
    Var_out = np.array([
        phi,
        vg,
        ml,
        mg,
        mox,
        mf,
        dhdphi(Tg, phi),
        dhdT(Tg, phi),
        dphidx,
        dmldx,
        dSoldx[0],
        dSoldx[1],
        dSoldx[2],
        h_gas,
        Kgas,
        rhogas,
    ], dtype=float)

    return dSoldx, Var_out


# ─────────────────────────────────────────────────────────────────────────────
# ODE right-hand side wrapper for scipy.integrate.solve_ivp
# (only dSoldx is returned; Var_out is computed separately in post-processing)
# ─────────────────────────────────────────────────────────────────────────────
def model_rhs(x, sol):
    """Wrapper returning only dSoldx for the ODE solver."""
    dSoldx, _ = model_core(x, sol)
    return dSoldx


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: evaluate model_core over the full solution array
# ─────────────────────────────────────────────────────────────────────────────
def model_full_array(x_arr, sol_arr):
    """
    Evaluate model_core at every axial position in x_arr.

    Parameters
    ----------
    x_arr   : 1-D array of length N_steps
    sol_arr : array of shape (3, N_steps)  →  rows: [D, vd, Tg]

    Returns
    -------
    dSoldx_all  : (3,  N_steps)
    Var_out_all : (16, N_steps)
    """
    N = len(x_arr)
    dSoldx_all  = np.zeros((3,  N))
    Var_out_all = np.zeros((16, N))

    for i in range(N):
        dSoldx_all[:, i], Var_out_all[:, i] = model_core(x_arr[i], sol_arr[:, i])

    return dSoldx_all, Var_out_all


# ─────────────────────────────────────────────────────────────────────────────
# Main script
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Problem data (from homework assignment) ───────────────────────────────
    vd_0       = 10.0              # Droplet inlet velocity                 [m/s]
    phi_0      = 0.45              # Inlet equivalence ratio                [-]
    phi_T      = 1.139             # Target (total) equivalence ratio       [-]
    A_cs       = 0.157             # Chamber cross-section area             [m^2]
    A_injector = 0.0157            # Injector area                          [m^2]
    L          = 0.75              # Chamber length                         [m]
    Tg_inlet   = 600.0             # Gas inlet temperature                  [K]
    P          = 3.4474e6          # Pressure                               [Pa]
    R          = 8.314             # Universal gas constant                 [J/(mol.K)]
    D_range    = np.array([30, 50, 70, 90, 110]) * 1e-6  # Initial droplet diameters [m]

    # ── Cantera objects – instantiated ONCE to avoid performance overhead ──────
    CH4_data = ct.Methane()         # Pure-fluid methane (replaces MATLAB Methane())
    O2_data  = ct.Oxygen()          # Pure-fluid oxygen  (replaces MATLAB Oxygen())
    gas      = ct.Solution('gri30.yaml')   # GRI 3.0 reaction mechanism
    gas.transport_model = 'Mix'            # Mixture-averaged transport (replaces 'Mix' flag)
    nsp      = gas.n_species
    ich4     = gas.species_index('CH4')
    io2      = gas.species_index('O2')

    # ── Liquid methane properties ─────────────────────────────────────────────
    # Sweep T from 91 to 200 K at pressure P to locate the boiling point.
    T_sweep     = np.linspace(91, 200, 1000)
    ch4_density = np.zeros(len(T_sweep))
    for k, T in enumerate(T_sweep):
        CH4_data.TP = T, P
        ch4_density[k] = CH4_data.density

    # Boiling point identified by the maximum density gradient
    index           = np.argmax(np.abs(np.diff(ch4_density)))
    T_boil_droplets = T_sweep[index]       # ≈ 181.45 K

    CH4_data.TP = T_boil_droplets, P
    rho_ch4_l = CH4_data.density          # ≈ 270.04 kg/m^3  – liquid CH4 density at (P, T_boil)
    h_ch4_l   = CH4_data.enthalpy_mass    # ≈ -5.26e6 J/kg   – liquid CH4 enthalpy at (P, T_boil)
    MWCH4     = CH4_data.mean_molecular_weight  # 16.05 g/mol (independent of T, P)

    # ── Stoichiometry: CH4 + 2 O2 → CO2 + 2 H2O ─────────────────────────────
    MWO2   = O2_data.mean_molecular_weight  # 31.999 g/mol
    FOR_st = MWCH4 / (2.0 * MWO2)          # stoichiometric fuel-to-oxidizer ratio ≈ 0.2508
    MWO2   = MWO2  / 1000.0                 # [kg/mol]
    MWCH4  = MWCH4 / 1000.0                 # [kg/mol]

    Dhc = 50.0115e6   # heat of combustion        [J/kg]
    hfg = 509272.25   # latent heat of vaporization [J/kg]

    # ── Inlet conditions at x = 0 ─────────────────────────────────────────────
    ml_0  = rho_ch4_l * A_injector * vd_0           # liquid mass flow rate     [kg/s] ≈ 42.4
    mox_0 = ml_0 / (FOR_st * (phi_T - phi_0))       # oxidizer mass flow rate   [kg/s] ≈ 245.4
    mf_0  = phi_0 * ml_0 / (phi_T - phi_0)          # gas-phase fuel flow rate  [kg/s] ≈ 27.7
    mg_0  = mox_0 + mf_0                            # total gas mass flow rate  [kg/s] ≈ 273.1

    # Inlet gas temperature via HP equilibrium
    Y = np.zeros(nsp)
    Y[io2]  = 1.0
    Y[ich4] = phi_0 * FOR_st
    gas.TPY = Tg_inlet, P, Y
    gas.equilibrate('HP')
    Tg0 = gas.T   # gas temperature at x = 0  ≈ 3196 K

    # ── Populate the shared parameter dictionary (replaces MATLAB globals) ────
    params.update({
        'gas'             : gas,
        'nsp'             : nsp,
        'ich4'            : ich4,
        'io2'             : io2,
        'FOR_st'          : FOR_st,
        'phi_T'           : phi_T,
        'P'               : P,
        'A_cs'            : A_cs,
        'R'               : R,
        'MWCH4'           : MWCH4,
        'MWO2'            : MWO2,
        'h_ch4_l'         : h_ch4_l,
        'T_boil_droplets' : T_boil_droplets,
        'Dhc'             : Dhc,
        'hfg'             : hfg,
        'rho_ch4_l'       : rho_ch4_l,
        'ml_0'            : ml_0,
        'mg_0'            : mg_0,
        'mox_0'           : mox_0,
        'mf_0'            : mf_0,
    })

    # ── Integrate ODE for each initial droplet diameter ───────────────────────
    # Storage dictionary C mirrors the MATLAB cell array C{k, col}:
    #   C[k, 0]  = x          axial positions
    #   C[k, 1]  = D(x)       droplet diameter
    #   C[k, 2]  = vd(x)      droplet velocity
    #   C[k, 3]  = Tg(x)      gas temperature
    #   C[k, 4]  = phi(x)     equivalence ratio
    #   C[k, 5]  = vg(x)      gas velocity
    #   C[k, 6]  = ml(x)      liquid mass flow rate
    #   C[k, 7]  = mg(x)      gas mass flow rate
    #   C[k, 8]  = mox(x)     oxidizer mass flow rate
    #   C[k, 9]  = mf(x)      fuel mass flow rate
    #   C[k, 10] = dhdphi     dh/dphi
    #   C[k, 11] = dhdT       dh/dT
    #   C[k, 12] = dphidx     dphi/dx
    #   C[k, 13] = dmldx      dml/dx
    #   C[k, 14] = dD/dx      droplet diameter derivative
    #   C[k, 15] = dvd/dx     droplet velocity derivative
    #   C[k, 16] = dTg/dx     gas temperature derivative
    #   C[k, 17] = h_gas      gas enthalpy
    #   C[k, 18] = K_gas      evaporation constant
    #   C[k, 19] = rhogas     gas density
    C = {}

    # Terminal event: stop the ODE integration the moment D reaches zero.
    # This prevents the solver from stepping into negative-D territory and
    # avoids the discontinuous jump in auxiliary variables at burnout.
    def event_D_zero(x, sol):
        return sol[0]          # crosses zero when D = 0
    event_D_zero.terminal  = True   # stop integration when triggered
    event_D_zero.direction = -1     # only trigger on downward crossing

    for k in range(len(D_range)):
        D0 = D_range[k]
        params['D0'] = D0     # update D0 before each ODE integration

        sol = solve_ivp(
            model_rhs,
            [0.0, L],
            [D0, vd_0, Tg0],
            method='RK45',
            rtol=1e-5,
            dense_output=False,
            events=event_D_zero,   # halt cleanly at droplet burnout
        )

        x   = sol.t    # axial positions, shape (N_steps,)
        Sol = sol.y    # solution,        shape (3, N_steps)

        # Clamp any residual sub-zero diameter values (terminal event stops
        # integration at burnout, but the solver may land fractionally below 0).
        Sol[0, Sol[0, :] < 0] = 0.0

        C[k, 0] = x
        C[k, 1] = Sol[0, :]   # D(x)
        C[k, 2] = Sol[1, :]   # vd(x)
        C[k, 3] = Sol[2, :]   # Tg(x)

        # Evaluate auxiliary variables along the full solution trajectory
        _, Var_out = model_full_array(x, Sol)
        C[k, 4]  = Var_out[0, :]    # phi
        C[k, 5]  = Var_out[1, :]    # vg
        C[k, 6]  = Var_out[2, :]    # ml
        C[k, 7]  = Var_out[3, :]    # mg
        C[k, 8]  = Var_out[4, :]    # mox
        C[k, 9]  = Var_out[5, :]    # mf
        C[k, 10] = Var_out[6, :]    # dh/dphi
        C[k, 11] = Var_out[7, :]    # dh/dT
        C[k, 12] = Var_out[8, :]    # dphi/dx
        C[k, 13] = Var_out[9, :]    # dml/dx
        C[k, 14] = Var_out[10, :]   # dD/dx
        C[k, 15] = Var_out[11, :]   # dvd/dx
        C[k, 16] = Var_out[12, :]   # dTg/dx
        C[k, 17] = Var_out[13, :]   # h_gas
        C[k, 18] = Var_out[14, :]   # K_gas
        C[k, 19] = Var_out[15, :]   # rhogas

    # ── Plot Results ──────────────────────────────────────────────────────────
    colors = ['b', 'r', 'g', 'm', 'k']
    D0_um  = D_range * 1e6                                 # [µm] for legend
    L_plot = 0.8                                           # x-axis limit
    labels = [f'D0 = {round(d)} µm' for d in D0_um]  # round() avoids 29.999…→29 truncation
    N      = len(D_range)

    # ── Figure 1: physical / thermodynamic profiles ──────────────────────────
    fig1, axes1 = plt.subplots(3, 3, figsize=(16, 12))
    fig1.suptitle('Figure 1 – Combustion Chamber Profiles', fontsize=14, fontweight='bold')

    # ── Figure 2: flow rates and derivatives ─────────────────────────────────
    # MATLAB used subplot(3,3,...) but only filled 6 of the 9 slots (rows 1&2).
    # A 2×3 grid cleanly matches the actual layout.
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
    fig2.suptitle('Figure 2 – Flow Rates and Derivatives', fontsize=14, fontweight='bold')

    for k in range(N):
        # For the 30 µm case (k=0) the ODE solver's final step lands on/past
        # burnout and produces a spurious outlier point; drop it on every plot.
        trim = -1 if k == 0 else None   # slice endpoint: -1 drops last, None keeps all
        x = C[k, 0][:trim]
        c = colors[k]

        # Convenience: pre-slice all stored arrays with [:trim] so the final
        # spurious point is excluded for the 30 µm case (k==0) on every plot.
        D_plot    = C[k,  1][:trim]
        vd_plot   = C[k,  2][:trim]
        Tg_plot   = C[k,  3][:trim]
        phi_plot  = C[k,  4][:trim]
        vg_plot   = C[k,  5][:trim]
        ml_plot   = C[k,  6][:trim]
        mg_plot   = C[k,  7][:trim]
        mox_plot  = C[k,  8][:trim]
        mf_plot   = C[k,  9][:trim]
        dhdphi_plot = C[k, 10][:trim]
        dhdT_plot   = C[k, 11][:trim]
        dphidx_plot = C[k, 12][:trim]
        dmldx_plot  = C[k, 13][:trim]
        dDdx_plot   = C[k, 14][:trim]
        dvddx_plot  = C[k, 15][:trim]
        dTgdx_plot  = C[k, 16][:trim]
        hgas_plot   = C[k, 17][:trim]
        Kgas_plot   = C[k, 18][:trim]
        rho_plot    = C[k, 19][:trim]

        # ── Figure 1: subplot (1,1)  D(x) / D0 ──────────────────────────────
        ax = axes1[0, 0]
        ax.plot(x, D_plot / D_range[k], c, label=labels[k])
        ax.set_title(r'$D(x)/D_0$', fontsize=14)
        ax.set_ylabel(r'$D(x)/D_0\ [-]$')
        ax.set_xlim([0, L_plot]);  ax.set_ylim([0, 1])
        ax.grid(True);  ax.legend(fontsize=9)

        # ── Figure 1: subplot (1,2)  T(x) ────────────────────────────────────
        ax = axes1[0, 1]
        ax.plot(x, Tg_plot, c, label=labels[k])
        ax.set_title(r'$T(x)$', fontsize=14)
        ax.set_ylabel(r'$T(x)\ [K]$')
        ax.grid(True)
        if k == N - 1:
            ax.legend(fontsize=9)

        # ── Figure 1: subplot (1,3)  h_g(x) ──────────────────────────────────
        # MATLAB used semilogy, but h_g is negative (formation enthalpies dominate),
        # so semilogy silently drops them and the plot degrades to a regular scale.
        # Regular plot with scientific notation matches the MATLAB visual output.
        ax = axes1[0, 2]
        ax.plot(x, hgas_plot, c, label=labels[k])
        ax.set_title(r'$h_g(x)$', fontsize=14)
        ax.set_ylabel(r'$h_g(x)\ [J/kg]$')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True)

        # ── Figure 1: subplot (2,1)  dD/dx ───────────────────────────────────
        ax = axes1[1, 0]
        ax.plot(x, dDdx_plot, c, label=labels[k])
        ax.set_title(r'$dD(x)/dx$', fontsize=14)
        ax.set_ylabel(r'$dD(x)/dx$')
        ax.set_xlim([0, L_plot]);  ax.set_ylim([-4e-3, 0])
        ax.grid(True)

        # ── Figure 1: subplot (2,2)  dphi/dx (titled dT/dx in original) ──────
        # NOTE: MATLAB code plots C{k,13}=dphidx but mis-titles it 'dT(x)/dx'.
        # MATLAB uses semilogy then immediately calls axis([0,L,0,10]) which
        # silently overrides the log scale → effectively a linear plot [0,10].
        ax = axes1[1, 1]
        ax.plot(x, dphidx_plot, c, label=labels[k])
        ax.set_title(r'$d\phi(x)/dx$ [titled $dT/dx$ in original]', fontsize=11)
        ax.set_ylabel(r'$d\phi(x)/dx\ [1/m]$')
        ax.set_xlim([0, L_plot]);  ax.set_ylim([0, 10])
        ax.grid(True)

        # ── Figure 1: subplot (2,3)  dh/dT ───────────────────────────────────
        ax = axes1[1, 2]
        ax.plot(x, dhdT_plot, c, label=labels[k])
        ax.set_title(r'$dh(x)/dT$', fontsize=14)
        ax.set_ylabel(r'$dh(x)/dT\ [J/(kg \cdot K)]$')
        ax.grid(True)

        # ── Figure 1: subplot (3,1)  K_gas ───────────────────────────────────
        ax = axes1[2, 0]
        ax.plot(x, Kgas_plot, c, label=labels[k])
        ax.set_title(r'Evaporation Constant $K$', fontsize=12)
        ax.set_ylabel(r'$K_{gas}\ [m^2/s]$')
        ax.set_xlabel('Distance (x) [m]')
        ax.grid(True)

        # ── Figure 1: subplot (3,2)  rho_gas ─────────────────────────────────
        ax = axes1[2, 1]
        ax.plot(x, rho_plot, c, label=labels[k])
        ax.set_title(r'Gas density $\rho_{gas}$', fontsize=12)
        ax.set_ylabel(r'$\rho_g\ [kg/m^3]$')
        ax.set_xlabel('Distance (x) [m]')
        ax.grid(True)

        # ── Figure 1: subplot (3,3)  dh/dphi ─────────────────────────────────
        ax = axes1[2, 2]
        ax.plot(x, dhdphi_plot, c, label=labels[k])
        ax.set_title(r'$dh(x)/d\phi$', fontsize=14)
        ax.set_ylabel(r'$dh(x)/d\phi\ [J/kg]$')
        ax.set_xlabel('Distance (x) [m]')
        ax.grid(True)

        # ── Figure 2: subplot (1,1)  phi(x) ──────────────────────────────────
        ax = axes2[0, 0]
        ax.plot(x, phi_plot, c, label=labels[k])
        ax.set_title(r'$\phi(x)$', fontsize=14)
        ax.set_ylabel(r'$\phi(x)\ [-]$')
        ax.grid(True)
        if k == N - 1:
            ax.legend(fontsize=9)

        # ── Figure 2: subplot (1,2)  mass flow rates ─────────────────────────
        ax = axes2[0, 1]
        ax.plot(x, ml_plot,  c,         label='liquid'   if k == 0 else '')
        ax.plot(x, mg_plot,  c + '--',  label='gas'      if k == 0 else '')
        ax.plot(x, mox_plot, c + '-.',  label='ox gas'   if k == 0 else '')
        ax.plot(x, mf_plot,  c + ':',   label='fuel gas' if k == 0 else '')
        ax.set_title(r'$\dot{m}_i(x)$', fontsize=14)
        ax.set_ylabel(r'$\dot{m}_i\ [kg/s]$')
        ax.set_xlim([0, L_plot]);  ax.set_ylim([0, 350])
        ax.legend(fontsize=9);  ax.grid(True)

        # ── Figure 2: subplot (1,3)  velocities ──────────────────────────────
        ax = axes2[0, 2]
        ax.plot(x, vd_plot, c,        label='Droplet' if k == 0 else '')
        ax.plot(x, vg_plot, c + '--', label='Gas'     if k == 0 else '')
        ax.set_title(r'$v_d(x)$', fontsize=14)
        ax.set_ylabel(r'$v_d(x)\ [m/s]$')
        ax.legend(fontsize=9);  ax.grid(True)

        # ── Figure 2: subplot (2,1)  dphi/dx ─────────────────────────────────
        # MATLAB: semilogy + axis([0,L,0,10]) → axis() overrides log scale,
        # so the effective result is a LINEAR plot clipped to [0, 10].
        ax = axes2[1, 0]
        ax.plot(x, dphidx_plot, c, label=labels[k])
        ax.set_title(r'$d\phi(x)/dx$', fontsize=14)
        ax.set_ylabel(r'$d\phi(x)/dx\ [1/m]$')
        ax.set_xlim([0, L_plot]);  ax.set_ylim([0, 10])
        ax.set_xlabel('Distance (x) [m]')
        ax.grid(True)

        # ── Figure 2: subplot (2,2)  dml/dx ──────────────────────────────────
        ax = axes2[1, 1]
        ax.plot(x, dmldx_plot, c, label=labels[k])
        ax.set_title(r'$d\dot{m}_l(x)/dx$', fontsize=14)
        ax.set_ylabel(r'$d\dot{m}_l(x)/dx\ [kg/(s \cdot m)]$')
        ax.set_xlim([0, L_plot]);  ax.set_ylim([-600, 0])
        ax.set_xlabel('Distance (x) [m]')
        ax.grid(True)
        if k == N - 1:
            ax.legend(fontsize=9)

        # ── Figure 2: subplot (2,3)  dvd/dx ──────────────────────────────────
        # MATLAB: semilogy + axis([0,L,0,3500]) → axis() overrides log scale,
        # so the effective result is a LINEAR plot clipped to [0, 3500].
        ax = axes2[1, 2]
        ax.plot(x, dvddx_plot, c, label=labels[k])
        ax.set_title(r'$dv_d(x)/dx$', fontsize=14)
        ax.set_ylabel(r'$dv_d(x)/dx\ [Hz]$')
        ax.set_xlim([0, L_plot]);  ax.set_ylim([0, 3500])
        ax.set_xlabel('Distance (x) [m]')
        ax.grid(True)

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()