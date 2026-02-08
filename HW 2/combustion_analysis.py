import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g, R

# ============================================================================
# Constants
# ============================================================================

g0 = g  # Standard gravity, m/s^2
R_u = R  # Universal gas constant, J/(mol·K)

# ============================================================================
# Propellant Definitions
# ============================================================================

# Molecular weights (g/mol)
MW_O2 = 32.0    # Oxygen
MW_CH4 = 16.04  # Methane
MW_N2O4 = 92.01 # Nitrogen tetroxide
MW_N2H4 = 32.05 # Hydrazine

# Stoichiometric O/F ratios
# CH4 + 2O2 -> CO2 + 2H2O => O/F = (2*32)/16 = 4.0
OF_STOICH_LOX_CH4 = (2 * MW_O2) / MW_CH4

# N2O4 + 2N2H4 -> 3N2 + 4H2O => O/F = 92/(2*32.05) = 1.435
OF_STOICH_N2O4_N2H4 = MW_N2O4 / (2 * MW_N2H4)

# ============================================================================
# Configuration
# ============================================================================

# O/F ratio range
OF_min = 0.75
OF_max = 5.0
n_OF = 60  # Number of O/F points

OF_range = np.linspace(OF_min, OF_max, n_OF)

# Chamber pressures (atm -> Pa)
P_chambers_atm = [80, 120, 160]  # atm
P_chambers = [p * ct.one_atm for p in P_chambers_atm]

# Initial temperature (K) - reactants at room temperature
T_initial = 300.0

# Mechanism file
mechanism_file = 'Ae121_V1.0.yaml'

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_cstar(T_c, gamma, M_w):
    """
    Calculate characteristic velocity c*.
    
    c* = sqrt(R_u * T_c / (M_w * gamma)) * ((gamma + 1) / 2)^((gamma + 1) / (2 * (gamma - 1)))
    
    Parameters:
        T_c: Chamber/combustion temperature (K)
        gamma: Ratio of specific heats (Cp/Cv)
        M_w: Mean molecular weight (kg/mol)
    
    Returns:
        c*: Characteristic velocity (m/s)
    """
    term1 = np.sqrt(R_u * T_c / (M_w * gamma))
    term2 = ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))
    return term1 * term2


def calculate_isp_ideal(cstar, gamma):
    """
    Calculate ideal specific impulse for vacuum expansion.
    
    Isp = (c* / g0) * sqrt(2 * gamma^2 / (gamma - 1))
    
    This assumes optimal expansion to vacuum (Pe -> 0).
    
    Parameters:
        cstar: Characteristic velocity (m/s)
        gamma: Ratio of specific heats
    
    Returns:
        Isp: Ideal specific impulse (s)
    """
    cf_ideal = np.sqrt(2 * gamma**2 / (gamma - 1))
    return cstar * cf_ideal / g0


def analyze_propellant(oxidizer, fuel, OF_range, P_chambers, T_initial, mechanism_file, name=""):
    """
    Analyze a propellant combination over a range of O/F ratios and pressures.
    
    Parameters:
        oxidizer: Oxidizer species name (e.g., 'O2', 'N2O4')
        fuel: Fuel species name (e.g., 'CH4', 'N2H4')
        OF_range: Array of O/F ratios
        P_chambers: List of chamber pressures (Pa)
        T_initial: Initial temperature (K)
        mechanism_file: Cantera mechanism file
        name: Name for display
    
    Returns:
        results: Dictionary with arrays for each property
    """
    n_OF = len(OF_range)
    n_P = len(P_chambers)
    
    # Initialize result arrays
    results = {
        'OF': OF_range,
        'P_atm': [p / ct.one_atm for p in P_chambers],
        'MW': np.zeros((n_P, n_OF)),      # Molecular weight (g/mol)
        'Tc': np.zeros((n_P, n_OF)),      # Combustion temperature (K)
        'gamma': np.zeros((n_P, n_OF)),   # Ratio of specific heats
        'cstar': np.zeros((n_P, n_OF)),   # Characteristic velocity (m/s)
        'Isp': np.zeros((n_P, n_OF)),     # Ideal specific impulse (s)
    }
    
    print(f"\n{'='*70}")
    print(f"Analyzing {name}: {oxidizer}/{fuel}")
    print(f"{'='*70}")
    
    for j, P in enumerate(P_chambers):
        P_atm = P / ct.one_atm
        print(f"\nPressure: {P_atm:.0f} atm")
        print(f"{'O/F':<8} {'MW (g/mol)':<12} {'Tc (K)':<10} {'gamma':<8} {'c* (m/s)':<12} {'Isp (s)':<10}")
        print("-" * 60)
        
        for i, OF in enumerate(OF_range):
            try:
                # Create gas mixture
                gas = ct.Solution(mechanism_file)
                
                # Set composition based on O/F ratio
                # O/F = mass_oxidizer / mass_fuel
                # For 1 kg of fuel, we have OF kg of oxidizer
                # Mole fractions: n_ox / n_fuel = (OF / MW_ox) / (1 / MW_fuel)
                
                if oxidizer == 'O2' and fuel == 'CH4':
                    # n_ox / n_fuel = OF * MW_fuel / MW_ox
                    mole_ratio = OF * MW_CH4 / MW_O2
                    composition = f'CH4:1, O2:{mole_ratio}'
                elif oxidizer == 'N2O4' and fuel == 'N2H4':
                    mole_ratio = OF * MW_N2H4 / MW_N2O4
                    composition = f'N2H4:1, N2O4:{mole_ratio}'
                else:
                    raise ValueError(f"Unknown propellant: {oxidizer}/{fuel}")
                
                # Set initial state
                gas.TPX = T_initial, P, composition
                
                # Store initial enthalpy for HP equilibration
                h_initial = gas.enthalpy_mass
                
                # Equilibrate at constant H and P (adiabatic combustion)
                # Use robust solver as suggested in the problem hint
                gas.equilibrate('HP', solver='gibbs')
                
                # Extract properties
                MW = gas.mean_molecular_weight  # g/mol
                Tc = gas.T  # K
                gamma = gas.cp / gas.cv
                
                # Convert MW to kg/mol for c* calculation
                MW_kg = MW / 1000.0
                
                # Calculate c* and Isp
                cstar = calculate_cstar(Tc, gamma, MW_kg)
                Isp = calculate_isp_ideal(cstar, gamma)
                
                # Store results
                results['MW'][j, i] = MW
                results['Tc'][j, i] = Tc
                results['gamma'][j, i] = gamma
                results['cstar'][j, i] = cstar
                results['Isp'][j, i] = Isp
                
                # Print sample points
                if i % 10 == 0:
                    print(f"{OF:<8.2f} {MW:<12.2f} {Tc:<10.1f} {gamma:<8.4f} {cstar:<12.1f} {Isp:<10.1f}")
                    
            except Exception as e:
                print(f"Error at O/F = {OF}: {e}")
                results['MW'][j, i] = np.nan
                results['Tc'][j, i] = np.nan
                results['gamma'][j, i] = np.nan
                results['cstar'][j, i] = np.nan
                results['Isp'][j, i] = np.nan
    
    return results


def plot_results(results, name, stoich_OF, save_prefix):
    """
    Generate plots for combustion analysis results.
    """
    OF = results['OF']
    P_atm = results['P_atm']
    colors = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{name} Combustion Analysis', fontsize=14, fontweight='bold')
    
    # Properties to plot
    properties = [
        ('MW', 'Molecular Weight (g/mol)', axes[0, 0]),
        ('Tc', 'Combustion Temperature (K)', axes[0, 1]),
        ('gamma', 'Ratio of Specific Heats γ', axes[0, 2]),
        ('cstar', 'Characteristic Velocity c* (m/s)', axes[1, 0]),
        ('Isp', 'Specific Impulse Isp (s)', axes[1, 1]),
    ]
    
    for prop_name, ylabel, ax in properties:
        for j, (P, color) in enumerate(zip(P_atm, colors)):
            ax.plot(OF, results[prop_name][j, :], '-', color=color, 
                   linewidth=2, label=f'{P:.0f} atm')
        
        # Mark stoichiometric point
        ax.axvline(x=stoich_OF, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Stoich O/F = {stoich_OF:.2f}')
        
        ax.set_xlabel('O/F Ratio')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(OF.min(), OF.max())
    
    # Find and display max Isp
    ax_summary = axes[1, 2]
    ax_summary.axis('off')
    
    summary_text = f"{name} Summary\n" + "=" * 30 + "\n\n"
    summary_text += f"Stoichiometric O/F: {stoich_OF:.3f}\n\n"
    summary_text += "Maximum Isp:\n"
    
    for j, P in enumerate(P_atm):
        idx_max = np.nanargmax(results['Isp'][j, :])
        max_Isp = results['Isp'][j, idx_max]
        opt_OF = OF[idx_max]
        summary_text += f"  {P:.0f} atm: {max_Isp:.1f} s at O/F = {opt_OF:.2f}\n"
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_prefix}_analysis.png")
    
    return fig


# ============================================================================
# Main Analysis
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ROCKET PROPELLANT COMBUSTION ANALYSIS")
    print("SpaceX Raptor (LOX/CH4) vs Apollo-style (N2O4/N2H4)")
    print("=" * 70)
    print(f"\nO/F ratio range: {OF_min} to {OF_max}")
    print(f"Chamber pressures: {P_chambers_atm} atm")
    print(f"Initial temperature: {T_initial} K")
    print(f"Mechanism file: {mechanism_file}")
    
    # Analyze LOX/CH4
    results_lox_ch4 = analyze_propellant(
        oxidizer='O2', fuel='CH4',
        OF_range=OF_range, P_chambers=P_chambers,
        T_initial=T_initial, mechanism_file=mechanism_file,
        name="LOX/CH4 (Raptor)"
    )
    
    # Analyze N2O4/N2H4
    results_n2o4_n2h4 = analyze_propellant(
        oxidizer='N2O4', fuel='N2H4',
        OF_range=OF_range, P_chambers=P_chambers,
        T_initial=T_initial, mechanism_file=mechanism_file,
        name="N2O4/N2H4 (Apollo-style)"
    )
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_results(results_lox_ch4, "LOX/CH₄", OF_STOICH_LOX_CH4, "lox_ch4")
    plot_results(results_n2o4_n2h4, "N₂O₄/N₂H₄", OF_STOICH_N2O4_N2H4, "n2o4_n2h4")
    
    # ========================================================================
    # Comparison Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("COMPARISON: LOX/CH4 vs N2O4/N2H4")
    print("=" * 70)
    
    print(f"\nStoichiometric O/F ratios:")
    print(f"  LOX/CH4:    {OF_STOICH_LOX_CH4:.3f}")
    print(f"  N2O4/N2H4:  {OF_STOICH_N2O4_N2H4:.3f}")
    
    print(f"\nMaximum Specific Impulse Comparison:")
    print(f"{'Pressure':<12} {'LOX/CH4':<20} {'N2O4/N2H4':<20} {'Winner':<15}")
    print("-" * 67)
    
    for j, P in enumerate(P_chambers_atm):
        # LOX/CH4
        idx_lox = np.nanargmax(results_lox_ch4['Isp'][j, :])
        max_isp_lox = results_lox_ch4['Isp'][j, idx_lox]
        opt_of_lox = OF_range[idx_lox]
        
        # N2O4/N2H4
        idx_n2o4 = np.nanargmax(results_n2o4_n2h4['Isp'][j, :])
        max_isp_n2o4 = results_n2o4_n2h4['Isp'][j, idx_n2o4]
        opt_of_n2o4 = OF_range[idx_n2o4]
        
        winner = "LOX/CH4" if max_isp_lox > max_isp_n2o4 else "N2O4/N2H4"
        
        print(f"{P:<12.0f} {max_isp_lox:.1f} s (O/F={opt_of_lox:.2f}){'':<4} "
              f"{max_isp_n2o4:.1f} s (O/F={opt_of_n2o4:.2f}){'':<4} {winner:<15}")
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    # Determine overall winner
    avg_isp_lox = np.nanmax(results_lox_ch4['Isp'], axis=1).mean()
    avg_isp_n2o4 = np.nanmax(results_n2o4_n2h4['Isp'], axis=1).mean()
    
    print(f"""
   - LOX/CH4 average max Isp: {avg_isp_lox:.1f} s
   - N2O4/N2H4 average max Isp: {avg_isp_n2o4:.1f} s
   - {"LOX/CH4" if avg_isp_lox > avg_isp_n2o4 else "N2O4/N2H4"} provides higher specific impulse capability.
    """)
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    plt.show()
