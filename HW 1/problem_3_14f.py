"""
Problem 3.14(f): Gibbs Free Energy

Calculate Gibbs function for NH3/N2/H2 mixture and identify equilibrium composition.
"""

import numpy as np
from thermo_utils import (load_thermo_data, get_entropy_at_T, get_enthalpy_at_T,
                          calculate_mole_fractions, R)

def calculate_gibbs_energy(alpha, T, H_NH3, H_N2, H_H2, S_NH3, S_N2, S_H2, P):
    """
    Calculate TOTAL (extensive) Gibbs free energy for the mixture.
    
    G_total = Σ n_i * μ_i where μ_i = G°_i + RT ln(x_i P/P0)
    
    Returns G_total in kJ (for 2 initial moles of NH3 basis).
    """
    # Enthalpies of formation at 298 K (kJ/mol)
    Hf_NH3 = -45.90
    Hf_N2 = 0.0
    Hf_H2 = 0.0
    
    # Mole numbers (basis: 2 moles initial NH3)
    n_NH3 = 2 * (1 - alpha)
    n_N2 = alpha
    n_H2 = 3 * alpha
    n_total = 2 + 2 * alpha
    
    # Mole fractions
    x_NH3 = n_NH3 / n_total if n_total > 0 else 0
    x_N2 = n_N2 / n_total if n_total > 0 else 0
    x_H2 = n_H2 / n_total if n_total > 0 else 0
    
    # Total enthalpy = Hf° + (H - H°) [kJ/mol]
    H_total_NH3 = Hf_NH3 + H_NH3
    H_total_N2 = Hf_N2 + H_N2
    H_total_H2 = Hf_H2 + H_H2
    
    R_kJ = R / 1000.0  # R in kJ/mol-K
    
    def safe_log(x):
        return np.log(x) if x > 1e-10 else np.log(1e-10)
    
    # Chemical potentials (kJ/mol)
    mu_NH3 = H_total_NH3 - T * S_NH3 / 1000.0 + R_kJ * T * safe_log(x_NH3 * P) if n_NH3 > 0 else 0
    mu_N2 = H_total_N2 - T * S_N2 / 1000.0 + R_kJ * T * safe_log(x_N2 * P) if n_N2 > 0 else 0
    mu_H2 = H_total_H2 - T * S_H2 / 1000.0 + R_kJ * T * safe_log(x_H2 * P) if n_H2 > 0 else 0
    
    # Total Gibbs energy (kJ)
    G_total = n_NH3 * mu_NH3 + n_N2 * mu_N2 + n_H2 * mu_H2
    
    # Intensive quantities for display
    H_mix = (n_NH3 * H_total_NH3 + n_N2 * H_total_N2 + n_H2 * H_total_H2) / n_total
    S_mix = (n_NH3 * S_NH3 + n_N2 * S_N2 + n_H2 * S_H2) / n_total
    
    return G_total, H_mix, S_mix, n_total

def main():
    print("=" * 100)
    print("Problem 3.14(f): GIBBS FREE ENERGY CALCULATION")
    print("=" * 100)
    
    # Target conditions
    temperatures = [500, 866, 1644]
    pressures = [1, 10]
    alphas = np.arange(0, 1.1, 0.1)
    
    # Load thermodynamic data
    T_NH3, S_NH3_data, H_NH3_data = load_thermo_data('NH3')
    T_N2, S_N2_data, H_N2_data = load_thermo_data('N2')
    T_H2, S_H2_data, H_H2_data = load_thermo_data('H2')
    
    equilibrium_results = []
    
    for P in pressures:
        for T in temperatures:
            S_NH3 = get_entropy_at_T(T_NH3, S_NH3_data, T)
            S_N2 = get_entropy_at_T(T_N2, S_N2_data, T)
            S_H2 = get_entropy_at_T(T_H2, S_H2_data, T)
            
            H_NH3 = get_enthalpy_at_T(T_NH3, H_NH3_data, T)
            H_N2 = get_enthalpy_at_T(T_N2, H_N2_data, T)
            H_H2 = get_enthalpy_at_T(T_H2, H_H2_data, T)
            
            print(f"\n{'='*100}")
            print(f"T = {T} K, P = {P} atm")
            print(f"{'='*100}")
            print(f"{'α':^6} {'x_NH3':^8} {'x_N2':^8} {'x_H2':^8} {'n_total':^8} {'H_mix':^12} {'S_mix':^12} {'G_total':^12}")
            print(f"{'':^6} {'':^8} {'':^8} {'':^8} {'[mol]':^8} {'[kJ/mol]':^12} {'[J/mol-K]':^12} {'[kJ]':^12}")
            print("-" * 100)
            
            min_G = float('inf')
            
            for alpha in alphas:
                x_NH3, x_N2, x_H2, n_total = calculate_mole_fractions(alpha)
                G_total, H_mix, S_mix, n_t = calculate_gibbs_energy(
                    alpha, T, H_NH3, H_N2, H_H2, S_NH3, S_N2, S_H2, P
                )
                
                if G_total < min_G:
                    min_G = G_total
                
                print(f"{alpha:^6.1f} {x_NH3:^8.4f} {x_N2:^8.4f} {x_H2:^8.4f} {n_t:^8.2f} "
                      f"{H_mix:^12.3f} {S_mix:^12.2f} {G_total:^12.3f}")
            
            # Fine grid for precise equilibrium
            alphas_fine = np.linspace(0, 1, 1001)
            G_vals = [calculate_gibbs_energy(a, T, H_NH3, H_N2, H_H2, S_NH3, S_N2, S_H2, P)[0] 
                      for a in alphas_fine]
            
            min_idx = np.argmin(G_vals)
            alpha_eq = alphas_fine[min_idx]
            G_eq = G_vals[min_idx]
            x_NH3_eq, x_N2_eq, x_H2_eq, _ = calculate_mole_fractions(alpha_eq)
            
            equilibrium_results.append({
                'T': T, 'P': P, 'alpha_eq': alpha_eq, 'G_eq': G_eq,
                'x_NH3': x_NH3_eq, 'x_N2': x_N2_eq, 'x_H2': x_H2_eq
            })
            
            print("-" * 100)
            print(f"  *** EQUILIBRIUM: α_eq = {alpha_eq:.3f}, G_min = {G_eq:.3f} kJ ***")
            print(f"      Composition: x_NH3 = {x_NH3_eq:.4f}, x_N2 = {x_N2_eq:.4f}, x_H2 = {x_H2_eq:.4f}")
    
    # Summary
    print("\n" + "=" * 100)
    print("EQUILIBRIUM SUMMARY")
    print("=" * 100)
    print(f"{'T (K)':^10} {'P (atm)':^10} {'α_eq':^10} {'G_min [kJ]':^12} {'x_NH3':^10} {'x_N2':^10} {'x_H2':^10}")
    print("-" * 100)
    
    for res in equilibrium_results:
        print(f"{res['T']:^10} {res['P']:^10} {res['alpha_eq']:^10.3f} {res['G_eq']:^12.3f} "
              f"{res['x_NH3']:^10.4f} {res['x_N2']:^10.4f} {res['x_H2']:^10.4f}")

if __name__ == "__main__":
    main()
