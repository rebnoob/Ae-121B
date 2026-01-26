import numpy as np
from thermo_utils import (load_entropy_data, get_entropy_at_T, 
                          calculate_mole_fractions, calculate_entropy_terms)

def main():
    print("=" * 90)
    print("Problem 3.14(c): Entropy of NH3/N2/H2 Mixture")
    print("Ammonia Dissociation: 2NH3 → N2 + 3H2")
    print("=" * 90)
    
    # Load thermodynamic data
    T_NH3, S_NH3_data = load_entropy_data('NH3')
    T_N2, S_N2_data = load_entropy_data('N2')
    T_H2, S_H2_data = load_entropy_data('H2')
    
    print("\nThermodynamic data loaded from ig_thermo.xls")
    
    # Target conditions
    temperatures = [500, 866, 1644]  # K
    pressures = [1, 10]              # atm
    alphas = np.arange(0, 1.1, 0.1)  # Dissociation fraction
    
    # Get entropy values at target temperatures
    print("\n" + "-" * 90)
    print("Pure Species Standard Entropies S°(T) [J/mol-K]:")
    print("-" * 90)
    print(f"{'T (K)':<10} {'S°_NH3':<15} {'S°_N2':<15} {'S°_H2':<15}")
    
    entropy_data = {}
    for T in temperatures:
        S_NH3 = get_entropy_at_T(T_NH3, S_NH3_data, T)
        S_N2 = get_entropy_at_T(T_N2, S_N2_data, T)
        S_H2 = get_entropy_at_T(T_H2, S_H2_data, T)
        entropy_data[T] = {'NH3': S_NH3, 'N2': S_N2, 'H2': S_H2}
        print(f"{T:<10} {S_NH3:<15.2f} {S_N2:<15.2f} {S_H2:<15.2f}")
    
    # Calculate and print tables
    for P in pressures:
        for T in temperatures:
            S_NH3 = entropy_data[T]['NH3']
            S_N2 = entropy_data[T]['N2']
            S_H2 = entropy_data[T]['H2']
            
            print("\n" + "=" * 90)
            print(f"T = {T} K, P = {P} atm")
            print("=" * 90)
            print(f"{'α':^6} {'x_NH3':^8} {'x_N2':^8} {'x_H2':^8} {'Term 1':^12} {'Term 2':^12} {'Term 3':^12} {'S_total':^12}")
            print(f"{'':^6} {'':^8} {'':^8} {'':^8} {'(Σx·S°)':^12} {'(-RΣx·lnx)':^12} {'(-Rln(P/P0))':^12} {'[J/mol-K]':^12}")
            print("-" * 90)
            
            for alpha in alphas:
                x_NH3, x_N2, x_H2, n_total = calculate_mole_fractions(alpha)
                S_total, term1, term2, term3, _ = calculate_entropy_terms(
                    alpha, T, S_NH3, S_N2, S_H2, P
                )
                
                print(f"{alpha:^6.1f} {x_NH3:^8.4f} {x_N2:^8.4f} {x_H2:^8.4f} "
                      f"{term1:^12.2f} {term2:^12.2f} {term3:^12.2f} {S_total:^12.2f}")

if __name__ == "__main__":
    main()
