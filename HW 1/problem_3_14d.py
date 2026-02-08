import numpy as np
import matplotlib.pyplot as plt
from thermo_utils import (load_entropy_data, get_entropy_at_T,
                          calculate_entropy_terms)

def main():
    print("Problem 3.14(d): Generating entropy plot at T = 500 K")
    
    # Load thermodynamic data
    T_NH3, S_NH3_data = load_entropy_data('NH3')
    T_N2, S_N2_data = load_entropy_data('N2')
    T_H2, S_H2_data = load_entropy_data('H2')
    
    T = 500  # K
    S_NH3 = get_entropy_at_T(T_NH3, S_NH3_data, T)
    S_N2 = get_entropy_at_T(T_N2, S_N2_data, T)
    S_H2 = get_entropy_at_T(T_H2, S_H2_data, T)
    
    # Fine grid for smooth curves
    alphas = np.linspace(0, 1, 101)
    
    # Calculate entropy components
    term1_vals = []
    term1_plus_term2_vals = []
    total_10atm_vals = []
    
    for alpha in alphas:
        S_total, term1, term2, term3, _ = calculate_entropy_terms(
            alpha, T, S_NH3, S_N2, S_H2, P=10
        )
        term1_vals.append(term1)
        term1_plus_term2_vals.append(term1 + term2)
        total_10atm_vals.append(term1 + term2 + term3)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(alphas, term1_vals, 'b-', linewidth=2, 
            label=r'i) Linear mixing rule (Term 1 only): $\sum x_i S_i^\circ$')
    ax.plot(alphas, term1_plus_term2_vals, 'g--', linewidth=2,
            label=r'ii) With mixing entropy (Term 1 + Term 2): $\sum x_i S_i^\circ - R\sum x_i \ln x_i$')
    ax.plot(alphas, total_10atm_vals, 'r-.', linewidth=2,
            label=r'iii) Total at 10 atm (all terms): $\sum x_i S_i^\circ - R\sum x_i \ln x_i - R\ln(P/P_0)$')
    
    ax.set_xlabel('Dissociation Fraction, α', fontsize=12)
    ax.set_ylabel('Entropy [J/mol-K]', fontsize=12)
    ax.set_title('Problem 3.14(d): Entropy vs Dissociation Fraction\n'
                 f'T = {T} K, Ammonia Dissociation: 2NH₃ → N₂ + 3H₂', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    ax.annotate(f'α=0: {term1_vals[0]:.1f}', xy=(0, term1_vals[0]), 
                xytext=(0.1, term1_vals[0]+3), fontsize=9, color='blue')
    ax.annotate(f'α=1: {term1_vals[-1]:.1f}', xy=(1, term1_vals[-1]), 
                xytext=(0.85, term1_vals[-1]+3), fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig('problem_3_14d_plot.png', dpi=150, bbox_inches='tight')
    plt.savefig('problem_3_14d_plot.pdf', bbox_inches='tight')
    print("Plot saved to: problem_3_14d_plot.png and problem_3_14d_plot.pdf")
    plt.show()

if __name__ == "__main__":
    main()
