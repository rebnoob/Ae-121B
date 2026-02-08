"""
Ammonia Dissociation Equilibrium Analysis

This script analyzes the equilibrium composition of ammonia (NH3) as a function of
temperature at two pressures (1 atm and 10 atm).

The dissociation reaction is: 2 NH3(g) ⇌ N2(g) + 3 H2(g)

Based on the Cantera equil.m demo concept:
- Set initial composition to pure NH3
- Use equilibrate('TP') for constant temperature and pressure equilibration
- Extract mole fractions of NH3, N2, H2

Equilibration method: 'TP' since temperature and pressure are specified.
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Setup
# ============================================================================

# Temperature range: 300K to 1650K
T_min = 300  # K
T_max = 1650  # K
n_temps = 100  # Number of temperature points
T_range = np.linspace(T_min, T_max, n_temps)

# Pressures to analyze
P_1atm = ct.one_atm  # 1 atm in Pa
P_10atm = 10 * ct.one_atm  # 10 atm in Pa

# Load the Cantera mechanism (using local Ae121 mechanism)
# This mechanism includes NH3, N2, H2 and other species
mechanism_file = 'Ae121_V1.0.yaml'

print("=" * 70)
print("Ammonia Dissociation Equilibrium Analysis")
print("=" * 70)
print(f"Temperature range: {T_min} K to {T_max} K")
print(f"Pressures: 1 atm and 10 atm")
print(f"Mechanism file: {mechanism_file}")
print("=" * 70)

# ============================================================================
# Equilibrium Calculations
# ============================================================================

# Species of interest
species_of_interest = ['NH3', 'N2', 'H2']

# Initialize storage arrays for 1 atm
X_NH3_1atm = np.zeros(n_temps)
X_N2_1atm = np.zeros(n_temps)
X_H2_1atm = np.zeros(n_temps)

# Initialize storage arrays for 10 atm
X_NH3_10atm = np.zeros(n_temps)
X_N2_10atm = np.zeros(n_temps)
X_H2_10atm = np.zeros(n_temps)

print("\nCalculating equilibrium compositions...")
print(f"{'T (K)':<10} {'P (atm)':<10} {'X_NH3':<12} {'X_N2':<12} {'X_H2':<12} {'Sum':<10}")
print("-" * 66)

# Calculate equilibrium for each temperature and pressure
for i, T in enumerate(T_range):
    # --- 1 atm ---
    gas_1atm = ct.Solution(mechanism_file)
    gas_1atm.TPX = T, P_1atm, 'NH3:1.0'  # Pure ammonia initially
    gas_1atm.equilibrate('TP')  # Equilibrate at constant T and P
    
    # Get mole fractions
    X_NH3_1atm[i] = gas_1atm['NH3'].X[0]
    X_N2_1atm[i] = gas_1atm['N2'].X[0]
    X_H2_1atm[i] = gas_1atm['H2'].X[0]
    
    # --- 10 atm ---
    gas_10atm = ct.Solution(mechanism_file)
    gas_10atm.TPX = T, P_10atm, 'NH3:1.0'  # Pure ammonia initially
    gas_10atm.equilibrate('TP')  # Equilibrate at constant T and P
    
    # Get mole fractions
    X_NH3_10atm[i] = gas_10atm['NH3'].X[0]
    X_N2_10atm[i] = gas_10atm['N2'].X[0]
    X_H2_10atm[i] = gas_10atm['H2'].X[0]
    
    # Print a few sample points
    if i % 20 == 0:
        print(f"{T:<10.1f} {1:<10} {X_NH3_1atm[i]:<12.6f} {X_N2_1atm[i]:<12.6f} {X_H2_1atm[i]:<12.6f} {X_NH3_1atm[i] + X_N2_1atm[i] + X_H2_1atm[i]:<10.6f}")
        print(f"{T:<10.1f} {10:<10} {X_NH3_10atm[i]:<12.6f} {X_N2_10atm[i]:<12.6f} {X_H2_10atm[i]:<12.6f} {X_NH3_10atm[i] + X_N2_10atm[i] + X_H2_10atm[i]:<10.6f}")

# ============================================================================
# Normalize Results
# ============================================================================
# For ammonia dissociation (2 NH3 -> N2 + 3 H2), only NH3, N2, H2 are present
# Sum of their mole fractions should already be 1
# But we normalize to ensure this is exactly true

print("\n" + "=" * 70)
print("Normalizing mole fractions...")
print("=" * 70)

# Calculate sums
sum_1atm = X_NH3_1atm + X_N2_1atm + X_H2_1atm
sum_10atm = X_NH3_10atm + X_N2_10atm + X_H2_10atm

# Normalize
X_NH3_1atm_norm = X_NH3_1atm / sum_1atm
X_N2_1atm_norm = X_N2_1atm / sum_1atm
X_H2_1atm_norm = X_H2_1atm / sum_1atm

X_NH3_10atm_norm = X_NH3_10atm / sum_10atm
X_N2_10atm_norm = X_N2_10atm / sum_10atm
X_H2_10atm_norm = X_H2_10atm / sum_10atm

# Verify normalization
print(f"Sum of mole fractions (1 atm) - min: {(X_NH3_1atm_norm + X_N2_1atm_norm + X_H2_1atm_norm).min():.6f}, max: {(X_NH3_1atm_norm + X_N2_1atm_norm + X_H2_1atm_norm).max():.6f}")
print(f"Sum of mole fractions (10 atm) - min: {(X_NH3_10atm_norm + X_N2_10atm_norm + X_H2_10atm_norm).min():.6f}, max: {(X_NH3_10atm_norm + X_N2_10atm_norm + X_H2_10atm_norm).max():.6f}")

# ============================================================================
# Generate Plot
# ============================================================================

print("\nGenerating plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Colors for species
colors = {'NH3': 'blue', 'N2': 'green', 'H2': 'red'}

# 1 atm (solid lines)
ax.plot(T_range, X_NH3_1atm_norm, '-', color=colors['NH3'], linewidth=2, label='NH₃ (1 atm)')
ax.plot(T_range, X_N2_1atm_norm, '-', color=colors['N2'], linewidth=2, label='N₂ (1 atm)')
ax.plot(T_range, X_H2_1atm_norm, '-', color=colors['H2'], linewidth=2, label='H₂ (1 atm)')

# 10 atm (dashed lines)
ax.plot(T_range, X_NH3_10atm_norm, '--', color=colors['NH3'], linewidth=2, label='NH₃ (10 atm)')
ax.plot(T_range, X_N2_10atm_norm, '--', color=colors['N2'], linewidth=2, label='N₂ (10 atm)')
ax.plot(T_range, X_H2_10atm_norm, '--', color=colors['H2'], linewidth=2, label='H₂ (10 atm)')

# Labels and formatting
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('Mole Fraction', fontsize=12)
ax.set_title('Ammonia Dissociation Equilibrium: 2 NH₃ ⇌ N₂ + 3 H₂\n(Normalized Mole Fractions)', fontsize=14)
ax.legend(loc='center right', fontsize=10)
ax.set_xlim(T_min, T_max)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Add annotation explaining the trends
ax.annotate('Higher T → More dissociation\nHigher P → Less dissociation', 
            xy=(0.02, 0.98), xycoords='axes fraction',
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_file = 'ammonia_equilibrium.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key observations from the ammonia dissociation equilibrium:

1. TEMPERATURE EFFECT:
   - Dissociation increases with temperature (endothermic reaction)
   - At low T (~300K): NH₃ dominates, N₂ and H₂ are negligible
   - At high T (~1650K): Significant dissociation into N₂ and H₂

2. PRESSURE EFFECT:
   - Higher pressure suppresses dissociation (Le Chatelier's principle)
   - 2 mol NH₃ → 4 mol products (N₂ + 3H₂)
   - At any T, 10 atm shows more NH₃ than 1 atm

3. STOICHIOMETRY CHECK:
   - The ratio of N₂:H₂ mole fractions should be approximately 1:3
   - This confirms the reaction: 2 NH₃ → N₂ + 3 H₂
""")

# Stoichiometry check at high temperature
T_check_idx = -1  # Last temperature point
ratio_1atm = X_H2_1atm_norm[T_check_idx] / X_N2_1atm_norm[T_check_idx] if X_N2_1atm_norm[T_check_idx] > 0 else 0
ratio_10atm = X_H2_10atm_norm[T_check_idx] / X_N2_10atm_norm[T_check_idx] if X_N2_10atm_norm[T_check_idx] > 0 else 0

print(f"Stoichiometry verification at T = {T_range[T_check_idx]:.0f} K:")
print(f"  H₂/N₂ ratio at 1 atm:  {ratio_1atm:.3f} (expected: 3.0)")
print(f"  H₂/N₂ ratio at 10 atm: {ratio_10atm:.3f} (expected: 3.0)")
print("=" * 70)