"""
Problem 3.14(b): Thermodynamic Properties of Ammonia (NH3)

Calculate (H - H°) [kJ/mol] and S° [J/mol-K] at 1 atm for ammonia using 
numerical integration of Cp data from the ig.thermo.xls tables.

From the problem statement:
- Enthalpy: h(T) - h_0 = ∫[T0 to T] Cp dT   (Eq. 3)
- Entropy: S(T,P) = ∫[T0 to T] Cp/T dT + s_i0   (Eq. 5, at constant P=P0)

For an ideal gas at 1 atm (P = P0), the entropy simplifies since P/P0 = 1.
"""

import pandas as pd
import numpy as np
from scipy import integrate

# Read the NH3 data from the Excel file
print("=" * 70)
print("Problem 3.14(b): Thermodynamic Properties of Ammonia (NH3)")
print("=" * 70)

# Load the data
xls = pd.ExcelFile('ig_thermo.xls')
df_raw = pd.read_excel(xls, 'NH3')

# Extract the data starting from row 8 (0-indexed), skipping the header rows
# Columns: T (K), Cp (J/mol-K), h(T)-h0 (kJ/mol), s (J/mol-K), mu0 (kJ/mol)
data_start = 8
df = df_raw.iloc[data_start:].copy()
df.columns = ['T', 'Cp', 'H_minus_H0', 'S', 'mu0']
df = df.dropna()

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col])

# Sort by temperature
df = df.sort_values('T').reset_index(drop=True)

print(f"\nLoaded {len(df)} temperature data points from {df['T'].min()} K to {df['T'].max()} K")

# Reference conditions
T0 = 298.15  # K (reference temperature)
s_i0 = df[df['T'] == T0]['S'].values[0]  # S° at 298.15 K

print(f"\nReference conditions:")
print(f"  T0 = {T0} K")
print(f"  s_i0 (S° at 298.15 K) = {s_i0} J/mol-K")

# Extract temperature and Cp arrays
T_arr = df['T'].values
Cp_arr = df['Cp'].values

# Find the index for T0
T0_idx = np.where(T_arr == T0)[0][0]

print("\n" + "=" * 70)
print("NUMERICAL INTEGRATION RESULTS")
print("=" * 70)
print(f"\n{'T (K)':<10} {'Cp':>10} {'(H-H°) calc':>15} {'(H-H°) tab':>15} "
      f"{'Error %':>10} {'S° calc':>12} {'S° tab':>10} {'Error %':>10}")
print("-" * 92)

# Store results
results = []

# Calculate for each temperature
for i in range(len(T_arr)):
    T = T_arr[i]
    Cp = Cp_arr[i]
    H_tab = df.iloc[i]['H_minus_H0']  # Tabulated (H - H°) in kJ/mol
    S_tab = df.iloc[i]['S']            # Tabulated S° in J/mol-K
    
    if T < T0:
        # For T < T0, integrate backwards (negative contribution)
        # Use trapezoidal rule for the segment from T to T0
        mask = (T_arr >= T) & (T_arr <= T0)
        T_segment = T_arr[mask]
        Cp_segment = Cp_arr[mask]
        
        # Integrate Cp dT from T to T0 (will give positive value)
        H_calc = -integrate.trapezoid(Cp_segment, T_segment) / 1000  # Convert to kJ/mol
        
        # Integrate Cp/T dT from T to T0
        S_calc = s_i0 - integrate.trapezoid(Cp_segment / T_segment, T_segment)
        
    elif T == T0:
        H_calc = 0.0
        S_calc = s_i0
    else:
        # For T > T0, integrate forward
        mask = (T_arr >= T0) & (T_arr <= T)
        T_segment = T_arr[mask]
        Cp_segment = Cp_arr[mask]
        
        # Integrate Cp dT from T0 to T
        H_calc = integrate.trapezoid(Cp_segment, T_segment) / 1000  # Convert to kJ/mol
        
        # Integrate Cp/T dT from T0 to T
        S_calc = s_i0 + integrate.trapezoid(Cp_segment / T_segment, T_segment)
    
    # Calculate errors
    if abs(H_tab) > 0.001:
        H_error = abs((H_calc - H_tab) / H_tab) * 100
    else:
        H_error = abs(H_calc - H_tab) * 100  # Absolute error for near-zero values
    
    S_error = abs((S_calc - S_tab) / S_tab) * 100
    
    results.append({
        'T': T,
        'Cp': Cp,
        'H_calc': H_calc,
        'H_tab': H_tab,
        'H_error': H_error,
        'S_calc': S_calc,
        'S_tab': S_tab,
        'S_error': S_error
    })
    
    print(f"{T:<10.2f} {Cp:>10.2f} {H_calc:>15.3f} {H_tab:>15.3f} "
          f"{H_error:>10.2f} {S_calc:>12.2f} {S_tab:>10.2f} {S_error:>10.3f}")

# Summary statistics
df_results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"\nEnthalpy (H - H°) [kJ/mol]:")
print(f"  Mean absolute error: {df_results['H_error'].mean():.3f}%")
print(f"  Max absolute error:  {df_results['H_error'].max():.3f}%")

print(f"\nEntropy S° [J/mol-K]:")
print(f"  Mean absolute error: {df_results['S_error'].mean():.4f}%")
print(f"  Max absolute error:  {df_results['S_error'].max():.4f}%")

print("\n" + "=" * 70)
print("METHODOLOGY")
print("=" * 70)
print("""
For numerical integration, we used the trapezoidal rule to approximate:

1. Enthalpy change (Eq. 3):
   h(T) - h_0 = ∫[T0 to T] Cp dT

2. Entropy (from Eq. 5 at P = P0 = 1 atm):
   S(T) = s_i0 + ∫[T0 to T] (Cp/T) dT
   
   where s_i0 is the entropy at the reference temperature T0 = 298.15 K.

The slight differences between calculated and tabulated values arise from:
- The discrete nature of the trapezoidal integration
- The tabulated values may use more sophisticated integration methods
  or polynomial fits for Cp(T)
""")

print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
