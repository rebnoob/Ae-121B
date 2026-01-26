"""
Shared utility functions for Problem 3.14
"""

import pandas as pd
import numpy as np
from scipy import interpolate

# Constants
R = 8.314  # J/mol-K, universal gas constant

def load_thermo_data(species, excel_file='ig_thermo.xls'):
    """Load thermodynamic data (entropy and enthalpy) from tables."""
    xls = pd.ExcelFile(excel_file)
    df_raw = pd.read_excel(xls, species)
    data_start = 8
    df = df_raw.iloc[data_start:].copy()
    df.columns = ['T', 'Cp', 'H_minus_H0', 'S', 'mu0']
    df = df.dropna()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    df = df.sort_values('T').reset_index(drop=True)
    return df['T'].values, df['S'].values, df['H_minus_H0'].values

def load_entropy_data(species, excel_file='ig_thermo.xls'):
    """Load entropy data from thermodynamic tables (backward compatible)."""
    T, S, _ = load_thermo_data(species, excel_file)
    return T, S

def get_entropy_at_T(T_data, S_data, T_target):
    """Interpolate entropy at target temperature."""
    f = interpolate.interp1d(T_data, S_data, kind='linear', fill_value='extrapolate')
    return float(f(T_target))

def get_enthalpy_at_T(T_data, H_data, T_target):
    """Interpolate enthalpy at target temperature."""
    f = interpolate.interp1d(T_data, H_data, kind='linear', fill_value='extrapolate')
    return float(f(T_target))

def calculate_mole_fractions(alpha):
    """
    Calculate mole fractions for NH3/N2/H2 mixture.
    
    Starting with 2 moles NH3:
    2NH3 -> N2 + 3H2 (extent α)
    
    n_NH3 = 2(1 - α)
    n_N2 = α
    n_H2 = 3α
    n_total = 2 + 2α
    """
    n_NH3 = 2 * (1 - alpha)
    n_N2 = alpha
    n_H2 = 3 * alpha
    n_total = 2 + 2 * alpha
    
    x_NH3 = n_NH3 / n_total
    x_N2 = n_N2 / n_total
    x_H2 = n_H2 / n_total
    
    return x_NH3, x_N2, x_H2, n_total

def xlnx(x):
    """Calculate x*ln(x), handling x=0 case (limit is 0)."""
    if x <= 0:
        return 0.0
    return x * np.log(x)

def calculate_entropy_terms(alpha, T, S_NH3, S_N2, S_H2, P):
    """
    Calculate the three entropy terms from Eq. (6).
    
    Returns total entropy and individual terms (all in J/mol-K).
    """
    x_NH3, x_N2, x_H2, n_total = calculate_mole_fractions(alpha)
    
    # Term 1: Weighted pure species entropy
    term1 = x_NH3 * S_NH3 + x_N2 * S_N2 + x_H2 * S_H2
    
    # Term 2: Entropy of mixing (-R * Σ x_i * ln(x_i))
    term2 = -R * (xlnx(x_NH3) + xlnx(x_N2) + xlnx(x_H2))
    
    # Term 3: Pressure correction (-R * ln(P/P0))
    P0 = 1.0
    term3 = -R * np.log(P / P0)
    
    # Total entropy
    S_total = term1 + term2 + term3
    
    return S_total, term1, term2, term3, n_total
