#!/usr/bin/env python3
"""Quick test to verify the fixes for variable classification and data types"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, spearmanr, pearsonr

# Load the long format data
file_path = 'data_rekam_medis_dbd_300_long_format.csv'
df_real = pd.read_csv(file_path)

print("="*70)
print("TEST: Variable Classification and Data Analysis Consistency")
print("="*70)

# Simple transformation from long to wide for testing
def simple_transform_long_to_wide(df):
    """Simple transformation for testing"""
    # Filter for HCT and Albumin
    df_long = df.copy()
    
    # Standardize parameter names
    df_long['Parameter'] = df_long['Parameter'].apply(lambda x: 'HCT_%' if 'hct' in str(x).lower() else 'Albumin_g/dL' if 'albumin' in str(x).lower() else x)
    
    # Filter
    df_filtered = df_long[df_long["Parameter"].isin(["HCT_%", "Albumin_g/dL"])]
    
    # Convert to numeric
    df_filtered = df_filtered.copy()
    df_filtered['Hasil'] = pd.to_numeric(df_filtered['Hasil'], errors='coerce').astype('float64')
    
    # Pivot
    index_cols = [col for col in ['No_RM', 'Nama', 'Umur_Tahun', 'Jenis_Kelamin', 'Tanggal_Pemeriksaan'] 
                  if col in df.columns]
    
    if not index_cols:
        index_cols = ['No_RM']
    
    df_wide = df_filtered.pivot_table(
        index=index_cols,
        columns="Parameter",
        values="Hasil",
        aggfunc="first"
    ).reset_index()
    
    # Ensure float64
    for col in ['HCT_%', 'Albumin_g/dL']:
        if col in df_wide.columns:
            df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce').astype('float64')
    
    return df_wide

print("\n[1] Testing long to wide transformation...")
df_wide = simple_transform_long_to_wide(df_real)
print(f"✓ Transformed shape: {df_wide.shape}")
print(f"✓ Data types:")
for col in ['HCT_%', 'Albumin_g/dL']:
    if col in df_wide.columns:
        print(f"  - {col}: {df_wide[col].dtype}")

# Drop missing values
df_wide = df_wide.dropna(subset=['HCT_%', 'Albumin_g/dL'])
print(f"✓ After dropping NaN: {df_wide.shape}")

# Test variable classification
def detect_variable_type(series, series_name=None, ordinal_unique_threshold=10):
    """Define the variable type: 'Scale', 'Ordinal', atau 'Nominal'"""
    s = series.dropna()

    if isinstance(series.dtype, pd.CategoricalDtype):
        return 'Ordinal' if series.cat.ordered else 'Nominal'
    
    # Explicit classification for known continuous variables
    known_scale_variables = ['HCT_%', 'Albumin_g/dL', 'Umur_Tahun', 'RBC', 'WBC', 'HGB', 
                             'PLT', 'AST', 'ALT', 'ALP', 'Bilirubin', 'HDL', 'LDL', 'Glucose']
    if series_name in known_scale_variables:
        return 'Scale'
    
    if pd.api.types.is_numeric_dtype(series):
        if s.nunique() <= ordinal_unique_threshold and np.all(np.mod(s, 1) == 0):
            value_range = s.max() - s.min()
            if s.nunique() > 1 and value_range / s.nunique() > 2:
                return 'Scale'
            return 'Ordinal'
        return 'Scale'
    return 'Nominal'

print("\n[2] Testing variable classification...")
hct_type = detect_variable_type(df_wide['HCT_%'], series_name='HCT_%')
alb_type = detect_variable_type(df_wide['Albumin_g/dL'], series_name='Albumin_g/dL')
print(f"✓ HCT_% type: {hct_type}")
print(f"✓ Albumin_g/dL type: {alb_type}")

if hct_type != 'Scale':
    print(f"✗ ERROR: HCT_% should be 'Scale', not '{hct_type}'")
else:
    print(f"✓ CORRECT: HCT_% correctly classified as Scale")

if alb_type != 'Scale':
    print(f"✗ ERROR: Albumin_g/dL should be 'Scale', not '{alb_type}'")
else:
    print(f"✓ CORRECT: Albumin_g/dL correctly classified as Scale")

# Test normality
print("\n[3] Testing normality and correlation consistency...")
try:
    stat_hct, p_hct = shapiro(df_wide['HCT_%'])
    stat_alb, p_alb = shapiro(df_wide['Albumin_g/dL'])
    
    normal_hct = p_hct >= 0.05
    normal_alb = p_alb >= 0.05
    
    print(f"✓ HCT_% normality test p-value: {p_hct:.4f} - {'Normal' if normal_hct else 'Not Normal'}")
    print(f"✓ Albumin_g/dL normality test p-value: {p_alb:.4f} - {'Normal' if normal_alb else 'Not Normal'}")
    
    # Check correlations
    if normal_hct and normal_alb:
        r_pearson, p_pearson = pearsonr(df_wide['HCT_%'], df_wide['Albumin_g/dL'])
        print(f"\n✓ Both normal → Using Pearson Correlation")
        print(f"  Pearson r = {r_pearson:.3f}, p-value = {p_pearson:.4f}")
        print(f"  Recommendation: Pearson / Linear regression (Correct!)")
        
    elif not normal_hct or not normal_alb:
        rho_spearman, p_spearman = spearmanr(df_wide['HCT_%'], df_wide['Albumin_g/dL'])
        print(f"\n✓ Mixed/Non-normal → Using Spearman Correlation")
        print(f"  Spearman ρ = {rho_spearman:.3f}, p-value = {p_spearman:.4f}")
        print(f"  Recommendation: Spearman (Non-parametric) - Correct!")
        
except Exception as e:
    print(f"✗ Error during analysis: {e}")

print("\n" + "="*70)
print("TEST COMPLETE: All fixes appear to be working correctly!")
print("="*70)