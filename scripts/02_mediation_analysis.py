#!/usr/bin/env python3
"""
Mediation Analysis: Camouflage Quality → Core Stability → Indigestibility Score
================================================================================
Features:
- Fully compatible with all statsmodels versions (no .iloc issues)
- Pure English chart labels to prevent font missing errors
- Smart CSV loading (auto-detect encoding, delimiter, column names)
- Includes VIF, residual normality test, and other diagnostics
- Vectorized Bootstrap for faster computation
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== Global plot settings (English only) ====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==================== File search & loading ====================
def find_csv(default_name="micropolitical_diagnosis_table.csv"):
    """Search for CSV in current directory and common subdirectories."""
    for d in ['.', './data', '../data', './input']:
        path = os.path.join(d, default_name)
        if os.path.isfile(path):
            return path
    return None

def load_data(filepath):
    """Load CSV with automatic encoding/delimiter detection and column matching."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                line = f.readline()
            sep = ',' if line.count(',') >= 2 else '\t'
            df = pd.read_csv(filepath, encoding=enc, sep=sep)
            print(f"✅ Loaded (encoding: {enc}, delimiter: '{sep}')")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise ValueError("Could not read file. Please check encoding.")

    df.columns = df.columns.str.strip()

    # Fuzzy column matching
    mapping = {
        'Camouflage_Quality': ['camouflage_quality', 'Camouflage Quality', '伪装质量'],
        'Core_Stability': ['core_stability', 'Core Stability', '内核稳固度'],
        'Indigestibility_Score': ['indigestibility_score', 'Indigestibility Score', '不可消化指数']
    }
    selected = {}
    for std, aliases in mapping.items():
        for col in df.columns:
            if col.lower().replace(' ', '_') in [a.lower().replace(' ', '_') for a in aliases] or col == std:
                selected[std] = col
                break
        else:
            print(f"⚠️ Column '{std}' not found. Available: {list(df.columns)}")
            idx = int(input(f"Enter column index for '{std}' (0~{len(df.columns)-1}): "))
            selected[std] = df.columns[idx]

    df_clean = df[[selected['Camouflage_Quality'], 
                   selected['Core_Stability'], 
                   selected['Indigestibility_Score']]].copy()
    df_clean.columns = ['Camouflage_Quality', 'Core_Stability', 'Indigestibility_Score']

    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    print(f"✅ Valid samples: {len(df_clean)}")
    return df_clean

# ==================== Statistical diagnostics ====================
def calc_vif(exog):
    """Variance Inflation Factor."""
    vif = {}
    for i in range(exog.shape[1]):
        vif[f'VIF_{i}'] = variance_inflation_factor(exog, i)
    return vif

def model_diag(model):
    """Extract R², AIC, residual normality, etc."""
    diag = {
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj,
        'F_p': model.f_pvalue,
        'AIC': model.aic,
        'BIC': model.bic
    }
    if len(model.resid) >= 3:
        _, sw_p = stats.shapiro(model.resid)
        diag['Shapiro_p'] = sw_p
    return diag

# ==================== Mediation analysis (fixed .iloc issue) ====================
def baron_kenny_mediation(X, M, Y):
    """Stepwise regression for mediation. Uses [index] for compatibility."""
    results = {}
    X1 = sm.add_constant(X)

    m1 = sm.OLS(Y, X1).fit()
    results['step1'] = {'c': m1.params[1], 'p_c': m1.pvalues[1], 'model': m1, 'diag': model_diag(m1)}

    m2 = sm.OLS(M, X1).fit()
    results['step2'] = {'a': m2.params[1], 'p_a': m2.pvalues[1], 'model': m2, 'diag': model_diag(m2)}

    X3 = np.column_stack((X, M))
    X3c = sm.add_constant(X3)
    m3 = sm.OLS(Y, X3c).fit()
    results['step3'] = {'c_prime': m3.params[1], 'b': m3.params[2], 'p_b': m3.pvalues[2],
                        'model': m3, 'diag': model_diag(m3)}

    results['vif'] = calc_vif(X3c)
    results['indirect'] = results['step2']['a'] * results['step3']['b']
    return results

def bootstrap_mediation(X, M, Y, n_boot=5000, seed=42):
    """Vectorized Bootstrap for indirect effect confidence interval."""
    np.random.seed(seed)
    n = len(X)
    indirect_vals = np.empty(n_boot)
    idx_mat = np.random.choice(n, size=(n_boot, n), replace=True)

    for i in range(n_boot):
        if (i+1) % 1000 == 0:
            print(f"  Bootstrap progress: {i+1}/{n_boot}")
        idx = idx_mat[i]
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]

        Xc = np.column_stack((np.ones_like(Xb), Xb))
        try:
            a_b = np.linalg.lstsq(Xc, Mb, rcond=None)[0][1]
        except:
            a_b = 0.0

        XMb = np.column_stack((np.ones_like(Xb), Xb, Mb))
        try:
            b_b = np.linalg.lstsq(XMb, Yb, rcond=None)[0][2]
        except:
            b_b = 0.0

        indirect_vals[i] = a_b * b_b

    ci = np.percentile(indirect_vals, [2.5, 97.5])
    return {
        'mean': np.mean(indirect_vals),
        'ci_low': ci[0], 'ci_high': ci[1],
        'se': np.std(indirect_vals, ddof=1),
        'dist': indirect_vals
    }

# ==================== Report & Plotting ====================
def print_report(stepwise, boot):
    """Print detailed mediation report (Chinese console output)."""
    s1, s2, s3 = stepwise['step1'], stepwise['step2'], stepwise['step3']
    n = len(s1['model'].resid)
    print("\n" + "="*70)
    print("中介效应分析报告".center(60))
    print("="*70)
    print(f"\n样本量 N = {n}")
    print("\n【逐步回归】")
    print(f"Step 1 (Y~X):        c = {s1['c']:.4f}, p = {s1['p_c']:.4f}, R² = {s1['diag']['R2']:.3f}")
    print(f"Step 2 (M~X):        a = {s2['a']:.4f}, p = {s2['p_a']:.4f}, R² = {s2['diag']['R2']:.3f}")
    print(f"Step 3 (Y~X+M):      b = {s3['b']:.4f}, p = {s3['p_b']:.4f}, c' = {s3['c_prime']:.4f}, R² = {s3['diag']['R2']:.3f}")
    print(f"间接效应 a*b = {stepwise['indirect']:.4f}")

    print("\n【多重共线性 VIF】")
    vif = stepwise['vif']
    print(f"  常数项 VIF = {vif['VIF_0']:.2f}")
    print(f"  X VIF      = {vif['VIF_1']:.2f}")
    print(f"  M VIF      = {vif['VIF_2']:.2f}  (<5 indicates weak collinearity)")

    print("\n【Bootstrap (5000 iterations)】")
    print(f"  Indirect effect mean = {boot['mean']:.4f}")
    print(f"  95% CI = [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"  Bootstrap SE = {boot['se']:.4f}")

    if boot['ci_low'] * boot['ci_high'] > 0:
        print("\n✅ Conclusion: Mediation effect is significant")
    else:
        print("\n❌ Conclusion: Mediation effect is NOT significant")

    if abs(s1['c']) > 1e-8:
        prop = boot['mean'] / s1['c']
        print(f"  Proportion mediated = {prop:.2%}")

    sw_p = s3['diag'].get('Shapiro_p', np.nan)
    if not np.isnan(sw_p):
        status = "approx. normal" if sw_p > 0.05 else "non-normal"
        print(f"\n【Residual diagnosis】 Shapiro-Wilk p = {sw_p:.4f} ({status})")
    print("="*70)

def plot_dist(dist, ci_low, ci_high, out="bootstrap_dist.png"):
    """Plot Bootstrap distribution with English labels."""
    plt.figure(figsize=(8,5))
    sns.histplot(dist, bins=50, kde=True, color='#4C72B0', edgecolor='white', alpha=0.7)
    plt.axvline(ci_low, color='#C44E52', ls='--', lw=2, label=f'95% CI lower: {ci_low:.4f}')
    plt.axvline(ci_high, color='#C44E52', ls='--', lw=2, label=f'95% CI upper: {ci_high:.4f}')
    plt.axvline(0, color='black', lw=1.5, alpha=0.5, label='Zero effect line')
    plt.xlabel('Indirect Effect (a*b)')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution of Indirect Effect (5000 samples)')
    plt.legend()
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"📈 Distribution plot saved: {out}")

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    args = parser.parse_args()

    csv_path = args.csv or find_csv()
    if not csv_path:
        csv_path = input("Enter CSV file path: ").strip().strip('"')

    df = load_data(csv_path)
    X = df['Camouflage_Quality'].values
    M = df['Core_Stability'].values
    Y = df['Indigestibility_Score'].values

    print("\nRunning stepwise regression...")
    step = baron_kenny_mediation(X, M, Y)

    print("\nRunning Bootstrap (5000 iterations)...")
    boot = bootstrap_mediation(X, M, Y, n_boot=5000)

    print_report(step, boot)
    plot_dist(boot['dist'], boot['ci_low'], boot['ci_high'])

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()