"""
Linear Hypothesis: Predicting linear relationship between Camouflage and Indigestibility
Trojan Horse Correlation Matrix: The Paradox of Camouflage and Resistance
==============================================================================
Testing the "90% Camouflage + 10% Heterogeneity" hypothesis.

Analysis:
1. Load micropolitical_diagnosis CSV and extract core variables
2. Compute Pearson correlation matrix
3. Generate regression scatter plot (Camouflage_Quality vs Indigestibility_Score)
4. Output descriptive statistics and correlation interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

# ==================== Configuration ====================
CSV_FILENAME = "micropolitical_diagnosis_table.csv"
OUTPUT_IMAGE = "trojan_horse_correlation_scatter.png"
DPI = 300
FIGSIZE = (8, 6)
STYLE = "seaborn-v0_8-whitegrid"

plt.style.use(STYLE)
sns.set_context("notebook", font_scale=1.2)

# ==================== Data Loading & Validation ====================
def load_and_validate_data(filepath):
    """Load CSV and validate required columns."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}\nPlease ensure '{CSV_FILENAME}' is in the current directory.")
    
    df = pd.read_csv(filepath)
    required_cols = ["Indigestibility_Score", "Core_Stability", "Camouflage_Quality"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nExisting columns: {list(df.columns)}")
    
    df_analysis = df[required_cols].copy()
    for col in required_cols:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
    
    if df_analysis.isnull().any().any():
        print("Warning: Missing values detected. Dropping affected rows.")
        df_analysis = df_analysis.dropna()
        if len(df_analysis) < 3:
            raise ValueError("Insufficient valid samples (<3). Cannot perform correlation analysis.")
    
    return df_analysis

# ==================== Correlation Matrix ====================
def compute_correlation_matrix(data):
    """Compute Pearson correlation matrix and p-values."""
    corr_matrix = data.corr(method='pearson')
    p_matrix = pd.DataFrame(np.ones_like(corr_matrix), 
                            index=corr_matrix.index, 
                            columns=corr_matrix.columns)
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if i != j:
                _, p = stats.pearsonr(data[i], data[j])
                p_matrix.loc[i, j] = p
            else:
                p_matrix.loc[i, j] = 0.0
    return corr_matrix, p_matrix

# ==================== Visualization: Scatter with Regression Line ====================
def plot_regression_scatter(x, y, xlabel, ylabel, title, output_path):
    """Scatter plot with linear regression line and confidence interval."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.scatter(x, y, alpha=0.7, edgecolors='w', linewidth=0.5, 
               color='steelblue', label='Sample points')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    
    ax.plot(line_x, line_y, color='darkred', linewidth=2.5, 
            label=f'Regression: y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r_value**2:.3f}, p = {p_value:.4f}')
    
    # 95% confidence interval
    n = len(x)
    y_pred = slope * x + intercept
    resid_std = np.sqrt(np.sum((y - y_pred)**2) / (n - 2))
    ci = 1.96 * resid_std * np.sqrt(1/n + (line_x - x.mean())**2 / np.sum((x - x.mean())**2))
    ax.fill_between(line_x, line_y - ci, line_y + ci, color='darkred', alpha=0.15)
    
    r_pearson, _ = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f'Pearson r = {r_pearson:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel, fontweight='semibold')
    ax.set_ylabel(ylabel, fontweight='semibold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {output_path}")

# ==================== Main ====================
def main():
    print("\n" + "="*60)
    print("Trojan Horse Correlation Analysis: Camouflage vs Indigestibility".center(50))
    print("="*60 + "\n")
    
    try:
        df = load_and_validate_data(CSV_FILENAME)
        print(f"Data loaded successfully. Valid sample size N = {len(df)}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        sys.exit(1)
    
    print("\n【Descriptive Statistics】")
    print(df.describe().round(3).to_string())
    
    print("\n【Pearson Correlation Matrix】")
    corr_mat, p_mat = compute_correlation_matrix(df)
    print(corr_mat.round(3).to_string())
    
    print("\n【P-values Matrix】")
    print(p_mat.round(4).to_string())
    
    r_camo_indigest = corr_mat.loc["Camouflage_Quality", "Indigestibility_Score"]
    p_camo_indigest = p_mat.loc["Camouflage_Quality", "Indigestibility_Score"]
    
    print("\n" + "-"*40)
    print(f"Key relationship: Camouflage_Quality ↔ Indigestibility_Score")
    print(f"   Pearson r = {r_camo_indigest:.3f}")
    print(f"   p = {p_camo_indigest:.4f}")
    if p_camo_indigest < 0.05:
        direction = "positive" if r_camo_indigest > 0 else "negative"
        strength = "strong" if abs(r_camo_indigest) > 0.7 else ("moderate" if abs(r_camo_indigest) > 0.4 else "weak")
        print(f"   Conclusion: Significant {direction} correlation ({strength}).")
        if r_camo_indigest > 0:
            print("   → Supports hypothesis: higher camouflage predicts higher indigestibility.")
        else:
            print("   → Contradicts hypothesis: higher camouflage predicts lower indigestibility (beware over-adaptation).")
    else:
        print("   Conclusion: No significant correlation. Hypothesis not supported by linear model.")
    print("-"*40)
    
    # Scatter plot
    plot_regression_scatter(
        x=df["Camouflage_Quality"],
        y=df["Indigestibility_Score"],
        xlabel="Camouflage Quality",
        ylabel="Indigestibility Score",
        title="Trojan Horse Paradox: Camouflage vs. Indigestibility",
        output_path=OUTPUT_IMAGE
    )
    
    # Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix Heatmap", fontweight='bold')
    plt.tight_layout()
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=DPI)
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()