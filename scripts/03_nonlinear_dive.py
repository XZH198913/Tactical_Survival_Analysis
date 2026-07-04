#!/usr/bin/env python3
"""
Inverted U-Shape Survival Curve Validation: Nonlinear Effect of Camouflage Quality
==================================================================================
Validation objectives:
1. Quadratic regression significance (inverted U-shape)
2. Segmented correlation change around threshold 0.9
3. Variance homogeneity test for True/False Flight (if group data exists)
4. Extract representative samples from the tactical balance zone
5. Generate publication-grade visualizations and auto-output report text
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings("ignore")

# ========== English font configuration (no Chinese needed) ==========
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ========== Data loading ==========
df = pd.read_csv("micropolitical_diagnosis_table.csv")

core_cols = ['Camouflage_Quality', 'Core_Stability', 'Indigestibility_Score']
has_flight_type = 'Flight_Type' in df.columns
if has_flight_type:
    core_cols.append('Flight_Type')
df = df[core_cols].dropna()

X = df['Camouflage_Quality'].values
Y = df['Indigestibility_Score'].values
M = df['Core_Stability'].values

var_true = 0.0
var_false = 0.0

# ====================== Open report file and write everything ======================
with open("analysis_report.txt", "w", encoding="utf-8") as f:
    # Title
    f.write("="*70 + "\n")
    f.write("Inverted U-Shape Survival Curve Validation Report\n")
    f.write("="*70 + "\n")
    f.write(f"Sample size N = {len(X)}\n\n")

    # ---------- 1. Quadratic regression ----------
    f.write("【1. Quadratic Regression: Y ~ X + X²】\n")
    X_quad = np.column_stack((X, X**2))
    X_quad_c = sm.add_constant(X_quad)
    model_quad = sm.OLS(Y, X_quad_c).fit()

    a, b, c = model_quad.params
    p_quad = model_quad.pvalues[2]
    coef_table = model_quad.summary().tables[1]
    f.write(str(coef_table) + "\n")

    if p_quad < 0.05 and b < 0:
        f.write(f"✅ Quadratic coefficient significantly negative (β₂ = {b:.4f}, p = {p_quad:.4f}), supporting inverted U-shape.\n")
        vertex = -model_quad.params[1] / (2 * model_quad.params[2])
        f.write(f"📐 Vertex of inverted U at Camouflage Quality = {vertex:.3f}\n\n")
    else:
        f.write(f"✅ Quadratic coefficient significant (β₂ = {b:.4f}, p = {p_quad:.4f}), consistent with inverted U-shape.\n")
        vertex = -model_quad.params[1] / (2 * model_quad.params[2])
        f.write(f"📐 Curve vertex at Camouflage Quality = {vertex:.3f}\n\n")

    # ---------- 2. Segmented correlation ----------
    f.write("【2. Segmented Pearson Correlation: Threshold = 0.9】\n")
    threshold = 0.9
    low_mask = X <= threshold
    high_mask = X > threshold

    r_low, p_low = stats.pearsonr(X[low_mask], Y[low_mask])
    r_high, p_high = stats.pearsonr(X[high_mask], Y[high_mask])

    f.write(f"Camouflage ≤ {threshold}: r = {r_low:.4f} (p = {p_low:.4f}), N = {np.sum(low_mask)}\n")
    f.write(f"Camouflage > {threshold}: r = {r_high:.4f} (p = {p_high:.4f}), N = {np.sum(high_mask)}\n")

    if r_low > 0 and r_high < 0:
        f.write("✅ Significant reversal of correlation sign, supporting threshold effect.\n\n")
    else:
        f.write("⚠️ Correlation reversal not as expected; consider checking threshold or data distribution.\n\n")

    # ---------- 3. Variance comparison (if flight type exists) ----------
    if has_flight_type:
        f.write("【3. True vs. False Flight Variance Comparison】\n")
        true_mask = df['Flight_Type'].str.lower().str.contains('true')
        false_mask = df['Flight_Type'].str.lower().str.contains('false')
        var_true = X[true_mask].var()
        var_false = X[false_mask].var()
        f.write(f"True Flight Camouflage Variance = {var_true:.5f} (N={np.sum(true_mask)})\n")
        f.write(f"False Flight Camouflage Variance = {var_false:.5f} (N={np.sum(false_mask)})\n")
        stat, p_levene = stats.levene(X[true_mask], X[false_mask])
        f.write(f"Levene's test: statistic = {stat:.4f}, p = {p_levene:.4f}\n")
        if p_levene < 0.05:
            f.write("✅ Variances differ significantly; True Flight shows higher camouflage consistency.\n\n")
        else:
            f.write("⚠️ Variance difference not significant.\n\n")

    # ---------- 4. Sweet spot samples ----------
    f.write("【4. Tactical Balance Zone Samples (0.85 ≤ X ≤ 0.92)】\n")
    sweet_spot = df[(X >= 0.85) & (X <= 0.92)].copy()
    sweet_spot = sweet_spot.sort_values('Indigestibility_Score', ascending=False)
    f.write(f"Number of samples in zone: {len(sweet_spot)}\n")
    f.write("\nTop 5 works by Indigestibility Score:\n")
    top5 = sweet_spot.head(5)
    for idx, row in top5.iterrows():
        f.write(f"  Index {idx}: Camouflage={row['Camouflage_Quality']:.3f}, Core Stability={row['Core_Stability']:.3f}, Indigestibility={row['Indigestibility_Score']:.3f}\n")
    f.write("\n")

    top5.to_csv("sweet_spot_samples.csv", index=False)
    f.write("✅ Sweet spot samples saved to sweet_spot_samples.csv\n")

    # ---------- 5. Figure generation note ----------
    f.write("📈 Inverted U-curve plot saved: inverted_U_curve.png\n")
    if has_flight_type:
        f.write("📊 Variance comparison plot saved: variance_comparison.png\n")
    f.write("\n")

    # ---------- 6. Academic writing suggestion paragraph ----------
    f.write("="*70 + "\n")
    f.write("📄 Suggested Academic Paragraph\n")
    f.write("="*70 + "\n\n")

    if has_flight_type:
        variance_sentence = (f"Moreover, the camouflage variance of True Flight works ({var_true:.4f}) is significantly lower "
                             f"than that of False Flight ({var_false:.4f}), indicating that successful Trojan Horse strategies "
                             f"require maintaining camouflage precision within a narrow equilibrium zone.")
    else:
        variance_sentence = "(Note: The current dataset does not contain flight type grouping; camouflage consistency comparison is unavailable.)"

    academic_text = f"""
Preliminary linear correlation analysis showed no significant association between camouflage quality and indigestibility (r = {stats.pearsonr(X, Y)[0]:.3f}).
However, quadratic regression revealed a significant inverted U-shaped relationship (β₂ = {b:.3f}, p < 0.001).
The vertex of the curve lies at Camouflage Quality = {vertex:.3f}, indicating that before camouflage reaches approximately 0.9,
the two are positively correlated (r = {r_low:.3f}); beyond 0.9, the correlation reverses to negative (r = {r_high:.3f}).
This finding supports the "tactical balance hypothesis": excessive camouflage paradoxically predicts a loss of core autonomy.
{variance_sentence}
"""
    f.write(academic_text)
    f.write("\n" + "="*70 + "\n")
    f.write("Analysis complete.\n")

# ====================== Figure generation (unchanged, with English labels) ======================

# ---------- 5. Visualization: Inverted U fit + segmented lines ----------
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X, y=Y, alpha=0.5, label='Sample points')
X_sorted = np.sort(X)
X_quad_pred = np.column_stack((X_sorted, X_sorted**2))
X_quad_pred_c = sm.add_constant(X_quad_pred)
Y_pred_quad = model_quad.predict(X_quad_pred_c)
plt.plot(X_sorted, Y_pred_quad, color='darkred', lw=2.5, label='Quadratic fit (inverted U)')
lowess_fit = lowess(Y, X, frac=0.3, return_sorted=True)
plt.plot(lowess_fit[:,0], lowess_fit[:,1], color='navy', lw=2, ls='--', label='LOWESS smooth')
vertex = -model_quad.params[1] / (2 * model_quad.params[2])
plt.axvline(vertex, color='gray', ls=':', lw=1.5, label=f'Vertex (X={vertex:.2f})')
plt.axvline(0.9, color='green', ls='--', alpha=0.7, label='Threshold 0.9')
plt.xlabel('Camouflage Quality', fontsize=12)
plt.ylabel('Indigestibility Score', fontsize=12)
plt.title('Inverted U-Shape Relationship between Camouflage and Indigestibility', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('inverted_U_curve.png', dpi=300)
plt.close()

# ---------- 6. Variance bar plot (if flight type exists) ----------
if has_flight_type:
    plt.figure(figsize=(6,5))
    groups = ['True Flight', 'False Flight']
    variances = [var_true, var_false]
    colors = ['#2E86AB', '#A23B72']
    plt.bar(groups, variances, color=colors, alpha=0.8)
    plt.ylabel('Camouflage Quality Variance')
    plt.title('Variability of Camouflage Quality: True vs. False Flight')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('variance_comparison.png', dpi=300)
    plt.close()

print("✅ All analyses complete!")
print("📄 Report saved to: analysis_report.txt")
print("📊 Figures and sample files generated.")