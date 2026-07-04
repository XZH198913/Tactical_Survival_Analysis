#!/usr/bin/env python3
"""
突围战术语义聚类分析（学术发表增强版 · 纯数据驱动）
================================================================
论文定位：第四章 4.4 节 - 突围战术的语义降维

增强特性：
- 自动确定最优聚类数（肘部法则+轮廓系数+CH指数+DB指数）
- 轮廓系数分布可视化与按簇统计
- 簇内代表性样本提取（最近/最远各3个）
- 聚类质量综合诊断图（四合一）
- Markdown报告包含完整统计量
- 支持PDF矢量图输出

运行前安装：
    pip install pandas numpy matplotlib seaborn scikit-learn


"""

import os
import sys
import sqlite3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ==================== 0. 全局配置 ====================
def configure_environment():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['pdf.fonttype'] = 42  # 确保PDF可编辑
    sns.set_style("whitegrid")
    np.random.seed(42)

configure_environment()

# 自动配色
CLUSTER_COLORS = plt.cm.tab10.colors

# ==================== 1. 数据加载 ====================
def load_data(source, table='micropolitical_diagnosis'):
    if source.endswith('.db'):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Database not found: {source}")
        conn = sqlite3.connect(source)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        print(f"✅ Loaded {len(df)} records from database")
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV file not found: {source}")
        df = pd.read_csv(source)
        print(f"✅ Loaded {len(df)} records from CSV")
    return df

def filter_true_flight(df):
    df.columns = df.columns.str.lower()
    verdict_col = 'verdict_type' if 'verdict_type' in df.columns else 'Verdict_Type'.lower()
    strategy_col = 'breakthrough_strategy' if 'breakthrough_strategy' in df.columns else 'Breakthrough_Strategy'.lower()
    
    df_true = df[df[verdict_col].astype(str).str.contains('True_Flight', case=False, na=False)].copy()
    df_true = df_true[df_true[strategy_col].notna()]
    df_true['cleaned_text'] = df_true[strategy_col].astype(str).str.replace('_', ' ').str.lower()
    print(f"✅ Valid True_Flight samples: {len(df_true)}")
    return df_true, strategy_col

# ==================== 2. 综合聚类数验证（四指标） ====================
def comprehensive_k_validation(X, max_k=8, output_prefix='cluster_validation'):
    """生成四合一验证图：肘部法则、轮廓系数、CH指数、DB指数"""
    inertias, sil_scores, ch_scores, db_scores = [], [], [], []
    K_range = range(2, min(max_k, X.shape[0]) + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X.toarray(), labels))
        db_scores.append(davies_bouldin_score(X.toarray(), labels))
    
    best_k_sil = K_range[np.argmax(sil_scores)]
    best_k_ch = K_range[np.argmax(ch_scores)]
    best_k_db = K_range[np.argmin(db_scores)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 肘部法则
    axes[0, 0].plot(K_range, inertias, 'bo-')
    axes[0, 0].set_xlabel('Number of clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    
    # 轮廓系数
    axes[0, 1].plot(K_range, sil_scores, 'go-')
    axes[0, 1].axvline(x=best_k_sil, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Number of clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title(f'Silhouette Score (best k={best_k_sil})')
    
    # CH指数
    axes[1, 0].plot(K_range, ch_scores, 'mo-')
    axes[1, 0].axvline(x=best_k_ch, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Number of clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Index')
    axes[1, 0].set_title(f'CH Index (best k={best_k_ch})')
    
    # DB指数（越小越好）
    axes[1, 1].plot(K_range, db_scores, 'co-')
    axes[1, 1].axvline(x=best_k_db, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Number of clusters (k)')
    axes[1, 1].set_ylabel('Davies-Bouldin Index')
    axes[1, 1].set_title(f'DB Index (best k={best_k_db})')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comprehensive.png', dpi=300)
    plt.savefig(f'{output_prefix}_comprehensive.pdf', format='pdf')
    plt.close()
    
    print(f"\n📊 Comprehensive k-validation saved: {output_prefix}_comprehensive.png/pdf")
    print(f"   Recommended k by Silhouette: {best_k_sil}")
    print(f"   Recommended k by CH Index: {best_k_ch}")
    print(f"   Recommended k by DB Index: {best_k_db}")
    
    return best_k_sil

# ==================== 3. 轮廓系数详细分析 ====================
def silhouette_detailed_analysis(X, labels, k, output_prefix='silhouette'):
    """生成轮廓系数分布图，并输出按簇统计"""
    sil_vals = silhouette_samples(X, labels)
    sil_avg = np.mean(sil_vals)
    
    # 按簇统计
    cluster_sil_stats = {}
    for i in range(k):
        cluster_sil = sil_vals[labels == i]
        cluster_sil_stats[i] = {
            'mean': np.mean(cluster_sil),
            'std': np.std(cluster_sil),
            'min': np.min(cluster_sil),
            'max': np.max(cluster_sil),
            'negative_ratio': np.sum(cluster_sil < 0) / len(cluster_sil)
        }
    
    # 绘制轮廓系数分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for i in range(k):
        cluster_sil = sil_vals[labels == i]
        cluster_sil.sort()
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_lower + len(cluster_sil)),
                         0, cluster_sil, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), f'Cluster {i}', fontsize=10)
        y_lower += len(cluster_sil) + 20
    
    ax.axvline(x=sil_avg, color='red', linestyle='--', label=f'Average ({sil_avg:.3f})')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Samples (stacked by cluster)')
    ax.set_title(f'Silhouette Analysis for k={k}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_distribution.png', dpi=300)
    plt.savefig(f'{output_prefix}_distribution.pdf', format='pdf')
    plt.close()
    
    print(f"\n📈 Silhouette analysis saved: {output_prefix}_distribution.png/pdf")
    print(f"   Overall Average: {sil_avg:.3f}")
    for i, stats_dict in cluster_sil_stats.items():
        print(f"   Cluster {i}: mean={stats_dict['mean']:.3f}, std={stats_dict['std']:.3f}, "
              f"negative_ratio={stats_dict['negative_ratio']:.1%}")
    
    return sil_vals, cluster_sil_stats

# ==================== 4. 置信椭圆 ====================
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if len(x) != len(y):
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# ==================== 5. 纯数据驱动聚类（核心） ====================
def perform_pure_clustering(df, k=None):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    if k is None:
        k = comprehensive_k_validation(X)
        print(f"\n🤖 Auto-selected k = {k}")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = kmeans.fit_predict(X)
    df['cluster_id'] = labels
    
    # 聚类质量评估
    sil_avg = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X.toarray(), labels)
    db_score = davies_bouldin_score(X.toarray(), labels)
    print(f"\n📊 Clustering Quality Metrics:")
    print(f"   Silhouette Score: {sil_avg:.3f}")
    print(f"   Calinski-Harabasz Index: {ch_score:.1f}")
    print(f"   Davies-Bouldin Index: {db_score:.3f} (lower is better)")
    
    # 轮廓系数详细分析
    sil_vals, cluster_sil_stats = silhouette_detailed_analysis(X, labels, k)
    df['silhouette'] = sil_vals
    
    # 自动生成簇标签
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_labels = {}
    print("\n📌 Cluster top keywords (auto-generated labels):")
    for i in range(k):
        top_words = [terms[ind] for ind in order_centroids[i, :5]]
        label = '/'.join(top_words[:3])
        cluster_labels[i] = label
        print(f"   Cluster {i}: {label}  (top5: {', '.join(top_words)})")
    
    df['cluster_label'] = df['cluster_id'].map(cluster_labels)
    return df, X, kmeans, vectorizer, sil_avg, ch_score, db_score, k, cluster_sil_stats

# ==================== 6. 可视化 ====================
def visualize_pure_clusters(df, X, kmeans, k, output_path='figure4_4_clusters_pure'):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    df['pca_x'], df['pca_y'] = X_pca[:, 0], X_pca[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 散点图
    for cluster_id in range(k):
        subset = df[df['cluster_id'] == cluster_id]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        label = subset['cluster_label'].iloc[0] if len(subset) > 0 else f'Cluster {cluster_id}'
        ax1.scatter(subset['pca_x'], subset['pca_y'], c=[color], label=label, s=70, alpha=0.7, edgecolor='w')
        if len(subset) > 2:
            confidence_ellipse(subset['pca_x'].values, subset['pca_y'].values, ax1, n_std=2.0, edgecolor=color, linewidth=2, alpha=0.5)
    
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='X', s=300, c='black', edgecolor='white', label='Centroid')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.set_title(f'Semantic Clustering of Breakthrough Strategies (k={k})')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3, linestyle=':')
    
    # 柱状图
    counts = df['cluster_label'].value_counts()
    bars = ax2.bar(range(len(counts)), counts.values, color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(counts))])
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels(counts.index, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Distribution per Cluster')
    for bar, count in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{count}\n({count/len(df):.1%})', ha='center', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.suptitle('Figure 4-4: Data-Driven Clustering of Escape Tactics', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_path}.png', dpi=300)
    plt.savefig(f'{output_path}.pdf', format='pdf')
    plt.close()
    print(f"\n📈 Cluster visualization saved: {output_path}.png/pdf")

# ==================== 7. 簇内代表性样本提取 ====================
def extract_representative_samples(df, X, kmeans, k, top_n=3):
    """提取每个簇距离中心最近和最远的样本"""
    dist_matrix = pairwise_distances(X, kmeans.cluster_centers_)
    df['dist_to_center'] = dist_matrix[np.arange(len(df)), df['cluster_id']]
    
    representatives = {}
    for i in range(k):
        subset = df[df['cluster_id'] == i].copy()
        if len(subset) == 0:
            continue
        # 最近（最典型）
        closest = subset.nsmallest(top_n, 'dist_to_center')
        # 最远（边界样本）
        farthest = subset.nlargest(top_n, 'dist_to_center')
        representatives[i] = {
            'label': subset['cluster_label'].iloc[0],
            'closest': closest,
            'farthest': farthest
        }
    return representatives

# ==================== 8. 增强版Markdown报告 ====================
def generate_enhanced_report(df, X, kmeans, vectorizer, sil_avg, ch_score, db_score, 
                             k, cluster_sil_stats, representatives, output_md='cluster_report_enhanced.md'):
    lines = ["# Chapter 4.4 Cluster Analysis Report (Enhanced)\n"]
    lines.append(f"**Sample Size**: {len(df)} True_Flight works\n")
    lines.append(f"**Optimal Number of Clusters**: k = {k}\n\n")
    
    lines.append("## Clustering Quality Metrics\n")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")
    lines.append(f"| Silhouette Score | {sil_avg:.3f} | {'Fair' if sil_avg > 0.2 else 'Moderate' if sil_avg > 0.1 else 'Weak'} structure |")
    lines.append(f"| Calinski-Harabasz Index | {ch_score:.1f} | Higher = better separation |")
    lines.append(f"| Davies-Bouldin Index | {db_score:.3f} | Lower = better separation |\n")
    
    lines.append("## Silhouette Statistics by Cluster\n")
    lines.append("| Cluster | Mean Silhouette | Std Dev | Negative Ratio |")
    lines.append("|---------|-----------------|---------|----------------|")
    for i, stats_dict in cluster_sil_stats.items():
        label = df[df['cluster_id'] == i]['cluster_label'].iloc[0]
        lines.append(f"| {label} | {stats_dict['mean']:.3f} | {stats_dict['std']:.3f} | {stats_dict['negative_ratio']:.1%} |")
    
    lines.append("\n## Sample Distribution\n")
    lines.append("| Cluster Label | Count | Percentage |")
    lines.append("|---------------|-------|------------|")
    for label, count in df['cluster_label'].value_counts().items():
        lines.append(f"| {label} | {count} | {count/len(df):.1%} |")
    
    lines.append("\n## Cluster Profiles (Top 10 TF-IDF Keywords)\n")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(k):
        label = df[df['cluster_id'] == i]['cluster_label'].iloc[0]
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        lines.append(f"### Cluster {i}: {label}\n")
        lines.append(f"**Top Keywords**: {', '.join(top_words)}\n")
    
    lines.append("\n## Most Representative Samples (Closest to Centroid)\n")
    lines.append("| Cluster | Artwork ID | Breakthrough Strategy | Distance | Silhouette |")
    lines.append("|---------|------------|-----------------------|----------|------------|")
    for i, rep in representatives.items():
        for _, row in rep['closest'].iterrows():
            artwork_id = row.get('artwork_id', 'N/A')
            strategy = row.get('breakthrough_strategy', '')[:45]
            lines.append(f"| {rep['label']} | {artwork_id} | {strategy}... | {row['dist_to_center']:.3f} | {row['silhouette']:.3f} |")
    
    lines.append("\n## Boundary Samples (Farthest from Centroid)\n")
    lines.append("| Cluster | Artwork ID | Breakthrough Strategy | Distance | Silhouette |")
    lines.append("|---------|------------|-----------------------|----------|------------|")
    for i, rep in representatives.items():
        for _, row in rep['farthest'].iterrows():
            artwork_id = row.get('artwork_id', 'N/A')
            strategy = row.get('breakthrough_strategy', '')[:45]
            lines.append(f"| {rep['label']} | {artwork_id} | {strategy}... | {row['dist_to_center']:.3f} | {row['silhouette']:.3f} |")
    
    lines.append("\n---\n*Report generated by data-driven clustering pipeline (v5.0)*")
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"📄 Enhanced report saved: {output_md}")

# ==================== 9. 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="Data-Driven Semantic Clustering (Enhanced)")
    parser.add_argument('--source', type=str, default='westbund_advanced.0315.db')
    parser.add_argument('--k', type=int, default=None, help='Number of clusters (auto if not specified)')
    parser.add_argument('--output_prefix', type=str, default='cluster_analysis')
    args = parser.parse_args()
    
    df_raw = load_data(args.source)
    df_true, _ = filter_true_flight(df_raw)
    
    df_final, X, kmeans, vectorizer, sil_avg, ch_score, db_score, k_used, cluster_sil_stats = perform_pure_clustering(df_true, k=args.k)
    
    print("\n📊 Final Cluster Distribution:")
    for label, count in df_final['cluster_label'].value_counts().items():
        print(f"   {label}: {count} ({count/len(df_final):.1%})")
    
    visualize_pure_clusters(df_final, X, kmeans, k_used, output_path=f'{args.output_prefix}_clusters')
    
    representatives = extract_representative_samples(df_final, X, kmeans, k_used, top_n=3)
    
    generate_enhanced_report(df_final, X, kmeans, vectorizer, sil_avg, ch_score, db_score, 
                             k_used, cluster_sil_stats, representatives, output_md=f'{args.output_prefix}_report.md')
    
    df_final.to_csv(f'{args.output_prefix}_labeled_data.csv', index=False)
    print(f"💾 Labeled data saved: {args.output_prefix}_labeled_data.csv")
    
    print("\n✅ Enhanced data-driven analysis completed successfully.")

if __name__ == "__main__":
    main()