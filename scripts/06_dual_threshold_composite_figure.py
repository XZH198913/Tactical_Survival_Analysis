#!/usr/bin/env python3
"""
双阈值约束模型综合证据图：伪装质量的边际效应与场域筛选
================================================================
功能：
- 双Y轴展示二次拟合曲线与Johnson-Neyman边际效应
- 标注统计顶点、J-N临界点、战术甜点位
- 包含样本分布地毯图与LOWESS平滑参考线
- 直接输出论文级矢量图（PDF/SVG/PNG）

"""

import os
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings("ignore")

# ==================== 0. 学术级视觉配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.format'] = 'png'       # 可改为 'pdf' 获得矢量图
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 12

# 学术期刊常用配色（色盲友好，低饱和度）
COLOR_Y = '#2C3E50'          # 深蓝灰 —— Y值曲线
COLOR_SLOPE = '#A93226'      # 暗红 —— 边际效应
COLOR_SCATTER = '#5D6D7E'    # 灰蓝 —— 散点
COLOR_LOWESS = '#1ABC9C'     # 青绿 —— LOWESS平滑
COLOR_VERTEX = '#7F8C8D'     # 灰色 —— 顶点线
COLOR_SWEET = '#F39C12'      # 金色 —— 甜点位填充
COLOR_JN = '#E74C3C'         # 红色 —— JN负向区

# ==================== 1. 数据加载模块 ====================
def load_data(source, table_name='micropolitical_diagnosis'):
    """智能加载：优先尝试数据库，回退到CSV"""
    if source.endswith('.db'):
        if not os.path.exists(source):
            raise FileNotFoundError(f"数据库不存在: {source}")
        conn = sqlite3.connect(source)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        print(f"✅ 从数据库加载 {len(df)} 条记录。")
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV文件不存在: {source}")
        df = pd.read_csv(source)
        print(f"✅ 从CSV加载 {len(df)} 条记录。")
    return df

def prepare_variables(df, x_col='camouflage_quality', y_col='indigestibility_score'):
    """列名兼容、类型转换、缺失值处理"""
    df.columns = df.columns.str.lower()
    x_col = x_col.lower()
    y_col = y_col.lower()
    
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"缺少必要列。现有列: {list(df.columns)}")
    
    X = pd.to_numeric(df[x_col], errors='coerce')
    Y = pd.to_numeric(df[y_col], errors='coerce')
    mask = (~X.isna()) & (~Y.isna())
    return X[mask].values, Y[mask].values

# ==================== 2. 核心计算 ====================
def fit_quadratic_model(X, Y):
    """二次回归并提取参数"""
    X_quad = np.column_stack((X, X**2))
    X_quad = sm.add_constant(X_quad)
    model = sm.OLS(Y, X_quad).fit()
    return model

def compute_jn_curve(model, X_grid):
    """计算Johnson-Neyman斜率及置信带"""
    b0 = model.params[0]
    b1 = model.params[1]
    b2 = model.params[2]
    cov = model.cov_params()
    # 兼容不同的cov_params()返回类型
    if hasattr(cov, 'iloc'):
        var_b1 = cov.iloc[1, 1]
        var_b2 = cov.iloc[2, 2]
        cov_b1b2 = cov.iloc[1, 2]
    else:
        var_b1 = cov[1, 1]
        var_b2 = cov[2, 2]
        cov_b1b2 = cov[1, 2]
    
    slopes = b1 + 2 * b2 * X_grid
    se_slopes = np.sqrt(var_b1 + 4 * (X_grid**2) * var_b2 + 4 * X_grid * cov_b1b2)
    t_crit = stats.t.ppf(0.975, df=model.df_resid)
    ci_low = slopes - t_crit * se_slopes
    ci_high = slopes + t_crit * se_slopes
    return slopes, ci_low, ci_high

def find_jn_boundaries(X_grid, ci_low, ci_high):
    """定位J-N临界点（置信区间首次包含0或离开0）"""
    # 负向显著起点：上界 < 0
    neg_start = None
    neg_mask = ci_high < 0
    if np.any(neg_mask):
        idx = np.where(neg_mask)[0][0]
        neg_start = X_grid[idx]
    # 正向显著终点：下界 > 0
    pos_end = None
    pos_mask = ci_low > 0
    if np.any(pos_mask):
        idx = np.where(pos_mask)[0][-1]
        pos_end = X_grid[idx]
    return pos_end, neg_start

# ==================== 3. 绘图主函数 ====================
def plot_dual_axis_composite(X, Y, model, X_grid, Y_fit, slopes, ci_low, ci_high, 
                             pos_end, neg_start, output_path='dual_threshold_model.png'):
    """生成双Y轴综合证据图"""
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    # ---------- 左轴：不可消化性 (Y) ----------
    # 散点（半透明，展示原始分布）
    ax1.scatter(X, Y, color=COLOR_SCATTER, alpha=0.25, s=25, edgecolors='none', 
                label='样本观察点 (N={})'.format(len(X)), rasterized=True)
    
    # 二次拟合曲线
    ax1.plot(X_grid, Y_fit, color=COLOR_Y, linewidth=2.8, label='二次回归拟合 $Y = \\beta_0 + \\beta_1 X + \\beta_2 X^2$')
    
    # LOWESS平滑参考线（非参数稳健）
    lowess_fit = lowess(Y, X, frac=0.3, return_sorted=True)
    ax1.plot(lowess_fit[:,0], lowess_fit[:,1], color=COLOR_LOWESS, linewidth=1.8, 
             linestyle='-.', label='LOWESS 平滑 (非参数)')
    
    # 统计顶点标注
    vertex_x = -model.params[1] / (2 * model.params[2])
    vertex_y = model.params[0] + model.params[1]*vertex_x + model.params[2]*(vertex_x**2)
    ax1.axvline(vertex_x, color=COLOR_VERTEX, linestyle=':', linewidth=1.5, alpha=0.9)
    ax1.scatter([vertex_x], [vertex_y], color=COLOR_VERTEX, s=60, zorder=5, 
                marker='D', edgecolors='white', linewidth=1)
    ax1.annotate(f'统计顶点 X={vertex_x:.3f}', 
                 xy=(vertex_x, vertex_y), xytext=(vertex_x-0.15, vertex_y-0.12),
                 arrowprops=dict(arrowstyle='->', color=COLOR_VERTEX, lw=1),
                 fontsize=10, color=COLOR_VERTEX, fontweight='bold')
    
    # 样本分布地毯图（底部）
    y_min, y_max = ax1.get_ylim()
    y_range = y_max - y_min
    ax1.scatter(X, np.full_like(X, y_min - y_range*0.02), marker='|', 
                color='gray', alpha=0.4, s=80)
    
    # 左轴标签与样式
    ax1.set_xlabel('伪装质量 (Camouflage Quality)', fontsize=13, fontweight='semibold')
    ax1.set_ylabel('不可消化指数 (Indigestibility Score)', color=COLOR_Y, fontsize=13, fontweight='semibold')
    ax1.tick_params(axis='y', labelcolor=COLOR_Y, colors=COLOR_Y)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.05, 1.10)
    ax1.grid(True, linestyle=':', alpha=0.4, axis='y')
    
    # ---------- 右轴：边际效应 (斜率) ----------
    ax2 = ax1.twinx()
    ax2.plot(X_grid, slopes, color=COLOR_SLOPE, linewidth=2.2, linestyle='--', 
             label='边际效应 $\\partial Y/\\partial X = \\beta_1 + 2\\beta_2 X$')
    ax2.fill_between(X_grid, ci_low, ci_high, color=COLOR_SLOPE, alpha=0.12, 
                     label='95% 置信带 (Johnson-Neyman)')
    ax2.axhline(0, color='black', linewidth=1.0, alpha=0.5, linestyle='-')
    ax2.set_ylabel('边际效应 (简单斜率)', color=COLOR_SLOPE, fontsize=13, fontweight='semibold')
    ax2.tick_params(axis='y', labelcolor=COLOR_SLOPE, colors=COLOR_SLOPE)
    
    # ---------- 关键区域着色 ----------
    # J-N 显著负向区 (红色半透明)
    if neg_start is not None:
        ax1.axvspan(neg_start, 1.0, color=COLOR_JN, alpha=0.06, zorder=0)
        ax1.text(neg_start + 0.01, 0.06, f'J-N临界点\nX={neg_start:.3f}', 
                 color=COLOR_JN, fontsize=9, rotation=90, va='bottom', fontweight='bold')
    if pos_end is not None:
        ax1.axvspan(0.0, pos_end, color='#2ECC71', alpha=0.04, zorder=0)
    
    # 战术甜点位 (金色高亮)
    sweet_low, sweet_high = 0.85, 0.92
    ax1.axvspan(sweet_low, sweet_high, color=COLOR_SWEET, alpha=0.18, zorder=0, 
                label='战术平衡“甜点位” [0.85, 0.92]')
    
    # 甜点位内样本的高亮标记（可选）
    sweet_mask = (X >= sweet_low) & (X <= sweet_high)
    if sweet_mask.sum() > 0:
        ax1.scatter(X[sweet_mask], Y[sweet_mask], color=COLOR_SWEET, alpha=0.5, 
                    s=30, edgecolors='white', linewidth=0.5, zorder=4, 
                    label=f'甜点位样本 (N={sweet_mask.sum()})')
    
    # 添加解释性文本（理论注释）
    ax1.text(0.05, 0.95, '场域入场券门槛', transform=ax1.transAxes, 
             fontsize=10, color='#E67E22', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.9))
    ax1.text(0.72, 0.95, '过度适应陷阱', transform=ax1.transAxes, 
             fontsize=10, color=COLOR_JN, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', alpha=0.9))
    
    # ---------- 图例合并 ----------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 自定义图例顺序和位置
    custom_lines = lines1[:3] + lines2[:2] + [mpatches.Patch(color=COLOR_SWEET, alpha=0.3, label=labels1[-1])]
    custom_labels = labels1[:3] + labels2[:2] + [labels1[-1]]
    ax1.legend(custom_lines, custom_labels, loc='upper right', fontsize=10, 
               frameon=True, fancybox=True, framealpha=0.9)
    
    # ---------- 标题与调整 ----------
    plt.title('图X：伪装质量的“双阈值约束模型”——边际效应与场域筛选的综合证据', 
              fontsize=15, fontweight='bold', pad=20)
    
    # 添加模型公式标注
    formula = f'$Y = {model.params[0]:.3f} + {model.params[1]:.3f}X - {abs(model.params[2]):.3f}X^2$\n$R^2 = {model.rsquared:.3f}$'
    ax1.text(0.98, 0.02, formula, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📈 综合证据图已保存: {output_path}")

# ==================== 4. 报告输出 ====================
def print_summary(model, X, Y, pos_end, neg_start, vertex_x):
    print("\n" + "="*65)
    print("双阈值约束模型 · 统计摘要")
    print("="*65)
    print(f"样本量 N = {len(X)}")
    print(f"二次模型 R² = {model.rsquared:.3f}，F({int(model.df_model)},{int(model.df_resid)}) = {model.fvalue:.2f}")
    print(f"\n统计顶点: X = {vertex_x:.3f}, Y = {model.predict([1, vertex_x, vertex_x**2])[0]:.3f}")
    if pos_end:
        print(f"J-N 正向显著区终点: X = {pos_end:.3f}")
    if neg_start:
        print(f"J-N 负向显著区起点: X = {neg_start:.3f}")
    print(f"\n场域甜点位 (经验): [0.85, 0.92]")
    print("="*65)

# ==================== 5. 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="生成双阈值约束模型综合证据图")
    parser.add_argument('--source', type=str, default='westbund_advanced.0315.db',
                        help='数据源：.db 或 .csv 文件')
    parser.add_argument('--x', type=str, default='camouflage_quality')
    parser.add_argument('--y', type=str, default='indigestibility_score')
    parser.add_argument('--output', type=str, default='dual_threshold_model.png')
    args = parser.parse_args()
    
    # 加载数据
    try:
        df = load_data(args.source)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        sys.exit(1)
    
    X, Y = prepare_variables(df, args.x, args.y)
    print(f"✅ 有效样本: {len(X)}")
    
    # 拟合模型
    model = fit_quadratic_model(X, Y)
    X_grid = np.linspace(0, 1, 500)
    Y_fit = model.predict(sm.add_constant(np.column_stack((X_grid, X_grid**2))))
    
    # 计算J-N
    slopes, ci_low, ci_high = compute_jn_curve(model, X_grid)
    pos_end, neg_start = find_jn_boundaries(X_grid, ci_low, ci_high)
    vertex_x = -model.params[1] / (2 * model.params[2])
    
    # 打印摘要
    print_summary(model, X, Y, pos_end, neg_start, vertex_x)
    
    # 绘图
    plot_dual_axis_composite(X, Y, model, X_grid, Y_fit, slopes, ci_low, ci_high,
                             pos_end, neg_start, args.output)
    
    print("\n✅ 分析完成。")

if __name__ == "__main__":
    main()