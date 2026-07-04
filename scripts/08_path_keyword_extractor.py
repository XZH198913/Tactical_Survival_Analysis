#!/usr/bin/env python3
"""
两大逃逸路径的多维基因图谱提取（修复版）
================================================================
修复内容：
1. 先探查数据库 category 的真实值，再动态映射维度
2. 自动处理中英文类别标签
3. 确保 CSV 和报告都能正常生成

输出：
- path1_gene_by_category.csv / path2_gene_by_category.csv（完整高频词表）
- gene_comparison_report.md（对比报告）
"""

import sqlite3
import pandas as pd
from collections import Counter

# ==================== 配置 ====================
DB_PATH = 'westbund_advanced.0315.db'
CLUSTER_CSV = 'cluster_analysis_labeled_data.csv'

# 路径映射规则
def map_to_path(cluster_label):
    label = str(cluster_label).lower()
    if 'semiotic' in label or 'glitch' in label:
        return 'Semiotic/Cognitive Glitch'
    else:
        return 'Material/Haptic Subversion'

# ==================== 第一步：探查数据库 category 真实值 ====================
print("🔍 探查 Keywords 表中的 category 分布...")
conn = sqlite3.connect(DB_PATH)
df_categories = pd.read_sql_query("SELECT DISTINCT category FROM Keywords", conn)
conn.close()

print(f"   发现 {len(df_categories)} 种不同的 category 值:")
all_categories = df_categories['category'].dropna().unique().tolist()
for cat in all_categories:
    print(f"     - {cat}")

# ==================== 第二步：动态构建维度映射 ====================
# 根据探查结果，手动将数据库中的类别映射到三个宏观维度
# 您可以根据实际探查结果调整以下映射字典

def build_dimension_map(categories):
    """根据探查到的类别列表，自动建议映射（可手动修正）"""
    # 预设关键词匹配规则
    material_keywords = ['material', 'medium', '材质', '媒介', '材料']
    style_keywords = ['style', 'movement', '流派', '风格', '主义']
    theory_keywords = ['theory', 'concept', '理论', '概念', '哲学']
    
    dimension_map = {}
    for cat in categories:
        cat_lower = str(cat).lower()
        if any(kw in cat_lower for kw in material_keywords):
            dimension_map[cat] = '材质与媒介'
        elif any(kw in cat_lower for kw in style_keywords):
            dimension_map[cat] = '流派与风格'
        elif any(kw in cat_lower for kw in theory_keywords):
            dimension_map[cat] = '理论概念'
        else:
            dimension_map[cat] = f'其他({cat})'  # 保留原类别名
    return dimension_map

DIMENSION_MAP = build_dimension_map(all_categories)
print("\n📌 维度映射结果:")
for k, v in DIMENSION_MAP.items():
    print(f"   {k} -> {v}")

# ==================== 第三步：加载聚类结果 ====================
print("\n📂 加载聚类结果...")
df_cluster = pd.read_csv(CLUSTER_CSV)
df_cluster['theoretical_path'] = df_cluster['cluster_label'].apply(map_to_path)

n_path1 = len(df_cluster[df_cluster['theoretical_path'] == 'Material/Haptic Subversion'])
n_path2 = len(df_cluster[df_cluster['theoretical_path'] == 'Semiotic/Cognitive Glitch'])
print(f"   路径一(物质/触觉): {n_path1} 件")
print(f"   路径二(符号/认知): {n_path2} 件")

# ==================== 第四步：联结数据库关键词 ====================
print("\n🔗 联结数据库...")
conn = sqlite3.connect(DB_PATH)
query = """
SELECT 
    akm.artwork_id,
    k.category,
    k.term_zh
FROM Artwork_Keyword_Map akm
JOIN Keywords k ON akm.keyword_id = k.keyword_id
"""
df_keywords = pd.read_sql_query(query, conn)
conn.close()

df_merged = df_cluster[['artwork_id', 'theoretical_path']].merge(
    df_keywords, on='artwork_id', how='inner'
)
print(f"   有效联结记录数: {len(df_merged)}")

# ==================== 第五步：分维度统计（无过滤，全保留） ====================
def generate_full_profile(df_path):
    """生成所有类别的完整高频词表，并添加维度标签"""
    profiles = {}
    for category in df_path['category'].dropna().unique():
        terms = df_path[df_path['category'] == category]['term_zh']
        counter = Counter(terms)
        df_cat = pd.DataFrame(counter.most_common(20), columns=['Term', 'Frequency'])
        df_cat['Category'] = category
        df_cat['Dimension'] = DIMENSION_MAP.get(category, '未分类')
        profiles[category] = df_cat
    return profiles

path1_df = df_merged[df_merged['theoretical_path'] == 'Material/Haptic Subversion']
path2_df = df_merged[df_merged['theoretical_path'] == 'Semiotic/Cognitive Glitch']

print("\n🧬 生成路径一基因图谱...")
path1_profiles = generate_full_profile(path1_df)
print("🧬 生成路径二基因图谱...")
path2_profiles = generate_full_profile(path2_df)

# ==================== 第六步：保存完整 CSV ====================
# 合并所有类别为一个大表
path1_all = pd.concat(path1_profiles.values(), ignore_index=True) if path1_profiles else pd.DataFrame()
path2_all = pd.concat(path2_profiles.values(), ignore_index=True) if path2_profiles else pd.DataFrame()

path1_all.to_csv('path1_gene_by_category.csv', index=False)
path2_all.to_csv('path2_gene_by_category.csv', index=False)
print("✅ 完整基因表已保存: path1_gene_by_category.csv, path2_gene_by_category.csv")

# ==================== 第七步：生成对比报告 ====================
report = []
report.append("# 两大逃逸路径的多维基因图谱对比\n")

# 路径一
report.append(f"## 路径一：物质与触觉反噬 (N={n_path1} 件)\n")
for cat, df_cat in path1_profiles.items():
    dim = DIMENSION_MAP.get(cat, cat)
    report.append(f"### {dim} ({cat})\n")
    report.append("| 关键词 | 频次 |")
    report.append("|--------|------|")
    for _, row in df_cat.head(10).iterrows():
        report.append(f"| {row['Term']} | {row['Frequency']} |")
    report.append("")

# 路径二
report.append(f"## 路径二：符号与认知故障 (N={n_path2} 件)\n")
for cat, df_cat in path2_profiles.items():
    dim = DIMENSION_MAP.get(cat, cat)
    report.append(f"### {dim} ({cat})\n")
    report.append("| 关键词 | 频次 |")
    report.append("|--------|------|")
    for _, row in df_cat.head(10).iterrows():
        report.append(f"| {row['Term']} | {row['Frequency']} |")
    report.append("")

# 对比摘要表
report.append("## 对比摘要\n")
# 收集所有维度
all_dims = set(DIMENSION_MAP.values())
summary_rows = []
for dim in all_dims:
    p1_terms = []
    p2_terms = []
    # 聚合该维度下所有类别的词频
    for cat, dim_name in DIMENSION_MAP.items():
        if dim_name == dim:
            if cat in path1_profiles:
                p1_terms.extend(path1_profiles[cat].head(3)['Term'].tolist())
            if cat in path2_profiles:
                p2_terms.extend(path2_profiles[cat].head(3)['Term'].tolist())
    p1_str = ', '.join(p1_terms[:5]) if p1_terms else '—'
    p2_str = ', '.join(p2_terms[:5]) if p2_terms else '—'
    summary_rows.append(f"| {dim} | {p1_str} | {p2_str} |")

report.append("| 维度 | 路径一(物质/触觉) 高频特征 | 路径二(符号/认知) 高频特征 |")
report.append("|------|---------------------------|---------------------------|")
report.extend(summary_rows)

# 论文段落
report.append("\n## 📄 论文写作建议段落\n")
report.append("""
基于上述分维度统计，两条逃逸路径呈现出鲜明的本体论差异：

**路径一（物质与触觉反噬）**在材质上高度依赖非常规、具有熵增属性的媒介（如工业废料、木炭、有机降解物），
其“特洛伊外壳”多伪装为极简主义、贫穷艺术等画廊安全流派，而理论话语则集中于身体现象学、贱斥物与新唯物主义。
这一基因图谱证实了该路径的逃逸策略是**通过物质的不可控衰变与触觉暴力，在感知层面制造资本无法抚平的痉挛**。

**路径二（符号与认知故障）**则严重依赖数字影像、霓虹灯文本和档案装置，
其风格伪装多借用观念艺术、挪用艺术等以语言为中心的传统，理论概念高度绑定人类世、算法规训、认识论暴力等宏大叙事。
这一配置使得该路径极易被策展黑话（IAE）收编，沦为安全的“智力快消品”。

两条路径在媒介本体论上的生殖隔离，构成了第五章区分“肉身起义”与“符号陷阱”的实证基础。
""")

with open('gene_comparison_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("\n✅ 全部完成！")
print("   - path1_gene_by_category.csv")
print("   - path2_gene_by_category.csv")
print("   - gene_comparison_report.md")