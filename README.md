特洛伊木马策略数据分析 Pipeline

本代码库负责论文第四章的全部量化与定性数据驱动分析，探究“伪装质量”与“不可消化性”之间的非线性生存机制。

## 📊 数据流向与执行顺序

### Phase 1: 机制探索 (01 - 02)
- 运行 `01_correlation_scatter.py`：探寻直接线性关系（确立悖论起点）。
- 运行 `02_mediation_analysis.py`：确立“内核稳固度”的中介作用机制。

### Phase 2: 非线性转折与对比 (03 - 04)
- 运行 `03_nonlinear_dive.py`：论证核心的“倒U型关系”，定位战术甜点位 (0.85-0.92)。
- 运行 `04_evidence_generator.py`：对比真/假逃逸样本，证明真逃逸策略在伪装精度上的极低方差（高度纪律性）。

### Phase 3: 边界精算与终极制图 (05 - 06)
- 运行 `05_jn_analysis.py`：通过 Johnson-Neyman 寻找边际效应由正转负的精确临界点。
- 运行 `06_dual_threshold_composite_figure.py`：生成论文核心图表（双阈值约束模型双Y轴综合图）。

### Phase 4: 语义降维 (07)
- 运行 `tactical_clustering_pipeline.py`：对成功突围的策略文本进行 TF-IDF + KMeans 聚类，提炼战术类型。

## 📁 关键输出物说明
- `dual_threshold_model.png`：第四章核心理论模型图。
- `analysis_report.txt` / `cluster_report_enhanced.md`：自动生成的统计检验结果与学术报告，可直接引用至正文。
- `sweet_spot_samples.csv`：甜点位区间内的代表性艺术作品清单。