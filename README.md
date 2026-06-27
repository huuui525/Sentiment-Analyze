# 多语言情感分析项目

一个从英文影评情感分类逐步扩展到中英双语、多源文本场景的 NLP 工程项目。项目覆盖数据构建、BERT 系列模型微调、长文本截断实验、模型横向对比、错误诊断、实验报告和推理脚本。

## 项目亮点

- 基于 `bert-base-uncased`、`roberta-base`、`distilbert-base-uncased`、`xlm-roberta-base`、`bert-base-multilingual-cased` 完成多轮模型对比。
- 设计 `head`、`tail`、`head_tail` 三种长文本截断策略，并验证 `head_tail_256` 在 IMDb 长文本场景中的收益。
- 构建多源中英双语情感数据集，融合 IMDb、TweetEval、微博情感数据、外卖评论数据，并保留 emoji 相关字段。
- 使用“数据源 + 标签”均衡采样策略，减少单一数据源或单一类别对训练的支配。
- 提供统一训练、评估、错误分析、推理脚本，所有实验通过 YAML 配置管理。

## 当前主模型

当前推荐主模型为：

```text
xlm-roberta-base + 数据源-标签均衡多语言情感数据集 + head_tail_256
```

第六阶段均衡数据集测试结果：

| 模型 | 验证集 F1 | 测试集 Accuracy | 测试集 F1 | 测试错误数 |
|---|---:|---:|---:|---:|
| `xlm-roberta-base` | 0.9425 | 0.9399 | 0.9366 | 613 / 10198 |
| `bert-base-multilingual-cased` | 0.9329 | 0.9217 | 0.9178 | 799 / 10198 |

因此，本项目当前选择 `xlm-roberta-base` 作为中英双语情感分析主模型。

## 项目结构

```text
p3_SentientAnalyze/
|-- configs/                         # 训练配置
|   |-- train.yaml                    # IMDb BERT 基线
|   |-- phase4_*.yaml                 # 长文本截断实验
|   |-- phase5_*.yaml                 # 英文模型对比实验
|   |-- phase6_xlm_roberta_balanced.yaml
|   `-- phase6_mbert_balanced.yaml
|-- data/
|   |-- README.md                     # 数据来源与重建说明
|   `-- sample/                       # 可提交的小样本，仅用于查看字段结构
|-- docs/
|   |-- project_architecture.md       # 架构与模块说明
|   |-- experiment_summary.md         # 实验总览
|   |-- phase6_multilingual_report.md # 中英双语模型对比报告
|   `-- reproduction_guide.md         # 复现指南
|-- scripts/
|   |-- run_phase6_balanced_queue.py  # 自动串联第六阶段均衡训练/评估
|   `-- watch_early_stop.py           # 外部早停监控工具
|-- src/
|   |-- data_preprocessing.py         # 数据集封装与截断策略
|   |-- prepare_multilingual_dataset.py
|   |-- train.py
|   |-- evaluate.py
|   |-- error_analysis.py
|   |-- inference.py
|   |-- compare_phase4.py
|   `-- compare_phase5.py
|-- MODEL_CARD.md
|-- README.md
|-- requirements.txt
`-- .gitignore
```

运行训练和评估后会在本地生成以下目录：

```text
data/raw/
data/processed/
models/
outputs/
.venv/
```

## 环境安装

建议使用虚拟环境：

```powershell
cd D:\code\p3_SentientAnalyze
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 PowerShell 禁止激活虚拟环境，也可以不执行 `activate`，直接使用：

```powershell
.\.venv\Scripts\python.exe src\train.py --config configs\train.yaml
```

## 数据准备

需要复现第六阶段数据集时运行：

```powershell
python src\prepare_multilingual_dataset.py
```

如果本地已经有 `data/raw/` 下的原始压缩包：

```powershell
python src\prepare_multilingual_dataset.py --skip-download
```

生成的数据目录：

```text
data/processed/phase6_multilingual_sentiment/
|-- balanced/
|-- emoji_sentiment_lexicon.parquet
|-- dataset_card.md
`-- summary.json
```

当前 balanced 数据规模：

| 训练集 | 验证集 | 测试集 | 总计 |
|---:|---:|---:|---:|
| 66682 | 7809 | 10198 | 84689 |

## 训练

IMDb 基线训练：

```powershell
python src\train.py --config configs\train.yaml
```

英文模型对比训练：

```powershell
python src\train.py --config configs\phase5_roberta_head_tail256.yaml
python src\evaluate.py --config configs\phase5_roberta_head_tail256.yaml --model-path models\phase5_roberta_head_tail256 --output-dir outputs\phase5\model_comparison\roberta_head_tail256\eval
```

中英双语均衡数据主模型：

```powershell
python src\train.py --config configs\phase6_xlm_roberta_balanced.yaml
python src\evaluate.py --config configs\phase6_xlm_roberta_balanced.yaml --model-path models\phase6_xlm_roberta_balanced --output-dir outputs\phase6\xlm_roberta_balanced\eval
```

自动串联第六阶段均衡数据上的两个多语言模型：

```powershell
python scripts\run_phase6_balanced_queue.py --wait-pid <xlm_roberta_training_pid>
```

## 推理

训练完成后可以直接推理：

```powershell
python src\inference.py --text "这个电影节奏很好，演员表演也很有感染力。"
python src\inference.py --text "I hated this film. It was boring and predictable."
```

默认模型路径为：

```text
models/phase6_xlm_roberta_balanced
```

如果模型未在本地训练或下载，需要先完成训练，或通过 `--model-path` 指向已有模型目录。

## 评估与错误分析

测试集评估：

```powershell
python src\evaluate.py --config configs\phase6_xlm_roberta_balanced.yaml --model-path models\phase6_xlm_roberta_balanced --output-dir outputs\phase6\xlm_roberta_balanced\eval --split test
```

错误诊断：

```powershell
python src\error_analysis.py --input outputs\phase6\xlm_roberta_balanced\eval\predictions.csv --output-dir outputs\phase6\xlm_roberta_balanced\diagnosis
```

主要产物：

```text
evaluation_metrics.json
confusion_matrix.png
evaluation_report.md
diagnosis_summary.json
error_by_length.csv
key_error_cases.csv
```

## 实验结论

| 阶段 | 目标 | 关键结论 |
|---|---|---|
| 第一至三阶段 | IMDb 基线、评估、错误诊断 | 建立完整训练/评估/诊断闭环 |
| 第四阶段 | 长文本截断策略 | `head_tail_256` 在整体 F1 和长文本错误率上更均衡 |
| 第五阶段 | 英文预训练模型对比 | `roberta-base + head_tail_256` 在 IMDb 上取得测试 F1 0.9513 |
| 第六阶段 | 中英双语多源模型对比 | `xlm-roberta-base` 在均衡多源测试集上取得测试 F1 0.9366，优于 mBERT |

详细报告见：

- [项目架构说明](docs/project_architecture.md)
- [实验总览](docs/experiment_summary.md)
- [第六阶段中英双语模型报告](docs/phase6_multilingual_report.md)
- [模型卡](MODEL_CARD.md)
- [复现指南](docs/reproduction_guide.md)
