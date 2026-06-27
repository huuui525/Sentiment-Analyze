# 项目架构说明

## 1. 项目定位

本项目是一个面向情感二分类任务的 NLP 工程项目，演进路线从 IMDb 英文影评 baseline 开始，逐步扩展到长文本策略、多模型横向对比、中英双语多源数据集和错误诊断。

核心任务：

```text
输入：一段英文或中文文本
输出：negative / positive 以及模型置信度
```

## 2. 总体流程

```text
数据下载/归档
  -> 数据清洗与统一 schema
  -> balanced 数据集构建
  -> tokenizer 编码与截断策略
  -> Transformer 分类模型微调
  -> 验证集选取 best checkpoint
  -> 测试集评估
  -> 错误诊断与实验报告
  -> 推理脚本调用
```

## 3. 核心模块

| 文件 | 作用 |
|---|---|
| `src/prepare_multilingual_dataset.py` | 下载并归档原始数据，构建 Phase 6 多源中英双语数据集 |
| `src/data_preprocessing.py` | 定义 `IMDBDataset`，完成文本编码、padding、`head/tail/head_tail` 截断 |
| `src/train.py` | 通用训练入口，支持 Hugging Face 数据集和本地 Parquet 数据集 |
| `src/evaluate.py` | 加载训练好的模型，在指定 split 上输出指标、混淆矩阵和预测明细 |
| `src/error_analysis.py` | 基于预测结果统计错误率、长文本错误、高置信错误样本 |
| `src/inference.py` | 单条文本或示例文本推理入口 |
| `src/compare_phase4.py` | 汇总长文本截断实验 |
| `src/compare_phase5.py` | 汇总英文模型横向对比实验 |
| `scripts/run_phase6_balanced_queue.py` | 自动串联 XLM-R 评估和 mBERT balanced 训练/评估 |
| `scripts/watch_early_stop.py` | 外部早停监控工具 |

## 4. 数据设计

Phase 6 使用统一 schema：

| 字段 | 含义 |
|---|---|
| `text` | 清洗后的文本 |
| `label` | 二分类标签，0 为 negative，1 为 positive |
| `label_name` | 标签名称 |
| `language` | `en` 或 `zh` |
| `domain` | 文本场景，如 movie_review、social_media、food_delivery |
| `source` | 数据来源 |
| `original_split` | 原始数据集 split |
| `split` | 统一后的 train / validation / test |
| `has_emoji` | 是否包含 Unicode emoji 或中文社交媒体表情标记 |
| `emoji_count` | emoji / 表情标记数量 |
| `char_count` | 字符数 |
| `word_count` | 空格分词后的词数 |

balanced 数据集生成方式：

- 先将多个数据源统一为相同字段结构。
- 对没有官方 split 的数据源，按 `source + label` 分层划分 train / validation / test。
- 再按 `split + source + label` 分组抽样，训练集每组最多 10000 条，验证/测试每组最多 1500 条。

balanced 设计目的不是最大化样本量，而是减少样本量大的来源对训练目标的支配，使模型对跨语言、跨场景表现更稳定。

## 5. 截断策略

`src/data_preprocessing.py` 支持三种截断：

| 策略 | 说明 | 适用场景 |
|---|---|---|
| `head` | 保留文本开头 | 常规 baseline |
| `tail` | 保留文本结尾 | 情感结论常出现在末尾的长评论 |
| `head_tail` | 保留开头和结尾 | 长影评、长评论中兼顾上下文和最终态度 |

Phase 4 实验证明，`head_tail_256` 在 IMDb 任务中整体 F1 和长文本错误率更均衡，因此后续 Phase 5、Phase 6 固定使用该策略。

## 6. 训练机制

`src/train.py` 的主要能力：

- YAML 配置驱动模型、数据、训练参数和输出目录。
- 支持 Hugging Face 在线数据集和本地 Parquet 数据集。
- 固定随机种子，提高实验可复现性。
- 支持 CUDA 和 mixed precision。
- 使用 AdamW、线性 warmup、梯度裁剪。
- 根据验证集指标保存 best model。
- 支持早停配置：

```yaml
early_stopping:
  enabled: true
  min_delta: 0.002
  patience: 1
  min_epochs: 2
```

## 7. 评估机制

`src/evaluate.py` 输出：

- Accuracy
- Precision
- Recall
- F1
- 混淆矩阵
- 每条样本的预测标签、概率、置信度
- Markdown 评估报告

`src/error_analysis.py` 进一步输出：

- 总错误数和错误率
- 高置信错误样本
- 按文本长度统计的错误率
- false positive / false negative 数量

## 8. 本地生成产物

运行数据处理、训练和评估后会生成以下本地目录：

```text
data/raw/
data/processed/
models/
outputs/
```
