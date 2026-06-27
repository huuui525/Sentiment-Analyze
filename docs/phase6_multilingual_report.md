# Phase 6 中英双语多源情感分析实验报告

## 1. 实验目标

Phase 6 的目标是将项目从单一英文影评情感分类扩展为中英双语、多源文本情感分类任务，并比较两个多语言预训练模型：

- `xlm-roberta-base`
- `bert-base-multilingual-cased`

实验重点是验证模型在跨语言、跨场景 balanced 数据上的稳定性。

## 2. 数据来源

| 数据源 | 语言 | 场景 | 用途 |
|---|---|---|---|
| IMDb | 英文 | 电影长评论 | 英文长文本情感基础数据 |
| TweetEval sentiment | 英文 | 社交媒体短文本 | 英文短文本与社交媒体表达 |
| weibo_senti_100k | 中文 | 微博文本 | 中文社交媒体情感 |
| waimai_10k | 中文 | 外卖评论 | 中文生活服务短评论 |
| Emoji Sentiment Ranking | emoji | 情感词典 | 作为后续 emoji 情感增强资源 |

## 3. balanced 数据集

| Train | Validation | Test | Total |
|---:|---:|---:|---:|
| 66682 | 7809 | 10198 | 84689 |

balanced 版本的构建方式：

```text
按 split + source + label 分组抽样
训练集每组最多 10000 条
验证/测试集每组最多 1500 条
```

这样可以减少样本量大的数据源主导训练目标的问题。

## 4. 实验配置

| 项目 | 设置 |
|---|---|
| 任务 | 二分类情感分析 |
| 标签 | negative / positive |
| 输入长度 | 256 tokens |
| 截断策略 | `head_tail` |
| batch size | 8 |
| epoch | 3 |
| optimizer | AdamW |
| learning rate | 2e-5 |
| mixed precision | enabled |
| best metric | validation F1 |
| early stopping | min_delta=0.002, patience=1, min_epochs=2 |

## 5. 结果

| 模型 | Best Epoch | Val F1 | Test Accuracy | Test F1 | Precision | Recall | 错误数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xlm-roberta-base | 2 | 0.9425 | 0.9399 | 0.9366 | 0.9495 | 0.9241 | 613 |
| bert-base-multilingual-cased | 3 | 0.9329 | 0.9217 | 0.9178 | 0.9259 | 0.9098 | 799 |

XLM-R 相比 mBERT：

```text
Test F1 提升：0.9366 - 0.9178 = 0.0188
错误数减少：799 - 613 = 186
```

## 6. 混淆矩阵

XLM-R：

| True / Pred | negative | positive |
|---|---:|---:|
| negative | 5057 | 241 |
| positive | 372 | 4528 |

mBERT：

| True / Pred | negative | positive |
|---|---:|---:|
| negative | 4941 | 357 |
| positive | 442 | 4458 |

XLM-R 同时减少了 false positive 和 false negative，说明提升不是单纯偏向某一类，而是整体判别能力更强。

## 7. 结论

1. `xlm-roberta-base` 是当前多语言阶段的主模型。
2. balanced 数据集通过限制 `split + source + label` 分组样本上限，降低了数据源分布偏移对训练的影响。
3. 后续可以围绕 emoji 情感增强、分组评估、轻量化推理和错误驱动数据增强继续扩展。
