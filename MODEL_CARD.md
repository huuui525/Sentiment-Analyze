# Model Card

## 模型名称

```text
phase6_xlm_roberta_balanced
```

## 基础模型

```text
xlm-roberta-base
```

## 任务

中英双语多源文本情感二分类：

```text
negative / positive
```

## 训练数据

使用项目构建的 `phase6_multilingual_sentiment_balanced` 数据集。

数据来源：

- IMDb
- TweetEval sentiment
- weibo_senti_100k
- waimai_10k

Emoji Sentiment Ranking 当前作为 emoji 情感词典保留，尚未作为监督训练样本直接参与分类训练。

balanced 数据规模：

| Split | Samples |
|---|---:|
| train | 66682 |
| validation | 7809 |
| test | 10198 |

## 输入处理

| 项目 | 设置 |
|---|---|
| tokenizer | `xlm-roberta-base` tokenizer |
| max_length | 256 |
| truncation_strategy | `head_tail` |
| padding | `max_length` |

`head_tail` 策略保留文本开头和结尾，适合长评论中前文交代背景、末尾给出情感结论的情况。

## 训练配置

| 项目 | 设置 |
|---|---|
| batch_size | 8 |
| epochs | 3 |
| learning_rate | 2e-5 |
| weight_decay | 0.01 |
| warmup_ratio | 0.1 |
| mixed_precision | true |
| best_metric | validation F1 |
| best_epoch | 2 |

## 测试集表现

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.9399 |
| Precision | 0.9495 |
| Recall | 0.9241 |
| F1 | 0.9366 |
| Errors | 613 / 10198 |

混淆矩阵：

| True / Pred | negative | positive |
|---|---:|---:|
| negative | 5057 | 241 |
| positive | 372 | 4528 |

## 适用场景

适合：

- 英文影评、英文社交媒体文本、中文微博文本、中文短评论的正负向情感判断。
- 作为中英双语情感分析项目的研究型 baseline。
- 作为后续 emoji 情感增强、错误诊断、轻量化部署的主模型。

不适合：

- 细粒度情绪分类，例如愤怒、悲伤、惊喜等多分类。
- 强领域专业文本，例如金融研报、医疗问答、法律文本。
- 讽刺、反语、复杂上下文依赖很强的长对话情感判断。

## 已知限制

- 当前标签仅为二分类，无法表达中性或混合情感。
- 数据源虽然覆盖中英双语，但仍以公开数据集为主，不代表所有真实业务场景。
- emoji 字段已保留，但 emoji 情感词典尚未直接融入模型结构或损失函数。
- 当前结论基于 source-label balanced 数据集。

## 后续改进

- 按语言、数据源、emoji 是否出现进行分组评估。
- 尝试 emoji 文本归一化或情感 token 注入。
- 加入中性类别，扩展为三分类情感分析。
- 增加按语言、数据源、文本长度、emoji 是否出现的分组评估。
