# Data

本仓库不提交完整原始数据和处理后数据，只提交少量 `data/sample/` 样例用于查看字段结构。

## 目录约定

```text
data/
|-- raw/        # 原始压缩包和数据源快照
|-- processed/  # 处理后的 Parquet 数据集
|-- interim/    # 临时目录
`-- sample/     # 小样本，可提交，用于 schema 展示
```

## 重建多语言数据集

完整下载并处理：

```powershell
python src\prepare_multilingual_dataset.py
```

如果 `data/raw/` 下已经有原始压缩包：

```powershell
python src\prepare_multilingual_dataset.py --skip-download
```

生成文件：

```text
data/processed/phase6_multilingual_sentiment/
|-- balanced/
|   |-- train.parquet
|   |-- validation.parquet
|   `-- test.parquet
|-- emoji_sentiment_lexicon.parquet
|-- dataset_card.md
`-- summary.json
```

## 数据来源

| 数据源 | 语言 | 场景 | 用途 |
|---|---|---|---|
| Stanford IMDb Large Movie Review Dataset | 英文 | 电影评论 | 英文长文本情感分析 |
| TweetEval sentiment | 英文 | Twitter 短文本 | 英文社交媒体情感分析 |
| weibo_senti_100k | 中文 | 微博文本 | 中文社交媒体情感分析 |
| waimai_10k | 中文 | 外卖评论 | 中文短评论情感分析 |
| Emoji Sentiment Ranking | emoji | 情感词典 | 后续 emoji 情感增强 |

## 字段说明

| 字段 | 含义 |
|---|---|
| `text` | 清洗后的文本 |
| `label` | 0 表示 negative，1 表示 positive |
| `label_name` | 标签名称 |
| `language` | `en` 或 `zh` |
| `domain` | 文本场景 |
| `source` | 数据来源 |
| `split` | train / validation / test |
| `has_emoji` | 是否包含 emoji 或中文社交媒体表情标记 |
| `emoji_count` | emoji / 表情标记数量 |
| `char_count` | 字符数 |
| `word_count` | 词数 |

## balanced 数据生成方式

处理流程：

1. 读取 IMDb、TweetEval、weibo_senti_100k、waimai_10k 四个监督数据源。
2. 统一字段为 `text`、`label`、`language`、`domain`、`source`、`split` 等。
3. IMDb 和 TweetEval 使用官方 split；weibo_senti_100k 和 waimai_10k 按 `source + label` 分层划分为 train / validation / test。
4. 按 `split + source + label` 分组抽样，避免某个数据源或标签占比过高。
5. 训练集每个分组最多保留 `10000` 条，验证集和测试集每个分组最多保留 `1500` 条。

## 当前 balanced 规模

| Train | Validation | Test | Total |
|---:|---:|---:|---:|
| 66682 | 7809 | 10198 | 84689 |

## 本地生成目录

数据处理过程中会生成：

```text
data/raw/
data/processed/
data/interim/
```

其中 `raw/` 可用于离线重建数据集，`processed/` 是训练脚本实际读取的数据目录。
