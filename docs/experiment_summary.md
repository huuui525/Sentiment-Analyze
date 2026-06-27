# 实验总览

## 1. Phase 1-3：IMDb Baseline、评估与错误诊断

目标是把原始脚本升级为完整训练闭环：

- 使用 `bert-base-uncased` 微调 IMDb 情感二分类。
- 增加 train / validation / test 划分。
- 保存 best checkpoint 和 `metrics.json`。
- 增加独立评估脚本。
- 增加错误诊断脚本。

baseline 测试结果：

```text
test accuracy: 0.9215
test f1:       0.9223
```

## 2. Phase 4：长文本截断策略实验

固定模型为 BERT，比较不同输入策略。

| 实验 | Accuracy | F1 | 错误数 | 501+ 长文本错误率 |
|---|---:|---:|---:|---:|
| head-256 baseline | 0.9215 | 0.9223 | 1963 | 0.1576 |
| head-512 | 0.9369 | 0.9374 | 1577 | 0.1056 |
| head+tail-256 | 0.9385 | 0.9387 | 1537 | 0.0944 |
| tail-256 | 0.9365 | 0.9372 | 1587 | 0.0923 |

结论：

- `head-512` 通过更长上下文提升明显，但计算成本更高。
- `tail-256` 对长文本有帮助，但整体略低于 `head_tail_256`。
- `head_tail_256` 在保持 256 tokens 成本的同时，获得最高整体 F1，因此作为后续默认输入策略。

## 3. Phase 5：英文预训练模型对比

固定输入策略为 `head_tail_256`，比较不同 encoder。

| 模型 | Accuracy | F1 | 错误数 | 模型目录大小 |
|---|---:|---:|---:|---:|
| bert-base-uncased | 0.9385 | 0.9387 | 1537 | 417.92 MB |
| distilbert-base-uncased | 0.9300 | 0.9302 | 1751 | 256.33 MB |
| roberta-base | 0.9512 | 0.9513 | 1221 | 478.72 MB |

结论：

- `roberta-base` 在 IMDb 英文影评上表现最好。
- `distilbert-base-uncased` 体积更小，但当前任务下性能损失明显。
- 对英文影评场景，强 encoder 能带来稳定收益。

## 4. Phase 6：中英双语多源模型对比

构建中英双语多源 balanced 数据集，并使用该数据集进行模型对比。

数据规模：

| Train | Validation | Test | Total |
|---:|---:|---:|---:|
| 66682 | 7809 | 10198 | 84689 |

模型对比：

| 模型 | Best Epoch | Val F1 | Test Accuracy | Test F1 | 错误数 |
|---|---:|---:|---:|---:|---:|
| xlm-roberta-base | 2 | 0.9425 | 0.9399 | 0.9366 | 613 |
| bert-base-multilingual-cased | 3 | 0.9329 | 0.9217 | 0.9178 | 799 |

结论：

- `xlm-roberta-base` 在中英双语多源 balanced 测试集上明显优于 mBERT。
- XLM-R 第 3 轮验证 F1 略低于第 2 轮，best checkpoint 来自 epoch 2。

## 5. 当前主线结论

当前项目可定义为：

```text
一个支持中英双语、多源文本、emoji 字段保留、可复现实验和错误诊断的情感分析工程项目。
```

当前主模型：

```text
xlm-roberta-base + 数据源-标签均衡多语言情感数据集 + head_tail_256
```
