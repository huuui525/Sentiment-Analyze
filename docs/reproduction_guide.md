# 复现指南

## 1. 环境

建议使用 Python 3.10+。当前本地环境使用 Python 3.12 和 CUDA 版 PyTorch。

```powershell
cd D:\code\p3_SentientAnalyze
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 PowerShell 禁止脚本激活，可以不激活虚拟环境，直接使用 `.venv` 内的 Python。

## 2. 准备数据

完整重建：

```powershell
python src\prepare_multilingual_dataset.py
```

使用已有原始压缩包离线重建：

```powershell
python src\prepare_multilingual_dataset.py --skip-download
```

检查生成文件：

```text
data/processed/phase6_multilingual_sentiment/balanced/train.parquet
data/processed/phase6_multilingual_sentiment/balanced/validation.parquet
data/processed/phase6_multilingual_sentiment/balanced/test.parquet
```

## 3. 训练 Phase 6 主模型

```powershell
python src\train.py --config configs\phase6_xlm_roberta_balanced.yaml
```

训练输出：

```text
models/phase6_xlm_roberta_balanced/
outputs/phase6/xlm_roberta_balanced/train/metrics.json
```

## 4. 评估

```powershell
python src\evaluate.py --config configs\phase6_xlm_roberta_balanced.yaml --model-path models\phase6_xlm_roberta_balanced --output-dir outputs\phase6\xlm_roberta_balanced\eval --split test
```

评估输出：

```text
outputs/phase6/xlm_roberta_balanced/eval/evaluation_metrics.json
outputs/phase6/xlm_roberta_balanced/eval/confusion_matrix.png
outputs/phase6/xlm_roberta_balanced/eval/evaluation_report.md
outputs/phase6/xlm_roberta_balanced/eval/predictions.csv
```

## 5. 错误诊断

```powershell
python src\error_analysis.py --input outputs\phase6\xlm_roberta_balanced\eval\predictions.csv --output-dir outputs\phase6\xlm_roberta_balanced\diagnosis
```

诊断输出：

```text
diagnosis_summary.json
error_by_length.csv
key_error_cases.csv
error_diagnosis_report.md
```

## 6. 推理

```powershell
python src\inference.py --text "这个电影节奏很好，演员表演也很有感染力。"
python src\inference.py --text "I hated this film. It was boring and predictable."
```

## 7. 本地生成产物

复现过程中会生成以下目录：

```text
.venv/
data/raw/
data/processed/
models/
outputs/
```

如果只需要重新运行实验，保留 `data/raw/` 可以避免重复下载原始数据；保留 `models/` 可以直接复用已经训练好的模型。
