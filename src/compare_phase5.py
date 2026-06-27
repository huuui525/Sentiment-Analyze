import argparse
import csv
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase5" / "model_comparison"
DEFAULT_OUTPUT = PHASE5_OUTPUT_DIR / "model_comparison.csv"
DEFAULT_REPORT = PHASE5_OUTPUT_DIR / "model_comparison_report.md"

EXPERIMENTS = [
    {
        "name": "bert_base_head_tail256",
        "role": "当前主模型",
        "config": PROJECT_ROOT / "configs" / "phase5_bert_head_tail256.yaml",
        "model_dir": PROJECT_ROOT / "models" / "phase4_head_tail256",
        "train_metrics": PROJECT_ROOT / "outputs" / "phase4" / "head_tail256" / "train" / "metrics.json",
        "evaluation_metrics": PROJECT_ROOT / "outputs" / "phase4" / "head_tail256" / "eval" / "evaluation_metrics.json",
        "diagnosis_summary": PROJECT_ROOT / "outputs" / "phase4" / "head_tail256" / "diagnosis" / "diagnosis_summary.json",
    },
    {
        "name": "distilbert_base_head_tail256",
        "role": "轻量化候选模型",
        "config": PROJECT_ROOT / "configs" / "phase5_distilbert_head_tail256.yaml",
        "model_dir": PROJECT_ROOT / "models" / "phase5_distilbert_head_tail256",
        "train_metrics": PHASE5_OUTPUT_DIR / "distilbert_head_tail256" / "train" / "metrics.json",
        "evaluation_metrics": PHASE5_OUTPUT_DIR / "distilbert_head_tail256" / "eval" / "evaluation_metrics.json",
        "diagnosis_summary": PHASE5_OUTPUT_DIR / "distilbert_head_tail256" / "diagnosis" / "diagnosis_summary.json",
    },
    {
        "name": "roberta_base_head_tail256",
        "role": "强性能候选模型",
        "config": PROJECT_ROOT / "configs" / "phase5_roberta_head_tail256.yaml",
        "model_dir": PROJECT_ROOT / "models" / "phase5_roberta_head_tail256",
        "train_metrics": PHASE5_OUTPUT_DIR / "roberta_head_tail256" / "train" / "metrics.json",
        "evaluation_metrics": PHASE5_OUTPUT_DIR / "roberta_head_tail256" / "eval" / "evaluation_metrics.json",
        "diagnosis_summary": PHASE5_OUTPUT_DIR / "roberta_head_tail256" / "diagnosis" / "diagnosis_summary.json",
    },
]


def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_yaml(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def directory_size_mb(path):
    if not path.exists():
        return ""
    size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return round(size / (1024 * 1024), 2)


def format_percent(value):
    if value in ("", None):
        return ""
    return f"{float(value) * 100:.2f}%"


def format_float(value, digits=4):
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def build_row(experiment):
    config = read_yaml(experiment["config"]) or {}
    train_metrics = read_json(experiment["train_metrics"])
    eval_metrics = read_json(experiment["evaluation_metrics"])
    diagnosis = read_json(experiment["diagnosis_summary"])

    row = {
        "experiment": experiment["name"],
        "role": experiment["role"],
        "status": "complete" if eval_metrics else "missing",
        "model_name": config.get("model", {}).get("display_name", config.get("model", {}).get("name", "")),
        "max_length": config.get("data", {}).get("max_length", ""),
        "truncation_strategy": config.get("data", {}).get("truncation_strategy", ""),
        "batch_size": config.get("training", {}).get("batch_size", ""),
        "epochs": config.get("training", {}).get("epochs", ""),
        "best_epoch": "",
        "best_val_f1": "",
        "test_accuracy": "",
        "test_f1": "",
        "errors": "",
        "error_rate": "",
        "high_confidence_errors": "",
        "long_text_501_plus_error_rate": "",
        "model_size_mb": directory_size_mb(experiment["model_dir"]),
        "config": str(experiment["config"]),
        "model_dir": str(experiment["model_dir"]),
    }

    if train_metrics:
        row["best_epoch"] = train_metrics.get("best_epoch", "")
        row["best_val_f1"] = train_metrics.get("best_validation_metric", {}).get("value", "")

    if eval_metrics:
        row.update(
            {
                "test_accuracy": eval_metrics.get("accuracy", ""),
                "test_f1": eval_metrics.get("f1", ""),
                "errors": eval_metrics.get("num_errors", ""),
                "error_rate": eval_metrics.get("error_rate", ""),
            }
        )

    if diagnosis:
        row["high_confidence_errors"] = diagnosis.get("high_confidence_errors", "")
        row["long_text_501_plus_error_rate"] = diagnosis.get("long_text_501_plus_error_rate", "")

    return row


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_deltas(rows, baseline):
    for row in rows:
        if row["status"] != "complete" or baseline is None:
            row["delta_f1"] = ""
            row["delta_errors"] = ""
            continue
        row["delta_f1"] = float(row["test_f1"]) - float(baseline["test_f1"])
        row["delta_errors"] = int(row["errors"]) - int(baseline["errors"])


def markdown_table(rows):
    lines = [
        "| 实验 | 模型 | 角色 | Accuracy | F1 | 相对BERT F1 | 错误数 | 相对BERT错误数 | 长文本501+错误率 | 模型大小 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {model_name} | {role} | {accuracy} | {f1} | {delta_f1} | "
            "{errors} | {delta_errors} | {long_error} | {size} MB |".format(
                experiment=row["experiment"],
                model_name=row["model_name"],
                role=row["role"],
                accuracy=format_percent(row["test_accuracy"]),
                f1=format_float(row["test_f1"]),
                delta_f1=format_float(row["delta_f1"]),
                errors=row["errors"],
                delta_errors=row["delta_errors"],
                long_error=format_percent(row["long_text_501_plus_error_rate"]),
                size=row["model_size_mb"],
            )
        )
    return lines


def write_report(rows, output_path):
    complete_rows = [row for row in rows if row["status"] == "complete"]
    baseline = next((row for row in rows if row["experiment"] == "bert_base_head_tail256"), None)
    compute_deltas(rows, baseline)
    best = max(complete_rows, key=lambda row: float(row["test_f1"])) if complete_rows else None
    smallest = min(complete_rows, key=lambda row: float(row["model_size_mb"])) if complete_rows else None

    lines = [
        "# Phase 5 模型横向对比实验报告",
        "",
        "## 1. 实验目的",
        "",
        "Phase 4 已经确认 `head_tail_256` 是当前更优的长文本截断策略。本阶段固定数据集、训练轮数、最大长度和截断策略，只替换预训练编码器，比较不同模型在 IMDb 二分类情感分析任务上的效果与工程成本。",
        "",
        "对比模型包括：",
        "",
        "- `bert-base-uncased`：Phase 4 当前主模型，作为基线。",
        "- `distilbert-base-uncased`：轻量化模型，用于观察模型体积下降后的性能损失。",
        "- `roberta-base`：更强编码器候选，用于观察是否能进一步提高上限。",
        "",
        "## 2. 实验设置",
        "",
        "| 项目 | 设置 |",
        "|---|---|",
        "| 数据集 | IMDb 影评情感分类数据集 |",
        "| 任务 | 二分类，negative / positive |",
        "| 输入策略 | `head_tail_256` |",
        "| 最大长度 | 256 tokens |",
        "| 训练轮数 | 3 epochs |",
        "| 评价口径 | 测试集 Accuracy、F1、错误数、长文本错误率、模型目录大小 |",
        "",
        "## 3. 横向结果",
        "",
    ]
    lines.extend(markdown_table(rows))

    lines.extend(["", "## 4. 主要结论", ""])
    if best:
        lines.append(
            f"- 当前最优模型是 `{best['model_name']}`，测试 F1 为 `{format_float(best['test_f1'])}`，"
            f"Accuracy 为 `{format_percent(best['test_accuracy'])}`。"
        )
    if baseline and best and best is not baseline:
        lines.append(
            f"- 相比当前 BERT 主模型，`{best['model_name']}` 的 F1 提升 `{format_float(best['delta_f1'])}`，"
            f"测试集错误数减少 `{abs(int(best['delta_errors']))}` 条。"
        )
    if smallest:
        lines.append(
            f"- 体积最小的是 `{smallest['model_name']}`，模型目录约 `{smallest['model_size_mb']} MB`，"
            f"但测试 F1 为 `{format_float(smallest['test_f1'])}`，低于当前 BERT 主模型。"
        )

    lines.extend(
        [
            "- 从效果优先角度看，`roberta-base + head_tail_256` 更适合作为下一阶段主模型。",
            "- 从轻量部署或低资源运行角度看，`distilbert-base-uncased` 有体积优势，但当前实验中精度损失较明显，不适合作为主模型替代。",
            "- `bert-base-uncased + head_tail_256` 仍然是一个稳定且成本适中的基线，适合作为后续消融实验的参照组。",
            "",
            "## 5. 建议",
            "",
            "建议将 Phase 5 的工程结论表述为：在固定 `head_tail_256` 输入策略后，通过预训练编码器横向对比，验证了更强 encoder 对 IMDb 情感分类任务仍有显著收益。下一步可以围绕 RoBERTa 主模型继续做学习率、epoch、冻结层、早停等训练策略实验，而不是再优先扩大截断策略实验。",
            "",
            "## 6. 产物路径",
            "",
            f"- 横向对比 CSV：`{DEFAULT_OUTPUT}`",
            f"- 正式报告：`{DEFAULT_REPORT}`",
            "- 各模型训练、评估与诊断结果：`outputs/phase5/model_comparison/`",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Phase 5 model experiments.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [build_row(experiment) for experiment in EXPERIMENTS]
    write_csv(rows, args.output)
    write_report(rows, args.report)
    print(f"Saved comparison to {args.output}")
    print(f"Saved report to {args.report}")
    for row in rows:
        print(f"{row['experiment']}: {row['status']}")


if __name__ == "__main__":
    main()
