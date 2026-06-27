import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "phase2_evaluation" / "predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3_error_diagnosis"


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_predictions(input_path):
    df = pd.read_csv(input_path)
    numeric_columns = ["is_correct", "confidence", "prob_negative", "prob_positive", "word_count"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["is_error"] = df["is_correct"] == 0
    df["text"] = df["text"].fillna("")
    return df


def make_length_summary(df):
    bins = [0, 50, 100, 200, 300, 500, float("inf")]
    labels = ["0-50", "51-100", "101-200", "201-300", "301-500", "501+"]
    temp = df.copy()
    temp["length_bucket"] = pd.cut(temp["word_count"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        temp.groupby("length_bucket", observed=False)
        .agg(
            samples=("is_correct", "size"),
            errors=("is_error", "sum"),
            accuracy=("is_correct", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    grouped["error_rate"] = grouped["errors"] / grouped["samples"]
    return grouped


def export_key_error_cases(df, output_path, limit, high_confidence_threshold):
    errors = df[df["is_error"]].copy()
    errors = errors.sort_values("confidence", ascending=False).head(limit)
    errors["is_high_confidence"] = errors["confidence"] >= high_confidence_threshold
    columns = [
        "index",
        "true_label",
        "pred_label",
        "confidence",
        "is_high_confidence",
        "word_count",
        "error_type",
        "text",
    ]
    errors[columns].to_csv(output_path, index=False, encoding="utf-8")


def get_long_text_error_rate(length_summary):
    row = length_summary[length_summary["length_bucket"].astype(str) == "501+"]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["error_rate"])


def save_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_report(summary, length_summary, output_path):
    lines = [
        "# 错误诊断报告",
        "",
        "## 关键结论",
        "",
        f"- 样本数：`{summary['samples']}`",
        f"- 错误数：`{summary['errors']}`",
        f"- 准确率：`{summary['accuracy']:.4f}`",
        f"- 错误率：`{summary['error_rate']:.4f}`",
        f"- 高置信错误数：`{summary['high_confidence_errors']}`",
        f"- 假阳性错误：`{summary['false_positive_errors']}`",
        f"- 假阴性错误：`{summary['false_negative_errors']}`",
        f"- 501+ 长文本错误率：`{summary['long_text_501_plus_error_rate']:.4f}`",
        "",
        "## 文本长度影响",
        "",
        "| 文本长度 | 样本数 | 错误数 | 准确率 | 错误率 | 平均置信度 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in length_summary.iterrows():
        lines.append(
            f"| {row['length_bucket']} | {int(row['samples'])} | {int(row['errors'])} | "
            f"{row['accuracy']:.4f} | {row['error_rate']:.4f} | {row['avg_confidence']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 保留产物",
            "",
            "- `diagnosis_summary.json`：错误诊断总览",
            "- `error_by_length.csv`：按文本长度统计错误率",
            "- `key_error_cases.csv`：置信度最高的一批错误样本，用于人工复盘",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_error_analysis(args):
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(input_path)
    length_summary = make_length_summary(df)

    correct = int(df["is_correct"].sum())
    errors = int(df["is_error"].sum())
    high_confidence_errors = int(((df["is_error"]) & (df["confidence"] >= args.high_confidence_threshold)).sum())
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "samples": int(len(df)),
        "correct": correct,
        "errors": errors,
        "accuracy": float(correct / len(df)) if len(df) else 0.0,
        "error_rate": float(errors / len(df)) if len(df) else 0.0,
        "high_confidence_threshold": args.high_confidence_threshold,
        "high_confidence_errors": high_confidence_errors,
        "false_positive_errors": int((df["error_type"] == "false_positive").sum()),
        "false_negative_errors": int((df["error_type"] == "false_negative").sum()),
        "long_text_501_plus_error_rate": get_long_text_error_rate(length_summary),
    }

    save_json(summary, output_dir / "diagnosis_summary.json")
    length_summary.to_csv(output_dir / "error_by_length.csv", index=False, encoding="utf-8")
    export_key_error_cases(df, output_dir / "key_error_cases.csv", args.case_limit, args.high_confidence_threshold)
    write_report(summary, length_summary, output_dir / "error_diagnosis_report.md")

    print(f"Samples: {summary['samples']}")
    print(f"Errors: {summary['errors']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"High-confidence errors: {summary['high_confidence_errors']}")
    print(f"Long-text 501+ error rate: {summary['long_text_501_plus_error_rate']:.4f}")
    print(f"Saved diagnosis artifacts to {output_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze core sentiment model errors from predictions.csv.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to evaluation predictions.csv.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--case-limit", type=int, default=100, help="Number of key error cases to export.")
    return parser.parse_args()


if __name__ == "__main__":
    run_error_analysis(parse_args())
