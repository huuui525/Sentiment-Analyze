import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_preprocessing import IMDBDataset
from train import DEFAULT_CONFIG_PATH, load_config, prepare_datasets, resolve_project_path, select_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2_evaluation"
LABEL_NAMES = ["negative", "positive"]


def get_split_dataset(config, split_name, max_samples=None):
    train_split, validation_split, test_split = prepare_datasets(config)
    split_map = {
        "train": train_split,
        "validation": validation_split,
        "test": test_split,
    }
    dataset = split_map[split_name]
    if max_samples is not None:
        dataset = select_samples(dataset, max_samples, config["training"]["seed"])
    return dataset


def build_eval_loader(dataset, tokenizer, config, batch_size=None):
    return DataLoader(
        IMDBDataset(
            dataset,
            tokenizer,
            max_length=config["data"]["max_length"],
            truncation_strategy=config["data"].get("truncation_strategy", "head"),
        ),
        batch_size=batch_size or config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )


def run_predictions(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    true_labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += outputs.loss.item()
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    return {
        "loss": float(total_loss / len(data_loader)),
        "true_labels": true_labels,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def summarize_results(true_labels, predictions, average_loss):
    report = metrics.classification_report(
        true_labels,
        predictions,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": float(average_loss),
        "accuracy": float(metrics.accuracy_score(true_labels, predictions)),
        "precision": float(metrics.precision_score(true_labels, predictions, zero_division=0)),
        "recall": float(metrics.recall_score(true_labels, predictions, zero_division=0)),
        "f1": float(metrics.f1_score(true_labels, predictions, zero_division=0)),
        "class_metrics": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABEL_NAMES
        },
        "confusion_matrix": metrics.confusion_matrix(true_labels, predictions).tolist(),
    }


def save_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_confusion_matrix(true_labels, predictions, output_path):
    matrix = metrics.confusion_matrix(true_labels, predictions)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_NAMES)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def normalize_text(text):
    return " ".join(str(text).split())


def build_prediction_rows(dataset, true_labels, predictions, probabilities):
    rows = []
    for idx, (true_label, pred_label, probs) in enumerate(zip(true_labels, predictions, probabilities)):
        prob_negative = float(probs[0])
        prob_positive = float(probs[1])
        confidence = max(prob_negative, prob_positive)
        is_correct = int(true_label == pred_label)
        error_type = ""
        if not is_correct:
            error_type = "false_positive" if pred_label == 1 else "false_negative"

        sample = dataset[idx]
        text = normalize_text(sample["text"])
        rows.append(
            {
                "index": idx,
                "true_label": LABEL_NAMES[true_label],
                "pred_label": LABEL_NAMES[pred_label],
                "is_correct": is_correct,
                "confidence": confidence,
                "prob_negative": prob_negative,
                "prob_positive": prob_positive,
                "word_count": len(text.split()),
                "error_type": error_type,
                "text": text,
            }
        )
    return rows


def save_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(summary, output_path):
    matrix = summary["confusion_matrix"]
    lines = [
        "# 模型评估报告",
        "",
        "## 核心指标",
        "",
        f"- 数据集切分：`{summary['split']}`",
        f"- 样本数：`{summary['num_samples']}`",
        f"- Accuracy：`{summary['accuracy']:.4f}`",
        f"- F1：`{summary['f1']:.4f}`",
        f"- 错误数：`{summary['num_errors']}`",
        f"- 错误率：`{summary['error_rate']:.4f}`",
        "",
        "## 混淆矩阵",
        "",
        "| 真实 / 预测 | negative | positive |",
        "|---|---:|---:|",
        f"| negative | {matrix[0][0]} | {matrix[0][1]} |",
        f"| positive | {matrix[1][0]} | {matrix[1][1]} |",
        "",
        "## 类别表现",
        "",
        "| 类别 | Precision | Recall | F1 | 样本数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in summary["class_metrics"].items():
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']} |"
        )
    lines.extend(
        [
            "",
            "## 保留产物",
            "",
            f"- 指标文件：`{summary['artifacts']['metrics']}`",
            f"- 混淆矩阵图：`{summary['artifacts']['confusion_matrix']}`",
            f"- 诊断输入：`{summary['artifacts']['predictions']}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_model(args):
    config = load_config(args.config)
    model_path = resolve_project_path(args.model_path)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_split_dataset(config, args.split, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Using device: {device}")
    print(f"Model path: {model_path}")
    print(f"Evaluating split: {args.split}")
    print(f"Samples: {len(dataset)}")

    data_loader = build_eval_loader(dataset, tokenizer, config, batch_size=args.batch_size)
    prediction_output = run_predictions(model, data_loader, device)

    true_labels = prediction_output["true_labels"]
    predictions = prediction_output["predictions"]
    probabilities = prediction_output["probabilities"]

    summary = summarize_results(true_labels, predictions, prediction_output["loss"])
    rows = build_prediction_rows(dataset, true_labels, predictions, probabilities)
    num_errors = sum(1 for row in rows if row["is_correct"] == 0)

    artifacts = {
        "metrics": str(output_dir / "evaluation_metrics.json"),
        "confusion_matrix": str(output_dir / "confusion_matrix.png"),
        "predictions": str(output_dir / "predictions.csv"),
        "markdown_report": str(output_dir / "evaluation_report.md"),
    }
    summary.update(
        {
            "split": args.split,
            "num_samples": len(dataset),
            "num_errors": num_errors,
            "error_rate": float(num_errors / len(dataset)) if len(dataset) else 0.0,
            "model_path": str(model_path),
            "config_path": str(Path(args.config).resolve()),
            "artifacts": artifacts,
        }
    )

    save_json(summary, output_dir / "evaluation_metrics.json")
    save_confusion_matrix(true_labels, predictions, output_dir / "confusion_matrix.png")
    save_csv(rows, output_dir / "predictions.csv")
    write_markdown_report(summary, output_dir / "evaluation_report.md")

    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"F1: {summary['f1']:.4f}")
    print(f"Errors: {num_errors}/{len(dataset)}")
    print(f"Saved evaluation artifacts to {output_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to training config.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to trained model.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Evaluation output directory.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only N samples.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_model(parse_args())
