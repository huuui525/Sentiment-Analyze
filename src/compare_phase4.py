import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE4_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase4"
DEFAULT_OUTPUT = PHASE4_OUTPUT_DIR / "long_text_comparison.csv"

EXPERIMENTS = [
    {
        "name": "head_256_baseline",
        "config": PROJECT_ROOT / "configs" / "train.yaml",
        "evaluation_metrics": PROJECT_ROOT / "outputs" / "phase2_evaluation" / "evaluation_metrics.json",
        "diagnosis_summary": PROJECT_ROOT / "outputs" / "phase3_error_diagnosis" / "diagnosis_summary.json",
        "error_by_length": PROJECT_ROOT / "outputs" / "phase3_error_diagnosis" / "error_by_length.csv",
    },
    {
        "name": "head_512",
        "config": PROJECT_ROOT / "configs" / "phase4_head512.yaml",
        "evaluation_metrics": PHASE4_OUTPUT_DIR / "head512" / "eval" / "evaluation_metrics.json",
        "diagnosis_summary": PHASE4_OUTPUT_DIR / "head512" / "diagnosis" / "diagnosis_summary.json",
        "error_by_length": PHASE4_OUTPUT_DIR / "head512" / "diagnosis" / "error_by_length.csv",
    },
    {
        "name": "head_tail_256",
        "config": PROJECT_ROOT / "configs" / "phase4_head_tail256.yaml",
        "evaluation_metrics": PHASE4_OUTPUT_DIR / "head_tail256" / "eval" / "evaluation_metrics.json",
        "diagnosis_summary": PHASE4_OUTPUT_DIR / "head_tail256" / "diagnosis" / "diagnosis_summary.json",
        "error_by_length": PHASE4_OUTPUT_DIR / "head_tail256" / "diagnosis" / "error_by_length.csv",
    },
    {
        "name": "tail_256",
        "config": PROJECT_ROOT / "configs" / "phase4_tail256.yaml",
        "evaluation_metrics": PHASE4_OUTPUT_DIR / "tail256" / "eval" / "evaluation_metrics.json",
        "diagnosis_summary": PHASE4_OUTPUT_DIR / "tail256" / "diagnosis" / "diagnosis_summary.json",
        "error_by_length": PHASE4_OUTPUT_DIR / "tail256" / "diagnosis" / "error_by_length.csv",
    },
]


def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_long_text_error_rate(path):
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("length_bucket") == "501+":
                return row.get("error_rate", "")
    return ""


def build_row(experiment):
    metrics = read_json(experiment["evaluation_metrics"])
    diagnosis = read_json(experiment["diagnosis_summary"])
    row = {
        "experiment": experiment["name"],
        "config": str(experiment["config"]),
        "status": "complete" if metrics else "missing",
        "accuracy": "",
        "f1": "",
        "errors": "",
        "error_rate": "",
        "high_confidence_errors": "",
        "false_positive_errors": "",
        "false_negative_errors": "",
        "long_text_501_plus_error_rate": "",
    }
    if metrics:
        row.update(
            {
                "accuracy": metrics.get("accuracy", ""),
                "f1": metrics.get("f1", ""),
                "errors": metrics.get("num_errors", ""),
                "error_rate": metrics.get("error_rate", ""),
            }
        )
    if diagnosis:
        row.update(
            {
                "high_confidence_errors": diagnosis.get("high_confidence_errors", ""),
                "false_positive_errors": diagnosis.get("false_positive_errors", ""),
                "false_negative_errors": diagnosis.get("false_negative_errors", ""),
            }
        )
    row["long_text_501_plus_error_rate"] = read_long_text_error_rate(experiment["error_by_length"])
    return row


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Phase 4 long-text experiments.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Comparison CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [build_row(experiment) for experiment in EXPERIMENTS]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    write_csv(rows, output_path)
    print(f"Saved comparison to {output_path}")
    for row in rows:
        print(f"{row['experiment']}: {row['status']}")


if __name__ == "__main__":
    main()
