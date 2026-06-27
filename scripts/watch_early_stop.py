import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def log(message: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def read_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return str(pid) in result.stdout


def stop_process_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)


def run_test_evaluation(args, log_path: Path) -> None:
    if not (args.eval_config and args.eval_model_path and args.eval_output_dir):
        return

    stdout_path = Path(args.eval_stdout)
    stderr_path = Path(args.eval_stderr)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        "src/evaluate.py",
        "--config",
        args.eval_config,
        "--model-path",
        args.eval_model_path,
        "--output-dir",
        args.eval_output_dir,
        "--split",
        "test",
    ]
    log(f"Starting test evaluation: {' '.join(command)}", log_path)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            command,
            cwd=args.project_root,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    log(f"Test evaluation finished with return code {result.returncode}.", log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--check-interval", type=int, default=60)
    parser.add_argument("--log", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--eval-config")
    parser.add_argument("--eval-model-path")
    parser.add_argument("--eval-output-dir")
    parser.add_argument("--eval-stdout", default="outputs/phase6/logs/early_stop_eval.out.log")
    parser.add_argument("--eval-stderr", default="outputs/phase6/logs/early_stop_eval.err.log")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    log_path = Path(args.log)

    log(
        f"Watcher started. pid={args.pid}, metrics={metrics_path}, min_delta={args.min_delta}",
        log_path,
    )

    while True:
        metrics = read_metrics(metrics_path)
        history = metrics.get("history", []) if metrics else []

        if len(history) >= 2:
            previous_best = max(item["validation"]["f1"] for item in history[:-1])
            latest = history[-1]
            latest_f1 = latest["validation"]["f1"]
            latest_epoch = latest.get("epoch", len(history))
            delta = latest_f1 - previous_best

            if delta < args.min_delta:
                log(
                    (
                        f"Early stop triggered at epoch {latest_epoch}: "
                        f"latest_f1={latest_f1:.6f}, previous_best={previous_best:.6f}, "
                        f"delta={delta:.6f} < min_delta={args.min_delta:.6f}. "
                        f"Stopping pid tree {args.pid}."
                    ),
                    log_path,
                )
                stop_process_tree(args.pid)
                run_test_evaluation(args, log_path)
            else:
                log(
                    (
                        f"No early stop at epoch {latest_epoch}: "
                        f"latest_f1={latest_f1:.6f}, previous_best={previous_best:.6f}, "
                        f"delta={delta:.6f} >= min_delta={args.min_delta:.6f}."
                    ),
                    log_path,
                )
            return

        if not process_exists(args.pid):
            log(f"Training pid {args.pid} exited before epoch 2 metrics were available.", log_path)
            return

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
