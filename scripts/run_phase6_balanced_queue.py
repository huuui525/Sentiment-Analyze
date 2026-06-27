import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


def write_log(message: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return str(pid) in result.stdout


def run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, log_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    write_log(f"Running: {' '.join(command)}", log_path)

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        write_log(f"Started pid={process.pid}", log_path)
        return_code = process.wait()

    write_log(f"Finished with return code {return_code}: {' '.join(command)}", log_path)
    return return_code


def evaluate_model(
    python_path: str,
    project_root: Path,
    config_path: str,
    model_path: str,
    output_dir: str,
    log_prefix: str,
    log_path: Path,
) -> int:
    return run_command(
        [
            python_path,
            "src/evaluate.py",
            "--config",
            config_path,
            "--model-path",
            model_path,
            "--output-dir",
            output_dir,
            "--split",
            "test",
        ],
        cwd=project_root,
        stdout_path=project_root / "outputs" / "phase6" / "logs" / f"{log_prefix}_eval.out.log",
        stderr_path=project_root / "outputs" / "phase6" / "logs" / f"{log_prefix}_eval.err.log",
        log_path=log_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python", default=".\\.venv\\Scripts\\python.exe")
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--check-interval", type=int, default=60)
    parser.add_argument("--log", default="outputs/phase6/logs/balanced_queue.log")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    log_path = project_root / args.log

    write_log(f"Queue started. Waiting for XLM-R pid={args.wait_pid}", log_path)
    while process_exists(args.wait_pid):
        time.sleep(args.check_interval)

    write_log("XLM-R training process exited. Starting XLM-R test evaluation.", log_path)
    evaluate_model(
        args.python,
        project_root,
        "configs/phase6_xlm_roberta_balanced.yaml",
        "models/phase6_xlm_roberta_balanced",
        "outputs/phase6/xlm_roberta_balanced/eval",
        "xlm_roberta_balanced",
        log_path,
    )

    write_log("Starting mBERT balanced training.", log_path)
    mbert_train_code = run_command(
        [
            args.python,
            "-u",
            "src/train.py",
            "--config",
            "configs/phase6_mbert_balanced.yaml",
        ],
        cwd=project_root,
        stdout_path=project_root / "outputs" / "phase6" / "logs" / "mbert_balanced_train.out.log",
        stderr_path=project_root / "outputs" / "phase6" / "logs" / "mbert_balanced_train.err.log",
        log_path=log_path,
    )

    if mbert_train_code == 0:
        write_log("mBERT training finished. Starting mBERT test evaluation.", log_path)
        evaluate_model(
            args.python,
            project_root,
            "configs/phase6_mbert_balanced.yaml",
            "models/phase6_mbert_balanced",
            "outputs/phase6/mbert_balanced/eval",
            "mbert_balanced",
            log_path,
        )
    else:
        write_log("mBERT training failed. Skipping mBERT evaluation.", log_path)

    write_log("Balanced queue finished. Full training was not started.", log_path)


if __name__ == "__main__":
    main()
