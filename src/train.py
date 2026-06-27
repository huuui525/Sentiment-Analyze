import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from data_preprocessing import IMDBDataset
from datasets import load_dataset
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_samples(dataset, sample_count, seed):
    if sample_count is None:
        return dataset
    sample_count = min(int(sample_count), len(dataset))
    return dataset.shuffle(seed=seed).select(range(sample_count))


def split_train_validation(dataset, validation_size, seed):
    try:
        return dataset.train_test_split(
            test_size=validation_size,
            seed=seed,
            stratify_by_column="label",
        )
    except ValueError:
        return dataset.train_test_split(test_size=validation_size, seed=seed)


def prepare_datasets(config):
    data_config = config["data"]
    seed = config["training"]["seed"]

    if data_config.get("dataset_format") == "local_parquet":
        data_dir = resolve_project_path(data_config["data_dir"])
        data_files = {
            "train": str(data_dir / "train.parquet"),
            "validation": str(data_dir / "validation.parquet"),
            "test": str(data_dir / "test.parquet"),
        }
        dataset = load_dataset("parquet", data_files=data_files)
        train_split = select_samples(dataset["train"], data_config.get("train_samples"), seed)
        val_split = select_samples(dataset["validation"], data_config.get("validation_samples"), seed)
        test_split = select_samples(dataset["test"], data_config.get("test_samples"), seed)
        return train_split, val_split, test_split

    dataset = load_dataset(data_config["dataset_name"])
    split = split_train_validation(dataset["train"], data_config["validation_size"], seed)

    train_split = select_samples(split["train"], data_config.get("train_samples"), seed)
    val_split = select_samples(split["test"], data_config.get("validation_samples"), seed)
    test_split = select_samples(dataset["test"], data_config.get("test_samples"), seed)

    return train_split, val_split, test_split


def build_loader(dataset, tokenizer, config, shuffle=False):
    training_config = config["training"]
    data_config = config["data"]
    torch_generator = torch.Generator()
    torch_generator.manual_seed(training_config["seed"])

    return DataLoader(
        IMDBDataset(
            dataset,
            tokenizer,
            max_length=data_config["max_length"],
            truncation_strategy=data_config.get("truncation_strategy", "head"),
        ),
        batch_size=training_config["batch_size"],
        shuffle=shuffle,
        num_workers=training_config.get("num_workers", 0),
        generator=torch_generator if shuffle else None,
    )


def train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, epochs, use_amp=False, scaler=None):
    model.train()
    total_loss = 0.0
    predictions = []
    true_labels = []

    progress = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        batch_predictions = torch.argmax(outputs.logits, dim=1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        true_labels.extend(labels.detach().cpu().tolist())

        running_accuracy = metrics.accuracy_score(true_labels, predictions)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_accuracy:.4f}")

    return summarize_metrics(true_labels, predictions, total_loss / len(data_loader))


def evaluate(model, data_loader, device, desc="Evaluating", use_amp=False):
    model.eval()
    total_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            batch_predictions = torch.argmax(outputs.logits, dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    return summarize_metrics(true_labels, predictions, total_loss / len(data_loader))


def summarize_metrics(true_labels, predictions, average_loss):
    return {
        "loss": float(average_loss),
        "accuracy": float(metrics.accuracy_score(true_labels, predictions)),
        "precision": float(metrics.precision_score(true_labels, predictions, zero_division=0)),
        "recall": float(metrics.recall_score(true_labels, predictions, zero_division=0)),
        "f1": float(metrics.f1_score(true_labels, predictions, zero_division=0)),
        "classification_report": metrics.classification_report(
            true_labels,
            predictions,
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_best_model(model, tokenizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_model(config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    output_dir = resolve_project_path(config["output"]["output_dir"])
    best_model_dir = resolve_project_path(config["output"]["best_model_dir"])
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "train_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {config['data']['dataset_name']}")

    train_split, val_split, test_split = prepare_datasets(config)
    print(f"Train samples: {len(train_split)}")
    print(f"Validation samples: {len(val_split)}")
    print(f"Test samples: {len(test_split)}")

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["model"]["num_labels"],
    )
    model.to(device)

    train_loader = build_loader(train_split, tokenizer, config, shuffle=True)
    val_loader = build_loader(val_split, tokenizer, config)
    test_loader = build_loader(test_split, tokenizer, config)

    training_config = config["training"]
    use_amp = bool(training_config.get("mixed_precision", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if use_amp:
        print("Using mixed precision: fp16")

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    total_steps = len(train_loader) * training_config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_config["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    best_metric_name = training_config["best_metric"]
    best_metric_value = -1.0
    best_epoch = None
    history = []
    early_stopping_config = training_config.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_config.get("enabled", False))
    early_stopping_min_delta = float(early_stopping_config.get("min_delta", 0.0))
    early_stopping_patience = int(early_stopping_config.get("patience", 1))
    early_stopping_min_epochs = int(early_stopping_config.get("min_epochs", 1))
    epochs_without_significant_improvement = 0
    stopped_early = False
    early_stop_reason = None

    for epoch in range(1, training_config["epochs"] + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            training_config["epochs"],
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, device, desc="Validating", use_amp=use_amp)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(epoch_record)

        current_metric = val_metrics[best_metric_name]
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_{best_metric_name}={current_metric:.4f}"
        )

        previous_best_metric_value = best_metric_value
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch
            save_best_model(model, tokenizer, best_model_dir)
            print(f"Saved new best model to {best_model_dir}")

        save_json(
            {
                "config": config,
                "best_epoch": best_epoch,
                "best_validation_metric": {
                    "name": best_metric_name,
                    "value": best_metric_value,
                },
                "history": history,
            },
            metrics_path,
        )

        if early_stopping_enabled and epoch >= early_stopping_min_epochs:
            has_significant_improvement = (
                previous_best_metric_value < 0
                or current_metric >= previous_best_metric_value + early_stopping_min_delta
            )
            if has_significant_improvement:
                epochs_without_significant_improvement = 0
            else:
                epochs_without_significant_improvement += 1

            if epochs_without_significant_improvement >= early_stopping_patience:
                stopped_early = True
                early_stop_reason = (
                    f"Stopped after epoch {epoch}: validation {best_metric_name} "
                    f"improvement was smaller than {early_stopping_min_delta} "
                    f"for {early_stopping_patience} epoch(s)."
                )
                print(f"Early stopping: {early_stop_reason}")
                break

    print("Loading best model for final test evaluation.")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device, desc="Testing", use_amp=use_amp)

    final_metrics = {
        "config": config,
        "splits": {
            "train": len(train_split),
            "validation": len(val_split),
            "test": len(test_split),
        },
        "best_epoch": best_epoch,
        "best_validation_metric": {
            "name": best_metric_name,
            "value": best_metric_value,
        },
        "history": history,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "stopped_early": stopped_early,
            "reason": early_stop_reason,
            "min_delta": early_stopping_min_delta,
            "patience": early_stopping_patience,
            "min_epochs": early_stopping_min_epochs,
        },
        "test": test_metrics,
        "artifacts": {
            "best_model_dir": str(best_model_dir),
            "metrics_path": str(metrics_path),
        },
    }
    save_json(final_metrics, metrics_path)

    print(f"Best epoch: {best_epoch}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test f1: {test_metrics['f1']:.4f}")
    print(f"Metrics saved to {metrics_path}")

    return final_metrics


def train_bert(config_path=DEFAULT_CONFIG_PATH):
    return train_model(config_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for IMDb sentiment analysis.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML training config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.config)
