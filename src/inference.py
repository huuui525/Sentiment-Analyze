import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_preprocessing import IMDBDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "phase6_xlm_roberta_balanced"
DEFAULT_MAX_LENGTH = 256
DEFAULT_TRUNCATION_STRATEGY = "head_tail"


def encode_for_inference(text, tokenizer, max_length=DEFAULT_MAX_LENGTH, truncation_strategy=DEFAULT_TRUNCATION_STRATEGY):
    dataset = IMDBDataset(
        [{"text": text, "label": 0}],
        tokenizer,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
    )
    return dataset.encode_text(text)


def predict_sentiment(
    text,
    model_path=DEFAULT_MODEL_PATH,
    max_length=DEFAULT_MAX_LENGTH,
    truncation_strategy=DEFAULT_TRUNCATION_STRATEGY,
):
    """Predict sentiment using the current main model and its truncation strategy."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoding = encode_for_inference(text, tokenizer, max_length, truncation_strategy)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    label = "positive" if prediction.item() == 1 else "negative"
    return {
        "label": label,
        "confidence": float(confidence.item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run sentiment inference with a trained model.")
    parser.add_argument("--text", help="Input text. If omitted, built-in examples are used.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to a trained model directory.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--truncation-strategy", choices=["head", "tail", "head_tail"], default=DEFAULT_TRUNCATION_STRATEGY)
    return parser.parse_args()


def main():
    args = parse_args()
    texts = [args.text] if args.text else [
        "This movie is absolutely fantastic! The acting is great and the story is compelling.",
        "I hated this film. It was boring and the characters were flat.",
        "这个电影节奏很好，演员表演也很有感染力。",
        "太失望了，剧情拖沓，体验很差 😞",
    ]

    for text in texts:
        result = predict_sentiment(
            text,
            model_path=args.model_path,
            max_length=args.max_length,
            truncation_strategy=args.truncation_strategy,
        )
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} ({result['confidence']:.4f})\n")


if __name__ == "__main__":
    main()
