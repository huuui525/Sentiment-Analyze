import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    """PyTorch dataset wrapper for text classification samples.

    Each source sample must contain a text field and a label field. The wrapper
    tokenizes text lazily so it remains simple and works with Hugging Face
    Dataset objects as well as list-like collections of dictionaries.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=512,
        truncation_strategy="head",
        text_column="text",
        label_column="label",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample[self.text_column]
        label = sample[self.label_column]

        encoding = self.encode_text(text)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def encode_text(self, text):
        if self.truncation_strategy == "head":
            return self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = self.truncate_token_ids(token_ids)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }

    def truncate_token_ids(self, token_ids):
        content_length = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if content_length <= 0:
            raise ValueError("max_length must be greater than 2.")
        if len(token_ids) <= content_length:
            return token_ids

        strategy = self.truncation_strategy
        if strategy == "tail":
            return token_ids[-content_length:]
        if strategy == "head_tail":
            head_length = content_length // 2
            tail_length = content_length - head_length
            return token_ids[:head_length] + token_ids[-tail_length:]

        raise ValueError(
            "truncation_strategy must be one of: head, tail, head_tail."
        )


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("imdb")["train"].select(range(4))
    imdb_dataset = IMDBDataset(dataset, tokenizer, max_length=256, truncation_strategy="head_tail")

    sample = imdb_dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label: {sample['labels'].item()}")
