import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "phase6_multilingual_sentiment"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

DIRECT_DOWNLOADS = {
    "imdb_aclImdb_v1.tar.gz": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "waimai_10k.csv": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
}

HF_DATASETS = {
    "tweet_eval_sentiment": ("cardiffnlp/tweet_eval", "sentiment"),
    "weibo_senti_100k": ("dirtycomputer/weibo_senti_100k", None),
}

EMOJI_SENTIMENT_API = "https://api.figshare.com/v2/articles/1600931"

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]"
)
WEIBO_EMOTION_PATTERN = re.compile(r"\[[^\[\]\s]{1,12}\]")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, output_path, chunk_size=1024 * 1024):
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skip existing raw file: {output_path}")
        return

    print(f"Downloading {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if downloaded // (50 * chunk_size) != (downloaded - len(chunk)) // (50 * chunk_size):
                    print(f"  {output_path.name}: {downloaded / 1024 / 1024:.1f} MB")
    temp_path.replace(output_path)
    print(f"Saved {output_path}")


def zip_single_file(input_path, output_path, arcname=None):
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skip existing archive: {output_path}")
        return
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(input_path, arcname=arcname or input_path.name)
    print(f"Archived {input_path.name} -> {output_path.name}")


def archive_hf_dataset(name, dataset_name, config_name):
    archive_path = RAW_DIR / f"{name}_hf_snapshot.zip"
    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"Skip existing HF snapshot: {archive_path}")
        return

    dataset = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)
    metadata = {
        "name": name,
        "dataset_name": dataset_name,
        "config_name": config_name,
        "splits": {split: len(dataset[split]) for split in dataset},
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        for split, split_dataset in dataset.items():
            lines = []
            for row in split_dataset:
                lines.append(json.dumps(row, ensure_ascii=False))
            archive.writestr(f"{split}.jsonl", "\n".join(lines) + "\n")
    print(f"Saved HF snapshot: {archive_path}")


def download_emoji_sentiment_archive():
    archive_path = RAW_DIR / "emoji_sentiment_ranking_figshare.zip"
    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"Skip existing emoji archive: {archive_path}")
        return

    response = requests.get(EMOJI_SENTIMENT_API, timeout=30)
    response.raise_for_status()
    article = response.json()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("figshare_article_metadata.json", json.dumps(article, ensure_ascii=False, indent=2))
        for file_info in article["files"]:
            file_response = requests.get(file_info["download_url"], timeout=60)
            file_response.raise_for_status()
            archive.writestr(file_info["name"], file_response.content)
    print(f"Saved emoji archive: {archive_path}")


def download_raw_sources():
    ensure_dirs()
    for filename, url in DIRECT_DOWNLOADS.items():
        download_file(url, RAW_DIR / filename)

    zip_single_file(RAW_DIR / "waimai_10k.csv", RAW_DIR / "waimai_10k.zip")

    for name, (dataset_name, config_name) in HF_DATASETS.items():
        archive_hf_dataset(name, dataset_name, config_name)

    download_emoji_sentiment_archive()


def normalize_text(text):
    return " ".join(str(text).replace("\ufeff", "").split())


def has_emoji_like_text(text):
    return bool(EMOJI_PATTERN.search(text) or WEIBO_EMOTION_PATTERN.search(text))


def emoji_like_count(text):
    return len(EMOJI_PATTERN.findall(text)) + len(WEIBO_EMOTION_PATTERN.findall(text))


def make_record(text, label, language, domain, source, original_split):
    text = normalize_text(text)
    return {
        "text": text,
        "label": int(label),
        "label_name": "positive" if int(label) == 1 else "negative",
        "language": language,
        "domain": domain,
        "source": source,
        "original_split": original_split,
        "has_emoji": has_emoji_like_text(text),
        "emoji_count": emoji_like_count(text),
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def records_from_imdb():
    dataset = load_dataset("imdb")
    records = []
    train_rows = [dict(row) for row in dataset["train"]]
    train_labels = [row["label"] for row in train_rows]
    train_rows, validation_rows = train_test_split(
        train_rows,
        test_size=0.1,
        random_state=42,
        stratify=train_labels,
    )
    for row in train_rows:
        records.append(make_record(row["text"], row["label"], "en", "movie_review", "imdb", "train"))
    for row in validation_rows:
        records.append(make_record(row["text"], row["label"], "en", "movie_review", "imdb", "validation"))
    for row in dataset["test"]:
        records.append(make_record(row["text"], row["label"], "en", "movie_review", "imdb", "test"))
    return records


def records_from_tweet_eval():
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    records = []
    for split in ["train", "validation", "test"]:
        for row in dataset[split]:
            if row["label"] == 1:
                continue
            label = 0 if row["label"] == 0 else 1
            records.append(make_record(row["text"], label, "en", "social_media", "tweet_eval_sentiment", split))
    return records


def records_from_weibo():
    dataset = load_dataset("dirtycomputer/weibo_senti_100k")
    return [
        make_record(row["review"], row["label"], "zh", "social_media", "weibo_senti_100k", "unsplit")
        for row in dataset["train"]
    ]


def records_from_waimai():
    csv_path = RAW_DIR / "waimai_10k.csv"
    df = pd.read_csv(csv_path)
    text_column = "review" if "review" in df.columns else "text"
    return [
        make_record(row[text_column], row["label"], "zh", "food_delivery", "waimai_10k", "unsplit")
        for _, row in df.iterrows()
    ]


def assign_splits(df, seed):
    fixed_split = df["original_split"].isin(["train", "validation", "test"])
    fixed = df[fixed_split].copy()
    unsplit = df[~fixed_split].copy()

    fixed["split"] = fixed["original_split"].replace({"validation": "validation"})
    if unsplit.empty:
        return fixed

    train_df, temp_df = train_test_split(
        unsplit,
        test_size=0.2,
        random_state=seed,
        stratify=unsplit[["source", "label"]],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df[["source", "label"]],
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    return pd.concat([fixed, train_df, val_df, test_df], ignore_index=True)


def make_balanced_subset(df, seed, train_cap=10000, eval_cap=1500):
    parts = []
    for (split, source, label), group in df.groupby(["split", "source", "label"]):
        cap = train_cap if split == "train" else eval_cap
        sample_size = min(len(group), cap)
        parts.append(group.sample(n=sample_size, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_split_parquet(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        split_df.to_parquet(output_dir / f"{split}.parquet", index=False)
    df.sample(n=min(100, len(df)), random_state=42).to_csv(
        SAMPLE_DIR / f"{output_dir.name}_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )


def load_emoji_sentiment_lexicon():
    archive_path = RAW_DIR / "emoji_sentiment_ranking_figshare.zip"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("Emoji_Sentiment_Data_v1.0.csv") as file:
            df = pd.read_csv(file)
    output_path = PROCESSED_DIR / "emoji_sentiment_lexicon.parquet"
    df.to_parquet(output_path, index=False)
    return df


def write_dataset_card(balanced_df, emoji_df, train_cap=10000, eval_cap=1500):
    def summary_table(df):
        summary_df = df.groupby(["split", "source", "label"]).size().rename("samples").reset_index()
        table_lines = [
            "| split | source | label | samples |",
            "|---|---|---:|---:|",
        ]
        for _, row in summary_df.iterrows():
            table_lines.append(
                f"| {row['split']} | {row['source']} | {int(row['label'])} | {int(row['samples'])} |"
            )
        return "\n".join(table_lines)

    lines = [
        "# Phase 6 Multilingual Sentiment Dataset",
        "",
        "## 数据来源",
        "",
        "| 数据集 | 语言 | 场景 | 用途 | 原始归档 |",
        "|---|---|---|---|---|",
        "| Stanford IMDb Large Movie Review Dataset | 英文 | 长影评 | 英文长文本情感基线 | `data/raw/imdb_aclImdb_v1.tar.gz` |",
        "| TweetEval sentiment | 英文 | Twitter短文本 | 英文社交媒体情感 | `data/raw/tweet_eval_sentiment_hf_snapshot.zip` |",
        "| weibo_senti_100k | 中文 | 微博短文本 | 中文社交文本与表情文本 | `data/raw/weibo_senti_100k_hf_snapshot.zip` |",
        "| waimai_10k | 中文 | 外卖评论 | 中文短评论情感 | `data/raw/waimai_10k.zip` |",
        "| Emoji Sentiment Ranking | emoji | 情绪词典 | emoji情绪增强与消融实验 | `data/raw/emoji_sentiment_ranking_figshare.zip` |",
        "",
        "## 字段说明",
        "",
        "- `text`: 原始文本清洗后的内容。",
        "- `label`: 二分类标签，0 表示 negative，1 表示 positive。",
        "- `language`: `en` 或 `zh`。",
        "- `domain`: 文本场景，例如 `movie_review`、`social_media`、`ecommerce`。",
        "- `source`: 数据集来源。",
        "- `split`: 统一后的 `train`、`validation`、`test`。",
        "- `has_emoji`: 是否包含 Unicode emoji 或微博方括号表情。",
        "- `emoji_count`: emoji/表情标记数量。",
        "",
        "## balanced 数据集生成方式",
        "",
        "处理流程：",
        "",
        "1. 读取 IMDb、TweetEval、weibo_senti_100k、waimai_10k 四个监督数据源。",
        "2. 统一字段为 `text`、`label`、`language`、`domain`、`source`、`split` 等。",
        "3. 对没有官方划分的数据源按 `source + label` 分层划分为 train / validation / test。",
        "4. 按 `split + source + label` 分组抽样，避免某个数据源或标签占比过高。",
        f"5. 训练集每个分组最多保留 `{train_cap}` 条，验证集和测试集每个分组最多保留 `{eval_cap}` 条。",
        "",
        "## balanced 数据规模",
        "",
        summary_table(balanced_df),
        "",
        "Emoji Sentiment Ranking 已保存为 `emoji_sentiment_lexicon.parquet`，后续可用于将 emoji 映射为情绪 token。",
        "",
        f"Emoji lexicon rows: {len(emoji_df)}",
    ]
    (PROCESSED_DIR / "dataset_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_processed_dataset(seed):
    records = []
    builders = [
        ("imdb", records_from_imdb),
        ("tweet_eval_sentiment", records_from_tweet_eval),
        ("weibo_senti_100k", records_from_weibo),
        ("waimai_10k", records_from_waimai),
    ]
    for name, builder in builders:
        print(f"Building records from {name}")
        source_records = builder()
        print(f"  {name}: {len(source_records)} records")
        records.extend(source_records)

    source_pool_df = pd.DataFrame(records)
    source_pool_df = source_pool_df[source_pool_df["text"].str.len() > 0].drop_duplicates(
        subset=["text", "label", "source"]
    )
    source_pool_df = assign_splits(source_pool_df, seed)
    source_pool_df = source_pool_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    balanced_df = make_balanced_subset(source_pool_df, seed)
    emoji_df = load_emoji_sentiment_lexicon()

    save_split_parquet(balanced_df, PROCESSED_DIR / "balanced")
    write_dataset_card(balanced_df, emoji_df)

    summary = {
        "balanced_rows": int(len(balanced_df)),
        "emoji_lexicon_rows": int(len(emoji_df)),
        "balanced_by_split": balanced_df["split"].value_counts().to_dict(),
        "sources": sorted(source_pool_df["source"].unique().tolist()),
        "balanced_sampling": {
            "group_by": ["split", "source", "label"],
            "train_cap_per_group": 10000,
            "eval_cap_per_group": 1500,
        },
        "processed_dir": str(PROCESSED_DIR),
    }
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def clean_interim():
    interim_dir = PROJECT_ROOT / "data" / "interim"
    if interim_dir.exists():
        shutil.rmtree(interim_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Phase 6 multilingual sentiment datasets.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing files in data/raw.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_download:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    ensure_dirs()
    if not args.skip_download:
        download_raw_sources()
    build_processed_dataset(args.seed)
    clean_interim()


if __name__ == "__main__":
    main()
