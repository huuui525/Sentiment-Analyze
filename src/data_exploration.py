from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import os

def explore_dataset():
    # 加载数据集
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # 查看数据样例
    print("\n=== Data Samples ===")
    for i in range(2):
        sample = train_dataset[i]
        sentiment = "Positive" if sample['label'] == 1 else "Negative"
        print(f"Sample {i + 1} - Sentiment: {sentiment}")
        print(f"Text preview: {sample['text'][:200]}...")

    # 转换为DataFrame进行更详细的分析
    train_df = pd.DataFrame(train_dataset)

    # 标签分布分析
    print("\n=== Label Distribution ===")
    label_counts = train_df['label'].value_counts()
    print(f"Positive reviews: {label_counts.get(1, 0)}")
    print(f"Negative reviews: {label_counts.get(0, 0)}")

    # 可视化标签分布
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    label_counts.plot(kind='bar', color=['red', 'green'])
    plt.title('Training Set Label Distribution')
    plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
    plt.ylabel('Count')

    # 文本长度分析
    print("\n=== Text Length Analysis ===")
    train_df['text_length'] = train_df['text'].apply(lambda x: len(x.split()))

    length_stats = train_df['text_length'].describe()
    print(length_stats)

    # 计算关键分位数
    for percentile in [50, 75, 90, 95, 99]:
        length = train_df['text_length'].quantile(percentile / 100)
        print(f"{percentile}% of reviews have length <= {length:.0f} words")

    # 可视化文本长度分布
    plt.subplot(1, 2, 2)
    plt.hist(train_df['text_length'], bins=50, range=[0, 500], alpha=0.7, color='skyblue')
    plt.xlabel('Text Length (words)')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.axvline(x=256, color='red', linestyle='--', label='Suggested max length (256)')
    plt.legend()

    plt.tight_layout()

    # 保存图表
    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 文本内容分析
    print("\n=== Text Content Analysis ===")
    # 查看正面和负面评论中最常见的单词（简单版本）
    positive_texts = ' '.join(train_df[train_df['label'] == 1]['text'].tolist())
    negative_texts = ' '.join(train_df[train_df['label'] == 0]['text'].tolist())

    print(f"Total words in positive reviews: {len(positive_texts.split())}")
    print(f"Total words in negative reviews: {len(negative_texts.split())}")

    return train_df, length_stats


if __name__ == "__main__":
    df, stats = explore_dataset()

    # 基于分析结果给出建议
    max_length = int(stats['75%'])
    print(f"\n=== Preprocessing Suggestions ===")
    print(f"Suggested maximum text length: {max_length} (based on 75th percentile)")
    print(f"Consider setting to 256 or 512 for better model performance")