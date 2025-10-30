import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from data_preprocessing import IMDBDataset
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn.metrics as metrics
import os
import numpy as np
import random


def debug_dataset_labels(dataset, name="数据集"):
    """调试数据集的标签分布"""
    print(f"\n=== {name}标签分析 ===")

    labels = []
    texts = []

    # 采样检查前100条数据
    for i in range(min(100, len(dataset))):
        item = dataset[i]
        if isinstance(item, dict) and 'label' in item:
            labels.append(item['label'])
            texts.append(item['text'][:50])  # 取前50个字符

    if labels:
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        print(f"采样检查 - 正面: {positive_count}, 负面: {negative_count}")

        # 显示一些样本内容
        print("样本预览:")
        for i in range(min(3, len(texts))):
            sentiment = "正面" if labels[i] == 1 else "负面"
            print(f"  {sentiment}: {texts[i]}...")
    else:
        print("无法获取标签信息")


def create_balanced_dataset(original_dataset, num_samples=1000, seed=42):
    """创建平衡的数据集，包含正负样本"""
    print(f"创建平衡数据集，目标样本数: {num_samples}")

    # 设置随机种子
    random.seed(seed)

    # 分离正负样本
    positive_samples = []
    negative_samples = []

    # 遍历数据集，分离正负样本
    for i in range(len(original_dataset)):
        item = original_dataset[i]
        if isinstance(item, dict) and 'label' in item:
            if item['label'] == 1:
                positive_samples.append(item)
            else:
                negative_samples.append(item)

    print(f"找到的正面样本数: {len(positive_samples)}")
    print(f"找到的负面样本数: {len(negative_samples)}")

    # 计算每个类别应该取多少样本
    samples_per_class = num_samples // 2

    # 如果某个类别的样本不够，调整采样数
    actual_positive = min(samples_per_class, len(positive_samples))
    actual_negative = min(samples_per_class, len(negative_samples))

    # 随机采样
    sampled_positive = random.sample(positive_samples, actual_positive)
    sampled_negative = random.sample(negative_samples, actual_negative)

    balanced_dataset = sampled_positive + sampled_negative

    # 打乱顺序
    random.shuffle(balanced_dataset)

    print(f"创建的平衡数据集大小: {len(balanced_dataset)}")
    print(f"其中正面: {actual_positive}, 负面: {actual_negative}")

    return balanced_dataset


# 训练函数
def train_bert():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 超参数设置
    BATCH_SIZE = 16
    EPOCHS = 3
    MAX_LENGTH = 256
    LEARNING_RATE = 2e-5

    # 1. 加载Tokenzier和模型
    print("加载 tokenizer 和模型...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(device)

    # 2. 加载并预处理数据
    print("加载IMDb数据集...")
    dataset = load_dataset('imdb')

    # 调试原始数据分布
    print("\n=== 原始数据集分布 ===")
    print(f"训练集总大小: {len(dataset['train'])}")
    print(f"测试集总大小: {len(dataset['test'])}")

    # 检查原始数据的标签分布
    debug_dataset_labels(dataset['train'], "原始训练集")
    debug_dataset_labels(dataset['test'], "原始测试集")

    # 创建平衡的训练和测试集
    print("\n=== 创建平衡数据集 ===")
    train_subset = create_balanced_dataset(dataset['train'], num_samples=1000)
    test_subset = create_balanced_dataset(dataset['test'], num_samples=500)

    # 验证平衡后的数据集
    print("\n=== 平衡后数据集验证 ===")
    train_labels = [item['label'] for item in train_subset]
    test_labels = [item['label'] for item in test_subset]

    print(f"训练集 - 正面: {sum(train_labels)}, 负面: {len(train_labels) - sum(train_labels)}")
    print(f"测试集 - 正面: {sum(test_labels)}, 负面: {len(test_labels) - sum(test_labels)}")

    # 显示一些样本验证
    print("\n训练集样本示例:")
    for i in range(3):
        sentiment = "正面" if train_subset[i]['label'] == 1 else "负面"
        print(f"  {i + 1}. {sentiment}: {train_subset[i]['text'][:100]}...")

    print("\n测试集样本示例:")
    for i in range(3):
        sentiment = "正面" if test_subset[i]['label'] == 1 else "负面"
        print(f"  {i + 1}. {sentiment}: {test_subset[i]['text'][:100]}...")

    # 创建数据集
    train_dataset = IMDBDataset(train_subset, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_subset, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    # 4. 训练循环
    print("\n开始训练...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            # 计算准确率
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            train_accuracy = correct_predictions / total_samples

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'Acc': f'{train_accuracy:.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        final_train_accuracy = correct_predictions / total_samples

        print(f'Epoch {epoch + 1} 完成:')
        print(f'  平均损失: {avg_loss:.4f}')
        print(f'  训练准确率: {final_train_accuracy:.4f}')

        # 测试集验证
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_accuracy = test_correct / test_total
        print(f'  测试准确率: {test_accuracy:.4f}')

    # 5. 最终评估
    print("\n开始最终评估...")
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    accuracy = metrics.accuracy_score(true_labels, predictions)
    print(f"\n最终测试准确率: {accuracy:.4f}")

    # 显示详细结果
    print("\n分类报告:")
    print(metrics.classification_report(true_labels, predictions,
                                        target_names=['负面', '正面']))

    # 6. 保存模型
    os.makedirs('../models', exist_ok=True)
    model.save_pretrained('../models/finetuned_bert_imdb')
    tokenizer.save_pretrained('../models/finetuned_bert_imdb')
    print("模型已保存至 '../models/finetuned_bert_imdb'")


if __name__ == "__main__":
    train_bert()