from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch


class IMDBDataset(Dataset):
    """自定义数据集类，用于加载和处理IMDB数据"""

    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']

        # 使用tokenizer对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 返回一个字典，包含模型需要的所有输入和标签
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


if __name__ == "__main__":
    from datasets import load_dataset

    # 加载tokenizer和数据集
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_dataset('imdb')
    train_dataset = dataset['train'].select(range(100))  # 取前100条做测试

    # 创建我们的自定义数据集实例
    imdb_train = IMDBDataset(train_dataset, tokenizer)

    # 创建DataLoader，用于批量加载数据
    train_loader = DataLoader(imdb_train, batch_size=4, shuffle=True)

    # 测试一个批次
    for batch in train_loader:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels: {batch['labels']}")
        break  # 只看一个批次