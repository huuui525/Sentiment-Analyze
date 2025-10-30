import torch
from transformers import BertTokenizer, BertForSequenceClassification


def predict_sentiment(text, model_path='../models/finetuned_bert_imdb'):
    """预测单条文本的情感"""
    # 加载已保存的模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 预处理输入文本
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    # 返回结果
    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    return sentiment


if __name__ == "__main__":
    # 测试一些例子
    test_texts = [
        "This movie is absolutely fantastic! The acting is great and the story is compelling.",
        "I hated this film. It was boring and the characters were flat.",
        "It's an okay movie, nothing special but not terrible either."
    ]

    for text in test_texts:
        result = predict_sentiment(text)
        print(f"Review: {text}")
        print(f"Sentiment: {result}\n")