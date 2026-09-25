import torch
import torch.nn as nn
from text_classifier import TextClassifier


def main():
    torch.manual_seed(0)

    batch_size = 64
    seq_len = 6
    vocab_size = 100
    d_model = 16
    num_heads = 2
    d_ff = 64
    num_layers = 2
    num_classes = 2
    num_steps = 500         
    lr = 1e-3               

    model = TextClassifier(vocab_size, d_model, num_heads, d_ff, num_layers, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for step in range(num_steps):
        # 生成有规律的数据
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = (input_ids[:, 0] % 2)   # 第一个 token 是奇数还是偶数

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

        logits, _ = model(input_ids, mask=causal_mask)  

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()