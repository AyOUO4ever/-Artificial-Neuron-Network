import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
import chardet

#配置参数
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "九章算经.txt"

#数据读取与预处理
def load_and_clean_text(filepath):
    with open(filepath, "rb") as f:
        raw_data = f.read()

    detected = chardet.detect(raw_data)
    encoding = detected.get('encoding', 'utf-8')
    confidence = detected.get('confidence', 0)
    print(f"检测到文件编码：{encoding} (置信度：{confidence:.2f})")

    try:
        raw = raw_data.decode(encoding)
    except UnicodeDecodeError:
        # 如果解码失败，尝试使用常见的中文编码
        print(f"编码 {encoding} 解码失败，尝试使用 gb18030...")
        raw = raw_data.decode('gb18030')

    #提取内容
    start_marker = "九章算術卷第一"
    start_idx = raw.find(start_marker)
    if start_idx == -1:
        start_idx = 0
    text = raw[start_idx:]
    end_marker = "\n\n\n九章算經點校"
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    #将多个连续换行和空格规范化
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', '', text)
    return text


# 读取并清洗文本
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"文件缺失：{DATA_FILE}")
text = load_and_clean_text(DATA_FILE)
print(f"清洗后的文本长度：{len(text)} 字符")

# 建立字符级词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"字符种类数：{vocab_size}")
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 将整个文本转换为索引序列
text_as_int = [char_to_idx[c] for c in text]


# 构建数据集
class TextDataset(Dataset):

    def __init__(self, text_int, seq_length):
        self.seq_length = seq_length
        self.data = text_int

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


dataset = TextDataset(text_as_int, SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# 定义模型
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.fc(self.dropout(out))
        return out, hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_DIM).to(DEVICE)
        c = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_DIM).to(DEVICE)
        return (h, c)


model = CharLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")


# 训练循环
def train():
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    loss = train()
    print(f"Epoch {epoch:2d}/{EPOCHS}  Loss: {loss:.4f}")


# 文本生成函数
def generate(model, start_text, gen_length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # 将 start_text 转换为 tensor
        start_seq = [char_to_idx.get(c, 0) for c in start_text]
        input_tensor = torch.tensor(start_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)

        # 先让模型“读入”起始序列，获取隐藏状态
        hidden = model.init_hidden(1)
        # 除最后一个字符外，用于更新隐藏状态
        if len(start_seq) > 1:
            _, hidden = model(input_tensor[:, :-1], hidden)

        # 最后一个字符作为输入
        current_char = input_tensor[:, -1:]
        generated = start_text

        for _ in range(gen_length):
            output, hidden = model(current_char, hidden)
            # 取最后一个时间步的输出
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
            # 按概率采样下一个字符
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = idx_to_char[next_idx]
            generated += next_char
            current_char = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)

            # 如果生成了换行或停止符可以提前结束，这里简单以连续两个换行为停止条件
            if generated.endswith("\n\n"):
                break
    return generated


# 测试与展示
print("\n===== 生成示例 =====")
# 选取几个书中存在的问题（作为 prompt），检查模型能否给出正确答案
test_prompts = [
    "今有田廣十五步，從十六步。問為田幾何？\n荅曰：",
    "今有粟一斗，欲為糲米。問得幾何？\n荅曰：",
    "今有勾三尺，股四尺，問為弦幾何？\n荅曰：",
    "今有圓田，週三十步，逕十步。問為田幾何？\n荅曰：",
    "今有雉兔同籠，上有三十五頭，下有九十四足。問雉兔各幾何？\n荅曰：",  # 这是一个干扰项，书中无此题
]

for prompt in test_prompts:
    print(f"Prompt: {prompt.strip()}")
    result = generate(model, prompt, gen_length=50, temperature=0.6)
    # 只打印生成部分（去除 prompt 后的内容）
    generated_part = result[len(prompt):]
    # 限制显示到第一个换行，使输出整洁
    first_line = generated_part.split('\n')[0]
    print(f"生成: {first_line}\n")

print("训练完成")