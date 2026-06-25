import torch
import numpy as np
from hw5 import *

# 重新构建分词器和配置（与训练时相同）
base_text = """子曰：学而时习之，不亦说乎。有朋自远方来，不亦乐乎。人不知而不愠，不亦君子乎。
    子曰：温故而知新，可以为师矣。
    子曰：学而不思则罔，思而不学则殆。
    孟子曰：天时不如地利，地利不如人和。
    孟子曰：得道者多助，失道者寡助。
    孟子曰：生于忧患，死于安乐。
    """ * 500

tokenizer = CharTokenizer(base_text)
config = LMConfig(
    vocab_size=tokenizer.vocab_size,
    dim=1536, n_layers=24, n_heads=16,
    max_seq_len=64,
    pad_token_id=tokenizer.pad_id
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载最终 DPO 模型
model = MiniMindLM(config).to(device)
model.load_state_dict(torch.load("dpo_0.6B.pth", map_location=device))
model.eval()

# 生成示例
prompts = ["子曰：", "孟子曰：", "曰：", "天地"]
for p in prompts:
    input_ids = torch.tensor(
        [tokenizer.bos_id] + tokenizer.encode_as_ids(p),
        dtype=torch.long
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=30, temperature=0.8)
    text = tokenizer.decode_ids(out[0].tolist())
    print(f"Prompt: {p}\n生成: {text}\n")