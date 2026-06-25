"""
minimind 0.6B 古文 LLM 全流程训练（含损失记录）
1. 从头预训练  2. 指令微调  3. DPO 后训练
每个阶段自动保存损失数组至 .npy 文件
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

#字符分词器
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars) + 4
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        self.char_to_id = {ch: i + 4 for i, ch in enumerate(chars)}
        self.id_to_char = {i + 4: ch for i, ch in enumerate(chars)}
        self.id_to_char[0] = "<pad>"
        self.id_to_char[1] = "<unk>"
        self.id_to_char[2] = "<bos>"
        self.id_to_char[3] = "<eos>"

    def encode_as_ids(self, text: str) -> List[int]:
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(i, "<unk>") for i in ids)

#模型配置
@dataclass
class LMConfig:
    vocab_size: int
    dim: int = 1536
    n_layers: int = 24
    n_heads: int = 16
    max_seq_len: int = 64
    dropout: float = 0.0
    pad_token_id: int = 0

#模型定义
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype) * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq, xk: (bsz, n_heads, seqlen, head_dim)
    # freqs_cis: (seqlen, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 调整为 (1, 1, seqlen, head_dim/2) 匹配 xq_ 形状
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, config.dim, bias=False)
        self.wv = nn.Linear(config.dim, config.dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        att = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att + mask
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, xv).transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.dim)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(self, x, freqs_cis, mask=None):
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class MiniMindLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.output.weight   # 权重共享

    def _get_freqs(self, seqlen, device):
        return precompute_freqs_cis(self.config.dim // self.config.n_heads, seqlen).to(device)

    def forward(self, input_ids, attention_mask=None):
        seqlen = input_ids.shape[1]
        freqs_cis = self._get_freqs(seqlen, input_ids.device)
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        h = self.tok_emb(input_ids)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        return self.output(h)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

#数据集
class PretrainDataset(Dataset):
    def __init__(self, text, tokenizer, max_seq_len=64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ids = tokenizer.encode_as_ids(text)
        remain = len(self.ids) % max_seq_len
        if remain != 0:
            self.ids = self.ids[:-remain]

    def __len__(self):
        return len(self.ids) // self.max_seq_len

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        block = self.ids[start:start + self.max_seq_len]
        x = torch.tensor(block[:-1], dtype=torch.long)
        y = torch.tensor(block[1:], dtype=torch.long)
        return x, y

class SFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len=64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = self._process(data_list)

    def _process(self, data_list):
        processed = []
        for item in data_list:
            instr = self.tokenizer.encode_as_ids(item["instruction"])
            out = self.tokenizer.encode_as_ids(item["output"])
            sep = self.tokenizer.encode_as_ids("\n")
            ids = [self.tokenizer.bos_id] + instr + sep + out + [self.tokenizer.eos_id]
            processed.append(ids[:self.max_seq_len])
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

class DPODataset(Dataset):
    def __init__(self, data_list, tokenizer, max_prompt_len=64, max_seq_len=64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = self._process(data_list)

    def _process(self, data_list):
        processed = []
        for item in data_list:
            prompt_ids = self.tokenizer.encode_as_ids(item["prompt"])[:64]
            chosen_ids = self.tokenizer.encode_as_ids(item["chosen"])
            rejected_ids = self.tokenizer.encode_as_ids(item["rejected"])
            chosen_full = prompt_ids + chosen_ids + [self.tokenizer.eos_id]
            rejected_full = prompt_ids + rejected_ids + [self.tokenizer.eos_id]
            processed.append({
                "prompt": prompt_ids,
                "chosen": chosen_full[:self.max_seq_len],
                "rejected": rejected_full[:self.max_seq_len],
            })
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_dpo(batch):
    prompts = [torch.tensor(item["prompt"], dtype=torch.long) for item in batch]
    chosens = [torch.tensor(item["chosen"], dtype=torch.long) for item in batch]
    rejecteds = [torch.tensor(item["rejected"], dtype=torch.long) for item in batch]
    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

#训练函数
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_optimizer(model, lr):
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    except:
        return torch.optim.AdamW(model.parameters(), lr=lr)

def pretrain(config, tokenizer, text, epochs=1, batch_size=1, lr=3e-4, save_path="pretrain_0.6B.pth"):
    device = torch.device("cuda")
    model = MiniMindLM(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    dataset = PretrainDataset(text, tokenizer, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = get_optimizer(model, lr)
    scaler = GradScaler()
    model.train()

    loss_history = []                      # 记录损失
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Pretrain epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1),
                                       ignore_index=config.pad_token_id)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loss_history.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    np.save("pretrain_loss.npy", np.array(loss_history))   # 保存损失
    print(f"预训练模型已保存至 {save_path}，损失保存至 pretrain_loss.npy")

def sft(config, tokenizer, sft_data, base_model_path, epochs=1, batch_size=1, lr=1e-5, save_path="sft_0.6B.pth"):
    device = torch.device("cuda")
    model = MiniMindLM(config).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))

    dataset = SFTDataset(sft_data, tokenizer, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = get_optimizer(model, lr)
    scaler = GradScaler()
    model.train()

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"SFT epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1),
                                       ignore_index=config.pad_token_id)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loss_history.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        print(f"SFT Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    np.save("sft_loss.npy", np.array(loss_history))
    print(f"SFT 模型已保存至 {save_path}，损失保存至 sft_loss.npy")

def dpo_train(config, tokenizer, dpo_data, base_model_path, epochs=1, batch_size=1, lr=1e-6, beta=0.1, save_path="dpo_0.6B.pth"):
    device = torch.device("cuda")
    model = MiniMindLM(config).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))

    # 参考模型（仅推理）
    ref_model = MiniMindLM(config).to(device)
    ref_model.load_state_dict(torch.load(base_model_path, map_location=device))
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    dataset = DPODataset(dpo_data, tokenizer, max_seq_len=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dpo)

    optimizer = get_optimizer(model, lr)
    scaler = GradScaler()

    def ref_forward(input_ids):
        """参考模型推理，每次生成正确的频率"""
        seqlen = input_ids.shape[1]
        freqs = precompute_freqs_cis(config.dim // config.n_heads, seqlen).to(input_ids.device)
        mask = None
        h = ref_model.tok_emb(input_ids)
        for layer in ref_model.layers:
            h = layer(h, freqs, mask)
        h = ref_model.norm(h)
        return ref_model.output(h)

    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"DPO epoch {epoch+1}")
        for batch in pbar:
            prompts = batch["prompt"]
            chosens = batch["chosen"]
            rejecteds = batch["rejected"]

            chosen_logps = rejected_logps = ref_chosen_logps = ref_rejected_logps = 0.0
            optimizer.zero_grad()
            for i in range(len(prompts)):
                chosen_full = chosens[i].unsqueeze(0).to(device)
                rejected_full = rejecteds[i].unsqueeze(0).to(device)
                prompt_len = len(prompts[i])

                with torch.no_grad():
                    with autocast():
                        ref_logits_c = ref_forward(chosen_full)
                        ref_logits_r = ref_forward(rejected_full)

                with autocast():
                    logits_c = model(chosen_full)
                    logits_r = model(rejected_full)

                    logp_c = F.log_softmax(logits_c, dim=-1)
                    logp_r = F.log_softmax(logits_r, dim=-1)
                    ref_logp_c = F.log_softmax(ref_logits_c, dim=-1)
                    ref_logp_r = F.log_softmax(ref_logits_r, dim=-1)

                    resp_c = logp_c[0, prompt_len-1:-1].gather(1, chosen_full[:, prompt_len:].T).sum()
                    resp_r = logp_r[0, prompt_len-1:-1].gather(1, rejected_full[:, prompt_len:].T).sum()
                    ref_resp_c = ref_logp_c[0, prompt_len-1:-1].gather(1, chosen_full[:, prompt_len:].T).sum()
                    ref_resp_r = ref_logp_r[0, prompt_len-1:-1].gather(1, rejected_full[:, prompt_len:].T).sum()

                chosen_logps += resp_c
                rejected_logps += resp_r
                ref_chosen_logps += ref_resp_c
                ref_rejected_logps += ref_resp_r

            chosen_logps /= len(prompts)
            rejected_logps /= len(prompts)
            ref_chosen_logps /= len(prompts)
            ref_rejected_logps /= len(prompts)

            log_ratio_chosen = chosen_logps - ref_chosen_logps
            log_ratio_rejected = rejected_logps - ref_rejected_logps
            loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loss_history.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        print(f"DPO Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    np.save("dpo_loss.npy", np.array(loss_history))
    print(f"DPO 模型已保存至 {save_path}，损失保存至 dpo_loss.npy")

#主程序
if __name__ == "__main__":
    # 1. 准备古文文本
    base_text = """子曰：学而时习之，不亦说乎。有朋自远方来，不亦乐乎。人不知而不愠，不亦君子乎。
    子曰：温故而知新，可以为师矣。
    子曰：学而不思则罔，思而不学则殆。
    孟子曰：天时不如地利，地利不如人和。
    孟子曰：得道者多助，失道者寡助。
    孟子曰：生于忧患，死于安乐。
    """ * 500

    # 2. 分词器
    tokenizer = CharTokenizer(base_text)
    print(f"字符词表大小: {tokenizer.vocab_size}")

    # 3. 模型配置
    config = LMConfig(
        vocab_size=tokenizer.vocab_size,
        dim=1536,
        n_layers=24,
        n_heads=16,
        max_seq_len=64,
        pad_token_id=tokenizer.pad_id
    )

    # 4. 从头预训练
    print(">>> 开始从头预训练...")
    pretrain(config, tokenizer, base_text, epochs=1, batch_size=1, lr=3e-4, save_path="pretrain_0.6B.pth")

    # 5. SFT 微调数据
    sft_data = [
        {"instruction": "子曰：", "output": "学而时习之，不亦说乎。"},
        {"instruction": "子曰：温故而知新，", "output": "可以为师矣。"},
        {"instruction": "孟子曰：天时不如地利，", "output": "地利不如人和。"},
        {"instruction": "孟子曰：生于忧患，", "output": "死于安乐。"},
    ] * 30
    print(">>> 开始 SFT 微调...")
    sft(config, tokenizer, sft_data, "pretrain_0.6B.pth", epochs=1, batch_size=1, lr=1e-5, save_path="sft_0.6B.pth")

    # 6. DPO 偏好数据
    dpo_data = [
        {"prompt": "子曰：", "chosen": "学而时习之，不亦说乎。", "rejected": "学习是一件快乐的事情。"},
        {"prompt": "孟子曰：", "chosen": "得道者多助，失道者寡助。", "rejected": "朋友多就好办事。"},
    ] * 20
    print(">>> 开始 DPO 后训练...")
    dpo_train(config, tokenizer, dpo_data, "sft_0.6B.pth", epochs=1, batch_size=1, lr=1e-6, save_path="dpo_0.6B.pth")

    for f in ["pretrain_loss", "sft_loss", "dpo_loss"]:
        arr = np.load(f"{f}.npy")
        print(f"{f}: steps={len(arr)}, start={arr[0]:.3f}, end={arr[-1]:.3f}")

    print("全部训练完成！损失文件：pretrain_loss.npy, sft_loss.npy, dpo_loss.npy")

#图像生成
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    phases = [
        ("预训练 (Pretrain)", "pretrain_loss.npy"),
        ("指令微调 (SFT)", "sft_loss.npy"),
        ("偏好后训练 (DPO)", "dpo_loss.npy")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, fname) in zip(axes, phases):
        loss = np.load(fname)
        ax.plot(loss, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("步数 (Step)")
        ax.set_ylabel("损失 (Loss)")
        ax.grid(True, alpha=0.3)

        # 自动标注早期快速下降阶段（Aha Moment）
        if len(loss) > 0:
            early_idx = int(len(loss) * 0.2)  # 取前20%步
            min_idx = np.argmin(loss[:early_idx]) if early_idx > 0 else 0
            ax.annotate('Aha Moment', (min_idx, loss[min_idx]),
                        xytext=(min_idx + 10, loss[min_idx] * 1.5),
                        arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig("training_loss_curves.png", dpi=150)
    print("损失曲线已保存为 training_loss_curves.png")