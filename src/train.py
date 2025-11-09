import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import math
import csv
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List

# -------------------------- 1. 数据集类（适配双语对格式）--------------------------
class BilingualTokenDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = torch.load(data_path)  # 格式：[(en_tensor, de_tensor), ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (源语言张量, 目标语言张量)

# -------------------------- 2. 核心工具函数 --------------------------
def load_vocab(vocab_path: str) -> Dict[str, int]:
    """加载词汇表"""
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    return vocab

def generate_masks(src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成Padding Mask和Future Mask"""
    batch_size, src_seq_len = src.size()
    batch_size, tgt_seq_len = tgt.size()

    # Padding Mask
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, src_seq_len)
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # (B, 1, tgt_seq_len, 1)

    # Future Mask
    future_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=src.device), diagonal=1)
    tgt_mask = tgt_pad_mask & (future_mask == 0)  # 合并掩码

    return src_mask, tgt_mask

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_perplexities: List[float],
    val_perplexities: List[float],
    ablation_tag: str,
    save_dir: str
):
    """绘制训练/验证曲线（损失+困惑度），保存为高清图片（作业直接插入）"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False    # 负号支持
    plt.rcParams['figure.dpi'] = 300              # 高清图
    plt.rcParams['savefig.dpi'] = 300

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 损失曲线
    ax1.plot(epochs, train_losses, 'o-', color='#2E86AB', label='训练损失', linewidth=2.5, markersize=6)
    ax1.plot(epochs, val_losses, 's-', color='#A23B72', label='验证损失', linewidth=2.5, markersize=6)
    ax1.set_xlabel('训练轮数（Epoch）', fontsize=12)
    ax1.set_ylabel('损失值（Loss）', fontsize=12)
    ax1.set_title(f'{ablation_tag} - 训练/验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # 2. 困惑度曲线
    ax2.plot(epochs, train_perplexities, 'o-', color='#F18F01', label='训练困惑度', linewidth=2.5, markersize=6)
    ax2.plot(epochs, val_perplexities, 's-', color='#C73E1D', label='验证困惑度', linewidth=2.5, markersize=6)
    ax2.set_xlabel('训练轮数（Epoch）', fontsize=12)
    ax2.set_ylabel('困惑度（Perplexity）', fontsize=12)
    ax2.set_title(f'{ablation_tag} - 训练/验证困惑度曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    # 保存图片
    save_path = os.path.join(save_dir, f'training_curves_{ablation_tag}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"📊 曲线已保存至：{save_path}")

def save_hyperparameter_table(hyperparams: Dict[str, any], save_path: str):
    """生成作业要求的超参数表格（CSV格式，可直接复制到Word）"""
    # 检查文件是否存在，不存在则创建并写入表头
    file_exists = os.path.exists(save_path)
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')  # 制表符分隔，方便Word粘贴
        if not file_exists:
            # 表头（对应作业表3的列）
            writer.writerow(['模型标签', '嵌入维度', '注意力头数', 'FFN维度', '层数', '批次大小', '学习率', '优化器', '学习率调度器', '训练轮数'])
        # 写入当前模型的超参数
        writer.writerow([
            hyperparams['ablation_tag'],
            hyperparams['d_model'],
            hyperparams['n_heads'],
            hyperparams['d_ff'],
            hyperparams['n_layers'],
            hyperparams['batch_size'],
            hyperparams['lr'],
            hyperparams['optimizer'],
            hyperparams['scheduler'],
            hyperparams['epochs']
        ])
    print(f"📋 超参数已记录至：{save_path}")

def save_experiment_results(results: Dict[str, any], save_path: str):
    """保存量化结果（损失+困惑度），用于模型对比"""
    file_exists = os.path.exists(save_path)
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(['模型标签', '最终训练损失', '最终验证损失', '最终训练困惑度', '最终验证困惑度', '最优验证损失', '最优验证困惑度'])
        writer.writerow([
            results['ablation_tag'],
            round(results['final_train_loss'], 4),
            round(results['final_val_loss'], 4),
            round(results['final_train_perp'], 2),
            round(results['final_val_perp'], 2),
            round(results['best_val_loss'], 4),
            round(results['best_val_perp'], 2)
        ])
    print(f"📈 实验结果已记录至：{save_path}")

def generate_translation_samples(
    model: torch.nn.Module,
    test_data: BilingualTokenDataset,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: torch.device,
    num_samples: int = 3
) -> List[Dict[str, str]]:
    model.eval()
    # 词汇表反向映射（index→word）
    src_idx2word = {idx: word for word, idx in src_vocab.items()}
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}

    samples = []
    with torch.no_grad():
        for i in range(num_samples):
            src_tensor, tgt_true_tensor = test_data[i]
            src_tensor = src_tensor.unsqueeze(0).to(device)  # (1, seq_len)

            # 贪心解码生成预测
            tgt_pred_tensor = torch.tensor([tgt_vocab['<sos>']], device=device).unsqueeze(0)  # 初始化<SOS>
            for _ in range(len(tgt_true_tensor)-1):
                if tgt_pred_tensor[0, -1].item() == tgt_vocab['<eos>']:
                    break  # 遇到<eos>停止
                output = model(src_tensor, tgt_pred_tensor)
                next_token = output.argmax(-1)[:, -1].unsqueeze(1)
                tgt_pred_tensor = torch.cat([tgt_pred_tensor, next_token], dim=1)

            # 转换为文字（过滤<PAD>、<SOS>、<EOS>）
            def tensor_to_sentence(tensor, idx2word):
                return ' '.join([
                    idx2word[idx.item()] for idx in tensor
                    # 修复：过滤的是单词，不是索引！
                    if idx2word[idx.item()] not in ['<pad>', '<sos>', '<eos>']
                ])

            src_sent = tensor_to_sentence(src_tensor[0], src_idx2word)
            tgt_true_sent = tensor_to_sentence(tgt_true_tensor, tgt_idx2word)
            tgt_pred_sent = tensor_to_sentence(tgt_pred_tensor[0], tgt_idx2word)

            samples.append({
                'source': src_sent,
                'prediction': tgt_pred_sent,
                'ground_truth': tgt_true_sent
            })
    return samples

# -------------------------- 3. Transformer核心组件 --------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=2, dropout=0.1, ablate_multi_head=False):
        super().__init__()
        self.n_heads = 1 if ablate_multi_head else n_heads
        self.d_k = d_model // self.n_heads
        # 新增：显式绑定 d_model 为实例属性（关键修复！）
        self.d_model = d_model  # 这一行必须添加
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask=None):
        residual = Q
        batch_size = Q.size(0)
        # 多头拆分
        Q = self.w_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 注意力计算
        attn_output, _ = self.attention(Q, K, V, mask)
        # 多头合并（使用 self.d_model，已通过 __init__ 定义）
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.dropout(self.w_o(attn_output))
        return self.layer_norm(residual + attn_output), None

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.w_2(self.dropout(self.relu(self.w_1(x))))
        return self.layer_norm(residual + output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_seq_len=10, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 生成位置编码
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=2, d_ff=512, dropout=0.1, ablate_multi_head=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, ablate_multi_head)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x, _ = self.self_attn(x, x, x, mask)
        x = self.ffn(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=2, d_ff=512, dropout=0.1, ablate_multi_head=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, ablate_multi_head)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, ablate_multi_head)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x, _ = self.self_attn(x, x, x, tgt_mask)
        x, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.ffn(x)
        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model=128, n_layers=2, n_heads=2, d_ff=512, max_seq_len=10, dropout=0.1, ablate_multi_head=False):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, ablate_multi_head) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model=128, n_layers=2, n_heads=2, d_ff=512, max_seq_len=10, dropout=0.1, ablate_pos_encoding=False, ablate_multi_head=False):
        super().__init__()
        self.ablate_pos_encoding = ablate_pos_encoding
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout) if not ablate_pos_encoding else None
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout, ablate_multi_head) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        x = self.embedding(tgt)
        if not self.ablate_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_layers=2, n_heads=2, d_ff=512, max_seq_len=10, dropout=0.1, ablate_pos_encoding=False, ablate_multi_head=False):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout, ablate_multi_head)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout, ablate_pos_encoding, ablate_multi_head)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = generate_masks(src, tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        output = self.fc_out(dec_output)
        return output

# -------------------------- 4. 训练主函数（核心）--------------------------
def train_transformer(args):
    # 1. 设备初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备：{device}")

    # 2. 数据加载
    src_vocab = load_vocab(args.src_vocab)
    tgt_vocab = load_vocab(args.tgt_vocab)
    train_dataset = BilingualTokenDataset(args.train_data)
    val_dataset = BilingualTokenDataset(args.val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True if device.type == "cuda" else False
    )

    print(f"📚 数据加载完成：")
    print(f"   - 源语言词汇表：{len(src_vocab)}词 | 目标语言词汇表：{len(tgt_vocab)}词")
    print(f"   - 训练集：{len(train_dataset)}样本 | 验证集：{len(val_dataset)}样本")
    print(f"   - 训练批次：{len(train_loader)} | 验证批次：{len(val_loader)}")

    # 3. 模型初始化
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        ablate_pos_encoding=args.ablate_pos_encoding,
        ablate_multi_head=args.ablate_multi_head
    ).to(device)

    # 4. 训练配置
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])  # 忽略<PAD>
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # 5. 训练记录初始化
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    best_val_loss = float('inf')
    os.makedirs(args.model_save_dir, exist_ok=True)

    # 6. 超参数记录（作业表格用）
    hyperparams = {
        'ablation_tag': args.ablation_tag,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'n_layers': args.n_layers,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'epochs': args.epochs
    }
    save_hyperparameter_table(hyperparams, "../results/hyperparameter_table.tsv")

    # 7. 训练循环
    print(f"\n🚀 开始训练（{args.ablation_tag}）：共{args.epochs}个Epoch")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_total_loss = 0.0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # 输入去掉<eos>
            loss = criterion(output.reshape(-1, len(tgt_vocab)), tgt[:, 1:].reshape(-1))  # 标签去掉<sos>
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_total_loss += loss.item()

            # 批次日志
            if (batch_idx + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        # 验证阶段
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, len(tgt_vocab)), tgt[:, 1:].reshape(-1))
                val_total_loss += loss.item()

        # 计算指标
        train_avg_loss = train_total_loss / len(train_loader)
        val_avg_loss = val_total_loss / len(val_loader)
        train_perp = torch.exp(torch.tensor(train_avg_loss, device=device)).item()
        val_perp = torch.exp(torch.tensor(val_avg_loss, device=device)).item()

        # 记录指标
        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)
        train_perplexities.append(train_perp)
        val_perplexities.append(val_perp)

        # 学习率调度
        scheduler.step(val_avg_loss)

        # 保存最优模型
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            model_path = os.path.join(args.model_save_dir, f"best_model_{args.ablation_tag}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"📥 保存最优模型：{model_path}（验证损失：{best_val_loss:.4f}）")

        # Epoch日志
        print(f"\n==================================================")
        print(f"Epoch {epoch+1}/{args.epochs} | {args.ablation_tag}")
        print(f"训练损失：{train_avg_loss:.4f} | 训练困惑度：{train_perp:.2f}")
        print(f"验证损失：{val_avg_loss:.4f} | 验证困惑度：{val_perp:.2f}")
        print(f"当前学习率：{optimizer.param_groups[0]['lr']:.6f}")
        print(f"==================================================\n")

    # 8. 训练后处理（作业核心要求）
    # 8.1 绘制并保存曲线
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_perplexities=train_perplexities,
        val_perplexities=val_perplexities,
        ablation_tag=args.ablation_tag,
        save_dir=args.model_save_dir
    )

    # 8.2 保存实验结果（量化对比用）
    experiment_results = {
        'ablation_tag': args.ablation_tag,
        'final_train_loss': train_avg_loss,
        'final_val_loss': val_avg_loss,
        'final_train_perp': train_perp,
        'final_val_perp': val_perp,
        'best_val_loss': best_val_loss,
        'best_val_perp': torch.exp(torch.tensor(best_val_loss)).item()
    }
    save_experiment_results(experiment_results, "../results/experiment_results.tsv")

    # 8.3 生成翻译示例（定性分析用）
    print(f"\n📝 生成翻译示例（{args.ablation_tag}）：")
    translation_samples = generate_translation_samples(model, val_dataset, src_vocab, tgt_vocab, device, num_samples=3)
    for i, sample in enumerate(translation_samples, 1):
        print(f"\n示例 {i}：")
        print(f"原文（英语）：{sample['source']}")
        print(f"预测（德语）：{sample['prediction']}")
        print(f"真实（德语）：{sample['ground_truth']}")

    print(f"\n🎉 训练完成！所有结果已保存至 ../results/")

# -------------------------- 5. 命令行参数解析 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer训练脚本（满足作业要求：曲线+表格+翻译示例）")
    # 数据路径
    parser.add_argument('--train_data', required=True, help="训练集路径（.pt）")
    parser.add_argument('--val_data', required=True, help="验证集路径（.pt）")
    parser.add_argument('--src_vocab', required=True, help="源语言词汇表（.txt）")
    parser.add_argument('--tgt_vocab', required=True, help="目标语言词汇表（.txt）")
    # 模型参数（匹配作业表3）
    parser.add_argument('--d_model', type=int, default=128, help="嵌入维度（作业要求128）")
    parser.add_argument('--n_layers', type=int, default=2, help="层数（作业要求2）")
    parser.add_argument('--n_heads', type=int, default=4, help="注意力头数（作业要求4）")
    parser.add_argument('--d_ff', type=int, default=512, help="FFN维度（作业要求512）")
    parser.add_argument('--max_seq_len', type=int, default=10, help="最大序列长度")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout率")
    # 训练参数（匹配作业要求）
    parser.add_argument('--batch_size', type=int, default=32, help="批次大小（作业要求32）")
    parser.add_argument('--lr', type=float, default=3e-4, help="学习率（作业要求3e-4）")
    parser.add_argument('--epochs', type=int, default=5, help="训练轮数")
    parser.add_argument('--clip', type=float, default=1.0, help="梯度裁剪阈值")
    # 实验参数
    parser.add_argument('--model_save_dir', required=True, help="模型保存目录")
    parser.add_argument('--ablation_tag', required=True, help="模型标签（如base、no_pos、no_multi）")
    parser.add_argument('--ablate_pos_encoding', action='store_true', help="消融位置编码（作业要求）")
    parser.add_argument('--ablate_multi_head', action='store_true', help="消融多头注意力（作业要求）")
    return parser.parse_args()

# -------------------------- 6. 主函数入口 --------------------------
if __name__ == "__main__":
    args = parse_args()
    train_transformer(args)