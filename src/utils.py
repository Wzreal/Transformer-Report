import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
from collections import Counter
from typing import Tuple, List, Dict, Optional


def calculate_perplexity(loss: float) -> float:
    """计算困惑度（Perplexity）：exp(loss)"""
    return torch.exp(torch.tensor(loss)).item()


def plot_training_curves(
        train_losses: List[float], val_losses: List[float],
        train_perplexities: List[float], val_perplexities: List[float],
        save_path: str = "../results/training_curves.png"
):
    """绘制训练/验证loss和困惑度曲线"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", color="red", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # 困惑度曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_perplexities, label="Train Perplexity", color="blue", marker="o")
    plt.plot(epochs, val_perplexities, label="Val Perplexity", color="red", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training and Validation Perplexity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 训练曲线已保存至：{save_path}")


def save_experiment_results(
        hyperparams: Dict, train_loss: float, val_loss: float,
        train_perp: float, val_perp: float, ablation_tag: str = "base",
        save_path: str = "../results/experiment_results.csv"
):
    """保存实验结果到CSV表格"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "ablation_tag": ablation_tag,
        "d_model": hyperparams["d_model"],
        "n_layers": hyperparams["n_layers"],
        "n_heads": hyperparams["n_heads"],
        "lr": hyperparams["lr"],
        "batch_size": hyperparams["batch_size"],
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_perplexity": train_perp,
        "val_perplexity": val_perp
    }

    # 若文件不存在，创建表头；否则追加
    if not Path(save_path).exists():
        df = pd.DataFrame(columns=result_dict.keys())
        df.to_csv(save_path, index=False)

    df = pd.read_csv(save_path)
    df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    df.to_csv(save_path, index=False)
    print(f"📋 实验结果已保存至：{save_path}")


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """加载词汇表（word→idx 和 idx→word）"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        words = f.read().splitlines()
    word2idx = {word: idx for idx, word in enumerate(words)}
    idx2word = {idx: word for idx, word in enumerate(words)}
    return word2idx, idx2word


def tokenize(text: List[str], word2idx: Dict[str, int], max_seq_len: int) -> torch.Tensor:
    """文本tokenize（转换为token ID）"""
    tokenized = []
    for sent in text:
        tokens = sent.split()[:max_seq_len]  # 截断长句子
        # 转换为ID，未登录词用<unk>（索引1），padding用0
        token_ids = [word2idx.get(word, 1) for word in tokens]
        # padding到max_seq_len
        if len(token_ids) < max_seq_len:
            token_ids += [0] * (max_seq_len - len(token_ids))
        tokenized.append(token_ids)
    return torch.tensor(tokenized, dtype=torch.long)


def build_vocab(text_paths: List[str], vocab_size: int) -> List[str]:
    """从文本文件构建词汇表（取top N高频词，含特殊符号）"""
    counter = Counter()
    # 读取所有文本并统计词频
    for path in text_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                counter.update(words)
    # 保留top N高频词，加上特殊符号（<pad>:0, <unk>:1, <sos>:2, <eos>:3）
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    # 确保词汇表总大小为vocab_size（特殊符号+高频词）
    top_k = vocab_size - len(special_tokens)
    top_words = [word for word, _ in counter.most_common(top_k)]
    vocab = special_tokens + top_words
    return vocab


def main():
    """命令行入口：处理 --generate_vocab 参数生成词汇表"""
    parser = argparse.ArgumentParser(description="工具函数：词汇表生成、指标计算、可视化等")
    # 词汇表生成相关参数
    parser.add_argument("--generate_vocab", action="store_true", help="生成词汇表开关（必选）")
    parser.add_argument("--src_data", type=str, required=False, help="源语言文本路径（如英语train_en.txt）")
    parser.add_argument("--tgt_data", type=str, required=False, help="目标语言文本路径（如德语train_de.txt）")
    parser.add_argument("--src_vocab", type=str, required=False, help="源语言词汇表保存路径（如src_vocab.txt）")
    parser.add_argument("--tgt_vocab", type=str, required=False, help="目标语言词汇表保存路径（如tgt_vocab.txt）")
    parser.add_argument("--vocab_size", type=int, default=10000, help="词汇表大小（包含4个特殊符号，默认10000）")

    args = parser.parse_args()

    # 执行词汇表生成
    if args.generate_vocab:
        # 检查必要参数是否齐全
        required_args = [args.src_data, args.tgt_data, args.src_vocab, args.tgt_vocab]
        if not all(required_args):
            raise ValueError("❌ 生成词汇表必须指定以下参数：--src_data、--tgt_data、--src_vocab、--tgt_vocab")

        # 生成源语言词汇表
        print(f"⏳ 正在生成源语言词汇表（保存路径：{args.src_vocab}）...")
        src_vocab = build_vocab(text_paths=[args.src_data], vocab_size=args.vocab_size)
        with open(args.src_vocab, "w", encoding="utf-8") as f:
            f.write("\n".join(src_vocab))

        # 生成目标语言词汇表
        print(f"⏳ 正在生成目标语言词汇表（保存路径：{args.tgt_vocab}）...")
        tgt_vocab = build_vocab(text_paths=[args.tgt_data], vocab_size=args.vocab_size)
        with open(args.tgt_vocab, "w", encoding="utf-8") as f:
            f.write("\n".join(tgt_vocab))

        # 输出结果提示
        print(f"\n✅ 词汇表生成完成！")
        print(f"📚 源语言词汇表：{len(src_vocab)} 个词（特殊符号4个 + 高频词 {len(src_vocab) - 4} 个）")
        print(f"📚 目标语言词汇表：{len(tgt_vocab)} 个词（特殊符号4个 + 高频词 {len(tgt_vocab) - 4} 个）")
        print(f"💾 源语言词汇表路径：{args.src_vocab}")
        print(f"💾 目标语言词汇表路径：{args.tgt_vocab}")


if __name__ == "__main__":
    main()