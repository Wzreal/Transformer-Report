import torch
from pathlib import Path
import argparse


def convert_bilingual_to_pt(src_text_path: str, tgt_text_path: str, output_path: str):
    """
    将双语文本文件转换为PyTorch .pt格式（保存为句子对列表）
    :param src_text_path: 源语言文本路径（如train_en.txt）
    :param tgt_text_path: 目标语言文本路径（如train_de.txt）
    :param output_path: 输出.pt文件路径（如train_en-de.pt）
    """
    # 读取源语言文本（每行一句）
    with open(src_text_path, "r", encoding="utf-8") as f:
        src_sents = [line.strip() for line in f if line.strip()]

    # 读取目标语言文本（每行一句，需与源语言句子一一对应）
    with open(tgt_text_path, "r", encoding="utf-8") as f:
        tgt_sents = [line.strip() for line in f if line.strip()]

    # 确保双语句子数量一致
    min_len = min(len(src_sents), len(tgt_sents))
    src_sents = src_sents[:min_len]
    tgt_sents = tgt_sents[:min_len]

    # 保存为PyTorch格式
    torch.save((src_sents, tgt_sents), output_path)
    print(f"✅ 双语数据集转换完成！")
    print(f"📊 样本数量：{len(src_sents)} 对句子")
    print(f"💾 输出路径：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将双语文本转换为PyTorch .pt格式")
    parser.add_argument("--src_text", required=True, help="源语言文本路径（如train_en.txt）")
    parser.add_argument("--tgt_text", required=True, help="目标语言文本路径（如train_de.txt）")
    parser.add_argument("--output", required=True, help="输出.pt文件路径（如train_en-de.pt）")
    args = parser.parse_args()

    # 执行转换
    convert_bilingual_to_pt(args.src_text, args.tgt_text, args.output)