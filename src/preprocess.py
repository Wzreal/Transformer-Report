import re
import random
import torch
from pathlib import Path
from typing import Tuple, Optional, Union
import argparse

# 设置随机种子（保证结果可复现）
random.seed(42)
torch.manual_seed(42)


class TEDDatasetPreprocessor:
    def __init__(self,
                 src_lang: str = "en",  # 源语言（en=英语，de=德语）
                 tgt_lang: Optional[str] = None,  # 目标语言（双语配对时设置，如de=德语→英语翻译）
                 min_sent_len: int = 5,  # 过滤短句子（少于5个词）
                 max_sent_len: int = 128,  # 截断长句子（最多128个词）
                 train_ratio: float = 0.8,  # 训练集比例
                 val_ratio: float = 0.1):  # 验证集比例（测试集=1-训练-验证）
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.min_len = min_sent_len
        self.max_len = max_sent_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

    def clean_text(self, text: str) -> str:
        """清洗单句文本：剔除HTML标签、注释、特殊符号，标准化格式"""
        # 1. 剔除HTML标签（<doc>、<url>等）和CDATA注释
        text = re.sub(r'<[^>]+>', '', text)  # 移除所有<>包裹的标签
        text = re.sub(r'<!\[CDATA\[|\]\]>', '', text)  # 移除CDATA注释
        # 2. 剔除特殊符号和多余空格
        text = re.sub(r'[^\w\s\.,!\?;\-]', '', text)  # 保留字母、数字、空格和常见标点
        text = re.sub(r'\s+', ' ', text).strip()  # 合并多个空格为一个，去除首尾空格
        # 3. 小写化（可选，根据任务调整，语言建模建议保留大小写）
        # text = text.lower()
        return text

    def split_sentences(self, text: str, lang: str) -> list[str]:
        """按语言拆分句子（处理英语/德语标点差异）"""
        if lang == "en":
            # 英语句子结束标点：. ! ? ;
            sentence_endings = re.compile(r'(?<=[.!?;])\s+')
        elif lang == "de":
            # 德语句子结束标点：. ! ? ; （注意德语标点后空格要求，这里统一拆分）
            sentence_endings = re.compile(r'(?<=[.!?;])\s+')
        else:
            raise ValueError(f"不支持的语言：{lang}（仅支持en/de）")

        sentences = sentence_endings.split(text)
        # 过滤空句子和过短/过长句子
        filtered = []
        for sent in sentences:
            sent = self.clean_text(sent)
            word_count = len(sent.split())
            if self.min_len <= word_count <= self.max_len:
                filtered.append(sent)
        return filtered

    def load_single_language(self, file_path: Union[str, Path]) -> list[str]:
        """加载单语言TED文件（用于Encoder-only任务：语言建模/文本分类）"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        # 读取文件所有内容
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 拆分句子并清洗
        sentences = self.split_sentences(raw_text, self.src_lang)
        print(f"✅ 加载{self.src_lang}文件完成：共{len(sentences)}个有效句子")
        return sentences

    def load_bilingual_pair(self, src_file: Union[str, Path], tgt_file: Union[str, Path]) -> list[Tuple[str, str]]:
        """加载双语配对文件（用于Encoder-Decoder任务：机器翻译）"""
        if not self.tgt_lang:
            raise ValueError("双语模式需设置tgt_lang（如--tgt_lang de）")

        # 加载源语言和目标语言句子
        src_sentences = self.load_single_language(src_file)
        tgt_sentences = self.load_single_language(tgt_file)

        # 对齐句子（仅保留长度一致的配对，避免错位）
        min_len = min(len(src_sentences), len(tgt_sentences))
        paired_sentences = list(zip(src_sentences[:min_len], tgt_sentences[:min_len]))
        print(f"✅ 双语配对完成：共{len(paired_sentences)}个有效翻译对")
        return paired_sentences

    def split_dataset(self, data: Union[list[str], list[Tuple[str, str]]]) -> Tuple[list, list, list]:
        """划分训练集、验证集、测试集（按比例随机拆分）"""
        random.shuffle(data)  # 随机打乱
        total = len(data)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)

        train = data[:train_size]
        val = data[train_size:train_size + val_size]
        test = data[train_size + val_size:]

        print(f"📊 数据集划分完成：")
        print(f" - 训练集：{len(train)} 样本")
        print(f" - 验证集：{len(val)} 样本")
        print(f" - 测试集：{len(test)} 样本")
        return train, val, test

    def save_text_format(self, data: Union[list[str], list[Tuple[str, str]]], save_path: Path, split_name: str):
        """保存为纯文本格式（一行一个样本，便于查看和后续处理）"""
        save_path.mkdir(parents=True, exist_ok=True)

        if self.tgt_lang:
            # 双语模式：保存为src-tgt配对文件（每行格式：源语言句子\t目标语言句子）
            file_path = save_path / f"{split_name}_{self.src_lang}-{self.tgt_lang}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                for src_sent, tgt_sent in data:
                    f.write(f"{src_sent}\t{tgt_sent}\n")
        else:
            # 单语言模式：保存为单文件（每行一个句子）
            file_path = save_path / f"{split_name}_{self.src_lang}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                for sent in data:
                    f.write(f"{sent}\n")
        print(f"💾 已保存{split_name}集文本文件：{file_path}")

    def save_torch_format(self, data: Union[list[str], list[Tuple[str, str]]], save_path: Path, split_name: str):
        """保存为PyTorch张量格式（便于直接加载训练）"""
        save_path.mkdir(parents=True, exist_ok=True)

        if self.tgt_lang:
            # 双语模式：保存为(src_tensor, tgt_tensor)（这里先保存句子索引，后续结合tokenizer）
            src_sents = [src for src, tgt in data]
            tgt_sents = [tgt for src, tgt in data]
            torch.save((src_sents, tgt_sents), save_path / f"{split_name}_{self.src_lang}-{self.tgt_lang}.pt")
        else:
            # 单语言模式：保存为句子列表张量
            torch.save(data, save_path / f"{split_name}_{self.src_lang}.pt")
        print(
            f"💾 已保存{split_name}集PyTorch文件：{save_path / f'{split_name}_{self.src_lang}{"-" + self.tgt_lang if self.tgt_lang else ""}.pt'}")

    def run(self,
            src_file: str,
            tgt_file: Optional[str] = None,
            output_dir: str = "../data/processed"):
        """执行完整预处理流程：加载→清洗→拆分→保存"""
        output_dir = Path(output_dir)
        print(f"🚀 开始预处理（源语言：{self.src_lang}，目标语言：{self.tgt_lang or '无'}）")

        # 1. 加载数据
        if tgt_file:
            data = self.load_bilingual_pair(src_file, tgt_file)
        else:
            data = self.load_single_language(src_file)

        # 2. 划分数据集
        train_data, val_data, test_data = self.split_dataset(data)

        # 3. 保存文件（同时保存文本格式和PyTorch格式，适配不同训练需求）
        self.save_text_format(train_data, output_dir, "train")
        self.save_text_format(val_data, output_dir, "validation")
        self.save_text_format(test_data, output_dir, "test")

        self.save_torch_format(train_data, output_dir, "train")
        self.save_torch_format(val_data, output_dir, "validation")
        self.save_torch_format(test_data, output_dir, "test")

        print(f"🎉 预处理全部完成！结果保存在：{output_dir}")


if __name__ == "__main__":
    # 解析命令行参数（方便作业中灵活配置）
    parser = argparse.ArgumentParser(description="TED数据集预处理（支持单语言/双语）")
    parser.add_argument("--src_file", required=True, help="源语言文件路径（英语/德语TED文件）")
    parser.add_argument("--tgt_file", default=None, help="目标语言文件路径（双语配对时使用，如翻译任务）")
    parser.add_argument("--src_lang", default="en", choices=["en", "de"], help="源语言（en=英语，de=德语）")
    parser.add_argument("--tgt_lang", default=None, choices=["en", "de"], help="目标语言（双语时设置，如de=德语→英语）")
    parser.add_argument("--output_dir", default="../data/processed", help="处理后数据集保存目录")
    parser.add_argument("--min_sent_len", type=int, default=5, help="最小句子长度（词数）")
    parser.add_argument("--max_sent_len", type=int, default=128, help="最大句子长度（词数）")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")

    args = parser.parse_args()

    # 初始化预处理工具并运行
    preprocessor = TEDDatasetPreprocessor(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        min_sent_len=args.min_sent_len,
        max_sent_len=args.max_sent_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    preprocessor.run(
        src_file=args.src_file,
        tgt_file=args.tgt_file,
        output_dir=args.output_dir
    )