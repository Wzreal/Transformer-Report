import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Union, List, Tuple


class TEDSingleLanguageDataset(Dataset):
    """单语言数据集（用于Encoder-only任务：语言建模/文本分类）"""

    def __init__(self, data_path: Union[str, Path]):
        self.data = torch.load(data_path)  # 加载预处理后的句子列表

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


class TEDBilingualDataset(Dataset):
    """双语数据集（用于Encoder-Decoder任务：机器翻译）"""

    def __init__(self, data_path: Union[str, Path]):
        self.src_sents, self.tgt_sents = torch.load(data_path)  # 加载（源语言句子，目标语言句子）

    def __len__(self) -> int:
        return len(self.src_sents)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.src_sents[idx], self.tgt_sents[idx]


def get_single_language_dataloader(
        data_dir: Union[str, Path],
        lang: str = "en",
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True
) -> DataLoader:
    """获取单语言数据加载器"""
    data_dir = Path(data_dir)
    data_path = data_dir / f"{split}_{lang}.pt"
    dataset = TEDSingleLanguageDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)


def get_bilingual_dataloader(
        data_dir: Union[str, Path],
        src_lang: str = "en",
        tgt_lang: str = "de",
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True
) -> DataLoader:
    """获取双语数据加载器"""
    data_dir = Path(data_dir)
    data_path = data_dir / f"{split}_{src_lang}-{tgt_lang}.pt"
    dataset = TEDBilingualDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: list(zip(*x)))


# 测试加载（运行此文件可验证）
if __name__ == "__main__":
    # 单语言数据加载示例（英语语言建模）
    en_dataloader = get_single_language_dataloader(
        data_dir="../data/processed",
        lang="en",
        split="train",
        batch_size=4
    )
    print("单语言数据示例（英语）：")
    for batch in en_dataloader:
        print(batch)
        break

    # 双语数据加载示例（英语→德语翻译）
    bilingual_dataloader = get_bilingual_dataloader(
        data_dir="../data/processed",
        src_lang="en",
        tgt_lang="de",
        split="train",
        batch_size=4
    )
    print("\n双语数据示例（英语→德语）：")
    for src_batch, tgt_batch in bilingual_dataloader:
        print("源语言（英语）：", src_batch)
        print("目标语言（德语）：", tgt_batch)
        break