#!/bin/bash
# 1. 创建并激活环境
conda create -n transformer python=3.10 -y
conda activate transformer
pip install -r ../requirements.txt

# 2. 数据预处理（生成词汇表+转.pt格式）
cd ../src
python utils.py --generate_vocab --src_data ../data/processed/train_en.txt --tgt_data ../data/processed/train_de.txt --src_vocab ../data/processed/src_vocab.txt --tgt_vocab ../data/processed/tgt_vocab.txt --vocab_size 10000
python convert_bilingual_to_pt.py --src_text ../data/processed/train_en.txt --tgt_text ../data/processed/train_de.txt --output ../data/processed/train_en-de.pt
python convert_bilingual_to_pt.py --src_text ../data/processed/validation_en.txt --tgt_text ../data/processed/validation_de.txt --output ../data/processed/validation_en-de.pt
python sample_data_gpu.py  # 生成小规模GPU数据

# 3. 固定随机种子（作业可复现要求）
export PYTHONHASHSEED=42
export numpy_random_seed=42
export torch_seed=42

# 4. 训练3个模型（基础+2个消融实验）
# 基础模型
python train.py --train_data ../data/processed/train_en-de_gpu_small.pt --val_data ../data/processed/validation_en-de_gpu_small.pt --src_vocab ../data/processed/src_vocab.txt --tgt_vocab ../data/processed/tgt_vocab.txt --d_model 128 --n_layers 2 --n_heads 4 --d_ff 512 --max_seq_len 10 --batch_size 32 --lr 3e-4 --epochs 5 --clip 1.0 --ablation_tag base --model_save_dir ../results/models
# 消融位置编码
python train.py --train_data ../data/processed/train_en-de_gpu_small.pt --val_data ../data/processed/validation_en-de_gpu_small.pt --src_vocab ../data/processed/src_vocab.txt --tgt_vocab ../data/processed/tgt_vocab.txt --d_model 128 --n_layers 2 --n_heads 4 --d_ff 512 --max_seq_len 10 --batch_size 32 --lr 3e-4 --epochs 5 --clip 1.0 --ablation_tag no_pos --ablate_pos_encoding --model_save_dir ../results/models
# 消融多头注意力
python train.py --train_data ../data/processed/train_en-de_gpu_small.pt --val_data ../data/processed/validation_en-de_gpu_small.pt --src_vocab ../data/processed/src_vocab.txt --tgt_vocab ../data/processed/tgt_vocab.txt --d_model 128 --n_layers 2 --n_heads 4 --d_ff 512 --max_seq_len 10 --batch_size 32 --lr 3e-4 --epochs 5 --clip 1.0 --ablation_tag no_multi --ablate_multi_head --model_save_dir ../results/models