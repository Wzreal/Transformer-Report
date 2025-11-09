#!/bin/bash
# 环境配置（首次运行）
# conda create -n transformer python=3.10
# pip install torch==2.0.1 pandas==2.1.0 matplotlib==3.7.2 numpy==1.24.3

# 生成词汇表（简化版：统计双语数据的词频，取Top 10000）
python ../src/utils.py --generate_vocab \
  --src_data ../data/processed/train_en.txt \
  --tgt_data ../data/processed/train_de.txt \
  --src_vocab ../data/processed/src_vocab.txt \
  --tgt_vocab ../data/processed/tgt_vocab.txt \
  --vocab_size 10000

# 基础模型训练（Encoder+Decoder，全功能）
python ../src/train.py \
  --train_data ../data/processed/train_en-de.pt \
  --val_data ../data/processed/validation_en-de.pt \
  --src_vocab ../data/processed/src_vocab.txt \
  --tgt_vocab ../data/processed/tgt_vocab.txt \
  --d_model 256 \
  --n_layers 3 \
  --n_heads 4 \
  --d_ff 1024 \
  --max_seq_len 64 \
  --batch_size 32 \
  --lr 3e-4 \
  --epochs 20 \
  --clip 1.0 \
  --ablation_tag base \
  --model_save_dir ../results/models

# 消融实验1：关闭位置编码
python ../src/train.py \
  --train_data ../data/processed/train_en-de.pt \
  --val_data ../data/processed/validation_en-de.pt \
  --src_vocab ../data/processed/src_vocab.txt \
  --tgt_vocab ../data/processed/tgt_vocab.txt \
  --d_model 256 \
  --n_layers 3 \
  --n_heads 4 \
  --d_ff 1024 \
  --max_seq_len 64 \
  --batch_size 32 \
  --lr 3e-4 \
  --epochs 20 \
  --clip 1.0 \
  --ablation_tag no_pos_encoding \
  --ablate_pos_encoding \
  --model_save_dir ../results/models

# 消融实验2：关闭多头注意力（强制单头）
python ../src/train.py \
  --train_data ../data/processed/train_en-de.pt \
  --val_data ../data/processed/validation_en-de.pt \
  --src_vocab ../data/processed/src_vocab.txt \
  --tgt_vocab ../data/processed/tgt_vocab.txt \
  --d_model 256 \
  --n_layers 3 \
  --n_heads 4 \
  --d_ff 1024 \
  --max_seq_len 64 \
  --batch_size 32 \
  --lr 3e-4 \
  --epochs 20 \
  --clip 1.0 \
  --ablation_tag no_multi_head \
  --ablate_multi_head \
  --model_save_dir ../results/models