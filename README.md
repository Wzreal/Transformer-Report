Transformer 手工实现与小规模文本建模实验
本仓库为《大模型基础与应用》期中作业，包含 Transformer 完整手工实现、消融实验及实验报告，支持一键复现所有实验结果。
Transformer-Midterm/
├── src/                # 核心代码目录
│   ├── model.py        # Transformer 模型架构（Encoder/Decoder/注意力机制）
│   ├── train.py        # 训练脚本（含 3 个模型训练+消融实验）
│   ├── utils.py        # 工具函数（绘图、指标计算、词汇表生成）
│   ├── data_loader.py  # 数据加载与预处理
│   └── data_utils.py   # 数据集转 .pt 格式工具
├── scripts/            # 运行脚本目录
│   └── run.sh          # 一键训练脚本（含 3 个模型训练命令）
├── data/               # 数据集目录
│   └── processed/      # 处理后数据集（.pt 文件 + 词汇表）
│       ├── train_en-de_gpu_small.pt   # 抽样训练集
│       ├── validation_en-de_gpu_small.pt  # 抽样验证集
│       ├── src_vocab.txt  # 源语言（英语）词汇表
│       └── tgt_vocab.txt  # 目标语言（德语）词汇表
├── results/            # 实验结果目录
│   ├── models/         # 最优模型权重
│   ├── training_curves/  # 训练曲线图片
│   ├── hyperparameter_table.tsv  # 超参数表格
│   └── experiment_results.tsv    # 量化结果表格
├── requirements.txt    # 依赖库清单
├── Transformer_Report.pdf  # 实验报告（LaTeX 编译版）
└── README.md           # 本说明文件
环境配置
依赖安装
创建并激活虚拟环境：
conda create -n transformer python=3.10
conda activate transformer
安装依赖库：
pip install -r requirements.txt

实验复现步骤
一键运行（推荐）
进入 scripts 目录，执行一键运行脚本：
该脚本会自动完成 3 个模型的训练（基础模型 + 2 个消融实验），并生成所有实验结果（曲线、表格、模型权重）。

分步训练（如需单独调试）
1. 基础模型训练
python ../src/train.py --train_data ../data/processed/train_en-de_gpu_small.pt \
                      --val_data ../data/processed/validation_en-de_gpu_small.pt \
                      --src_vocab ../data/processed/src_vocab.txt \
                      --tgt_vocab ../data/processed/tgt_vocab.txt \
                      --d_model 128 --n_layers 2 --n_heads 4 --d_ff 512 \
                      --max_seq_len 10 --batch_size 32 --lr 3e-4 --epochs 5 \
                      --clip 1.0 --ablation_tag base --model_save_dir ../results/models
2. 消融位置编码模型训练
python ../src/train.py --train_data ../data/processed/train_en-de_gpu_small.pt \
                      --val_data ../data/processed/validation_en-de_gpu_small.pt \
                      --src_vocab ../data/processed/src_vocab.txt \
                      --tgt_vocab ../data/processed/tgt_vocab.txt \
                      --d_model 128 --n_layers 2 --n_heads 4 --d_ff 512 \
                      --max_seq_len 10 --batch_size 32 --lr 3e-4 --epochs 5 \
                      --clip 1.0 --ablation_tag no_pos --ablate_pos_encoding \
                      --model_save_dir ../results/models
3.  消融多头注意力模型训练
python ../src/train.py --train_data ../data/processed/train_en-de_gpu_small.pt \
                      --val_data ../data/processed/validation_en-de_gpu_small.pt \
                      --src_vocab ../data/processed/src_vocab.txt \
                      --tgt_vocab ../data/processed/tgt_vocab.txt \
                      --d_model 128 --n_layers 2 --n_heads 4 --d_ff 512 \
                      --max_seq_len 10 --batch_size 32 --lr 3e-4 --epochs 5 \
                      --clip 1.0 --ablation_tag no_multi --ablate_multi_head \
                      --model_save_dir ../results/models

实验结果
所有实验结果自动保存在 results/ 目录下：
训练曲线：training_curves/ 目录下的 3 张图片（基础模型、消融位置编码、消融多头注意力）；
量化表格：hyperparameter_table.tsv（超参数对比）、experiment_results.tsv（损失 / 困惑度对比）；
模型权重：models/ 目录下的 3 个最优模型 .pth 文件；
实验报告：根目录的 Transformer_Report.pdf（LaTeX 编译版）。

数据集说明
来源：TED Talks 双语语料（Hugging Face 链接：https://huggingface.co/datasets/ted_talks_iwslt）；
规模：1000 对平行语料（训练 900 + 验证 100，抽样后训练 90 + 验证 30）；
预处理：分词 → 生成词汇表 → 转 .pt 格式 → 固定序列长度 10。

依赖库清单（requirements.txt）：
torch==2.0.1
numpy==1.24.3
matplotlib==3.8.0
pandas==2.1.0
scikit-learn==1.3.0