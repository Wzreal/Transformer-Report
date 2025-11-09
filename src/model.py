import torch
import torch.nn as nn
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, n_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len_q, d_v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
            self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = Q  # 残差连接
        batch_size = Q.size(0)

        # 线性投影 + 多头拆分 (batch_size, seq_len, d_model) → (batch_size, n_heads, seq_len, d_k)
        Q = self.w_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 多头合并 (batch_size, n_heads, seq_len_q, d_k) → (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.dropout(self.w_o(attn_output))
        output = self.layer_norm(residual + attn_output)  # 残差+LayerNorm

        return output, attn_weights


class PositionWiseFFN(nn.Module):
    """位置wise前馈网络"""

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.w_2(self.dropout(self.relu(self.w_1(x))))
        output = self.layer_norm(residual + output)  # 残差+LayerNorm
        return output


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int = 512, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 生成位置编码 (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Encoder单层"""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.self_attn(x, x, x, mask)  # 自注意力
        ffn_output = self.ffn(attn_output)
        return ffn_output, attn_weights


class DecoderLayer(nn.Module):
    """Decoder单层"""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 掩码自注意力（未来掩码）
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 交叉注意力（Encoder输出）
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(
            self, x: torch.Tensor, enc_output: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # 掩码自注意力
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        # 交叉注意力（Decoder输入 ↔ Encoder输出）
        cross_attn_output, cross_attn_weights = self.cross_attn(self_attn_output, enc_output, enc_output, src_mask)
        # 前馈网络
        ffn_output = self.ffn(cross_attn_output)
        return ffn_output, (self_attn_weights, cross_attn_weights)


class Encoder(nn.Module):
    """Encoder整体"""

    def __init__(
            self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
            n_heads: int = 8, d_ff: int = 2048, max_seq_len: int = 5000,
            dropout: float = 0.1, use_pos_encoding: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding

        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout) if use_pos_encoding else None
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # x: (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)  # 词嵌入缩放
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)

        return x, attn_weights_list


class Decoder(nn.Module):
    """Decoder整体"""

    def __init__(
            self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
            n_heads: int = 8, d_ff: int = 2048, max_seq_len: int = 5000,
            dropout: float = 0.1, use_pos_encoding: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding

        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout) if use_pos_encoding else None
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, x: torch.Tensor, enc_output: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # x: (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)  # 词嵌入缩放
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_weights_list = []
        for layer in self.layers:
            x, (self_attn_weights, cross_attn_weights) = layer(x, enc_output, tgt_mask, src_mask)
            attn_weights_list.append((self_attn_weights, cross_attn_weights))

        return x, attn_weights_list


class Transformer(nn.Module):
    """完整Transformer（Encoder+Decoder）"""

    def __init__(
            self, src_vocab_size: int, tgt_vocab_size: int,
            d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
            d_ff: int = 2048, max_seq_len: int = 5000, dropout: float = 0.1,
            use_pos_encoding: bool = True, use_multi_head: bool = True
    ):
        super().__init__()
        self.use_multi_head = use_multi_head

        # 若关闭多头注意力，强制n_heads=1
        self.n_heads = n_heads if use_multi_head else 1

        self.encoder = Encoder(
            vocab_size=src_vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=self.n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
            dropout=dropout, use_pos_encoding=use_pos_encoding
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=self.n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
            dropout=dropout, use_pos_encoding=use_pos_encoding
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)  # 输出层

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """生成padding mask和future mask"""
        batch_size, src_seq_len = src.size()
        batch_size, tgt_seq_len = tgt.size()

        # Padding mask（遮挡padding token）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, tgt_seq_len, 1)

        # Future mask（遮挡未来token，仅Decoder用）
        future_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=src.device), diagonal=1)
        future_mask = future_mask == 0  # (tgt_seq_len, tgt_seq_len)
        tgt_mask = tgt_mask & future_mask  # (batch_size, 1, tgt_seq_len, tgt_seq_len)

        return src_mask, tgt_mask

    def forward(
            self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]]:
        # 生成mask
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Encoder前向传播
        enc_output, enc_attn_weights = self.encoder(src, src_mask)

        # Decoder前向传播
        dec_output, dec_attn_weights = self.decoder(tgt[:, :-1], enc_output, tgt_mask[:, :, :-1, :-1], src_mask)
        # tgt[:, :-1]：Decoder输入去掉最后一个token；tgt_mask[:, :, :-1, :-1]：对应输入的mask

        # 输出层
        output = self.fc_out(dec_output)  # (batch_size, tgt_seq_len-1, tgt_vocab_size)

        return output, (enc_attn_weights, dec_attn_weights)