# FILE: src/mini_t/model.py
import torch
import torch.nn as nn
from .modules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
from typing import Optional, Tuple

class EncoderLayer(nn.Module):
    """인코더의 단일 레이어"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    """디코더의 단일 레이어"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_attn_output, self_attn_probs = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output, cross_attn_probs = self.cross_attn(x, enc_output, enc_output, padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x, self_attn_probs, cross_attn_probs

class Encoder(nn.Module):
    """N개의 EncoderLayer로 구성된 인코더"""
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """N개의 DecoderLayer로 구성된 디코더"""
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, _, _ = layer(x, enc_output, look_ahead_mask, padding_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """최종 트랜스포머 모델"""
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = generate_masks(src_ids, tgt_ids)
        
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src_ids)))
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt_ids)))

        enc_output = self.encoder(src_emb, src_mask)
        dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)
        
        output = self.final_linear(dec_output)
        return output

def generate_masks(src_ids: torch.Tensor, tgt_ids: torch.Tensor, pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    소스 및 타겟 시퀀스에 대한 패딩 마스크와 룩어헤드 마스크를 생성합니다.
    """
    # 소스 패딩 마스크
    src_mask = (src_ids != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, src_len)

    # 타겟 패딩 마스크
    tgt_pad_mask = (tgt_ids != pad_idx).unsqueeze(1).unsqueeze(2) # (B, 1, 1, tgt_len)

    # 타겟 룩어헤드 마스크 (causal mask)
    tgt_len = tgt_ids.size(1)
    look_ahead_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt_ids.device), diagonal=1).bool()
    look_ahead_mask = ~look_ahead_mask # (tgt_len, tgt_len)

    # 타겟 패딩 마스크와 룩어헤드 마스크 결합
    tgt_mask = tgt_pad_mask & look_ahead_mask # (B, 1, tgt_len, tgt_len)
    
    return src_mask, tgt_mask