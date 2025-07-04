# FILE: tests/test_mini_t.py
import pytest
import torch
from src.mini_t.modules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
from src.mini_t.model import EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer, generate_masks

@pytest.fixture
def config():
    """테스트를 위한 기본 설정 값을 제공하는 Fixture"""
    return {
        "d_model": 64,
        "n_heads": 8,
        "d_ff": 128,
        "dropout": 0.1,
        "n_layers": 2,
        "vocab_size": 1000
    }

@pytest.mark.parametrize("batch_size, seq_len", [(4, 50), (1, 10)])
def test_multi_head_attention_shape(config, batch_size, seq_len):
    """MultiHeadAttention 모듈의 출력 텐서 형태가 올바른지 검증"""
    mha = MultiHeadAttention(d_model=config["d_model"], num_heads=config["n_heads"], dropout=config["dropout"])
    q = torch.randn(batch_size, seq_len, config["d_model"])
    k = torch.randn(batch_size, seq_len, config["d_model"])
    v = torch.randn(batch_size, seq_len, config["d_model"])
    output, _ = mha(q, k, v)
    assert output.shape == (batch_size, seq_len, config["d_model"])

@pytest.mark.parametrize("batch_size, seq_len", [(4, 50), (1, 10)])
def test_positionwise_ffn_shape(config, batch_size, seq_len):
    """PositionwiseFeedForward 모듈의 출력 텐서 형태가 올바른지 검증"""
    ffn = PositionwiseFeedForward(d_model=config["d_model"], d_ff=config["d_ff"], dropout=config["dropout"])
    x = torch.randn(batch_size, seq_len, config["d_model"])
    output = ffn(x)
    assert output.shape == (batch_size, seq_len, config["d_model"])

@pytest.mark.parametrize("batch_size, seq_len", [(4, 50), (1, 10)])
def test_encoder_layer_shape(config, batch_size, seq_len):
    """EncoderLayer 모듈의 출력 텐서 형태가 올바른지 검증"""
    layer = EncoderLayer(d_model=config["d_model"], num_heads=config["n_heads"], d_ff=config["d_ff"], dropout=config["dropout"])
    x = torch.randn(batch_size, seq_len, config["d_model"])
    mask = torch.ones(batch_size, 1, 1, seq_len)
    output = layer(x, mask)
    assert output.shape == (batch_size, seq_len, config["d_model"])

@pytest.mark.parametrize("batch_size, seq_len", [(4, 50), (1, 10)])
def test_decoder_layer_shape(config, batch_size, seq_len):
    """DecoderLayer 모듈의 출력 텐서 형태가 올바른지 검증"""
    layer = DecoderLayer(d_model=config["d_model"], num_heads=config["n_heads"], d_ff=config["d_ff"], dropout=config["dropout"])
    x = torch.randn(batch_size, seq_len, config["d_model"])
    enc_output = torch.randn(batch_size, seq_len, config["d_model"])
    look_ahead_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    padding_mask = torch.ones(batch_size, 1, 1, seq_len)
    output, _, _ = layer(x, enc_output, look_ahead_mask, padding_mask)
    assert output.shape == (batch_size, seq_len, config["d_model"])

@pytest.mark.parametrize("batch_size, src_len, tgt_len", [(4, 50, 60)])
def test_transformer_shape(config, batch_size, src_len, tgt_len):
    """최종 Transformer 모델의 출력 텐서 형태가 올바른지 검증"""
    model = Transformer(
        src_vocab_size=config["vocab_size"],
        tgt_vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["n_heads"],
        num_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"]
    )
    src = torch.randint(0, config["vocab_size"], (batch_size, src_len))
    tgt = torch.randint(0, config["vocab_size"], (batch_size, tgt_len))
    output = model(src, tgt)
    assert output.shape == (batch_size, tgt_len, config["vocab_size"])

def test_generate_masks(config):
    """마스크 생성 함수의 출력 형태와 타입이 올바른지 검증"""
    src_ids = torch.tensor([[1, 2, 3, 0, 0]]) # (batch_size=1, seq_len=5)
    tgt_ids = torch.tensor([[4, 5, 0]])      # (batch_size=1, seq_len=3)
    src_mask, tgt_mask = generate_masks(src_ids, tgt_ids, pad_idx=0)
    assert src_mask.shape == (1, 1, 1, 5)
    assert tgt_mask.shape == (1, 1, 3, 3)
    assert src_mask.dtype == torch.bool
    assert tgt_mask.dtype == torch.bool