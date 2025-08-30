import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.RoPE = RoPE(dims=self.head_dim, seq_len=self.max_seq_len, base=self.theta)

        # x: B, L, E
        # q = linear(x, wq, bq) -> B, L, H_q, D
        # k = linear(x, wk, bk) -> B, L, H, D
        # v = linear(x, wv, bv) -> B, L, H, D
        # q = rope(q, offset=slice(0, L))
        # k = rope(k, offset=slice(0, L))
        # (transpose as needed)
        # x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
        # (transpose as needed)
        # x = linear(x, wo) -> B, L, E

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape # E = 32

        q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim)
        k = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.RoPE(q, slice(0,L))
        k = self.RoPE(k, slice(0,L))

        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        x = scaled_dot_product_attention_grouped(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), mask=mask).transpose((0, 2, 1, 3))
        x = x.reshape((B, L, E))
        return linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
