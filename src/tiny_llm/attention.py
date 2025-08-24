import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale_factor = 1/mx.sqrt(query.shape[-1])
    else:
        scale_factor = scale
    attention_scores = mx.matmul(query, key.swapaxes(-1, -2)) * scale_factor
    if mask is not None:
        attention_scores = attention_scores + mask
    return mx.matmul(softmax(attention_scores, -1), value)

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        assert wq.shape == wk.shape == wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape

        Q = linear(query, self.wq)
        K = linear(key, self.wk)
        V = linear(value, self.wv) # N x L x H x D

        Q = Q.reshape(N, L, self.num_heads, self.head_dim).transpose(0,2,1,3)
        K = K.reshape(N, L, self.num_heads, self.head_dim).transpose(0,2,1,3)
        V = V.reshape(N, L, self.num_heads, self.head_dim).transpose(0,2,1,3)

        x = scaled_dot_product_attention_simple(Q, K, V, self.scale, mask)
        x = x.transpose(0,2,1,3).reshape(N, L, self.num_heads * self.head_dim)
        return linear(x, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
