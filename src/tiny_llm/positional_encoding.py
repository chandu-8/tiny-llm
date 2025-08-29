import math

import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        exp = mx.arange(0, self.dims//2, dtype=mx.float32) / (self.dims//2)
        omega = mx.power(self.base, -exp)
        freq = mx.arange(0, self.seq_len)
        freq = mx.outer(freq, omega).astype(mx.float32)
        self.sin_freq = mx.sin(freq).astype(mx.float32)
        self.cos_freq = mx.cos(freq).astype(mx.float32)


    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        if offset is not None:
            sin_freq = self.sin_freq[offset, :]
            cos_freq = self.cos_freq[offset, :]
        else:
            sin_freq = self.sin_freq[:L, :]
            cos_freq = self.cos_freq[:L, :]

        if self.traditional:

            x = x.reshape(N, L, H, D//2, 2)
            cos_freq = cos_freq.reshape(1, L, 1, D//2)
            sin_freq = sin_freq.reshape(1, L, 1, D//2)

            x1 = x[..., 0]
            x2 = x[..., 1]
            o1 = mx.multiply(x1, cos_freq) - mx.multiply(x2, sin_freq)
            o2 = mx.multiply(x1, sin_freq) + mx.multiply(x2, cos_freq)

            y = mx.stack([o1, o2], axis=-1)
            y = y.reshape(N, L, H, D)
        else:
            cos_freq = cos_freq.reshape(1, L, 1, D//2)
            sin_freq = sin_freq.reshape(1, L, 1, D//2)
            x1 = x[..., 0:self.dims//2]
            x2 = x[..., self.dims//2:]
            o1 = mx.multiply(x1, cos_freq) - mx.multiply(x2, sin_freq)
            o2 = mx.multiply(x1, sin_freq) + mx.multiply(x2, cos_freq)

            y = mx.concat([o1, o2], axis=-1)
            y = y.reshape(N, L, H, D)
        return y.astype(x.dtype)


