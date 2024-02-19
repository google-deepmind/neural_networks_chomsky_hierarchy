# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Positional encodings, used in `transformer.py`."""

import enum
import math
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class PositionalEncodings(enum.Enum):
  """Enum for all the positional encodings implemented."""
  NONE = 0
  SIN_COS = 1
  ALIBI = 2
  RELATIVE = 3
  ROTARY = 4


# General type used throughout the class for pos enc parameters.
PositionalEncodingsParams = Any


@chex.dataclass
class SinCosParams:
  """Parameters for the classical sin/cos positional encoding."""
  # The maximum wavelength used.
  max_time: int = 10_000


# We will use this same class for Rotary and Relative.
RotaryParams = SinCosParams
RelativeParams = SinCosParams


POS_ENC_TABLE = {
    'NONE': PositionalEncodings.NONE,
    'SIN_COS': PositionalEncodings.SIN_COS,
    'ALIBI': PositionalEncodings.ALIBI,
    'RELATIVE': PositionalEncodings.RELATIVE,
    'ROTARY': PositionalEncodings.ROTARY,
}

POS_ENC_PARAMS_TABLE = {
    'NONE': SinCosParams,
    'SIN_COS': SinCosParams,
    'ALIBI': SinCosParams,
    'RELATIVE': RelativeParams,
    'ROTARY': RotaryParams,
}


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    memory_length: int = 0,
    max_timescale: float = 1e4,
    min_timescale: float = 2.0,
    clamp_length: int = 0,
    causal: bool = False,
):
  """Creates sinusoidal encodings.

  The time dimension is larger than sequence_length as we need to cover all
  cases of looking in either the future or past.

  Args:
    sequence_length: `int` sequence length, L
    hidden_size: `int` dimension of the positional encoding vectors, D
    memory_length: `int` size of the memory, M
    max_timescale: `int` maximum timescale for the frequency
    min_timescale: `int` minimum timescale for the frequency
    clamp_length: If greater than 0, any positions further apart than
      `clamp_length` are clamped to this value
    causal: If true then generates a smaller set (L vs 2 * L) of time-encodings
      for the use-case of causal attention.

  Returns:
    An array of shape [L + M, D] for causal and [2 * L + M, D] otherwise.
  """
  freqs = np.arange(0, hidden_size, min_timescale)
  inv_freq = max_timescale ** (-freqs / hidden_size)
  # Since inputs can look into the past and into the future, depending on the
  # permutation mask, we need to have relative encodings for both. The furthest
  # back an input can see is the final token, up to sequence_length +
  # memory_length - 1. The furthest ahead an input can see is for token 0 where
  # it can see up to sequence_length - 1 future tokens.
  if causal:
    pos_seq = np.arange(sequence_length + memory_length, 0, -1.0)
  else:
    pos_seq = np.arange(sequence_length + memory_length, -sequence_length, -1.0)
  if clamp_length:
    pos_seq = np.clip(pos_seq, a_min=-clamp_length, a_max=clamp_length)
  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return pos_emb


def _rel_shift_inner(logits: chex.Array, attention_length: int) -> chex.Array:
  """Shifts the relative logits.

  This is a more general than the original Transformer-XL implementation as
  inputs may also see the future. (The implementation does not rely on a
  causal mask removing the upper-right triangle.)

  Given attention length 3 and inputs:
      [[-3, -2, -1, 0, 1, 2],
       [-3, -2, -1, 0, 1, 2],
       [-3, -2, -1, 0, 1, 2]]

  The shifted output is:
      [[0, 1, 2],
       [-1, 0, 1],
       [-2, -1, 0]]

  Args:
    logits: input tensor of shape [T_q, T_v + T_q]
    attention_length: T_v `int` length of the attention, should be equal to
      memory size + sequence length.

  Returns:
    A shifted version of the input of size [T_q, T_v]. In each row, a window of
      size T_v elements is kept. The window starts at
      the rightmost end, for the first row. It then shifts left by 1 for each
      subsequent row.
  """
  if logits.ndim != 2:
    raise ValueError('`logits` needs to be an array of dimension 2.')
  tq, total_len = logits.shape
  assert total_len == tq + attention_length
  logits = jnp.reshape(logits, [total_len, tq])
  logits = jax.lax.slice(logits, (1, 0), logits.shape)  # logits[1:]
  logits = jnp.reshape(logits, [tq, total_len - 1])
  # Equiv to logits[:, :attention_length].
  logits = jax.lax.slice(logits, (0, 0), (tq, attention_length))
  return logits


def _rel_shift_causal(logits: chex.Array) -> chex.Array:
  """Shifts the relative logits, assuming causal attention.

  Given inputs:
      [[-4, -3, -2, -1],
       [-4, -3, -2, -1]]

  The shifted (and, later, masked) output is:
      [[-3, -2, -1,  0],
       [-4, -3, -2, -1]]

  Args:
    logits: input tensor of shape [T_q, T_v]

  Returns:
    A shifted version of the input of size [T_q, T_v].
  """
  t1, t2 = logits.shape
  # We prepend zeros on the final timescale dimension.
  to_pad = jnp.zeros_like(logits[..., :1])
  x = jnp.concatenate((to_pad, logits), axis=-1)

  # Reshape trick to  shift input.
  x = jnp.reshape(x, [t2 + 1, t1])

  # Remove extra time dimension and re-shape.
  x = jax.lax.slice(x, [1] + [0] * (x.ndim - 1), x.shape)

  return jnp.reshape(x, [t1, t2])


def relative_shift(
    logits: chex.Array, attention_length: int, causal: bool = False
) -> chex.Array:
  if causal:
    fn = _rel_shift_causal
  else:
    fn = lambda t: _rel_shift_inner(t, attention_length)
  return jax.vmap(jax.vmap(fn))(logits)


def apply_rotary_encoding(
    x: jnp.ndarray, position: jnp.ndarray, max_time: int = 10_000
) -> jnp.ndarray:
  """Applies RoPE positional encodings for the input.

  Paper: https://arxiv.org/abs/2104.09864

  Args:
    x: The input tensor on which RoPE will be applied. Usually it is either some
      queries q or some keys k.
    position: The positions to use. Usually it's an arange of the maximum
      length.
    max_time: Constant used to scale position by in the encodings.

  Returns:
    A tensor with the same shape as x.
  """
  # Expand dims for positions to support inputs of shapes BTC or BTHC.
  freq_seq = jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
  freq_seq = freq_seq / (x.shape[-1] // 2)
  inv_freq = max_time**-freq_seq
  inv_freq = jnp.repeat(inv_freq, 2, 0)
  # Produce position inputs to periodic functions.
  t = position[:, :, None, None] * inv_freq[None, None, None, :]
  x_rot = jnp.einsum('bthd,dD->bthD', x, _rope_kernel(x.shape[-1], x.dtype))
  return x * jnp.cos(t).astype(x.dtype) + jnp.sin(t).astype(x.dtype) * x_rot


def _rope_kernel(n: int, dtype: Any) -> np.ndarray:
  """Reorders the embedding dimension of an array, to make rotation easier."""
  # We implement the equivalent of
  #   even_dims, odd_dims,  = x[..., ::2], x[..., 1::2]
  #   return jnp.stack((-odd_dims, even_dims), axis=-1).reshape(x.shape)
  # with a custom kernel for einsum. This allows the computation to execute
  # on the MXU instead of producing a slow gather.
  assert n % 2 == 0, n
  kernel = np.zeros((n, n), dtype)
  for i in range(n):
    # Swap each neighbouring pair of values.
    if i % 2 == 0:
      kernel[i, i + 1] = 1
    else:
      kernel[i, i - 1] = -1
  return kernel


def compute_attention_with_relative_encodings(
    queries: chex.Array,
    keys: chex.Array,
    max_time: int = 10_000,
    causal: bool = False) -> chex.Array:
  """Returns attention with relative positional encodings.

  This code strictly follows what is described in the TransformerXL paper.
  https://arxiv.org/pdf/1901.02860.pdf

  Args:
    queries: The queries used for attention. Shape (b, t, h, d).
    keys: The keys used for attention. Shape (b, T, h, d).
    max_time: Constant used to scale position by in the sin/cos encodings.
    causal: Whether to use causal attention when shifting the relative logits.

  Returns:
    The attention logits. Shape (b, h, t, T).
  """
  batch_size, k_seq_len, num_heads, num_hiddens = keys.shape
  hiddens = num_hiddens * num_heads

  # First compute the content logits.
  content_bias = hk.get_parameter(
      name='relpos_contentbias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02))
  content_logits = jnp.einsum('bthd,bThd->bhtT', queries + content_bias, keys)

  positional_encodings = sinusoid_position_encoding(
      sequence_length=k_seq_len,
      hidden_size=hiddens,
      memory_length=0,
      max_timescale=max_time,
      min_timescale=2,
      clamp_length=0,
      causal=causal,
  )
  positional_encodings = jnp.broadcast_to(positional_encodings, (batch_size,) +
                                          positional_encodings.shape)
  relative_keys = hk.Linear(hiddens, with_bias=False)(positional_encodings)
  relative_keys = jnp.reshape(
      relative_keys, positional_encodings.shape[:-1] + (num_heads, num_hiddens))

  # Then compute the relative part.
  relative_bias = hk.get_parameter(
      name='relpos_relativebias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02))
  relative_logits = jnp.einsum('bthd,bThd->bhtT', queries + relative_bias,
                               relative_keys)
  # We shift the relative logits instead of the positional encoding matrix as
  # described in Appendix B of the paper (https://arxiv.org/pdf/1901.02860.pdf).
  relative_logits = relative_shift(
      relative_logits, attention_length=content_logits.shape[-1], causal=causal
  )
  assert content_logits.shape == relative_logits.shape
  return content_logits + relative_logits


def _get_alibi_slopes(num_heads: int) -> list[float]:
  """Returns the slopes for the different attention heads.

  While this does not exactly match the description of the [ALiBi
  paper](https://arxiv.org/pdf/2108.12409.pdf), it corresponds to the [official
  implementation](https://github.com/ofirpress/attention_with_linear_biases/blob/a06526fbfe557f9148e414b8569dcb97c7b182ba/fairseq/models/transformer.py#L742).

  Args:
    num_heads: The number of attention heads to create slopes for.
  """

  def get_slopes_power_of_2(n):
    start = (2**(-2**-(math.log2(n) - 3)))
    ratio = start
    return [start * ratio**i for i in range(n)]

  if math.log2(num_heads).is_integer():
    return get_slopes_power_of_2(num_heads)
  else:
    closest_power_of_2 = 2**math.floor(math.log2(num_heads))
    return (get_slopes_power_of_2(closest_power_of_2) + _get_alibi_slopes(
        2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2])


def compute_alibi_encodings_biases(
    attention_shape: tuple[int, ...]
) -> chex.Array:
  """Returns the biases following the ALiBi method.

  This code strictly follows what is described in the ALiBi paper.
  https://arxiv.org/pdf/2108.12409.pdf

  Args:
    attention_shape: The attention logits shape, without batch size, (h, t, T).

  Returns:
    The alibi biases, same shape as the input logits shape.
  """
  num_heads, q_seq_len, k_seq_len = attention_shape

  # Since we do not use causal masking, the upper triangle of the matrix has to
  # be nonzero. Therefore, we set it equal to the lower triangle, but we also
  # add a constant factor of 0.5 to the lower triangle, to (arbitrarily) break
  # the symmetry (otherwise, the model cannot distinguish left and right).
  alibi = np.zeros((q_seq_len, k_seq_len))
  alibi -= sum(np.tri(*alibi.shape, k=-i) for i in range(1, q_seq_len))
  alibi -= sum(np.tri(*alibi.T.shape, k=-i).T for i in range(1, k_seq_len))
  alibi += 0.5 * np.tri(*alibi.shape, k=-1)

  return alibi * jnp.array(_get_alibi_slopes(num_heads))[:, None, None]
