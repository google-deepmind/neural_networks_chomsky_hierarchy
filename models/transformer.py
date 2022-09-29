# Copyright 2022 DeepMind Technologies Limited
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

"""Transformer model."""

import enum
import math
from typing import Any, Callable, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


class PositionalEncodings(enum.Enum):
  NONE = 0
  SIN_COS = 1
  ALIBI = 2
  RELATIVE = 3
  ROTARY = 4


@chex.dataclass
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""
  # The size of the model output (i.e., the output vocabulary size).
  output_size: int
  # The dimension of the first embedding.
  embedding_dim: int = 64
  # The number of multi-head attention layers.
  num_layers: int = 5
  # The number of heads per layer.
  num_heads: int = 8
  # The number of hidden neurons per head. If None, it is set to be equal to
  # `embedding_dim // num_heads`.
  num_hiddens_per_head: Optional[int] = None
  # The probability that each element is discarded by the dropout modules.
  dropout_prob: float = 0.1
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # Whether to use the embeddings rather than raw inputs.
  use_embeddings: bool = True
  # Whether to share embeddings between the Encoder and the Decoder.
  share_embeddings: bool = False
  # The size of the sliding attention window. See MultiHeadDotProductAttention.
  attention_window: Optional[int] = None
  # The positional encoding used with default sin/cos (Vaswani et al., 2017).
  positional_encodings: PositionalEncodings = PositionalEncodings.SIN_COS
  # The maximum size of the context (used by the posiitonal encodings).
  max_time: int = 10_000
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4
  # The maximum length to sample from if noisy positional encodings are used.
  noise_max_length: Optional[int] = None

  def __post_init__(self) -> None:
    """Sets `num_hiddens_per_head` if it is `None`."""
    if self.num_hiddens_per_head is None:
      self.num_hiddens_per_head = self.embedding_dim // self.num_heads


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    memory_length: int = 0,
    max_timescale: float = 1e4,
    min_timescale: float = 2.,
    clamp_length: int = 0,
    causal: bool = False,
) -> np.ndarray:
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
  inv_freq = max_timescale**(-freqs / hidden_size)
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
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)
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


def relative_shift(logits: chex.Array,
                   attention_length: int,
                   causal: bool = False) -> chex.Array:
  if causal:
    fn = _rel_shift_causal
  else:
    fn = lambda t: _rel_shift_inner(t, attention_length)
  return jax.vmap(jax.vmap(fn))(logits)


def layer_norm(x: chex.Array) -> chex.Array:
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(x: chex.Array, output_size: int) -> chex.Array:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  x = jnp.argmax(x, axis=-1)

  # Add a time dimension for the single-output case (i.e., `ndim == 1`).
  if x.ndim == 1:
    x = jnp.expand_dims(x, axis=1)

  padded = jnp.pad(
      x, ((0, 0), (1, 0)), mode='constant', constant_values=output_size)

  return jnn.one_hot(padded[:, :-1], num_classes=output_size + 1)


def apply_rotary_encoding(x: chex.Array,
                          position: chex.Array,
                          max_time: int = 10_000) -> chex.Array:
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
  assert x.shape[1] == position.shape[1], (x.shape, position.shape)

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


def _fixed_encodings_to_relative(encodings: chex.Array) -> chex.Array:
  """Returns a matrix of shifted encodings.

  If the input is [[-2], [-1], [0], [1], [2]], the output will be
    [[[0], [1], [2]]
     [[-1], [0], [1]]
     [[-2], [-1], [0]]]

  Args:
    encodings: A tensor of encodings, of shape (length, encoding_size).

  Returns:
    A tensor of shifted encodings, of shape
    (length//2+1, length//2+1, encoding_size).
  Raises:
    ValueError if encodings is not in dimension 2.
  """
  if encodings.ndim != 2:
    raise ValueError('`logits` needs to be an array of dimension 2.')
  sequence_length, num_hiddens = encodings.shape
  if sequence_length == 1:
    return jnp.expand_dims(encodings, axis=0)
  sequence_length = sequence_length // 2 + 1
  index_matrix = jnp.sum(
      jnp.stack([
          k * jnp.eye(sequence_length, sequence_length, k=k, dtype=jnp.int32)
          for k in range(1, sequence_length)
      ]),
      axis=0)
  index_matrix = index_matrix - jnp.transpose(index_matrix)
  index_matrix += sequence_length - 1
  shifted = jnp.take(
      encodings, jnp.reshape(index_matrix, (sequence_length**2,)), axis=0)
  return jnp.reshape(shifted, (sequence_length, sequence_length, num_hiddens))


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

  # First compute the content logits.
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

  # Then compute the relative part.
  relative_bias = hk.get_parameter(
      name='relpos_relativebias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02))
  relative_logits = jnp.einsum('bthd,bThd->bhtT', queries + relative_bias,
                               relative_keys)
  # We shift the relative logits instead of the positional encoding matrix as
  # described in Appendix B of the paper (https://arxiv.org/pdf/1901.02860.pdf).
  relative_logits = relative_shift(
      relative_logits, attention_length=content_logits.shape[-1], causal=causal)
  assert content_logits.shape == relative_logits.shape
  return content_logits + relative_logits


def compute_alibi_encodings_biases(
    attention_shape: Tuple[int, ...]) -> chex.Array:
  """Returns the biases following the ALiBi method.

  This code strictly follows what is described in the ALiBi paper.
  https://arxiv.org/pdf/2108.12409.pdf

  Args:
    attention_shape: The attention logits shape, without batch size, (h, t, T).

  Returns:
    The alibi biases, same shape as the input logits shape.
  """
  num_heads, q_seq_len, k_seq_len = attention_shape

  # While this does not exactly match the description of the paper
  # (https://arxiv.org/pdf/2108.12409.pdf), it corresponds to the official
  # implementation
  # (https://github.com/ofirpress/attention_with_linear_biases/blob/a06526fbfe557f9148e414b8569dcb97c7b182ba/fairseq/models/transformer.py#L742).
  def get_slopes(n):

    def get_slopes_power_of_2(n):
      start = (2**(-2**-(math.log2(n) - 3)))
      ratio = start
      return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
      return get_slopes_power_of_2(n)
    else:
      closest_power_of_2 = 2**math.floor(math.log2(n))
      return (get_slopes_power_of_2(closest_power_of_2) +
              get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

  # Since we do not use causal masking, the upper triangle of the matrix has to
  # be nonzero. Therefore, we set it equal to the lower triangle, but we also
  # add a constant factor of 0.5 to the lower triangle, to (arbitrarily) break
  # the symmetry (otherwise, the model cannot distinguish left and right).
  alibi = np.zeros((q_seq_len, k_seq_len))
  alibi -= sum(np.tri(*alibi.shape, k=-i) for i in range(1, q_seq_len))
  alibi -= sum(np.tri(*alibi.T.shape, k=-i).T for i in range(1, k_seq_len))
  alibi += 0.5 * np.tri(*alibi.shape, k=-1)

  return alibi * jnp.array(get_slopes(num_heads))[:, None, None]


def compute_sliding_window_mask(sequence_length: int,
                                attention_window: int) -> chex.Array:
  """Returns a k-diagonal mask for a sliding window.

  Args:
    sequence_length: The length of the sequence, which will determine the shape
      of the output.
    attention_window: The size of the sliding window.

  Returns:
    A symmetric matrix of shape (sequence_length, sequence_length),
    attention_window-diagonal, with ones on the diagonal and on all the
    upper/lower diagonals up to attention_window // 2.

  Raises:
    ValueError if attention_window is <= 0.
  """
  if attention_window <= 0:
    raise ValueError(
        f'The attention window should be > 0. Got {attention_window}.')

  if attention_window == 1:
    return jnp.eye(sequence_length, sequence_length)

  attention_mask = jnp.sum(
      jnp.stack([
          jnp.eye(sequence_length, sequence_length, k=k, dtype=jnp.int32)
          for k in range(1, attention_window // 2 + 1)
      ]),
      axis=0)
  attention_mask = attention_mask + jnp.transpose(attention_mask)
  attention_mask += jnp.eye(sequence_length, sequence_length)
  return attention_mask


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      positional_encodings: PositionalEncodings,
      attention_window: Optional[int] = None,
      max_time: int = 10_000,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      positional_encodings: Which positional encodings to use in the attention.
      attention_window: Size of the attention sliding window. None means no
        sliding window is used (or equivalently, window=full_attention_length).
        We attend only on attention_window tokens around a given query token. We
        attend to tokens before AND after the query token. If attention_window
        is even, we use the value +1.
      max_time: Maximum size of the context, used in positional encodings.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._positional_encodings = positional_encodings
    self._attention_window = attention_window
    self._max_time = max_time

  def __call__(
      self,
      inputs_q: chex.Array,
      inputs_kv: chex.Array,
      mask: Optional[chex.Array] = None,
      causal: bool = False,
  ) -> chex.Array:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = inputs_q.shape

    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
    k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    # The second (sequence) dimension is undefined since it can differ between
    # queries and keys/values when decoding.
    new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    q = jnp.reshape(q, new_shape)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
    if self._positional_encodings == PositionalEncodings.RELATIVE:
      attention = compute_attention_with_relative_encodings(
          q, k, max_time=self._max_time, causal=causal)
    else:
      if self._positional_encodings == PositionalEncodings.ROTARY:
        q = apply_rotary_encoding(q, position=jnp.arange(q.shape[1])[None, :])
        k = apply_rotary_encoding(k, position=jnp.arange(k.shape[1])[None, :])
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1. / jnp.sqrt(self._num_hiddens_per_head)

    # ALiBi encodings are not scaled with the 1 / sqrt(d_k) factor.
    if self._positional_encodings == PositionalEncodings.ALIBI:
      attention += compute_alibi_encodings_biases(attention.shape[1:])

    if self._attention_window is not None:
      # We compute the sliding attention by just applying a mask on the values
      # that are outside our window.
      attention_mask = compute_sliding_window_mask(sequence_length,
                                                   self._attention_window)
      attention = jnp.where(attention_mask, attention,
                            jnp.finfo(jnp.float32).min)

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
    return hk.Linear(embedding_size, with_bias=False)(output)


class TransformerEncoder(hk.Module):
  """Transformer Encoder (Vaswani et al., 2017)."""

  def __init__(
      self,
      config: TransformerConfig,
      shared_embeddings_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the Transformer encoder.

    Args:
      config: The hyperparameters used in Transformer architectures.
      shared_embeddings_fn: Embedding function that is shared with the decoder.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._config = config
    self._shared_embeddings_fn = shared_embeddings_fn

  def __call__(self, x: chex.Array) -> chex.Array:
    """Returns the transformer encoder output, shape [B, T, E]."""
    if self._config.use_embeddings:
      if self._shared_embeddings_fn is not None:
        embeddings = self._shared_embeddings_fn(x)
      else:
        # Since `x` is one-hot encoded, using hk.Linear is equivalent to
        # hk.Embed with hk.EmbedLookupStyle.ONE_HOT.
        embs_init = hk.initializers.TruncatedNormal(
            stddev=self._config.emb_init_scale)
        embeddings = hk.Linear(
            self._config.embedding_dim, with_bias=False, w_init=embs_init)(
                x)

      embeddings *= jnp.sqrt(self._config.embedding_dim)

    else:
      embeddings = x

    _, sequence_length, embedding_size = embeddings.shape

    if self._config.positional_encodings == PositionalEncodings.SIN_COS:
      pos_encodings = sinusoid_position_encoding(
          sequence_length=sequence_length,
          hidden_size=embedding_size,
          memory_length=0,
          max_timescale=self._config.max_time,
          min_timescale=2,
          clamp_length=0,
          causal=True,
      )
      h = embeddings + pos_encodings
      h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
    else:
      h = embeddings

    for _ in range(self._config.num_layers):
      attention = MultiHeadDotProductAttention(
          num_heads=self._config.num_heads,
          num_hiddens_per_head=self._config.num_hiddens_per_head,
          positional_encodings=self._config.positional_encodings,
          attention_window=self._config.attention_window,
          max_time=self._config.max_time)(
              inputs_q=h, inputs_kv=h)
      attention = hk.dropout(hk.next_rng_key(), self._config.dropout_prob,
                             attention)
      attention = layer_norm(h + attention)

      # Position-wise feedforward network.
      h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
          attention)
      h = jnn.relu(h)
      h = hk.Linear(self._config.embedding_dim)(h)

      h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
      h = layer_norm(h + attention)
    return h


class TransformerDecoder(hk.Module):
  """Transformer Decoder (Vaswani et al., 2017)."""

  def __init__(
      self,
      config: TransformerConfig,
      shared_embeddings_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the Transformer decoder.

    Args:
      config: The hyperparameters used in Transformer architectures.
      shared_embeddings_fn: Embedding function that is shared with the encoder.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._config = config
    self._shared_embeddings_fn = shared_embeddings_fn

  def __call__(self, encoded: chex.Array, targets: chex.Array) -> chex.Array:
    """Returns the transformer decoder output, shape [B, T_O, E].

    Args:
      encoded: The output of the encoder, shape [B, T_I, E].
      targets: The one-hot encoded target values, shape [B, T_O, 2].
    """
    targets = shift_right(targets, self._config.output_size)

    if self._config.use_embeddings:
      if self._shared_embeddings_fn is not None:
        output_embeddings = self._shared_embeddings_fn(targets)
      else:
        # Since `x` is one-hot encoded, using hk.Linear is equivalent to
        # hk.Embed with hk.EmbedLookupStyle.ONE_HOT.
        embs_init = hk.initializers.TruncatedNormal(
            stddev=self._config.emb_init_scale)
        output_embeddings = hk.Linear(
            self._config.embedding_dim, with_bias=False, w_init=embs_init)(
                targets)

      output_embeddings *= jnp.sqrt(self._config.embedding_dim)

    else:
      output_embeddings = targets

    batch_size, output_sequence_length, embedding_size = output_embeddings.shape

    if self._config.positional_encodings == PositionalEncodings.SIN_COS:
      pos_encodings = sinusoid_position_encoding(
          sequence_length=output_sequence_length,
          hidden_size=embedding_size,
          memory_length=0,
          max_timescale=self._config.max_time,
          min_timescale=2,
          clamp_length=0,
          causal=True,
      )
      h = output_embeddings + pos_encodings
      h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
    else:
      h = output_embeddings

    # The causal mask is shared across heads.
    causal_mask = jnp.tril(
        jnp.ones(
            (batch_size, 1, output_sequence_length, output_sequence_length)))

    for _ in range(self._config.num_layers):
      self_attention = MultiHeadDotProductAttention(
          num_heads=self._config.num_heads,
          num_hiddens_per_head=self._config.num_hiddens_per_head,
          positional_encodings=self._config.positional_encodings,
          attention_window=self._config.attention_window,
          max_time=self._config.max_time,
      )(inputs_q=h, inputs_kv=h, mask=causal_mask, causal=True)
      self_attention = hk.dropout(hk.next_rng_key(), self._config.dropout_prob,
                                  self_attention)
      self_attention = layer_norm(h + self_attention)

      cross_attention = MultiHeadDotProductAttention(
          num_heads=self._config.num_heads,
          num_hiddens_per_head=self._config.num_hiddens_per_head,
          positional_encodings=self._config.positional_encodings,
          attention_window=self._config.attention_window,
          max_time=self._config.max_time,
      )(inputs_q=self_attention, inputs_kv=encoded, causal=True)
      cross_attention = hk.dropout(hk.next_rng_key(), self._config.dropout_prob,
                                   cross_attention)
      cross_attention = layer_norm(self_attention + cross_attention)

      # Position-wise feedforward network.
      h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
          cross_attention)
      h = jnn.relu(h)
      h = hk.Linear(self._config.embedding_dim)(h)

      h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
      h = layer_norm(h + cross_attention)

    return h


class Transformer(hk.Module):
  """Transformer (Vaswani et al., 2017)."""

  def __init__(self, config: TransformerConfig, name: Optional[str] = None):
    """Initializes the Transformer.

    Args:
      config: The hyperparameters used in Transformer architectures.
      name: The name of the module.
    """
    super().__init__(name=name)
    shared_embeddings_fn = None

    if config.share_embeddings:
      shared_embeddings_fn = hk.Linear(
          config.embedding_dim,
          with_bias=False,
          w_init=hk.initializers.TruncatedNormal(stddev=config.emb_init_scale),
          name='shared_embeddings')

    self._encoder = TransformerEncoder(config, shared_embeddings_fn)
    self._decoder = TransformerDecoder(config, shared_embeddings_fn)

  def __call__(self, inputs: chex.Array, targets: chex.Array) -> chex.Array:
    return self._decoder(self._encoder(inputs), targets)


def make_transformer_encoder(
    output_size: int,
    embedding_dim: int = 64,
    num_layers: int = 5,
    num_heads: int = 8,
    num_hiddens_per_head: Optional[int] = None,
    dropout_prob: float = 0.1,
    emb_init_scale: float = 0.02,
    use_embeddings: bool = True,
    share_embeddings: bool = False,
    attention_window: Optional[int] = None,
    positional_encodings: PositionalEncodings = PositionalEncodings.SIN_COS,
    max_time: int = 10_000,
    widening_factor: int = 4,
    noise_max_length: Optional[int] = None,
    return_all_outputs: bool = False,
) -> Callable[[chex.Array], chex.Array]:
  """Returns a transformer encoder model."""
  config = TransformerConfig(
      output_size=output_size,
      embedding_dim=embedding_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      num_hiddens_per_head=num_hiddens_per_head,
      dropout_prob=dropout_prob,
      emb_init_scale=emb_init_scale,
      use_embeddings=use_embeddings,
      share_embeddings=share_embeddings,
      attention_window=attention_window,
      positional_encodings=positional_encodings,
      max_time=max_time,
      widening_factor=widening_factor,
      noise_max_length=noise_max_length,
  )

  def transformer_encoder(inputs: chex.Array) -> chex.Array:
    output = TransformerEncoder(config)(inputs)
    if not return_all_outputs:
      output = output[:, -1, :]
    return hk.Linear(output_size)(output)

  return transformer_encoder


def make_transformer(
    output_size: int,
    embedding_dim: int = 64,
    num_layers: int = 5,
    num_heads: int = 8,
    num_hiddens_per_head: Optional[int] = None,
    dropout_prob: float = 0.1,
    emb_init_scale: float = 0.02,
    use_embeddings: bool = True,
    share_embeddings: bool = False,
    attention_window: Optional[int] = None,
    positional_encodings: PositionalEncodings = PositionalEncodings.SIN_COS,
    max_time: int = 10_000,
    widening_factor: int = 4,
    noise_max_length: Optional[int] = None,
    return_all_outputs: bool = False,
) -> Callable[[chex.Array, chex.Array], chex.Array]:
  """Returns a transformer model."""
  config = TransformerConfig(
      output_size=output_size,
      embedding_dim=embedding_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      num_hiddens_per_head=num_hiddens_per_head,
      dropout_prob=dropout_prob,
      emb_init_scale=emb_init_scale,
      use_embeddings=use_embeddings,
      share_embeddings=share_embeddings,
      attention_window=attention_window,
      positional_encodings=positional_encodings,
      max_time=max_time,
      widening_factor=widening_factor,
      noise_max_length=noise_max_length,
  )

  def transformer(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    output = Transformer(config)(inputs, targets)
    if not return_all_outputs:
      output = output[:, -1, :]
    return hk.Linear(output_size)(output)

  return transformer
