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
from typing import Callable, Optional, Tuple

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp


_INF_LOGITS = 10000


class PositionalEncodings(enum.Enum):
  NONE = 0
  SIN_COS = 1
  ALIBI = 2
  RELATIVE = 3


def sin_cos_positional_encodings(sequence_length: int,
                                 embedding_size: int,
                                 with_negative: bool = False,
                                 max_time: float = 10000.0) -> jnp.ndarray:
  """Generates positional encodings for the input.

  Args:
    sequence_length: The length of the output sequence.
    embedding_size: The size of the embedding to consider. Must be even.
    with_negative: Whether to also compute values before 0 (useful for
      shifting).
    max_time: (default 10000) Constant used to scale position by in the
      encodings.
  Returns:
    A tensor of size [seq_len, emb_size].

  Raises:
    ValueError if embedding_size is odd.
  """
  if embedding_size % 2 == 1:
    raise ValueError(
        'Embedding sizes must be even if using positional encodings.')

  # Generate a sequence of positions and frequencies.
  if not with_negative:
    pos = jnp.arange(0, sequence_length, dtype=jnp.float32)
  else:
    pos = jnp.arange(-sequence_length + 1, sequence_length, dtype=jnp.float32)
  freqs = jnp.arange(0, embedding_size, 2, dtype=jnp.float32)
  inverse_freqs = 1.0 / (max_time**(freqs / embedding_size))

  # We combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
  pos_emb = jnp.einsum('i,j->ij', pos, inverse_freqs)
  return jnp.concatenate([jnp.sin(pos_emb), jnp.cos(pos_emb)], -1)


def _fixed_encodings_to_relative(encodings: jnp.ndarray) -> jnp.ndarray:
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


def compute_attention_with_relative_encodings(queries: jnp.ndarray,
                                              keys: jnp.ndarray) -> jnp.ndarray:
  """Returns attention with relative positional encodings.

  This code strictly follows what is described in the TransformerXL paper.
  https://arxiv.org/pdf/1901.02860.pdf

  Args:
    queries: The queries used for attention. Shape (b, t, h, d).
    keys: The keys used for attention. Shape (b, t, h, d').

  Returns:
    The attention logits. Shape (b, h, t, t).
  """
  sequence_length, num_heads, num_hiddens = queries.shape[-3:]

  # First compute the content logits.
  content_bias = hk.get_parameter(
      name='relpos_contentbias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02))
  content_logits = jnp.einsum('bthd,bThd->bhtT', queries + content_bias, keys)

  # Then compute the relative part.
  relative_bias = hk.get_parameter(
      name='relpos_relativebias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02))
  sin_cos = sin_cos_positional_encodings(
      sequence_length, num_hiddens, with_negative=True)
  shifted_sin_cos = _fixed_encodings_to_relative(sin_cos)
  relative_keys = hk.Linear(num_hiddens, name='k_params')(shifted_sin_cos)
  relative_logits = jnp.einsum('bthd,Ttd->bhtT', queries + relative_bias,
                               relative_keys)  # No need to broadcast batch.
  return content_logits + relative_logits


def compute_alibi_encodings_biases(
    attention_shape: Tuple[int, int, int, int]) -> jnp.ndarray:
  """Returns the biases following the ALiBi method.

  This code strictly follows what is described in the ALiBi paper.
  https://arxiv.org/pdf/2108.12409.pdf

  Args:
    attention_shape: The attention logits shape. Shape (b, h, t, t).

  Returns:
    The alibi biases, same shape as the input logits shape.
  """
  batch_size, num_heads, sequence_length, _ = attention_shape

  base_coeff = 2**(-8 / num_heads)
  # Coeffs tensor of shape (h, 1, 1).
  coeffs = jnp.array([base_coeff**i for i in range(1, num_heads + 1)])
  coeffs = jnp.expand_dims(coeffs, -1)
  coeffs = jnp.expand_dims(coeffs, -1)

  # Biases tensor of shape (h, t, t).
  # The upper part of the matrix is not zero like in the paper because we
  # don't use causal attention.
  if sequence_length == 1:
    biases = jnp.zeros((1, 1))
  else:
    biases = jnp.sum(
        jnp.stack([
            k * jnp.eye(sequence_length, sequence_length, k=k)
            for k in range(1, sequence_length)
        ]),
        axis=0)
    biases -= jnp.transpose(biases)
    biases = jnp.stack([biases] * num_heads, axis=0)

  # Multiply the biases with the coeffs, and batch the resulting tensor.
  biases = coeffs * biases
  return jnp.stack([biases] * batch_size, axis=0)


def compute_sliding_window_mask(sequence_length: int,
                                attention_window: int) -> jnp.ndarray:
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
  attention_mask += jnp.transpose(attention_mask)
  attention_mask += jnp.eye(sequence_length, sequence_length)
  return attention_mask


class MultiHeadSelfAttention(hk.Module):
  """Classical attention module, using multiple heads."""

  def __init__(
      self,
      num_heads: int,
      hiddens_per_head: int,
      positional_encodings: PositionalEncodings,
      attention_window: Optional[int] = None,
      dropout_prob: float = 0.,
      name: Optional[str] = None,
  ):
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      hiddens_per_head: Number of hidden neurons per head.
      positional_encodings: Which positional encodings to use in the attention.
      attention_window: Size of the attention sliding window. None means no
        sliding window is used (or equivalently, window=full_attention_length).
        We attend only on attention_window tokens around a given query token. We
        attend to tokens before AND after the query token. If attention_window
        is even, we use the value +1.
      dropout_prob: Probability of dropout. Must be between 0 and 1.
      name: Name of the module.

    Raises:
      ValueError if dropout_prob is outside of [0, 1].
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._hiddens_per_head = hiddens_per_head
    if not 0 <= dropout_prob <= 1:
      raise ValueError(f'dropout_prob should be in [0, 1]. Got {dropout_prob}.')
    self._dropout_prob = dropout_prob
    self._positional_encodings = positional_encodings
    self._attention_window = attention_window

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = x.shape

    if self._positional_encodings == PositionalEncodings.SIN_COS:
      x += sin_cos_positional_encodings(sequence_length, embedding_size)

    if self._dropout_prob > 0:
      x = hk.dropout(hk.next_rng_key(), self._dropout_prob, x)

    hiddens = self._hiddens_per_head * self._num_heads
    q = hk.Linear(hiddens, with_bias=False)(x)
    k = hk.Linear(hiddens, with_bias=False, name='k_params')(x)
    v = hk.Linear(hiddens, with_bias=False)(x)
    new_shape = (batch_size, sequence_length, self._num_heads,
                 self._hiddens_per_head)
    q = jnp.reshape(q, new_shape)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    # In the following, b=batch_size, t=seq_len, h=num_heads, d=hiddens_per_head
    if self._positional_encodings == PositionalEncodings.RELATIVE:
      attention = compute_attention_with_relative_encodings(q, k)
    else:
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
      if self._positional_encodings == PositionalEncodings.ALIBI:
        attention += compute_alibi_encodings_biases(attention.shape)
    attention *= 1. / jnp.sqrt(hiddens)

    if self._attention_window is not None:
      # We compute the sliding attention by just applying a mask on the values
      # that are outside our window.
      attention_mask = compute_sliding_window_mask(sequence_length,
                                                   self._attention_window)
      attention_mask = -_INF_LOGITS * (1 - attention_mask)
      attention = attention_mask + attention

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, sequence_length, hiddens))
    return hk.Linear(embedding_size)(output)


class Transformer(hk.Module):
  """Transformer tower."""

  def __init__(
      self,
      vocab_size: int,
      embedding_dim: int = 128,
      num_layers: int = 2,
      num_heads: int = 8,
      hiddens_per_head: Optional[int] = None,
      dropout_prob: float = 0.1,
      emb_init_scale: float = 0.02,
      use_embeddings: bool = True,
      attention_window: Optional[int] = None,
      positional_encodings: PositionalEncodings = PositionalEncodings.SIN_COS,
      name: Optional[str] = None):
    """Initializes the transformer.

    Args:
      vocab_size: The number of tokens to consider (length of the one_hot).
      embedding_dim: The dimension of the first embedding.
      num_layers: The number of multi-head attention layers.
      num_heads: The number of heads per layer.
      hiddens_per_head: The number of hidden neurons per head. If None, equal to
        the embedding dimension divided by the number of heads.
      dropout_prob: The probability of dropout during training.
      emb_init_scale: Params initializer scale for the embeddings.
      use_embeddings: Whether to use embeddings rather than raw inputs.
      attention_window: Size of the attention sliding window. See
        MultiHeadSelfAttention.
      positional_encodings: Which positional encodings to use. Default is the
        same as in the seminal transformer paper, ie sin and cos values.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._num_vocab = vocab_size
    self._emb_dim = embedding_dim
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._emb_init_scale = emb_init_scale
    self._attention_window = attention_window
    self._positional_encodings = positional_encodings
    self._use_embeddings = use_embeddings
    self._hiddens_per_head = hiddens_per_head
    if hiddens_per_head is None:
      self._hiddens_per_head = embedding_dim // num_heads

  def __call__(self, x: jnp.ndarray, is_training: bool = True):
    """Returns the transformer tower output, shape [B, T, E]."""
    if self._use_embeddings:
      embs_init = hk.initializers.TruncatedNormal(stddev=self._emb_init_scale)
      embeddings = hk.Linear(
          self._emb_dim, with_bias=False, w_init=embs_init)(
              x)
    else:
      embeddings = x

    h = embeddings
    for _ in range(self._num_layers):
      attention = MultiHeadSelfAttention(
          num_heads=self._num_heads,
          hiddens_per_head=self._hiddens_per_head,
          dropout_prob=self._dropout_prob,
          positional_encodings=self._positional_encodings,
          attention_window=self._attention_window)(
              h)
      attention = h + attention
      attention = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True)(
              attention)
      h = jnn.relu(h)
      h = hk.Linear(self._emb_dim)(h)
      h = hk.dropout(hk.next_rng_key(), self._dropout_prob, h)
      h = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True)(
              h + attention)
    return h


def make_transformer(
    vocab_size: int,
    num_layers: int,
    output_size: int,
    return_all_outputs: bool = False,
    embedding_dim: int = 128,
    attention_window: Optional[int] = None,
    positional_encodings: PositionalEncodings = PositionalEncodings.SIN_COS,
    dropout_prob: float = 0.1,
    is_training: bool = True) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a transformer model."""
  def transformer(x):
    output = Transformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        positional_encodings=positional_encodings,
        dropout_prob=dropout_prob,
        attention_window=attention_window)(
            x, is_training=is_training)
    if not return_all_outputs:
      output = output[:, -1, :]
    return hk.Linear(output_size)(output)
  return transformer
