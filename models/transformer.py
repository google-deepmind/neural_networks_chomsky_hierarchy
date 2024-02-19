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

"""Transformer model."""

import dataclasses
from typing import Callable, Optional

import chex
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.models import positional_encodings as pos_encs_lib


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
  positional_encodings: pos_encs_lib.PositionalEncodings = dataclasses.field(
      default_factory=lambda: pos_encs_lib.PositionalEncodings.SIN_COS
  )
  # The maximum size of the context (used by the posiitonal encodings).
  max_time: int = 10_000
  # The parameters for the positional encodings, default sin/cos.
  positional_encodings_params: pos_encs_lib.PositionalEncodingsParams = (
      dataclasses.field(default_factory=pos_encs_lib.SinCosParams)
  )
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4
  # Add mask to make causal predictions.
  causal_masking: bool = False

  def __post_init__(self) -> None:
    """Sets `num_hiddens_per_head` if it is `None`."""
    if self.num_hiddens_per_head is None:
      self.num_hiddens_per_head = self.embedding_dim // self.num_heads


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
      positional_encodings: pos_encs_lib.PositionalEncodings,
      positional_encodings_params: pos_encs_lib.PositionalEncodingsParams,
      attention_window: Optional[int] = None,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      positional_encodings: Which positional encodings to use in the attention.
      positional_encodings_params: Parameters for the positional encodings.
      attention_window: Size of the attention sliding window. None means no
        sliding window is used (or equivalently, window=full_attention_length).
        We attend only on attention_window tokens around a given query token. We
        attend to tokens before AND after the query token. If attention_window
        is even, we use the value +1.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._positional_encodings = positional_encodings
    self._attention_window = attention_window
    self._positional_encodings_params = positional_encodings_params

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
    if self._positional_encodings == pos_encs_lib.PositionalEncodings.RELATIVE:
      # We type hint the params to match the if statement, for pytype.
      self._positional_encodings_params: pos_encs_lib.RelativeParams
      attention = pos_encs_lib.compute_attention_with_relative_encodings(
          q, k, self._positional_encodings_params.max_time, causal=causal
      )
    else:
      if self._positional_encodings == pos_encs_lib.PositionalEncodings.ROTARY:
        q = pos_encs_lib.apply_rotary_encoding(
            q, position=jnp.arange(q.shape[1])[None, :]
        )
        k = pos_encs_lib.apply_rotary_encoding(
            k, position=jnp.arange(k.shape[1])[None, :]
        )
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    # ALiBi encodings are not scaled with the 1 / sqrt(d_k) factor.
    if self._positional_encodings == pos_encs_lib.PositionalEncodings.ALIBI:
      attention += pos_encs_lib.compute_alibi_encodings_biases(
          attention.shape[1:]
      )

    if self._attention_window is not None:
      # We compute the sliding attention by just applying a mask on the values
      # that are outside our window.
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
    """Initializes the transformer encoder.

    Args:
      config: The hyperparameters used in Transformer architectures.
      shared_embeddings_fn: Embedding function that is shared with the decoder.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._config = config
    self._shared_embeddings_fn = shared_embeddings_fn

  def __call__(self, x: jnp.ndarray) -> chex.Array:
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

    batch_size, sequence_length, embedding_size = embeddings.shape

    pos_enc_params = self._config.positional_encodings_params
    if (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.SIN_COS
    ):
      pos_encodings = pos_encs_lib.sinusoid_position_encoding(
          sequence_length=sequence_length,
          hidden_size=embedding_size,
          memory_length=0,
          max_timescale=pos_enc_params.max_time,
          min_timescale=2,
          clamp_length=0,
          causal=True,
      )
      h = embeddings + pos_encodings
      h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
    else:
      h = embeddings

    # The causal mask is shared across heads.
    if self._config.causal_masking:
      causal_mask = jnp.tril(
          jnp.ones((batch_size, 1, sequence_length, sequence_length))
      )
    else:
      causal_mask = None

    for _ in range(self._config.num_layers):
      attention = MultiHeadDotProductAttention(
          num_heads=self._config.num_heads,
          num_hiddens_per_head=self._config.num_hiddens_per_head,
          positional_encodings=self._config.positional_encodings,
          positional_encodings_params=pos_enc_params,
          attention_window=self._config.attention_window,
      )(
          inputs_q=h,
          inputs_kv=h,
          mask=causal_mask,
          causal=self._config.causal_masking,
      )
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

    if (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.SIN_COS
    ):
      pos_encodings = pos_encs_lib.sinusoid_position_encoding(
          sequence_length=output_sequence_length,
          hidden_size=embedding_size,
          memory_length=0,
          max_timescale=self._config.positional_encodings_params.max_time,
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
          positional_encodings_params=self._config.positional_encodings_params,
          attention_window=self._config.attention_window,
      )(inputs_q=h, inputs_kv=h, mask=causal_mask, causal=True)
      self_attention = hk.dropout(hk.next_rng_key(), self._config.dropout_prob,
                                  self_attention)
      self_attention = layer_norm(h + self_attention)

      cross_attention = MultiHeadDotProductAttention(
          num_heads=self._config.num_heads,
          num_hiddens_per_head=self._config.num_hiddens_per_head,
          positional_encodings=self._config.positional_encodings,
          positional_encodings_params=self._config.positional_encodings_params,
          attention_window=self._config.attention_window,
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
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    widening_factor: int = 4,
    return_all_outputs: bool = False,
    causal_masking: bool = False,
) -> Callable[[chex.Array], chex.Array]:
  """Returns a transformer encoder model."""
  if positional_encodings is None:
    positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
    positional_encodings_params = pos_encs_lib.SinCosParams()
  elif positional_encodings_params is None:
    raise ValueError('No parameters for positional encodings are passed.')
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
      positional_encodings_params=positional_encodings_params,
      widening_factor=widening_factor,
      causal_masking=causal_masking,
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
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    widening_factor: int = 4,
    return_all_outputs: bool = False,
) -> Callable[[chex.Array, chex.Array], chex.Array]:
  """Returns a transformer model."""
  if positional_encodings is None:
    positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
    positional_encodings_params = pos_encs_lib.SinCosParams()
  elif positional_encodings_params is None:
    raise ValueError('No parameters for positional encodings are passed.')
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
      positional_encodings_params=positional_encodings_params,
      widening_factor=widening_factor,
  )

  def transformer(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    output = Transformer(config)(inputs, targets)
    if not return_all_outputs:
      output = output[:, -1, :]
    return hk.Linear(output_size)(output)

  return transformer
