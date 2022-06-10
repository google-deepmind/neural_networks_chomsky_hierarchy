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

"""Interlocked pairing task for generalization."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class InterlockedPairing(task.GeneralizationTask):
  """A task where the goal is to transform 0^n 1^m into 0^n 1^(m+n) 0^m.

  Examples:
    001111 -> 001111 110000
    0001 -> 0001 1110

  Note that the sampling is jittable so this task is fast.
  """

  def __init__(self, vocab_size: int, *args, **kwargs):
    """Initializes the interlocked_pairing task.

    Args:
      vocab_size: The size of the alphabet.
      *args: Args for the base task class.
      **kwargs: Kwargs for the base task class.
    """
    super().__init__(*args, **kwargs)

    self._vocab_size = vocab_size

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and their interlocked versions."""

    # Make random number generators.
    rng_first_token, rng_counts = jrandom.split(rng, 2)

    # Determine first & second token.
    first_token = jrandom.randint(
        rng_first_token, shape=(batch_size, 1),
        minval=0, maxval=self._vocab_size)
    second_token = 1 - first_token

    # Determine token separation location.
    token_sep = jrandom.randint(
        rng_counts, shape=(batch_size, 1), minval=1, maxval=length)

    # Build a binary matrix which masks the locations of the first token.
    counting_matrix = jnp.tile(
        jnp.arange(start=0, stop=length), (batch_size, 1))
    sep_matrix = (counting_matrix - token_sep) < 0

    # Build I/O strings using the binary separator matrix.
    input_strings = first_token*sep_matrix + second_token*(1-sep_matrix)
    output_strings = second_token*sep_matrix + first_token*(1-sep_matrix)
    output_strings = jnp.concatenate((input_strings, output_strings), axis=1)

    # Transform into one-hot-encodings.
    input_one_hot = jnn.one_hot(input_strings, num_classes=self._vocab_size)
    output_one_hot = jnn.one_hot(output_strings, num_classes=self._vocab_size)

    return {"input": input_one_hot, "output": output_one_hot}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self._vocab_size

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._vocab_size

  def output_length(self, input_length: int) -> int:
    """Returns the output length for a given input length."""
    return 2 * input_length
