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

"""Sort tokens from a fixed alphabet (i.e., bucket sort)."""

import functools

import chex
import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class BucketSort(task.GeneralizationTask):
  """A task with the goal of sorting tokens from a fixed alphabet.

  The input string is composed of tokens from a fixed-size alphabet, i.e.,
  `{0, 1, ..., vocab_size - 1}`, and the goal is to return the sorted string (in
  lexicographically increasing order).

  Examples:
    10204112  ->  00111224  (with `vocab_size = 5`)
    1110001   ->  0001111   (with `vocab_size = 2`)
  """

  def __init__(self, vocab_size: int = 5) -> None:
    """Initializes the task.

    Args:
      vocab_size: The size of the alphabet. We use 5 in the paper.
    """
    self._vocab_size = vocab_size

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      length: int,
  ) -> task.Batch:
    """Returns a batch of strings and tokens sorted by (inc.) occurrence."""
    strings = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=self._vocab_size)
    sorted_strings = jnp.sort(strings, axis=-1)

    return {
        'input': jnn.one_hot(strings, num_classes=self.input_size),
        'output': jnn.one_hot(sorted_strings, num_classes=self.output_size),
    }

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
    return input_length
