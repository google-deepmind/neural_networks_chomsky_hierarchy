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

"""Duplicate a string."""

import functools

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class DuplicateString(task.GeneralizationTask):
  """A task with the goal of duplicating a string.

  The input is a string s_1 ... s_n composed of symbols from a finite set S. The
  output is the same string outputted twice without any separator, ie:
  s_1 ... s_n s_1 ... s_n

  Examples:
    101 -> 101 101
    111111 -> 111111 111111

  In the paper, we use only binary strings (ie S = {0, 1}).
  Note that the sampling is jittable so this task is fast.
  """

  def __init__(self, vocab_size: int = 2) -> None:
    """Initializes the remember_string task.

    Args:
      vocab_size: The size of the alphabet. We use 2 in the paper.
    """
    self._vocab_size = vocab_size

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and their copies."""
    strings = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=self._vocab_size)
    one_hot_strings = jnn.one_hot(strings, num_classes=self._vocab_size)
    output = jnp.concatenate([one_hot_strings, one_hot_strings], axis=1)
    return {"input": one_hot_strings, "output": output}

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
