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

"""Sort a string by the parity of the indices (odd indices first)."""

import functools

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class OddsFirst(task.GeneralizationTask):
  """A task with the goal of outputting a string's tokens at odd indices first.

  The input is a string s_1 ... s_n composed of symbols from a finite set S. The
  output is the same string, but where the values at odd indexes have been put
  first: s_1 s_3 s_5 ... s_2 s_4 s_6 ...

  Examples:
    00110101 -> 0100 0111
    110 -> 10 1

  In the paper, we use only binary strings (ie S = {0, 1}).
  Note that the sampling is jittable so this task is fast.
  """

  def __init__(self, vocab_size: int = 2) -> None:
    """Initializes the odds_first task.

    Args:
      vocab_size: The size of the alphabet. We use 2 in the paper.
    """
    self._vocab_size = vocab_size

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and their outputs."""
    strings = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=self._vocab_size)
    one_hot_strings = jnn.one_hot(strings, num_classes=self._vocab_size)
    output = jnp.concatenate(
        [one_hot_strings[:, 1::2], one_hot_strings[:, ::2]], axis=1)
    return {"input": one_hot_strings, "output": output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the model."""
    return self._vocab_size

  @property
  def output_size(self) -> int:
    """Returns the output size for the model."""
    return self._vocab_size

  def output_length(self, input_length: int) -> int:
    """Returns the output length for the model."""
    return input_length
