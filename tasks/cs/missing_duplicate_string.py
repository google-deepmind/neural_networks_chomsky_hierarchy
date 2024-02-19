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

"""Predict the missing symbol in a duplicated string."""

import functools

import chex
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class MissingDuplicateString(task.GeneralizationTask):
  """A task with the goal of finding the missing symbol in a duplicated string.

  Given a binary string that is presented twice with exactly one element omitted
  (denoted by the placeholder token `2`), predict the value of that element.
  Thus, an agent trying to solve this task needs to recognize the underlying
  duplicated string to be able to produce the correct output.
  If the length is odd, the duplicated strings of length `length // 2` are
  padded with the empty token `3`.

  Examples
    01100210  ->  1   (the substring is 0110, so the missing value is 1)
    1011213   ->  0   (the subtring is 101, so the missing value is 0)
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      length: int,
  ) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    # For `length == 1`, we cannot meaningfully define substrings of length
    # `length // 2`, so we arbitrarily set the inputs and outputs to `1`.
    if length == 1:
      return {
          'input':
              jnn.one_hot(
                  jnp.ones((batch_size, length)), num_classes=self.input_size),
          'output':
              jnn.one_hot(
                  jnp.ones((batch_size,)), num_classes=self.output_size),
      }

    strings_rng, indices_rng = jrandom.split(rng)
    strings = jrandom.randint(
        strings_rng, shape=(batch_size, length // 2), minval=0, maxval=2)
    duplicated_strings = jnp.concatenate((strings, strings), axis=-1)
    indices = jrandom.randint(
        indices_rng,
        shape=(batch_size,),
        minval=0,
        maxval=duplicated_strings.shape[1])
    output = jax.vmap(lambda x, y: x[y])(duplicated_strings, indices)
    masked_strings = jax.vmap(lambda x, y: x.at[y].set(2))(duplicated_strings,
                                                           indices)

    # If `length` is odd, we pad the strings with the empty token `3` at the end
    # to ensure that the final input length is equal to `length` given the two
    # substrings of length `length // 2`.
    padding = jnp.full((batch_size, length % 2), fill_value=3)
    padded_strings = jnp.concatenate((masked_strings, padding), axis=-1)

    return {
        'input': jnn.one_hot(padded_strings, num_classes=self.input_size),
        'output': jnn.one_hot(output, num_classes=self.output_size)
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 4

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
