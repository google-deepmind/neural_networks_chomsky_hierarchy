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

"""Compute whether the number of 1s in a string is even."""

import functools

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class ParityCheck(task.GeneralizationTask):
  """A task with the goal of counting the number of '1' in a string, modulo 2.

  The input is a string, composed of 0s and 1s. If the result is even, the class
  is 0, otherwise it's 1.

  Examples:
    1010100 -> 3 1s (odd) -> class 1
    01111 -> 4 1s (even) -> class 0

  Note that the sampling is jittable so this task is fast.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    strings = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=2)
    n_b = jnp.sum(strings, axis=1) % 2
    n_b = jnn.one_hot(n_b, num_classes=2)
    one_hot_strings = jnn.one_hot(strings, num_classes=2)
    return {"input": one_hot_strings, "output": n_b}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
