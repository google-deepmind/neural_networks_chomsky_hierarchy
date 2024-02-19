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

"""Compute whether the number of 01's and 10's is even."""

import functools

import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class EvenPairs(task.GeneralizationTask):
  """A task with the goal of checking whether the number of 01s and 10s is even.

  The input is a binary string, composed of 0s and 1s. If the result is even,
  the class is 0, otherwise it's one.

  Examples:
    001110 -> 1 '10' and 1 '01' -> class 0
    0101001 -> 2 '10' and 3 '01' -> class 1

  Note the sampling is jittable so this task is fast.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    strings = jrandom.randint(
        rng,
        shape=(batch_size, length),
        minval=0,
        maxval=2,
    )
    one_hot_strings = jnn.one_hot(strings, num_classes=2)
    unequal_pairs = jnp.logical_xor(strings[:, :-1], strings[:, 1:])
    odd_unequal_pairs = jnp.sum(unequal_pairs, axis=-1) % 2
    return {
        'input': one_hot_strings,
        'output': jnn.one_hot(odd_unequal_pairs, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
