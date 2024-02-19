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

"""Compute the final state after randomly walking on a circle."""

import functools

import chex
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class CycleNavigation(task.GeneralizationTask):
  """A task with the goal of computing the final state on a circle.

  The input is a string of actions, composed of 0s, 1s or -1s. The actions give
  directions to take on a finite length circle (0 is for stay, 1 is for right,
  -1 is for left). The goal is to give the final position on the circle after
  all the actions have been taken. The agent starts at position 0.

  By default, the length the circle is 5.

  Examples:
    1 -1 0 -1 -1 -> -2 = class 3
    1 1 1 -1 -> 2 = class 2

  Note that the sampling is jittable so it is fast.
  """

  @property
  def _cycle_length(self) -> int:
    """Returns the cycle length, number of possible states."""
    return 5

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    actions = jrandom.randint(
        rng, shape=(batch_size, length), minval=0, maxval=3)
    final_states = jnp.sum(actions - 1, axis=1) % self._cycle_length
    final_states = jnn.one_hot(final_states, num_classes=self.output_size)
    one_hot_strings = jnn.one_hot(actions, num_classes=self.input_size)
    return {"input": one_hot_strings, "output": final_states}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 3

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._cycle_length
