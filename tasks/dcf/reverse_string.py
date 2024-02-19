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

"""Compute the reverse of an input string."""

import functools

import jax
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task
from neural_networks_chomsky_hierarchy.tasks.cs import duplicate_string


class ReverseString(duplicate_string.DuplicateString):
  """A task with the goal of reversing a given string.

  The input is a string s_1 ... s_n composed of symbols from a finite set S. The
  output is the string, reversed, ie s_n ... s_1.

  Examples:
    011010 -> 010110
    123021 -> 120321

  In the paper, we use only binary strings (ie S = {0, 1}).
  Note that the sampling is jittable so this task is fast.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and their reversed version."""
    batch = super().sample_batch(rng, batch_size, length)
    batch['output'] = jnp.flip(batch['input'], axis=1)
    return batch

  def output_length(self, input_length: int) -> int:
    """Returns the output length for a given input length."""
    return input_length
