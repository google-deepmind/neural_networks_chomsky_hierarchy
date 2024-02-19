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

"""Compute the floor of the square root of a binary number."""

import math
import random

import chex
import jax.nn as jnn
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task
from neural_networks_chomsky_hierarchy.tasks.cs import binary_addition


class ComputeSqrt(task.GeneralizationTask):
  """A task with the goal of computing the square root of a binary number.

  The input is a number in binary (big-endian), and the output is the floor of
  the square root of this number, also in binary.
  Note the output length ie the length of the square root in binary is always
  ceil(input_length / 2) (because log(sqrt(x)) = 1/2 log(x)).

  Examples:
   100101 = 37 -> square root is 6.08... -> floor(6.08) = 6 -> 101
   111 = 7 -> square root is 2.64 -> floor(2.64) = 2 -> 10
  """

  def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of binary numbers and their square roots, in binary."""
    del rng
    numbers = [random.randint(1, 2**length - 1) for _ in range(batch_size)]
    binary_numbers = binary_addition.numbers_to_fixed_length_binary(
        numbers, length=length, little_endian=False)

    sqrts = list(map(math.isqrt, numbers))
    binary_sqrts = binary_addition.numbers_to_fixed_length_binary(
        sqrts, length=self.output_length(length), little_endian=False)

    binary_numbers = jnp.array(binary_numbers, jnp.int32)
    binary_sqrts = jnp.array(binary_sqrts, jnp.int32)

    inputs = jnn.one_hot(binary_numbers, self.input_size)
    output = jnn.one_hot(binary_sqrts, self.output_size)
    return {'input': inputs, 'output': output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2

  def output_length(self, input_length: int) -> int:
    return math.ceil(input_length / 2)
