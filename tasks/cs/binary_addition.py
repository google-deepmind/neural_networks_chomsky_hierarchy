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

"""Add two binary numbers."""

import random
from typing import Sequence

import chex
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from neural_networks_chomsky_hierarchy.tasks import task


def numbers_to_variable_length_binary(
    numbers: Sequence[int],
    lengths: Sequence[int],
    little_endian: bool = True,
) -> list[list[int]]:
  """Returns the binary notation of a certain length for a sequence of numbers.

  Args:
    numbers: The numbers to be converted to binary.
    lengths: The lengths of the binary representations (every number uses its
      own length). This argument has no effect if the binary representation is
      longer than the specified length.
    little_endian: Whether to use little- or big-endian notation.
  """
  binary_strings = [f'{num:b}'.zfill(len) for num, len in zip(numbers, lengths)]

  if little_endian:
    binary_strings = [bin[::-1] for bin in binary_strings]

  return [list(map(int, bin)) for bin in binary_strings]


def numbers_to_fixed_length_binary(
    numbers: Sequence[int],
    length: int,
    little_endian: bool = True,
) -> list[list[int]]:
  """Returns the binary notation of a certain length for a sequence of numbers.

  Args:
    numbers: The numbers to be converted to binary.
    length: The length of the binary representations (all numbers use the same
      length). This argument has no effect if the binary representation is
      longer than the specified length.
    little_endian: Whether to use little- or big-endian notation.
  """
  return numbers_to_variable_length_binary(
      numbers=numbers,
      lengths=[length] * len(numbers),
      little_endian=little_endian,
  )


def expression_from_numbers(
    numbers_n: Sequence[list[int]],
    numbers_m: Sequence[list[int]],
) -> list[list[int]]:
  """Returns an expression with a placeholder value to denote the operation."""
  return [n + [2] + m for n, m in zip(numbers_n, numbers_m)]


class BinaryAddition(task.GeneralizationTask):
  """A task with the goal of summing two numbers in binary (little-endian).

  The input is a string of the form `first_number+second_number` in
  (little-endian) binary notation (e.g., `01001+011`). The goal of the agent is
  to output the result, also in (little-endian) binary form (i.e., in the
  example `18 + 6 = 24 = 00011`). The output is padded with 0s to match the
  input length, and the end of the sum is denoted with a termination token
  (i.e., the output has values in `{0, 1, 2}`).

  Examples:
   001 + 01101    = 010112000     (4 + 22 = 26)
   1001 + 000001  = 10010120000   (9 + 32 = 41)
  """

  def _sample_expressions_and_results(
      self,
      batch_size: int,
      length: int,
  ) -> tuple[Sequence[list[int]], Sequence[list[int]]]:
    """Samples pairs of numbers and sums them in (little-endian) binary.

    We use Python's bignums, which can represent arbitrary-precision integers to
    perform addition of two potentially very large values (roughly of the size
    `2 ** (length // 2)`).

    Args:
      batch_size: The number of expressions and results to sample.
      length: The length of the input expression containing the two numbers and
        the separation token.

    Returns:
      The expression and the sum of the two numbers. The expression has the
      format: `[first_number, 2, second_number]`, where the numbers are in
      (little-endian) binary notation. The sum is also in (little-endian) binary
      notation, without leading (i.e., ending) zeros.
    """
    # If `length <= 2`, we just sample a binary value and return it (without
    # leading zeros in little-endian notation).
    if length <= 2:
      # Since `length <= 2`, we can use `np.random`` without overflow errors.
      numbers = np.random.randint(0, 2**length - 1, size=(batch_size))
      expressions = numbers_to_fixed_length_binary(numbers, length)
      results = numbers_to_fixed_length_binary(numbers, 0)
      return expressions, results

    # We only use `length - 1` tokens for the two values to account for the `+`.
    length_n = np.random.randint(1, length - 1, size=(batch_size,))
    length_m = length - 1 - length_n

    integer_n = [random.randint(1, 2**int(len_n) - 1) for len_n in length_n]
    integer_m = [random.randint(1, 2**int(len_m) - 1) for len_m in length_m]

    binary_n = numbers_to_variable_length_binary(integer_n, length_n)
    binary_m = numbers_to_variable_length_binary(integer_m, length_m)

    expressions = expression_from_numbers(binary_n, binary_m)

    integer_sum = list(map(sum, zip(integer_n, integer_m)))
    results = numbers_to_fixed_length_binary(integer_sum, length=0)

    return expressions, results

  def sample_batch(
      self,
      rng: chex.PRNGKey,
      batch_size: int,
      length: int,
  ) -> task.Batch:
    """Returns a batch of binary additions and their results."""
    del rng

    expressions, results = self._sample_expressions_and_results(
        batch_size=batch_size, length=length)
    # Append the termination token to the result and pad the result with zeros
    # to match the output length (accounting for the termination token).
    results = [res + [2] + [0] * (length - len(res)) for res in results]

    expressions = jnp.array(expressions, dtype=jnp.int32)
    results = jnp.array(results, dtype=jnp.int32)

    return {
        'input': jnn.one_hot(expressions, self.input_size),
        'output': jnn.one_hot(results, self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 3

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 3

  def output_length(self, input_length: int) -> int:
    return input_length + 1

  def accuracy_mask(self, target: chex.Array) -> chex.Array:
    """Computes a mask that ignores everything after the termination token.

    Args:
      target: Target tokens of shape `(batch_size, output_length, output_size)`.

    Returns:
      The mask of shape `(batch_size, output_length)`.
    """
    batch_size, length, _ = target.shape
    termination_indices = jnp.argmax(
        jnp.argmax(target, axis=-1),
        axis=-1,
        keepdims=True,
    )
    indices = jnp.tile(jnp.arange(length), (batch_size, 1))
    return indices <= termination_indices
