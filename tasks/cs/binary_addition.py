# Copyright 2022 DeepMind Technologies Limited
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

"""Binary addition task for generalization."""

from typing import Mapping, List, Tuple

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from neural_networks_chomsky_hierarchy.tasks import task


class BinaryAddition(task.GeneralizationTask):
  """A task which goal is to sum two numbers in binary.

  The input is a string of the form number1+number2, in base 2, reversed order
  (small powers first). Eg:01001+011. The goal of the agent is to output the
  result, also in reversed binary form, which in this case is 18+6=24=00011.
  The output is padded with 0s to match the input length, and the end of the sum
  is denoted with a termination token (i.e., the output has values in {0, 1,
  2}).

  Examples:
    001+01101 = 010112000  (4+22 = 26)
    1001+000001 = 10010120000  (9+32 = 41)
  """

  def _sum_base_2(self, n: List[int], m: List[int]) -> List[int]:
    """Returns the sum of two numbers written in base 2, reversed.

    Args:
      n: First number, list of 0/1. Must be larger or equal than m.
      m: First number, list of 0/1.

    Raises:
      ValueError if len(n) < len(m).
    """
    if len(n) < len(m):
      raise ValueError('n must be the larger number of the two.')

    result = []
    carry = 0
    for i in range(len(n)):
      if i >= len(m):
        # We just have the digits of n left.
        local_sum = int(n[i]) + carry
      else:
        local_sum = int(n[i]) + int(m[i]) + carry
      carry = int(local_sum >= 2)
      result.append(local_sum % 2)
    if carry > 0:
      result.append(carry)
    return result

  def sample_expression_and_result(
      self, length: int) -> Tuple[jnp.ndarray, List[int]]:
    """Returns an expression of the form number1+number2 and its result."""
    if length <= 2:
      value = np.random.randint(low=0, high=2, size=(length,))
      return value, list(value)
    len_n = np.random.randint(low=1, high=length - 1)
    n = np.random.randint(low=0, high=2, size=(len_n,))
    m = np.random.randint(low=0, high=2, size=(length - 1 - len_n,))
    if len(n) > len(m):
      result = self._sum_base_2(list(n), list(m))
    else:
      result = self._sum_base_2(list(m), list(n))
    return np.concatenate([n, np.array([2]), m]), result

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of binary operations and their results."""
    expressions, results = [], []
    for _ in range(batch_size):
      expression, result = self.sample_expression_and_result(length)
      expressions.append(expression)
      # Append the termination token to the result.
      result.append(self.output_size - 1)
      # Pad the result with zeros to match the input length (accounting for the
      # termination token).
      result.extend([0] * (length + 1 - len(result)))
      results.append(result)
    expressions = jnp.array(expressions)
    results = jnp.array(results)

    inputs = jnn.one_hot(expressions, self.input_size)
    output = jnn.one_hot(results, self.output_size)
    return {'input': inputs, 'output': output}

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

  def accuracy_mask(self, target: jnp.ndarray) -> jnp.ndarray:
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
