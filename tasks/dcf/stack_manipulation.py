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

"""Manipulate an input stack, using the input actions."""

from typing import Mapping, Tuple, List

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from neural_networks_chomsky_hierarchy.tasks import task

ACTIONS = {'POP': 0, 'PUSH': 1}


class StackManipulation(task.GeneralizationTask):
  """A task which goal is to follow instructions and return the end stack.

  The input is composed of a stack of 0s and 1s, followed by a sequence of
  instructions POP/PUSH (represented by 2s and 3s). Note that the stack is given
  bottom to top.The agent needs to simply follow the instructions given and
  output the final stack, written from bottom to top. Note that the PUSH
  instruction means specifically to push a 1 to the stack.
  The output is padded with 0s to match the input length, and the end of the
  result is denoted with a termination symbol (i.e., the output has values in
  {0, 1, 2}).

  Examples:
    0 1 0 0 PUSH POP POP
      initial 0 0 1 0  (remember it's given reversed)
      PUSH    1 0 0 1 0
      POP     0 0 1 0
      POP     0 1 0
    -> 0 1 0 2

    1 1 0 POP POP POP
      initial 0 1 1
      POP     1 1
      POP     1
      POP
    -> 2 0 0 (empty stack!)
  """

  def _sample_expression_and_result(
      self, length: int) -> Tuple[np.ndarray, List[int]]:
    """Returns an expression with stack instructions, and the result stack."""
    if length == 1:
      value = np.random.randint(low=0, high=2, size=(1,))
      return value, list(value)

    # Initialize the stack content and the actions (POP/PUSH).
    stack_length = np.random.randint(low=1, high=length)
    stack = np.random.randint(low=0, high=2, size=(stack_length,))
    actions = np.random.randint(low=0, high=2, size=(length - stack_length,))

    # Apply the actions on the stack.
    current_stack = list(stack)[::-1]
    for action in actions:
      if action == ACTIONS['POP']:  # POP
        if len(current_stack) > 1:  # Otherwise, do nothing.
          current_stack.pop(0)
      elif action == ACTIONS['PUSH']:  # PUSH
        current_stack = [1] + current_stack

    return np.concatenate([stack, actions + 2]), current_stack

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    expressions, results = [], []
    for _ in range(batch_size):
      expression, result = self._sample_expression_and_result(length)
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
    """Returns the input size for the models.

    The value is 4 because we have two possible tokens in the stack, plus two
    tokens to describe the PUSH and POP actions.
    """
    return 4

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 3

  def output_length(self, input_length: int) -> int:
    """Returns the output length of the task."""
    return input_length + 1

  def accuracy_mask(self, target: jnp.ndarray) -> jnp.ndarray:
    """Computes mask that ignores everything after the termination tokens.

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
