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

"""Stack RNN core.

Following the paper from Joulin et al (2015):
https://arxiv.org/abs/1503.01007

The idea is to add a stack extension to a recurrent neural network to be able to
simulate a machine accepting context-free languages.
The stack is completely differentiable. The actions taken are probabilities
only and therefore no RL is required. The stack and state update are just linear
combinations of the last states and these probabilities.
"""

from typing import Any, Mapping, Optional

import einshape
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp


# First element is the stacks, second is the hidden internal state.
_StackRnnState = tuple[jnp.ndarray, jnp.ndarray]

# Number of actions the stack-RNN can take, namely POP, PUSH and NO_OP.
_NUM_ACTIONS = 3


def _update_stack(stack: jnp.ndarray, actions: jnp.ndarray,
                  push_value: jnp.ndarray) -> jnp.ndarray:
  """Updates the stack values.

  We update the stack in  two steps.
  In the first step, we update the top of the stack, and essentially do:
    stack[0] = push_action * push_value
               + pop_action * stack[1]
               + noop_action * stack[0]

  Then, in the second step, we update the rest of the stack and we move the
  elements up and down, depending on the action executed:
  * If push_action were 1, then we'd be purely pushing a new element
     to the top of the stack, so we'd move all elements down by one.
  * Likewise, if pop_action were 1, we'd be purely taking an element
     off the top of the stack, so we'd move all elements up by one.
  * Finally, if noop_action were 1, we'd leave elements where they were.
  The update is therefore essentially:
    stack[i] = push_action * stack[i-1]
               + pop_action * stack[i+1]
               + noop_action * stack[i]

  Args:
    stack: The current stack, shape (batch_size, stack_size, stack_cell_size).
    actions: The array of probabilities of the actions, shape (batch_size, 3).
    push_value: The vector to push on the stack, if the push action probability
      is positive, shape (batch_size, stack_cell_size).

  Returns:
    The new stack, same shape as the input stack.
  """
  batch_size, stack_size, stack_cell_size = stack.shape

  # Tiling the actions to match the top of the stack.
  # Shape (batch_size, stack_cell_size, _NUM_ACTIONS)
  cell_tiled_stack_actions = einshape.jax_einshape(
      'ba->bsa', actions, s=stack_cell_size)
  push_action = cell_tiled_stack_actions[..., 0]
  pop_action = cell_tiled_stack_actions[..., 1]
  pop_value = stack[..., 1, :]
  no_op_action = cell_tiled_stack_actions[..., 2]
  no_op_value = stack[..., 0, :]

  # Shape (batch_size, 1, stack_cell_size)
  top_new_stack = (
      push_action * push_value + pop_action * pop_value +
      no_op_action * no_op_value)
  top_new_stack = jnp.expand_dims(top_new_stack, axis=1)

  # Tiling the actions to match all of the stack except the top.
  # Shape (batch_size, stack_size,  stack_cell_size, _NUM_ACTIONS)
  stack_tiled_stack_actions = einshape.jax_einshape(
      'ba->bcsa', actions, s=stack_cell_size, c=stack_size - 1)
  push_action = stack_tiled_stack_actions[..., 0]
  push_value = stack[..., :-1, :]
  pop_action = stack_tiled_stack_actions[..., 1]
  pop_extra_zeros = jnp.zeros((batch_size, 1, stack_cell_size))
  pop_value = jnp.concatenate([stack[..., 2:, :], pop_extra_zeros], axis=1)
  no_op_action = stack_tiled_stack_actions[..., 2]
  no_op_value = stack[..., 1:, :]

  # Shape (batch_size, stack_size-1, stack_cell_size)
  rest_new_stack = (
      push_action * push_value + pop_action * pop_value +
      no_op_action * no_op_value)

  # Finally concatenate the new top with the new rest of the stack.
  return jnp.concatenate([top_new_stack, rest_new_stack], axis=1)


class StackRNNCore(hk.RNNCore):
  """Core for the stack RNN."""

  def __init__(
      self,
      stack_cell_size: int,
      stack_size: int = 30,
      n_stacks: int = 1,
      inner_core: type[hk.RNNCore] = hk.VanillaRNN,
      name: Optional[str] = None,
      **inner_core_kwargs: Mapping[str, Any]
  ):
    """Initializes.

    Args:
      stack_cell_size: The dimension of the vectors we put in the stack.
      stack_size: The total number of vectors we can stack.
      n_stacks: Number of stacks to use in the network.
      inner_core: The inner RNN core builder.
      name: See base class.
      **inner_core_kwargs: The arguments to be passed to the inner RNN core
        builder.
    """
    super().__init__(name=name)
    self._rnn_core = inner_core(**inner_core_kwargs)
    self._stack_cell_size = stack_cell_size
    self._stack_size = stack_size
    self._n_stacks = n_stacks

  def __call__(
      self, inputs: jnp.ndarray, prev_state: _StackRnnState
  ) -> tuple[jnp.ndarray, _StackRnnState]:
    """Steps the stack RNN core.

    See base class docstring.

    Args:
      inputs: An input array of shape (batch_size, input_size). The time
        dimension is not included since it is an RNNCore, which is unrolled over
        the time dimension.
      prev_state: A _StackRnnState tuple, consisting of the previous stacks and
        the previous state of the inner core. Each stack has shape (batch_size,
        stack_size, stack_cell_size), such that `stack[n][0]` represents the top
        of the stack for the nth batch item, and `stack[n][-1]` the bottom of
        the stack. The stacks are just the concatenation of all these tensors.

    Returns:
      - output: An output array of shape (batch_size, output_size).
      - next_state: Same format as prev_state.
    """
    stacks, old_core_state = prev_state

    # The network can always read the top of the stack.
    batch_size = stacks.shape[0]
    top_stacks = stacks[:, :, 0, :]
    top_stacks = jnp.reshape(
        top_stacks, (batch_size, self._n_stacks * self._stack_cell_size))
    inputs = jnp.concatenate([inputs, top_stacks], axis=-1)
    new_core_output, new_core_state = self._rnn_core(inputs, old_core_state)
    push_values = hk.Linear(self._n_stacks * self._stack_cell_size)(
        new_core_output)
    push_values = jnp.reshape(
        push_values, (batch_size, self._n_stacks, self._stack_cell_size))

    # Shape (batch_size, _NUM_ACTIONS)
    stack_actions = jnn.softmax(
        hk.Linear(self._n_stacks * _NUM_ACTIONS)(new_core_output), axis=-1)
    stack_actions = jnp.reshape(stack_actions,
                                (batch_size, self._n_stacks, _NUM_ACTIONS))

    new_stacks = jax.vmap(
        _update_stack, in_axes=1, out_axes=1)(stacks, stack_actions,
                                              push_values)
    return new_core_output, (new_stacks, new_core_state)

  def initial_state(self, batch_size: Optional[int]) -> _StackRnnState:
    """Returns the initial state of the core, a hidden state and an empty stack."""
    core_state = self._rnn_core.initial_state(batch_size)
    stacks = jnp.zeros(
        (batch_size, self._n_stacks, self._stack_size, self._stack_cell_size))
    return stacks, core_state
