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

"""Implements the Tape RNN."""

from typing import Any, Optional, Tuple, Type

import haiku as hk
from jax import nn as jnn
from jax import numpy as jnp

# The first element is the memory, the second is the hidden internal state, and
# the third is the input length, necessary for adaptive actions.
_TapeRNNState = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

# The actions are:
#  - write and stay at the current position
#  - write and move left
#  - write and move right
#  - move N steps to the left, where N is a constant
#  - move N steps to the right, where N is a constant
_NUM_ACTIONS = 5


def _update_memory(memory: jnp.ndarray, actions: jnp.ndarray,
                   write_values: jnp.ndarray,
                   steps_amplifier: int) -> jnp.ndarray:
  """Computes the new memory based on the `actions` and `values`.

  Args:
    memory: The current memory with shape `[batch_size, memory_size,
      memory_cell_size]`.
    actions: The action probabilities with shape `[batch_size, 5]`.
    write_values: The values added to the first memory entry with shape
      `[batch_size, memory_cell_size]`.
    steps_amplifier: The number of steps used for the 'big' move actions (4 and
      5).

  Returns:
    The new memory with shape `[batch_size, memory_size]`.
  """
  _, memory_size, _ = memory.shape

  memory_with_write = jnp.concatenate(
      [jnp.expand_dims(write_values, axis=1), memory[:, 1:]], axis=1)

  write_stay = jnp.eye(memory_size)
  write_left = jnp.roll(jnp.eye(memory_size), shift=-1, axis=0)
  write_right = jnp.roll(jnp.eye(memory_size), shift=1, axis=0)
  big_left = jnp.roll(jnp.eye(memory_size), shift=-steps_amplifier, axis=0)
  big_right = jnp.roll(jnp.eye(memory_size), shift=steps_amplifier, axis=0)

  memory_operations = jnp.stack([
      jnp.einsum('mM,bMc->bmc', write_stay, memory_with_write),
      jnp.einsum('mM,bMc->bmc', write_left, memory_with_write),
      jnp.einsum('mM,bMc->bmc', write_right, memory_with_write),
      jnp.einsum('mM,bMc->bmc', big_left, memory),
      jnp.einsum('mM,bMc->bmc', big_right, memory),
  ])
  return jnp.einsum('Abmc,bA->bmc', memory_operations, actions)


class TapeRNNCore(hk.RNNCore):
  """Core for the tape RNN."""

  def __init__(self,
               memory_cell_size: int,
               memory_size: int = 30,
               input_length: int = 1,
               inner_core: Type[hk.RNNCore] = hk.VanillaRNN,
               name: Optional[str] = None,
               **inner_core_kwargs: Any):
    """Initializes.

    Args:
      memory_cell_size: The dimension of the vectors we put in memory.
      memory_size: The size of the tape, fixed value along the episode.
      input_length: The length of the input, used to compute the actions
        jump-left and jump-right, which jump a number of cells equal to the
        input length.
      inner_core: The inner RNN core builder.
      name: See base class.
      **inner_core_kwargs: The arguments to be passed to the inner RNN core
        builder.
    """
    super().__init__(name=name)
    self._rnn_core = inner_core(**inner_core_kwargs)
    self._memory_cell_size = memory_cell_size
    self._memory_size = memory_size
    self._input_length = input_length

  def __call__(self, inputs: jnp.ndarray,
               prev_state: _TapeRNNState) -> Tuple[jnp.ndarray, _TapeRNNState]:
    """Steps the tape RNN core."""
    memory, old_core_state, input_length = prev_state

    # The network can always read the value of the current cell.
    current_memory = memory[:, 0, :]
    inputs = jnp.concatenate([inputs, current_memory], axis=-1)
    new_core_output, new_core_state = self._rnn_core(inputs, old_core_state)
    write_value = hk.Linear(self._memory_cell_size)(new_core_output)

    # Shape (batch_size, _NUM_ACTIONS)
    actions = jnn.softmax(hk.Linear(_NUM_ACTIONS)(new_core_output), axis=-1)

    new_memory = _update_memory(memory, actions, write_value, input_length[0])
    return new_core_output, (new_memory, new_core_state, input_length)

  def initial_state(self, batch_size: Optional[int]) -> _TapeRNNState:
    """Returns the initial state of the core."""
    core_state = self._rnn_core.initial_state(batch_size)
    memory = jnp.zeros((batch_size, self._memory_size, self._memory_cell_size))
    return memory, core_state, jnp.array([self._input_length])
