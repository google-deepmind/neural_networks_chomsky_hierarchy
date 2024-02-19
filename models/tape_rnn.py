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

"""Implements the Tape RNN."""

import abc
import functools
from typing import Any, Optional, Sequence

import chex
import haiku as hk
import jax
from jax import nn as jnn
from jax import numpy as jnp

# The first element is the memory, the second is the hidden internal state, and
# the third is the input length, necessary for adaptive actions.
_TapeRNNState = tuple[chex.Array, chex.Array, chex.Array]


class TapeRNNCore(hk.RNNCore, abc.ABC):
  """Core for the tape RNN."""

  def __init__(
      self,
      memory_cell_size: int,
      memory_size: int = 30,
      n_tapes: int = 1,
      mlp_layers_size: Sequence[int] = (64, 64),
      inner_core: type[hk.RNNCore] = hk.VanillaRNN,
      name: Optional[str] = None,
      **inner_core_kwargs: Any
  ):
    """Initializes.

    Args:
      memory_cell_size: The dimension of the vectors we put in memory.
      memory_size: The size of the tape, fixed value along the episode.
      n_tapes: Number of tapes to use. Default is 1.
      mlp_layers_size: Sizes for the inner MLP layers. Can be empty, in which
        case the MLP is a linear layer.
      inner_core: The inner RNN core builder.
      name: See base class.
      **inner_core_kwargs: The arguments to be passed to the inner RNN core
        builder.
    """
    super().__init__(name=name)
    self._rnn_core = inner_core(**inner_core_kwargs)
    self._mlp_layers_size = mlp_layers_size
    self._memory_cell_size = memory_cell_size
    self._memory_size = memory_size
    self._n_tapes = n_tapes

  @abc.abstractmethod
  def _tape_operations(
      self, eye_memory: chex.Array, input_length: int
  ) -> list[chex.Array]:
    """Returns a set of updated memory slots.

    An eye matrix is passed and corresponds to the positions of the memory
    slots. This method returns a matrix with the new positions associated with
    the actions. For instance, for a 'left' action, the new matrix will just be
    a roll(eye_memory, shift=-1). This is general enough to allow any
    permutation on the indexes.

    Args:
      eye_memory: An eye matrix of shape [memory_size, memory_size].
      input_length: The length of the input sequence. Can be useful for some
        operations.
    """

  @property
  @abc.abstractmethod
  def num_actions(self) -> int:
    """Returns the number of actions which can be taken on the tape."""

  def __call__(
      self, inputs: chex.Array, prev_state: _TapeRNNState
  ) -> tuple[chex.Array, _TapeRNNState]:
    """Steps the tape RNN core."""
    memories, old_core_state, input_length = prev_state

    # The network can always read the value of the current cell.
    batch_size = memories.shape[0]
    current_memories = memories[:, :, 0, :]
    current_memories = jnp.reshape(
        current_memories, (batch_size, self._n_tapes * self._memory_cell_size))
    inputs = jnp.concatenate([inputs, current_memories], axis=-1)
    new_core_output, new_core_state = self._rnn_core(inputs, old_core_state)
    readout_mlp = hk.nets.MLP(
        list(self._mlp_layers_size) + [self._n_tapes * self._memory_cell_size])
    write_values = readout_mlp(new_core_output)
    write_values = jnp.reshape(
        write_values, (batch_size, self._n_tapes, self._memory_cell_size))

    # Shape (batch_size, num_actions).
    actions = []
    for _ in range(self._n_tapes):
      actions.append(
          jnn.softmax(hk.Linear(self.num_actions)(new_core_output), axis=-1))
    actions = jnp.stack(actions, axis=1)

    update_memory = functools.partial(
        self._update_memory, input_length=input_length[0])
    new_memories = jax.vmap(
        update_memory, in_axes=1, out_axes=1)(memories, actions, write_values)
    return new_core_output, (new_memories, new_core_state, input_length)

  def initial_state(self, batch_size: Optional[int],
                    input_length: int) -> _TapeRNNState:  # pytype: disable=signature-mismatch
    """Returns the initial state of the core."""
    core_state = self._rnn_core.initial_state(batch_size)
    memories = jnp.zeros(
        (batch_size, self._n_tapes, self._memory_size, self._memory_cell_size))
    return memories, core_state, jnp.array([input_length])

  def _update_memory(self, memory: chex.Array, actions: chex.Array,
                     write_values: chex.Array, input_length: int) -> chex.Array:
    """Computes the new memory based on the `actions` and `write_values`.

    Args:
      memory: The current memory with shape `[batch_size, memory_size,
        memory_cell_size]`.
      actions: The action probabilities with shape `[batch_size, num_actions]`.
      write_values: The values added to the first memory entry with shape
        `[batch_size, memory_cell_size]`.
      input_length: The length of the input.

    Returns:
      The new memory with shape `[batch_size, memory_size]`.
    """
    _, memory_size, _ = memory.shape

    memory_with_write = jnp.concatenate(
        [jnp.expand_dims(write_values, axis=1), memory[:, 1:]], axis=1)

    eye_memory = jnp.eye(memory_size)
    operations = self._tape_operations(eye_memory, input_length)
    apply_operation = lambda x: jnp.einsum('mM,bMc->bmc', x, memory_with_write)
    memory_operations = jnp.stack(list(map(apply_operation, operations)))
    return jnp.einsum('Abmc,bA->bmc', memory_operations, actions)


class TapeInputLengthJumpCore(TapeRNNCore):
  """A tape-RNN with extra jumps of the length of the input.

  5 possible actions:
    - write and stay
    - write and move one cell left
    - write and move one cell right
    - write and move input_length cells left
    - write and move input_length cells right
  """

  @property
  def num_actions(self) -> int:
    """Returns the number of actions of the tape."""
    return 5

  def _tape_operations(
      self, eye_memory: chex.Array, input_length: int
  ) -> list[chex.Array]:
    write_stay = eye_memory
    write_left = jnp.roll(eye_memory, shift=-1, axis=0)
    write_right = jnp.roll(eye_memory, shift=1, axis=0)
    write_jump_left = jnp.roll(eye_memory, shift=-input_length, axis=0)
    write_jump_right = jnp.roll(eye_memory, shift=input_length, axis=0)
    return [
        write_stay, write_left, write_right, write_jump_left, write_jump_right
    ]
