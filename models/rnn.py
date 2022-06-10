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

"""Builders for RNN/LSTM cores."""

from typing import Callable, Any, Type

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.models import tape_rnn


def make_rnn(
    output_size: int,
    rnn_core: Type[hk.RNNCore],
    return_all_outputs: bool = False,
    **rnn_kwargs: Any) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns an RNN model, not haiku transformed.

  Only the last output in the sequence is returned. A linear layer is added to
  match the required output_size.

  Args:
    output_size: The output size of the model.
    rnn_core: The haiku RNN core to use. LSTM by default.
    return_all_outputs: Whether to return the whole sequence of outputs of the
      RNN, or just the last one.
    **rnn_kwargs: Kwargs to be passed to the RNN core.
  """

  def rnn_model(x: jnp.array, input_length: int = 1) -> jnp.ndarray:
    if rnn_core == tape_rnn.TapeRNNCore:
      rnn_kwargs['input_length'] = input_length
    core = rnn_core(**rnn_kwargs)

    initial_state = core.initial_state(x.shape[0])
    output, _ = hk.dynamic_unroll(
        core, x, initial_state, time_major=False, return_all_states=True)

    if not return_all_outputs:
      output = output[:, -1, :]  # (batch, time, alphabet_dim)
    output = jnn.relu(output)
    return hk.Linear(output_size)(output)

  return rnn_model
