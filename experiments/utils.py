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

"""Provides utility functions for training and evaluation."""

import inspect
from typing import Any, Callable

import chex
import haiku as hk
from jax import nn as jnn
from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task

COMPUTATION_EMPTY_TOKEN = 0
OUTPUT_EMPTY_TOKEN = 1


def make_model_with_empty_targets(
    model: Callable[[chex.Array], chex.Array],
    generalization_task: task.GeneralizationTask,
    computation_steps_mult: int = 0,
    single_output: bool = False,
) -> Callable[[chex.Array], chex.Array]:
  """Returns a wrapped model that pads the inputs to match the output length.

  For a given input tape `input_tape` of vocabulary size `vocab_size`, the
  wrapped model will process a tape of the format
  [`input_tape`, `empty_tape`], where the empty tape token is `vocab_size + 1`.
  The `empty_tape` has the same length as the task output.

  Args:
    model: A model function that converts inputs to outputs.
    generalization_task: The task that we train on.
    computation_steps_mult: The amount of empty cells to append to the input
      tape. This variable is a multiplier and the actual number of cells is
      `computation_steps_mult * input_length`.
    single_output: Whether to return the squeezed tensor of values.
  """

  def new_model(x: chex.Array) -> chex.Array:
    batch_size, input_length, input_size = x.shape
    output_length = generalization_task.output_length(input_length)
    extra_dims_onehot = 1 + int(computation_steps_mult > 0)
    final_input_size = input_size + extra_dims_onehot

    # Add trailing zeros to account for new final_input_size.
    extra_zeros_x = jnp.zeros(
        (batch_size, input_length, final_input_size - input_size)
    )
    x = jnp.concatenate([x, extra_zeros_x], axis=-1)

    computation_tape = jnp.full(
        (batch_size, computation_steps_mult * input_length),
        fill_value=input_size + COMPUTATION_EMPTY_TOKEN)
    computation_tape = jnn.one_hot(
        computation_tape, num_classes=final_input_size
    )

    output_tokens = jnp.full(
        (batch_size, output_length),
        fill_value=input_size
        + OUTPUT_EMPTY_TOKEN
        - int(computation_steps_mult == 0),
    )
    output_tokens = jnn.one_hot(output_tokens, num_classes=final_input_size)
    final_input = jnp.concatenate([x, computation_tape, output_tokens], axis=1)

    if 'input_length' in inspect.getfullargspec(model).args:
      output = model(final_input, input_length=input_length)  # pytype: disable=wrong-keyword-args
    else:
      output = model(final_input)
    output = output[:, -output_length:]
    if single_output:
      output = jnp.squeeze(output, axis=1)
    return output

  return new_model


def make_model_with_targets_as_input(
    model: Callable[[chex.Array], chex.Array], computation_steps_mult: int = 0
) -> Callable[[chex.Array, chex.Array], chex.Array]:
  """Returns a wrapped model that takes the targets as inputs.

  This function is useful for the autoregressive case where we pass the targets
  as inputs to the model. The final input looks like:
    [inputs, computation_tokens, output_token, targets]

  Args:
    model: A haiku model that takes 'x' as input.
    computation_steps_mult: The amount of computation tokens to append to the
      input tape. This variable is a multiplier and the actual number of cell is
      computation_steps_mult * input_length.
  """

  def new_model(x: chex.Array, y: chex.Array) -> chex.Array:
    """Returns an output from the inputs and targets.

    Args:
      x: One-hot input vectors, shape (B, T, input_size).
      y: One-hot target output vectors, shape (B, T, output_size).
    """
    batch_size, input_length, input_size = x.shape
    _, output_length, output_size = y.shape
    extra_dims_onehot = 1 + int(computation_steps_mult > 0)
    final_input_size = max(input_size, output_size) + extra_dims_onehot

    # Add trailing zeros to account for new final_input_size.
    extra_zeros_x = jnp.zeros(
        (batch_size, input_length, final_input_size - input_size)
    )
    x = jnp.concatenate([x, extra_zeros_x], axis=-1)
    extra_zeros_y = jnp.zeros(
        (batch_size, output_length, final_input_size - output_size)
    )
    y = jnp.concatenate([y, extra_zeros_y], axis=-1)

    computation_tape = jnp.full(
        (batch_size, computation_steps_mult * input_length),
        fill_value=input_size + COMPUTATION_EMPTY_TOKEN,
    )
    computation_tape = jnn.one_hot(
        computation_tape, num_classes=final_input_size
    )

    output_token = jnp.full(
        (batch_size, 1),
        fill_value=input_size
        + OUTPUT_EMPTY_TOKEN
        - int(computation_steps_mult == 0),
    )
    output_token = jnn.one_hot(output_token, num_classes=final_input_size)
    final_input = jnp.concatenate(
        [x, computation_tape, output_token, y], axis=1
    )

    if 'input_length' in inspect.getfullargspec(model).args:
      output = model(final_input, input_length=input_length)  # pytype: disable=wrong-keyword-args
    else:
      output = model(final_input)

    return output[:, -output_length - 1 : -1]

  return new_model


def add_sampling_to_autoregressive_model(
    model: Callable[[chex.Array, chex.Array], chex.Array],
    single_output: bool = False,
) -> Callable[[chex.Array, chex.Array, bool], chex.Array]:
  """Adds a 'sample' argument to the model, to use autoregressive sampling."""

  def new_model_with_sampling(
      x: chex.Array,
      y: chex.Array,
      sample: bool,
  ) -> chex.Array:
    """Returns an autoregressive model if `sample == True and output_size > 1`.

    Args:
      x: The input sequences of shape (b, t, i), where i is the input size.
      y: The target sequences of shape (b, t, o), where o is the output size.
      sample: Whether to evaluate the model using autoregressive decoding.
    """
    output_length = 1 if len(y.shape) == 2 else y.shape[1]
    output_size = y.shape[-1]

    if not sample or output_length == 1:
      output = model(x, y)

    else:

      def evaluate_model_autoregressively(
          idx: int,
          predictions: chex.Array,
      ) -> chex.Array:
        """Iteratively evaluates the model based on the previous predictions.

        Args:
          idx: The index of the target sequence that should be evaluated.
          predictions: The logits for the predictions up to but not including
            the index `idx`.

        Returns:
          The `predictions` array modified only at position `idx` where the
          logits for index `idx` have been inserted.
        """
        one_hot_predictions = jnn.one_hot(
            jnp.argmax(predictions, axis=-1),
            num_classes=output_size,
        )
        logits = model(x, one_hot_predictions)
        return predictions.at[:, idx].set(logits[:, idx])

      output = hk.fori_loop(
          lower=0,
          upper=output_length,
          body_fun=evaluate_model_autoregressively,
          init_val=jnp.empty_like(y),
      )

    if single_output:
      output = jnp.squeeze(output, axis=1)
    return output

  return new_model_with_sampling


def update_tree_with_new_containers(
    tree: Any, update_dict: dict[str, Any]
) -> None:
  """Updates a dataclass tree in place, adding new containers.

  This method is useful for the nested library to add fields to a tree, for
  which containers have not been created.
  For instance, if A is a dataclass with attribute architecture_params, and we
  want to add the value architecture_params.rnn_model.size, we need to create
  the container 'rnn_model' inside architecture_params.

  Args:
    tree: An object with attribute (typically a dataclass).
    update_dict: A dict of nested updates. See example above.
  """
  for key in update_dict:
    subkeys = key.split('.')
    if len(subkeys) >= 2:
      # Example: architecture.params.size
      for i in range(0, len(subkeys) - 2):
        getattr(tree, subkeys[i])[subkeys[i + 1]] = {}
