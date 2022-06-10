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

"""Divide by 2 task for generalization."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class DivideByTwo(task.GeneralizationTask):
  """A task requiring to divide a number by 2, with a one-hot encoding.

    Given a string of the form 0^(n-1) 1 0^m, output a string of the form
    0^ceil(n/2) 1 0^(floor(n/2) + m). Thus, the length of the inputs and the
    outputs are equal.

    Examples:
      000100 -> 001000
      00001 -> 00100
      010000 -> 010000

    Interpretation: the input corresponds to a one-hot encoding of a number, and
    the output is a one-hot encoding of this number, divided by two.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of monkey-throws-banana inputs and outputs."""

    # Sample target distances.
    position = jrandom.randint(
        rng,
        shape=(batch_size, 1),
        minval=0,
        maxval=length,
    )

    # Build target locations.
    input_strings = jnn.one_hot(position, num_classes=length)
    input_strings = jnp.squeeze(input_strings, axis=1)

    # Build outputs.
    output_strings = jnn.one_hot(jnp.ceil(position / 2), num_classes=length)
    output_strings = jnp.squeeze(output_strings, axis=1)

    # Transform into one-hot-encodings.
    input_one_hot = jnn.one_hot(input_strings, num_classes=self.input_size)
    output_one_hot = jnn.one_hot(output_strings, num_classes=self.output_size)

    return {"input": input_one_hot, "task_info": None, "output": output_one_hot}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2

  def output_length(self, input_length: int) -> int:
    return input_length

  def accuracy_mask(self, target: jnp.ndarray) -> jnp.ndarray:
    """Computes mask for the task.

    We need a mask for this task, otherwise the accuracies get artificially
    boosted to 1 easily. If a network always outputs zeros, as the output is
    itself full of zeros except at one position, it would get an accuracy very
    close to 1.
    Therefore, we only look at the position of the output 1 and the position
    just before it, which must be a 0. In practice this measure is sufficiently
    good to distinguish between bad and good models.

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
    return jnp.logical_or(indices == termination_indices,
                          indices == termination_indices - 1)
