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

"""Equal repeats task for generalization."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class EqualRepeats(task.GeneralizationTask):
  """A task which goal is to classify binary strings.

  The strings are of the form 0^l 1^m 0^n, where either l=m, m=n or l=n. The
  class is 0 if l=m, 1 if m=n and 2 if l=n. If m=n=l, the right output class is
  arbitrarily chosen to be 0.

  Examples:
    01000 -> l=m, m!=n and l!=n -> class 0
    0001100 -> l!=m, m=n and l!=n -> class 1
    011110 -> l!=m, m!=n and l=m -> class 2
    000111000 -> l=m, m=n and l=n -> class 0
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    # Case l=m
    max_index = jrandom.randint(rng, (batch_size // 3, 1), 1, length // 2 + 1)
    current_index = jnp.arange(length).reshape((1, length))
    bool_values = jnp.logical_and(current_index >= max_index,
                                  current_index < 2 * max_index)
    first_class_strings = jnp.array(bool_values, dtype=jnp.float32)
    # Case m=n
    second_class_strings = jnp.flip(first_class_strings, axis=-1)
    # Case l=n
    n_zeros = jrandom.randint(rng, (batch_size - 2 * (batch_size // 3), 1), 1,
                              length // 2 + 1)
    current_index = jnp.arange(length).reshape((1, length))
    bool_values = 1 - jnp.logical_or(current_index < n_zeros,
                                     current_index > length - n_zeros - 1)
    third_class_strings = jnp.array(bool_values, dtype=jnp.float32)
    strings = jnp.concatenate(
        [first_class_strings, second_class_strings, third_class_strings],
        axis=0)
    one_hot_strings = jnn.one_hot(strings, num_classes=2)

    classes = jnp.concatenate([
        jnp.zeros((batch_size // 3,)),
        jnp.ones((batch_size // 3,)),
        2 * jnp.ones((batch_size - 2 * (batch_size // 3),)),
    ])

    # Dealing with the limit case l=m=n there.
    if length % 3 == 0:  # Can only happen if the length is a multiple of 3.
      equal_strings = jnp.concatenate(
          [
              jnp.zeros((batch_size, length // 3)),
              jnp.ones((batch_size, length // 3)),
              jnp.zeros((batch_size, length // 3))
          ],
          axis=1,
      )
      classes = jnp.where(jnp.all(strings == equal_strings, axis=1), 0, classes)
    output = jnn.one_hot(classes, num_classes=3)

    return {"input": one_hot_strings, "output": output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 3
