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

"""Missing palindrome task for generalization."""

from typing import Mapping

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from neural_networks_chomsky_hierarchy.tasks import task


class MissingPalindrome(task.GeneralizationTask):
  """A task requiring to predict the missing symbol in a palindrome.

    Given a binary palindrome where exactly one element is omitted (denoted by a
    placeholder token), predict whether that element is 0 or 1.
    Thus, an agent trying to solve this task needs to recognize the underlying
    palindrome to be able to produce the correct output.
    The omitted element is represented by a 2 in the string.

    Examples:
      01102110 -> 0  (with a 1 it wouldn't be a palindrome)
      2111 -> 1
  """

  def sample_batch(
      self,
      rng: jnp.ndarray,
      batch_size: int,
      length: int,
  ) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    half_length, length_is_odd, = np.divmod(length, 2)
    tokens = [np.random.randint(low=0, high=2, size=(batch_size, half_length))]

    if length_is_odd:
      tokens.append(np.random.randint(low=0, high=2, size=(batch_size, 1)))

    # Flip the first half of the input to obtain a palindrome and concatentate.
    tokens.append(np.flip(tokens[0], axis=1))
    tokens = np.concatenate(tokens, axis=1)

    missing_index = np.random.randint(low=0, high=length, size=batch_size)
    output = tokens[np.arange(batch_size), missing_index]
    tokens[np.arange(batch_size), missing_index] = 2

    return {
        'input': jnn.one_hot(tokens, num_classes=self.input_size),
        'output': jnn.one_hot(output, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 3

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
