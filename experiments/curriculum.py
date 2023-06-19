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

"""Curricula over sequence lengths used to evaluate length generalization.

Allows to sample different sequence lengths along training. For instance,
one might want to start with length=1 and regularly increase the length by 1,
every 50k steps.
"""

import abc
from collections.abc import Collection
import random

import numpy as np


class Curriculum(abc.ABC):
  """Curriculum to sample lengths."""

  @abc.abstractmethod
  def sample_sequence_length(self, step: int) -> int:
    """Samples a sequence length from the current distribution."""


class FixedCurriculum(Curriculum):
  """A fixed curriculum, always sampling the same sequence length."""

  def __init__(self, sequence_length: int):
    """Initializes.

    Args:
      sequence_length: The sequence length to sample.
    """
    super().__init__()
    self._sequence_length = sequence_length

  def sample_sequence_length(self, step: int) -> int:
    """Returns a fixed sequence length."""
    del step
    return self._sequence_length


class UniformCurriculum(Curriculum):
  """A uniform curriculum, sampling different sequence lengths."""

  def __init__(self, values: Collection[int]):
    """Initializes.

    Args:
      values: The sequence lengths to sample.
    """
    super().__init__()
    self._values = tuple(values)

  def sample_sequence_length(self, step: int) -> int:
    """Returns a sequence length sampled from a uniform distribution."""
    del step
    return random.choice(self._values)


class ReverseExponentialCurriculum(Curriculum):
  """A reverse exponential curriculum, sampling different sequence lengths."""

  def __init__(self, values: Collection[int], tau: bool):
    """Initializes.

    Args:
      values: The sequence lengths to sample.
      tau: The exponential rate to use.
    """
    super().__init__()
    self._values = tuple(values)
    self._tau = tau

  def sample_sequence_length(self, step: int) -> int:
    """Returns a length sampled from a reverse exponential distribution."""
    del step
    probs = self._tau**np.array(self._values)
    probs = np.array(probs, dtype=np.float32)
    probs = probs / np.sum(probs)
    return np.random.choice(self._values, p=probs)


class RegularIncreaseCurriculum(Curriculum):
  """Curriculum for sequence lengths with a regular increase."""

  def __init__(self, initial_sequence_length: int, increase_frequency: int,
               increase_amount: int, sample_all_length: bool):
    """Initializes.

    Args:
      initial_sequence_length: The value of the sequence length at the beginning
        of the curriculum.
      increase_frequency: How often we increase the possible sequence length.
      increase_amount: The amount of the increase in length.
      sample_all_length: Whether to sample all length lower than the current one
        or just return the current one.
    """
    super().__init__()
    self._initial_sequence_length = initial_sequence_length
    self._increase_frequency = increase_frequency
    self._increase_amount = increase_amount
    self._sample_all_length = sample_all_length

  def sample_sequence_length(self, step: int) -> int:
    """Returns a sequence length from the curriculum with the current step."""
    if not self._sample_all_length:
      return self._initial_sequence_length + self._increase_amount * (
          step // self._increase_frequency
      )
    return (
        self._initial_sequence_length
        + self._increase_amount
        * np.random.randint(0, step // self._increase_frequency + 1)
    )
