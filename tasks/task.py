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

"""Base class for generalization tasks."""

import abc
import itertools
from typing import Iterator, Optional, TypedDict

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

from neural_networks_chomsky_hierarchy.training import curriculum as curriculum_lib


Batch = TypedDict('Batch', {'input': jnp.ndarray, 'output': jnp.ndarray})


class GeneralizationTask(abc.ABC):
  """A task for the generalization project.

  Contains a training dataset, a validation dataset and their specs. Also
  contains the objectives (loss and accuracy).
  """

  def __init__(self,
               training_data_seed: int,
               training_curriculum: curriculum_lib.Curriculum,
               training_batch_size: int,
               validation_data_seed: int,
               validation_sequence_length: int,
               validation_batch_size: int,
               pad_sequences_to: Optional[int] = None):
    """Initializes a generalization task.

    Args:
      training_data_seed: The seed used in the training data generator.
      training_curriculum: The curriculum of sequence lengths to use.
      training_batch_size: The batch size for the training dataset.
      validation_data_seed: The seed used in the validation data generator.
      validation_sequence_length: The length of the sequences in the validation
        dataset.
      validation_batch_size: The batch size for the validation dataset.
      pad_sequences_to: To which length the sequences should be padded. Can be
        None, in which case nothing is done.
    """
    self._training_data_seed = training_data_seed
    self._training_curriculum = training_curriculum
    self._training_batch_size = training_batch_size
    self._validation_data_seed = validation_data_seed
    self._validation_sequence_length = validation_sequence_length
    self._validation_batch_size = validation_batch_size
    self._pad_sequences_to = pad_sequences_to

  @abc.abstractmethod
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Batch:
    """Returns a batch of inputs/outputs."""

  def training_dataset(self) -> Iterator[Batch]:
    """Returns an iterator over random batches of strings and their class."""
    rng_seq = hk.PRNGSequence(self._training_data_seed)
    self._training_curriculum.set_seed(self._training_data_seed + 1)
    for step in itertools.count(0):
      length = self._training_curriculum.sample_sequence_length(step)
      yield self.sample_batch(next(rng_seq), self._training_batch_size, length)

  def validation_dataset(self) -> Iterator[Batch]:
    """Returns an iterator over random batches of strings and their class."""
    rng_seq = hk.PRNGSequence(self._validation_data_seed)
    for rng in rng_seq:
      yield self.sample_batch(
          rng, self._validation_batch_size, self._validation_sequence_length)

  def pointwise_loss_fn(self, output: jnp.ndarray,
                        target: jnp.ndarray) -> jnp.ndarray:
    """Returns the pointwise loss between an output and a target."""
    return -target * jnn.log_softmax(output)

  def accuracy_fn(self, output: jnp.ndarray,
                  target: jnp.ndarray) -> jnp.ndarray:
    """Returns the accuracy between an output and a target."""
    return (jnp.argmax(output,
                       axis=-1) == jnp.argmax(target,
                                              axis=-1)).astype(jnp.float32)

  def accuracy_mask(self, target: jnp.ndarray) -> jnp.ndarray:
    """Returns a mask to compute the accuracies, to remove the superfluous ones."""
    # Target is a shape of shape (B, T, C) where C is the number of classes.
    # We want a mask per input (B, T), so we take this shape.
    return jnp.ones(target.shape[:-1])

  @property
  @abc.abstractmethod
  def input_size(self) -> int:
    """Returns the size of the input of the models trained on this task."""

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    """Returns the size of the output of the models trained on this task."""

  def output_length(self, input_length: int) -> int:
    """Returns the length of the output, given an input length."""
    del input_length
    return 1
