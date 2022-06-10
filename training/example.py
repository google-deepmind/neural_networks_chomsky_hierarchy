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

"""Example script to train and evaluate a network."""

from absl import app

import haiku as hk
import jax.numpy as jnp
import numpy as np

from neural_networks_chomsky_hierarchy.training import constants
from neural_networks_chomsky_hierarchy.training import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.training import training
from neural_networks_chomsky_hierarchy.training import utils


def main(unused_argv) -> None:
  # Change your hyperparameters here. See constants.py for possible tasks and
  # architectures.
  batch_size = 128
  sequence_length = 40
  validation_sequence_length = 50  # Only used as a hint of performance.
  task = 'even_pairs'
  architecture = 'lstm'
  architecture_params = {'hidden_size': 256}

  # Create the task.
  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(1, sequence_length + 1)))
  task = constants.TASK_BUILDERS[task](
      training_data_seed=0,
      training_curriculum=curriculum,
      training_batch_size=batch_size,
      validation_data_seed=0,
      validation_sequence_length=validation_sequence_length,
      validation_batch_size=batch_size,
      pad_sequences_to=None)

  # Create the model.
  model = constants.MODEL_BUILDERS[architecture](
      output_size=task.output_size,
      return_all_outputs=True,
      **architecture_params)
  model = utils.wrap_model_with_pad(
      model=model, generalization_task=task,
      computation_steps_mult=0, single_output=True)
  model = hk.transform(model)

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=0,
      model_init_seed=0,
      training_steps=10_000,
      eval_frequency=200,
      training_dataset=task.training_dataset,
      validation_dataset=task.validation_dataset,
      sample_batch=task.sample_batch,
      model=model,
      loss_fn=loss_fn,
      learning_rate=1e-3,
      l2_weight=0.,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=500,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64)

  _, eval_results, _ = training.loop(
      training_params, use_tqdm=True)

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[validation_sequence_length + 1:])
  print(f'Network score: {score}')


if __name__ == '__main__':
  app.run(main)
