# Copyright 2023 DeepMind Technologies Limited
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
  # Change your hyperparameters here. See constants.py for possible tasks and
  # architectures.
  batch_size = 128
  sequence_length = 40
  task = 'even_pairs'
  architecture = 'tape_rnn'
  architecture_params = {
      'hidden_size': 256, 'memory_cell_size': 8, 'memory_size': 40}

  # Create the task.
  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(1, sequence_length + 1)))
  task = constants.TASK_BUILDERS[task]()

  # Create the model.
  is_autoregressive = False
  computation_steps_mult = 0
  single_output = task.output_length(10) == 1
  model = constants.MODEL_BUILDERS[architecture](
      output_size=task.output_size,
      return_all_outputs=True,
      **architecture_params)
  if is_autoregressive:
    if 'transformer' not in architecture:
      model = utils.make_model_with_targets_as_input(
          model, computation_steps_mult
      )
    model = utils.add_sampling_to_autoregressive_model(model, single_output)
  else:
    model = utils.make_model_with_empty_targets(
        model, task, computation_steps_mult, single_output
    )
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
      log_frequency=100,
      length_curriculum=curriculum,
      batch_size=batch_size,
      task=task,
      model=model,
      loss_fn=loss_fn,
      learning_rate=1e-3,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=100,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64,
      is_autoregressive=is_autoregressive)

  training_worker = training.TrainingWorker(training_params, use_tqdm=True)
  _, eval_results, _ = training_worker.run()

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[sequence_length + 1:])
  print(f'Network score: {score}')


if __name__ == '__main__':
  app.run(main)
