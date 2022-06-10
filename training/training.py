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

"""Training loop for base generalization experiments."""

import dataclasses
from typing import Tuple, List, Callable, Iterator, Mapping, Optional, Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from neural_networks_chomsky_hierarchy.training import range_evaluation


_Batch = Mapping[str, jnp.ndarray]
_LossMetrics = Optional[Mapping[str, jnp.ndarray]]


@dataclasses.dataclass
class ClassicTrainingParams:
  """Parameters needed to train classical architectures."""
  seed: int  # Used to sample during forward pass (e.g. from final logits).
  model_init_seed: int  # Used to initialize model parameters.
  training_steps: int
  eval_frequency: int

  training_dataset: Callable[[], Iterator[_Batch]]
  validation_dataset: Callable[[], Iterator[_Batch]]

  model: hk.Transformed
  loss_fn: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, _LossMetrics]]
  learning_rate: float
  l2_weight: float
  test_model: Optional[hk.Transformed] = None
  max_grad_norm: float = 1.

  compute_full_range_test: bool = False
  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100

  accuracy_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                 jnp.ndarray]] = None
  sample_batch: Optional[Callable[[jnp.ndarray, int, int], _Batch]] = None


def loop(
    training_params: ClassicTrainingParams,
    use_tqdm: bool = False
) -> Tuple[List[Mapping[str, Any]], Optional[List[Mapping[str, Any]]], Any]:
  """Trains the model with the provided parameters.

  Args:
    training_params: The training parameters.
    use_tqdm: Whether to add a progress bar to stdout.

  Returns:
    Results (various training and validation metrics), network parameters.
  """
  training_results = []
  model = training_params.model

  apply_fn = jax.jit(model.apply)
  training_dataset = training_params.training_dataset()
  validation_dataset = training_params.validation_dataset()

  optimizer = optax.chain(
      optax.clip_by_global_norm(training_params.max_grad_norm),
      optax.adam(training_params.learning_rate))

  rng_seq = hk.PRNGSequence(training_params.seed)
  dummy_batch = next(training_dataset)
  params = model.init(
      jax.random.PRNGKey(training_params.model_init_seed), dummy_batch["input"])
  opt_state = optimizer.init(params)

  @jax.jit
  def loss_fn(params: hk.Params, rng_key: jnp.ndarray,
              batch: _Batch) -> Tuple[float, _LossMetrics]:
    outputs = model.apply(params, rng_key, batch["input"])
    return training_params.loss_fn(outputs, batch["output"])

  @jax.jit
  def update(
      params: hk.Params, rng_key: jnp.ndarray, opt_state: optax.OptState,
      batch: _Batch
  ) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics]]:
    """Applies a single SGD update to the router parameters."""
    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(params, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, (loss, metrics)

  steps = range(training_params.training_steps + 1)
  if use_tqdm:
    steps = tqdm.tqdm(steps)
  for step in steps:
    train_batch = next(training_dataset)
    params, opt_state, (train_loss, train_metrics) = update(
        params=params,
        rng_key=next(rng_seq),
        opt_state=opt_state,
        batch=train_batch)

    eval_freq = training_params.eval_frequency
    if eval_freq > 0 and step % eval_freq == 0:
      valid_batch = next(validation_dataset)
      valid_outputs = apply_fn(params, next(rng_seq), valid_batch["input"])
      (valid_loss,
       valid_metrics) = training_params.loss_fn(valid_outputs,
                                                valid_batch["output"])

      log_data = {
          "train_loss": float(train_loss),
          "valid_loss": float(valid_loss),
          "step": step
      }

      for key, value in train_metrics.items():
        log_data[f"train_metrics.{key}"] = np.array(value)
      for key, value in valid_metrics.items():
        log_data[f"valid_metrics.{key}"] = np.array(value)

      if training_params.accuracy_fn is not None:
        log_data["valid_accuracy"] = float(
            jnp.mean(
                training_params.accuracy_fn(valid_outputs,
                                            valid_batch["output"])))
        train_outputs = apply_fn(params, next(rng_seq), train_batch["input"])
        log_data["train_accuracy"] = float(
            jnp.mean(
                training_params.accuracy_fn(train_outputs,
                                            train_batch["output"])))

      print(log_data)
      training_results.append(log_data)

  # Evaluation over all lengths just after training.
  evaluation_results = None
  if training_params.compute_full_range_test:
    eval_params = range_evaluation.EvaluationParams(
        model=training_params.test_model or model,
        params=params,
        accuracy_fn=training_params.accuracy_fn,
        sample_batch=training_params.sample_batch,
        max_test_length=training_params.max_range_test_length,
        total_batch_size=training_params.range_test_total_batch_size,
        sub_batch_size=training_params.range_test_sub_batch_size,
    )
    evaluation_results = range_evaluation.range_evaluation(
        eval_params, use_tqdm=use_tqdm)

  return training_results, evaluation_results, params
