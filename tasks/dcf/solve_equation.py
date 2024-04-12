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

"""Solve for the value of an unknown variable in an equation."""

import collections
from typing import Sequence

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import tqdm
import tree

from neural_networks_chomsky_hierarchy.tasks import task
from neural_networks_chomsky_hierarchy.tasks.dcf import modular_arithmetic_brackets as mab


def generate_equation_and_solution(
    modulus: int,
    length: int,
) -> tuple[str, int]:
  """Returns a modular arithmetic equation with brackets, and its solution.

  The values are in {0, 1, ..., modulus-1}, and the unknown
  value is x. The allowed operations are either {+, -} (mult=False) or
  {+, -, *} (mult=True).
  Warning: if mult=True, x might have multiple valid solutions.

  Args:
    modulus: The modulus to use for the expression.
    length: The length of the expression.

  Raises:
    ValueError if the length is < 3.
  """

  # Generate the expression.
  expr, val = mab.generate_one_expression_and_result(
      modulus,
      length - 2,
      # We use mult=False by default here, otherwise equations could have
      # multiple solutions if the variable 'x' or some expression containing the
      # variable is multiplied by 0.
      mult=False,
  )

  # Replace random digit with 'x'.
  idx = np.random.randint(low=0, high=len(expr))
  digits = [str(n) for n in range(modulus)]
  while expr[idx] not in digits:
    idx = (idx + 1) % (length - 2)
  solution = int(expr[idx])
  equation = f'{expr[:idx]}x{expr[idx + 1:]}={val}'
  return equation, solution


def generate_raw_dataset(
    n: int,
    lengths: Sequence[int],
    modulus: int,
    with_tqdm: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
  """Generates a dataset of equations and their solutions.

  Args:
    n: The number of datapoints in the dataset.
    lengths: The lengths of the sequences to generate. n is evenly distributed
      over these lengths.
    modulus: Modulus used to compute the expressions.
    with_tqdm: As the computation might be long, whether to add a tqdm progress
      bar or not.

  Returns:
    A dict which keys are the passed lengths, and the values are dicts with keys
    'equations' and 'solutions', and values are the data numpy arrays.
  """
  alphabet_to_int = {
      '+': modulus,
      '-': modulus + 1,
      '(': modulus + 2,
      ')': modulus + 3,
      'x': modulus + 4,
      '=': modulus + 5,
  }
  for x in range(modulus):
    alphabet_to_int[str(x)] = x

  sequences = collections.defaultdict(lambda: {  # pylint: disable=g-long-lambda
      'equations': [],
      'solutions': []
  })
  range_lengths = tqdm.tqdm(lengths) if with_tqdm else lengths
  for length in range_lengths:
    for _ in range(n // len(lengths)):
      seq, label = generate_equation_and_solution(modulus, length)
      seq = [alphabet_to_int[x] for x in seq]
      sequences[length]['equations'].append(seq)
      sequences[length]['solutions'].append(label)
  # Convert the list of numbers we have to arrays at the leaves.
  sequences = tree.traverse(
      lambda l: np.array(l, dtype=np.int32) if isinstance(l, list) else l,
      sequences,
      top_down=False,
  )
  return dict(sequences)


class SolveEquation(task.GeneralizationTask):
  """A task with the goal of solving an modular equation for an unknown.

  Note that the equations do not contain any multiplication as it could lead to
  multiple solutions (multiplication by zero).
  """

  def __init__(self, modulus: int = 5) -> None:
    """Initializes the modular arithmetic task.

    Args:
      modulus: The modulus used for the computation. We use 5 in the paper.
    """
    self._modulus = modulus

  def sample_batch(
      self,
      rng: jnp.ndarray,
      batch_size: int,
      length: int,
  ) -> task.Batch:
    """Returns a batch of inputs/outputs."""
    if length < 3:
      return {
          'input':
              jnn.one_hot(
                  jnp.zeros((batch_size, length)), num_classes=self.input_size),
          'output':
              jnn.one_hot(
                  jnp.zeros((batch_size,)), num_classes=self.output_size)
      }
    batch = generate_raw_dataset(
        batch_size,
        lengths=[length],
        modulus=self._modulus,
    )[length]
    inputs = jnn.one_hot(batch['equations'], self.input_size)
    output = jnn.one_hot(batch['solutions'], self.output_size)
    return {'input': inputs, 'output': output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self._modulus + 6

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._modulus
