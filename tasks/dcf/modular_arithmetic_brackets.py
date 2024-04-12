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

"""Modular arithmetic with brackets."""

import collections
from typing import Sequence

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import tqdm
import tree

from neural_networks_chomsky_hierarchy.tasks import task


def generate_one_expression_and_result(
    modulus: int, length: int, mult: bool = False
) -> tuple[str, int]:
  """Returns a modular arithmetic expression with brackets, and its result.

  The values in the expression are in {0, 1, ..., modulus-1}. The allowed
  operations are either {+, -} (mult=False) or {+, -, *} (mult=True).

  Args:
    modulus: The modulus to use for the expression.
    length: The length of the expression.
    mult: Whether to include the multiplication operator in the expressions.

  Raises:
    ValueError if length < 1.
  """

  # Generates a terminal (digit).
  def gen_terminal():
    terminal = np.random.randint(low=0, high=modulus)
    return str(terminal), terminal

  # If length is less than 1, issue an error.
  if length < 1:
    raise ValueError(
        f'Can\'t generate expressions of length < 1. Got {length}.')

  # If length is less than 5, generate a digit d, -d, (d), or (-d).
  if length == 1:
    return gen_terminal()
  elif length == 2:
    term_str, term_val = gen_terminal()
    return f'-{term_str}', -term_val % modulus
  elif length == 3:
    term_str, term_val = gen_terminal()
    return f'({term_str})', term_val % modulus
  elif length == 4:
    term_str, term_val = gen_terminal()
    return f'(-{term_str})', -term_val % modulus

  # First split the length into a left and right part.
  left_length = np.random.randint(low=1, high=length - 3)
  right_length = length - (left_length + 3)
  left_str, left_val = generate_one_expression_and_result(
      modulus, left_length, mult=mult)
  right_str, right_val = generate_one_expression_and_result(
      modulus, right_length, mult=mult)

  # Now sample an operator and return.
  maxop = 3 if mult else 2
  op = np.random.randint(low=0, high=maxop)
  if op == 0:
    return '(' + left_str + '+' + right_str + ')', (left_val +
                                                    right_val) % modulus
  elif op == 1:
    return '(' + left_str + '-' + right_str + ')', (left_val -
                                                    right_val) % modulus
  else:
    return '(' + left_str + '*' + right_str + ')', (left_val *
                                                    right_val) % modulus


def generate_raw_dataset(
    n: int,
    lengths: Sequence[int],
    modulus: int,
    mult: bool = False,
    with_tqdm: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
  """Generates a dataset of maths expressions with brackets, and their results.

  Args:
    n: The number of datapoints in the dataset.
    lengths: The lengths of the sequences to generate. n is evenly distributed
      over these lengths.
    modulus: Modulus used to compute the expressions.
    mult: Whether to include the multiplication operator in the expressions.
    with_tqdm: As the computation might be long, whether to add a tqdm progress
      bar or not.

  Returns:
    A dict which keys are the passed lengths, and the values are dicts with keys
    'equations' and 'solutions', and values are the data numpy arrays.
  """
  alphabet_to_int = {
      '+': modulus,
      '-': modulus + 1,
      '*': modulus + 2,
      '(': modulus + 3,
      ')': modulus + 4,
      'x': modulus + 5,
      '=': modulus + 6,
  }
  for x in range(modulus):
    alphabet_to_int[str(x)] = x

  make_default_dict = lambda: {'expressions': [], 'results': []}
  sequences = collections.defaultdict(make_default_dict)
  range_lengths = tqdm.tqdm(lengths) if with_tqdm else lengths
  for length in range_lengths:
    for _ in range(n // len(lengths)):
      seq, label = generate_one_expression_and_result(modulus, length, mult)
      seq = [alphabet_to_int[x] for x in seq]
      sequences[length]['expressions'].append(seq)
      sequences[length]['results'].append(label)
  sequences = tree.traverse(
      lambda l: np.array(l, dtype=np.int32) if isinstance(l, list) else l,
      sequences,
      top_down=False,
  )
  return dict(sequences)


class ModularArithmeticBrackets(task.GeneralizationTask):
  """A task with the goal of reducing an arithmetic expression with brackets."""

  def __init__(self, modulus: int = 5, mult: bool = False) -> None:
    """Initializes the modular arithmetic task.

    Args:
      modulus: The modulus used for the computation. We use 5 in the paper.
      mult: Whether to add multiplication or use only '+' and '-'.
    """
    self._modulus = modulus
    self._mult = mult

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of inputs/outputs."""
    del rng
    batch = generate_raw_dataset(
        batch_size, lengths=[length], modulus=self._modulus,
        mult=self._mult)[length]
    inputs = jnn.one_hot(batch['expressions'], self.input_size)
    output = jnn.one_hot(batch['results'], self.output_size)
    return {'input': inputs, 'output': output}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self._modulus + 6

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._modulus
