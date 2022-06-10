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

"""Nondeterministic Stack manipulation task for generalization."""

from typing import Any, List, Tuple

import numpy as np

from neural_networks_chomsky_hierarchy.tasks.dcf import stack_manipulation


class NondeterministicStackManipulation(stack_manipulation.StackManipulation):
  """A task which goal is to follow nondeterministic instructions on a stack.

  The input is composed of:
    - a stack with two initial tokens, 0 and 1
    - a sequence of POP (token 2) and PUSH (tokens
     `{3, ..., 2 + num_push_values}`) instructions, some of which are missing
     (represented by `3 + num_push_values`)
    - a delimiter (represented by `4 + num_push_values`)
    - the missing actions (represented by `{2, ..., 2 + num_push_values}`)
  Thus, the agent needs to follow the instructions that are provided and
  simulate all possible missing instructions by keeping multiple stacks (i.e., a
  nondeterministic stack). Once the delimiter symbol is reached, the agent can
  pick the stack that corresponds to the missing values. Finally, the agent just
  needs to output the final stack, written from bottom to top.
  The output is padded with 0s to match the input length, and the end of the
  result is denoted with a termination symbol (i.e., the output has values in
  `{0, 1, ..., 1 + num_push_values}`).

  Examples (similar to the classical stack_manipulation):
    0 1 0 0 PUSH _ POP | POP
      initial 0 0 1 0
      PUSH    1 0 0 1 0
      _       (POP) 0 0 1 0  &  (PUSH) 1 1 0 0 1 0  (two stacks are possible)
      POP           0 1 0    &         1 0 0 1 0
      |    ... wait and do nothing
      POP           0 1 0    -         - - - - -
    -> 0 1 0

    1 1 0 _ POP POP | POP
      initial 0 1 1
      _       (POP) 1 1  &  (PUSH)  1 0 1 1
      POP           1    &          0 1 1
      POP                &          1 1
      |    ... wait and do nothing
      POP                -          - -
    -> 2 0 0 (empty stack!)

  Note that in the main paper, we only consider num_push_values=1,
  num_missing_actions=1 and variable_missing_actions=False.
  """

  def __init__(
      self,
      *args: Any,
      num_push_values: int = 1,
      num_missing_actions: int = 1,
      variable_missing_actions: bool = False,
      **kwargs: Any,
  ) -> None:
    """Initializes the nondeterministic stack manipulation task.

    Args:
      *args: Args for the base task class.
      num_push_values: The number of different values that can be pushed.
      num_missing_actions: The number of actions that are removed from the list.
      variable_missing_actions: Whether the number of missing action is
        constant or not.
      **kwargs: Kwargs for the base task class.
    """
    super().__init__(*args, **kwargs)
    self._num_push_values = num_push_values
    self._num_missing_actions = num_missing_actions
    self._variable_missing_actions = variable_missing_actions

  def _sample_expression_and_result(
      self, length: int) -> Tuple[np.ndarray, List[int]]:
    """Returns an expression with stack instructions, and the result stack."""
    # Strings of length < 4 are too short to have missing actions.
    if length < 4:
      return super()._sample_expression_and_result(length)

    tokens = dict(
        initial_stack_values=(0, 1),
        stack_actions=(2, 2 + self._num_push_values),
        missing_action=3 + self._num_push_values,
        delimiter=4 + self._num_push_values,
    )
    num_missing_actions = self._num_missing_actions

    if self._variable_missing_actions:
      num_missing_actions = np.random.randint(1 + self._num_missing_actions)

    # Account for the delimiter and the missing actions appended at the end.
    disposable_length = length - 1 - num_missing_actions
    # Take the maximum to ensure that the stack is always nonempty.
    stack_length = np.random.randint(low=1, high=max(2, disposable_length))
    # Take the maximum to ensure that there is always at least one action.
    num_actions = max(1, disposable_length - stack_length)
    # Due to the maximum operations above, there may be fewer missing actions.
    num_missing_actions = min(
        num_missing_actions,
        length - 1 - stack_length - num_actions,
    )
    # Ensure that there are more actions than missing actions by swapping them
    # two if they are not in order.
    num_missing_actions, num_actions = sorted(
        (num_missing_actions, num_actions))

    # Initialize the stack content and the actions (POP/PUSH).
    stack = np.random.randint(
        low=tokens['initial_stack_values'][0],
        high=1 + tokens['initial_stack_values'][1],
        size=(stack_length,),
    )
    actions = np.random.randint(
        low=tokens['stack_actions'][0],
        high=1 + tokens['stack_actions'][1],
        size=(num_actions,),
    )

    # Apply the actions on the stack.
    current_stack = list(stack)[::-1]
    for action in actions:
      if action == stack_manipulation.ACTIONS['POP']:  # POP
        # We only pop if there is more than one element in the stack.
        if len(current_stack) > 1:
          current_stack.pop(0)
      else:  # PUSH
        current_stack = [action - 2] + current_stack

    # Sample which actions to remove from the list.
    missing_indices = np.random.choice(
        np.arange(num_actions),
        size=num_missing_actions,
        replace=False,
    )
    missing_indices.sort()
    missing_actions = actions[missing_indices]

    # Replace the designated missing actions with empty tokens.
    actions[missing_indices] = tokens['missing_action']

    # Insert the delimiter between the actions and the missing actions.
    return (
        np.concatenate([stack, actions, [tokens['delimiter']],
                        missing_actions]),
        current_stack,
    )

  @property
  def input_size(self) -> int:
    """Returns the input size for the models.

    There are six different input values:
      - 0/1 tokens in the stack
      - 2 POP action
      - {3, ..., 2 + `self._num_push_values`} PUSH actions
      - 3 + `self._num_push_values` indicates a missing action
      - 4 + `self._num_push_values` the delimiter between the actions provided
      and those that are missing
    """
    return 5 + self._num_push_values

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2 + self._num_push_values
