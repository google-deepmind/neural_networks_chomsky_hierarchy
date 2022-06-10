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

"""Constants for the generalization project.

This file is provided to simplify the matching from string names like 'lstm' or
'even_pairs' to actual code.
"""

import functools

import haiku as hk

from neural_networks_chomsky_hierarchy.models import ndstack_rnn
from neural_networks_chomsky_hierarchy.models import rnn
from neural_networks_chomsky_hierarchy.models import stack_rnn
from neural_networks_chomsky_hierarchy.models import tape_rnn
from neural_networks_chomsky_hierarchy.models import transformer
from neural_networks_chomsky_hierarchy.tasks.cs import binary_addition
from neural_networks_chomsky_hierarchy.tasks.cs import duplicate_string
from neural_networks_chomsky_hierarchy.tasks.cs import interlocked_pairing
from neural_networks_chomsky_hierarchy.tasks.cs import odds_first
from neural_networks_chomsky_hierarchy.tasks.dcf import compare_occurrence
from neural_networks_chomsky_hierarchy.tasks.dcf import modular_arithmetic_brackets
from neural_networks_chomsky_hierarchy.tasks.dcf import reverse_string
from neural_networks_chomsky_hierarchy.tasks.dcf import solve_equation
from neural_networks_chomsky_hierarchy.tasks.dcf import stack_manipulation
from neural_networks_chomsky_hierarchy.tasks.ndcf import divide_by_two
from neural_networks_chomsky_hierarchy.tasks.ndcf import equal_repeats
from neural_networks_chomsky_hierarchy.tasks.ndcf import missing_palindrome
from neural_networks_chomsky_hierarchy.tasks.ndcf import nondeterministic_stack_manipulation
from neural_networks_chomsky_hierarchy.tasks.regular import even_pairs
from neural_networks_chomsky_hierarchy.tasks.regular import modular_arithmetic
from neural_networks_chomsky_hierarchy.tasks.regular import parity_check
from neural_networks_chomsky_hierarchy.training import curriculum as curriculum_lib

MODEL_BUILDERS = {
    'rnn':
        functools.partial(rnn.make_rnn, rnn_core=hk.VanillaRNN),
    'lstm':
        functools.partial(rnn.make_rnn, rnn_core=hk.LSTM),
    'stack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=stack_rnn.StackRNNCore,
            inner_core=hk.VanillaRNN),
    'ndstack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=ndstack_rnn.NDStackRNNCore,
            inner_core=hk.VanillaRNN),
    'stack_lstm':
        functools.partial(
            rnn.make_rnn, rnn_core=stack_rnn.StackRNNCore, inner_core=hk.LSTM),
    'transformer':
        transformer.make_transformer,
    'tape_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=tape_rnn.TapeRNNCore,
            inner_core=hk.VanillaRNN)
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum
}

TASK_BUILDERS = {
    'modular_arithmetic':
        modular_arithmetic.ModularArithmetic,
    'parity_check':
        parity_check.ParityCheck,
    'even_pairs':
        even_pairs.EvenPairs,
    'modular_arithmetic_brackets':
        functools.partial(
            modular_arithmetic_brackets.ModularArithmeticBrackets, mult=True),
    'compare_occurrence':
        compare_occurrence.CompareOccurrence,
    'reverse_string':
        reverse_string.ReverseString,
    'equal_repeats':
        equal_repeats.EqualRepeats,
    'divide_by_two':
        divide_by_two.DivideByTwo,
    'nondeterministic_stack_manipulation':
        nondeterministic_stack_manipulation.NondeterministicStackManipulation,
    'missing_palindrome':
        missing_palindrome.MissingPalindrome,
    'duplicate_string':
        duplicate_string.DuplicateString,
    'binary_addition':
        binary_addition.BinaryAddition,
    'interlocked_pairing':
        interlocked_pairing.InterlockedPairing,
    'odds_first':
        odds_first.OddsFirst,
    'solve_equation':
        solve_equation.SolveEquation,
    'stack_manipulation':
        stack_manipulation.StackManipulation
}

# This dictionnary maps task to their respective level in the Chomsky hierarchy.
# dcf -> deterministic context-free
# ndcf -> non-deterministic context-free
# cs -> context-sensitive
TASK_LEVELS = {
    'modular_arithmetic': 'regular',
    'parity_check': 'regular',
    'even_pairs': 'regular',
    'modular_arithmetic_brackets': 'dcf',
    'modular_arithmetic_brackets_with_mult': 'dcf',
    'compare_occurrence': 'dcf',
    'reverse_string': 'dcf',
    'stack_manipulation': 'dcf',
    'solve_equation': 'dcf',
    'equal_repeats': 'ndcf',
    'divide_by_two': 'ndcf',
    'nondeterministic_stack_manipulation': 'ndcf',
    'missing_palindrome': 'ndcf',
    'duplicate_string': 'cs',
    'binary_addition': 'cs',
    'interlocked_pairing': 'cs',
    'odds_first': 'cs'
}

POS_ENC_TABLE = {
    'NONE': transformer.PositionalEncodings.NONE,
    'SIN_COS': transformer.PositionalEncodings.SIN_COS,
    'RELATIVE': transformer.PositionalEncodings.RELATIVE,
    'ALIBI': transformer.PositionalEncodings.ALIBI,
}
