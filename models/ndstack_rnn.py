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

"""Non-deterministic Stack RNN core.

Following the paper from DuSell et al (2020):
https://arxiv.org/abs/2010.04674

The idea is to add a non-deterministic stack extension to a recurrent neural
network to be able to simulate a machine accepting non-deterministic
context-free languages. It can be seen as an extension to the Stack-RNN
developed by Joulin et al (2015). However, it is far more complex and hard to
understand.
The non-deterministic stack is completely differentiable.

A non-deterministic Pushdown Automata (NDPDA) uses 'multiple stacks at the same
time'. The problem is that the number of possible stacks grows exponentially
with time, which makes a naive practical implementation impossible. However,
Lang et al proved in 1969, based on ideas from Context-Frees parsers like the
CYK, that a NDPDA can be simulated only using O(n³) memory, and not O(2^n).
The main idea is to reuse the content of the different stacks in a dynamic
programming manner. A stack with n values is a stack with n-1 values + an extra
value, so we can build a graph of possible stacks, which would reuse most of
the data.

Concretely, the graph is made of nodes (t, q, x) where t is a number (the time),
q is a state from a fixed, user-defined set and x is a symbol or value, also
from a finite, user-defined set. Then one path in this graph is exactly one
stack, which can simply be reconstructed by reading the value for each node
in the path. Each state q can be used as a 'branching' mechanism: the more
states, the more branching there can be and therefore the more stacks can be
used. The number of possible stacks is (#states * #symbols)^t.

To interact with this graph, ie do a push or a pop action, one uses transitions
on these nodes. For push, it is a function of the form (q1, x1) -> (q2, x2),
where q2 is the new state to go in (ie whether to branch to a new stack, or keep
the same) and x2 is the value to push. For pop, it is a function of the form
(q1, x1) -> q2, which again allows the network to choose whether to create a new
stack or not. No value should be passed there however. The functions are
modelled by transition matrices of shape (Q, S, Q, S) where Q=#states and
S=#symbols.
Once the action matrices are passed, the graph is updated. The update is done
via an internal transition matrix called gamma. This matrix is simple for the
push action (one can only push on the top of the stack, ie nodes for which t =
current timestep). It is far more complex for the pop action, as popping a value
from the current stack can completely change the structure of the graph: the
new stack after popping might be equal to a very old stack seen at the beginning
of the episode, and we must change the links accordingly. Roughly, the update
operation for gamma has a time complexity of O(Q⁴ S³ n³).
Finally, one the graph is updated via gamma, we update the probabilities of the
top of stacks, which gives us a tensor called alpha. From alpha we deduce the
average top of the stack to be sent to the agent.

As there are 3 actions (pop/push/no_op), unrolling this over
long sequences and using big batch sizes consumes too much memory and the
accelerators fail.

Notations:
  Q: number of states of the ND stack (not the number of states of the
    RNN).
  S: number of symbols which can be pushed on the stack.
  T: Sequence length.
  B: Batch size.
"""

from typing import Any, Mapping, NamedTuple, Optional

import chex
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp

_EPSILON = 0.001


class NDStack(NamedTuple):
  """The non-deterministic stack.

  Note that alpha and top_stack depend on gamma.
  """
  gamma: chex.Array  # Shape (B, T, T, Q, S, Q, S)
  alpha: chex.Array  # Shape (B, T, Q, S)
  top_stack: chex.Array  # Shape (B, S)


def _update_stack(ndstack: NDStack,
                  push_actions: chex.Array,
                  pop_actions: chex.Array,
                  replace_actions: chex.Array,
                  timestep: int,
                  read_states: bool = True) -> NDStack:
  """Returns an updated NDStack.

  Args:
    ndstack: See above. Contains the internals needed to simulate a
      non-deterministic stack.
    push_actions: A tensor of shape (B, Q, S, Q, S).
    pop_actions: A tensor of shape (B, Q, S, Q).
    replace_actions: A tensor of shape (B, Q, S, Q, S).
    timestep: The current timestep while processing the sequence.
    read_states: Whether to read the state of the NPDA as well.
  """
  stack_size = ndstack.gamma.shape[2]
  mask = jnp.zeros((stack_size, stack_size))
  mask = mask.at[timestep - 1, timestep].set(1)
  new_push_gamma_t = jnp.einsum('bqxry,tT->btTqxry', push_actions,
                                mask)[:, :, timestep]

  index_k = jnp.stack([jnp.arange(start=0, stop=stack_size)] * stack_size)
  index_i = jnp.transpose(index_k)
  timestep_arr = jnp.full((stack_size, stack_size), timestep)
  index_mask = jnp.logical_and(index_k > index_i, index_k < timestep_arr - 1)
  index_mask = jnp.einsum('tT,bqxry->btTqxry', index_mask,
                          jnp.ones(push_actions.shape))
  new_pop_gamma_t = jnp.einsum(
      'bikqxuy,bkuysz,bszr->biqxry',
      index_mask * ndstack.gamma,
      ndstack.gamma[:, :, timestep - 1],
      pop_actions,
  )

  new_replace_gamma_t = jnp.einsum('biqxsz,bszry->biqxry',
                                   ndstack.gamma[:, :,
                                                 timestep - 1], replace_actions)

  new_gamma = jax.vmap(jax.vmap(lambda x, y: x.at[timestep].set(y)))(
      ndstack.gamma, new_replace_gamma_t + new_pop_gamma_t + new_push_gamma_t)

  alpha_t = jnp.einsum('biqx,biqxry->bry', ndstack.alpha, new_gamma[:, :,
                                                                    timestep])
  new_alpha = jax.vmap(lambda x, y: x.at[timestep].set(y))(ndstack.alpha,
                                                           alpha_t)

  if read_states:
    batch_size, states, symbols = alpha_t.shape
    obs = jnp.reshape(alpha_t, (batch_size, states * symbols))
  else:
    obs = jnp.sum(alpha_t, axis=1)

  obs = obs / (jnp.sum(obs, axis=-1, keepdims=True) + _EPSILON)
  return NDStack(new_gamma, new_alpha, top_stack=obs)


# First element is the NDStack, second is the current timestep, third is the
# hidden internal state.
_NDStackRnnState = tuple[NDStack, chex.Array, chex.Array]


class NDStackRNNCore(hk.RNNCore):
  """Core for the non-deterministic stack RNN."""

  def __init__(
      self,
      stack_symbols: int,
      stack_states: int,
      stack_size: int = 30,
      inner_core: type[hk.RNNCore] = hk.VanillaRNN,
      read_states: bool = False,
      name: Optional[str] = None,
      **inner_core_kwargs: Mapping[str, Any]
  ):
    """Initializes.

    Args:
      stack_symbols: The number of symbols which can be used in the stack.
      stack_states: The number of states of the non-deterministic stack.
        Corresponds to the number of branching in the graph, ie roughly n_stacks
        = stack_states ^ t.
      stack_size: The total size of the stacks. Be careful when increasing this
        value since the computation is in O(stack_size ^ 3).
      inner_core: The inner RNN core builder.
      read_states: Whether to read the states on the NPDA or only the top of the
        stack.
      name: See base class.
      **inner_core_kwargs: The arguments to be passed to the inner RNN core
        builder.
    """
    super().__init__(name=name)
    self._rnn_core = inner_core(**inner_core_kwargs)
    self._stack_symbols = stack_symbols
    self._stack_states = stack_states
    self._stack_size = stack_size
    self._read_states = read_states

  def __call__(
      self, inputs: chex.Array, prev_state: _NDStackRnnState
  ) -> tuple[chex.Array, _NDStackRnnState]:
    """Steps the non-deterministic stack RNN core.

    See base class docstring.

    Args:
      inputs: An input array of shape (batch_size, input_size). The time
        dimension is not included since it is an RNNCore, which is unrolled over
        the time dimension.
      prev_state: A _NDStackRnnState tuple, consisting of the previous nd-stack,
        the previous timestep and the previous state of the inner core.

    Returns:
      - output: An output array of shape (batch_size, output_size).
      - next_state: Same format as prev_state.
    """
    ndstack, timestep, old_core_state = prev_state

    # The network can always read the top of the stack.
    batch_size = ndstack.gamma.shape[0]
    inputs = jnp.concatenate([inputs, ndstack.top_stack], axis=-1)
    new_core_output, new_core_state = self._rnn_core(inputs, old_core_state)

    n_push_actions = (self._stack_states * self._stack_symbols)**2
    n_pop_actions = self._stack_states**2 * self._stack_symbols
    n_replace_actions = (self._stack_states * self._stack_symbols)**2
    actions = hk.Linear(n_push_actions + n_pop_actions + n_replace_actions)(
        new_core_output)
    actions = jnn.softmax(actions, axis=-1)

    push_actions = jnp.reshape(
        actions[:, :n_push_actions],
        (batch_size, self._stack_states, self._stack_symbols,
         self._stack_states, self._stack_symbols))

    pop_actions = jnp.reshape(
        actions[:, n_push_actions:n_push_actions + n_pop_actions],
        (batch_size, self._stack_states, self._stack_symbols,
         self._stack_states))

    replace_actions = jnp.reshape(
        actions[:, -n_replace_actions:],
        (batch_size, self._stack_states, self._stack_symbols,
         self._stack_states, self._stack_symbols))

    new_ndstack = _update_stack(
        ndstack,
        push_actions,
        pop_actions,
        replace_actions, (timestep + 1)[0],
        read_states=self._read_states)
    return new_core_output, (new_ndstack, timestep + 1, new_core_state)

  def initial_state(self, batch_size: Optional[int]) -> _NDStackRnnState:
    """Returns the initial state of the core, a hidden state and an empty stack."""
    core_state = self._rnn_core.initial_state(batch_size)

    # Gamma, the transition matrix, is initialized to full zeros: there is no
    # connection in the graph at the beginning.
    gamma = jnp.zeros(
        (batch_size, self._stack_size, self._stack_size, self._stack_states,
         self._stack_symbols, self._stack_states, self._stack_symbols))

    # Alpha is zero everywhere except for the first node, which is (0, q0, 0).
    alpha = jnp.zeros(
        (batch_size, self._stack_size, self._stack_states, self._stack_symbols))
    alpha = jax.vmap(lambda x: x.at[0, 0, 0].set(1))(alpha)

    if self._read_states:
      top_stack = jnp.zeros(
          (batch_size, self._stack_states * self._stack_symbols))
    else:
      # The top of the stack is 0 as the first node contains the symbol 0.
      top_stack = jnp.zeros((batch_size, self._stack_symbols))

    ndstack = NDStack(gamma, alpha, top_stack)
    return ndstack, jnp.zeros((batch_size,), dtype=jnp.int32), core_state
