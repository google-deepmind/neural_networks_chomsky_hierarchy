# Neural Networks and the Chomsky Hierarchy

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmind/neural_networks_chomsky_hierarchy/master/chomsky.svg" alt="Overview figure"/>
</p>

This repository provides an implementation of the paper [Neural Networks and the Chomsky Hierarchy](https://arxiv.org/abs/2207.02098).

> Reliable generalization lies at the heart of safe ML and AI.
However, understanding when and how neural networks generalize remains one of the most important unsolved problems in the field.
In this work, we conduct an extensive empirical study (2200 models, 16 tasks) to investigate whether insights from the theory of computation can predict the limits of neural network generalization in practice.
We demonstrate that grouping tasks according to the Chomsky hierarchy allows us to forecast whether certain architectures will be able to generalize to out-of-distribution inputs.
This includes negative results where even extensive amounts of data and training time never led to any non-trivial generalization, despite models having sufficient capacity to perfectly fit the training data.
Our results show that, for our subset of tasks, RNNs and Transformers fail to generalize on non-regular tasks, LSTMs can solve regular and counter-language tasks, and only networks augmented with structured memory (such as a stack or memory tape) can successfully generalize on context-free and context-sensitive tasks.

It is based on [JAX](https://jax.readthedocs.io) and [Haiku](https://dm-haiku.readthedocs.io) and contains all code, datasets, and models necessary to reproduce the paper's results. 


## Content

```
.
├── models
|   ├── ndstack_rnn.py        - Nondeterministic Stack-RNN (DuSell & Chiang, 2021)
|   ├── rnn.py                - RNN (Elman, 1990)
|   ├── stack_rnn.py          - Stack-RNN (Joulin & Mikolov, 2015)
|   ├── tape_rnn.py           - Tape-RNN, loosely based on Baby-NTM (Suzgun et al., 2019) 
|   └── transformer.py        - Transformer (Vaswani et al., 2017)
├── tasks
|   ├── cs                    - Context-sensitive tasks
|   ├── dcf                   - Determinisitc context-free tasks
|   ├── ndcf                  - Nondeterministic context-free tasks
|   ├── regular               - Regular tasks
|   └── task.py               - Abstract GeneralizationTask 
├── training
|   ├── constants.py          - Training/Evaluation constants
|   ├── curriculum.py         - Training curricula (over sequence lengths)
|   ├── example.py            - Example training script (RNN on the Even Pairs task)
|   ├── range_evaluation.py   - Evaluation loop (over unseen sequence lengths)
|   ├── training.py           - Training loop
|   └── utils.py              - Utility functions
├── README.md
└── requirements.txt          - Dependencies
```

'tasks' contains all tasks, organized in their Chomsky hierarchy levels (regular, dcf, cs). They all inherit the abstract class GeneralizationTask, defined in tasks/task.py.

'models' contains all the models we use, written in [jax](https://github.com/google/jax) and [haiku](https://github.com/deepmind/dm-haiku), two open source libraries.

'training' contains the code for training models and evaluating them on a wide
range of lengths. We also included an example to train and evaluate an RNN
on the Even Pairs task. We use [optax](https://github.com/deepmind/optax) for our optimizers.

## Installation

`pip install -r requirements.txt`


## Usage Example

`python3 training/example.py`


## Citing This Work

```bibtex
@misc{deletang2022neural,
  author    = {Delétang, Grégoire and Ruoss, Anian and Grau-Moya, Jordi and Genewein, Tim and Wenliang, Li Kevin and Catt, Elliot and Hutter, Marcus and Legg, Shane and Ortega, Pedro A.},
  title     = {Neural Networks and the Chomsky Hierarchy},
  publisher = {arXiv},
  year      = {2022},
}
```


## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
