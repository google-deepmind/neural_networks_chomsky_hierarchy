# neural_networks_chomsky_hierarchy

This repository contains the code associated with the paper "Neural networks and the Chomsky hierarchy" (Deletang et al., 2022).

## Content

There are three folders: tasks, models and training.

'tasks' contains all tasks, organized in their Chomsky hierarchy levels (regular, dcf, ndcf, cs). They all inherit the abstract class GeneralizationTask, defined in tasks/task.py.

'models' contains all the models we use, written in [jax](https://github.com/google/jax) and [haiku](https://github.com/deepmind/dm-haiku), two open source libraries.

'training' contains the code for training models and evaluating them on a wide
range of lengths. We also included an example to train and evaluate an RNN
on the Even Pairs task. For training, we use [optax](https://github.com/deepmind/optax) for our optimizers.

## Installation

`pip install -r requirements.txt`

## Usage

`python3 training/example.py`

## Citing this work

Add citation details here, usually a pastable BibTeX snippet.

## License and disclaimer

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
