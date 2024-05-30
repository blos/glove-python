# Glove-python

This repository contains my approach of the implementation of the [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) model.
Basically GloVe is mathematically modelled as a regression model.
Counterintuitively, I went with PyTorch as implementation framework.
Due to this decision, there are some inconsistencies between the original paper and this implementation.


## Exploration
Basically, you can use the ``train.py`` file as reference point.
This file consists of one function in which the general intended workflow is depicted.

Orienting on the original paper a word-word co-occurrence matrix is created.
Following thereafter a model is trained.


## How it works (for reference, see ``train.py``):
1. A *tokenizer* is created: In order to build the tokenizer simple pre-processing steps as well as splitting text on word level is utilized.
2. The co-occurrence matrix is build: The resulting matrix is a sparse (scipy) matrix in the row-based list of lists ([lil](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html)) format. The dimensions are the obtained from the vocabulary size of the tokenizer. Regarding the actual matrix creation, multiprocessing is used in order to speed this up a bit. In general, this is done by spawning multiple "worker" processes that are fed by a "feeder" process. The "workers" do the heavy lifting. After the work is done all intermediate results of the separate workers are combined by the "collector" process.
3. Creating the `GloveDataset` which essentially samples batches from the word-word co-occurrence matrix created in the previous step.
4. Instantiating the GloVe model in addition to other needed things, e.g. a trainer and logger objects.


## Trade-Offs
In this section I will explain the deviations from the original paper to my implementation.
- The original paper used regression. However, in order to not implement the optimization steps myself I went with PyTorch. And therefore used ``Embedding`` layers as weights and biases.
- Originally Stochastic Gradient Descent ([SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)) is used for optimizing parameters. In this implementation SGD is traded with [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html). You can go back to SGD by importing it at the top of the ``glove.py`` file with `from torch.optim import SGD` and change `AdamW` to `SGD` in the `configure_optimizers` function.
- The [implementation](https://github.com/stanfordnlp/GloVe) provided by the authors of the original paper is by far faster than this one. Further, it is very efficient CPU-only scenarios, while my implementation is fast when `torch` is installed with CUDA support.


## Installation
1. Clone this repo
2. Install `torch` the way you like (gpu or cpu)
3. `pip install poetry`
4. `poetry install --no-root`


## Contribution
If you notice something that can be discussed, let me know via an issue.
