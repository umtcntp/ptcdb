# ptcdb

This dataset is created as a citational graph dataset, to be used with a graph
citational network implementation such as
[this.](https://github.com/tkipf/pygcn) See
[Kipf & Welling's paper](https://arxiv.org/abs/1609.02907) for more information
about graph convolutional networks.

## Usage

Under data batches file, there are 4 batches that we generated from USPTO patent
database files. Under the example implementation file, there is an
implementation of SVM and GCN(with PyTorch, from
[pygcn](https://github.com/tkipf/pygcn)). This particular implementation used
the Cora dataset, we modified it to use the patent citation dataset with the
same format as the Cora's. We also added an SVM test to compare the two
implementations with.

## Setup

```
$ cd example_implementation
$ python setup.py install
$ cd pygcn
$ python train.py
```

You can also change the dataset back to Cora, under utils.py to record the
difference in two datasets.
