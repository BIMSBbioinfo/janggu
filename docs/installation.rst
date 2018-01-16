============
Installation
============

`bluewhale2` depends on `tensorflow` which might be installed with
CPU-only or GPU support. When installing `bluewhale2`, please specify
the desired `tensorflow` alternative. Otherwise, `tensorflow` will not be
installed automatically.

For CPU-only use::

    pip install bluewhalecore[tf]

For GPU-support type::

    pip install bluewhalecore[tf_gpu]


HDF5 parallel
-------------

In particular, for very large datasets, it might not be feasible to
fetch all of the data into the CPU RAM. In this case, on could
fetch the data (e.g. the mini-batches during training) from disk
directly.
In order to speed up the process of fetching data from files in the
batches, we recommend to use HDF5 support to store the datasets
and enable parallel support for HDF5 and h5py. This in turn can be exploited
by fetching mini-batches in parallel using the provided generators.

To install HDF5 and h5py parallel support,
please consult `Build against parallel HDF5 <http://docs.h5py.org/en/latest/build.html#building-against-parallel-hdf5>`_.
