============
Installation
============

`janggo` depends on `tensorflow` which might be installed with
CPU-only or GPU support. When installing `janggo2`, please specify
the desired `tensorflow` alternative. Otherwise, `tensorflow` will not be
installed automatically.

For CPU-only use::

    pip install janggo[tf]

For GPU-support type::

    pip install janggo[tf_gpu]

Also follow the installation instructions on how to install CUDA
and cuDNN for the purpose of using tensorflow with GPU support.
