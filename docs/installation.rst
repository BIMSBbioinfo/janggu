============
Installation
============

`janggu` depends on `tensorflow` which might be installed with
CPU-only or GPU support. When installing `janggu2`, please specify
the desired `tensorflow` alternative. Otherwise, `tensorflow` will not be
installed automatically.

For CPU-only use::

    pip install janggu[tf]

For GPU-support type::

    pip install janggu[tf_gpu]

Also follow the installation instructions on how to install CUDA
and cuDNN for the purpose of using tensorflow with GPU support.
