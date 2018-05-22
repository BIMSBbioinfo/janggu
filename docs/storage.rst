================
Storage
================
Janggu automatically produces various output files as a result of model fitting
and evaluation which are placed in the :code:`<janggu_results>` root directory.
For example, the subdirectory :code:`<janggu_results>/models` contains the model
parameters and :code:`<janggu_results>/evaluation` contains evaluation results.
You can control the location of the root directory by setting the
environment variable globally

::

   export JANGGU_OUTPUT='/output/dir'

on startup of the script::

  JANGGU_OUTPUT='/output/dir' python classify.py

or inside your model script using

.. code:: python

   import os
   os.environ['JANGGU_OUTPUT']='/output/dir'

================
Genomic Datasets
================

Dataset storage
---------------
It is possible to maintain the datasets in various ways in memory.
One option is 'ndarray' which stores the entire dataset as numpy array
in memory. This option might require a large amount of memory resources
if the dataset is large. However, since the data is already in memory,
consuming mini-batches from the datasets will be quite fast.
A second option is 'hdf5' which stores the dataset into a hdf5 file
on disk. This option allows you to process datasets that would be too
large hold in memory. During training or evaluation, mini-batches will
be consumed directly from disk. Usually, processing dataset with
storage 'hdf5' will be slower than 'ndarray', but in many cases still practical.


Caching
--------
Genomic datasets might take a significant amount of time to load. For example,
it might take hours to load the coverage profile from a set of BAM files.
In order to avoid having to compute the coverage profile each time you want
to use the coverage track, by default the genomic dataset is cached
and will be reloaded later.
For both storage types, 'hdf5' and 'ndarray', the datasets will be cached by default.
In order to control the caching behavior, use the dataset options: :code:`cache=True/False`
to declare if the data should be cached and :code:`overwrite=False/True` if the
cached file should be overwritten.

The cache files will be stored in :code:`<janggu_results>/datasets`
under the specified dataset name and associated dataset tags.
For example,

.. code:: python

   Dna.create_from_fasta('hg19', refgenome, regions, datatags=['chr1'],
                         order=1)

might be a dataset that contains the sequence of chromosome 1 of hg19.
The actual cache file is then located at
:code:`<janggu_results>/datasets/hg19/chr1/order1/storage.npz`.
