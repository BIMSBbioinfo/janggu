.. _storage:

================
Storage
================
Janggu automatically produces various output files as a result of model fitting
and evaluation which are placed in the :code:`<janggu_results>` root directory.
For example, the subdirectory :code:`<janggu_results>/models` contains the model
parameters and :code:`<janggu_results>/evaluation` contains evaluation results.
You can control the location of the root directory by setting the
environment variable :code:`JANGGU_OUTPUT`

::

   export JANGGU_OUTPUT='/output/dir'

on startup of the script::

  JANGGU_OUTPUT='/output/dir' python classify.py

or inside your model script using

.. code:: python

   import os
   os.environ['JANGGU_OUTPUT']='/output/dir'

If  :code:`JANGGU_OUTPUT` is not set, root directory will be set
to :code:`/home/user/janggu_results`.

================
Genomic Datasets
================

Dataset storage
---------------
The genomic datasets can maintained in various storage modes,
including as numpy array, as sparse matrix or as hdf5 dataset.
The data is stored as numpy array using 'ndarray', which usually
yields faster access time compared to the other options
at the cost of large amount of memory consumption.
If the data is sufficiently sparse, the option 'sparse'
might yield a good compromise between speed and memory consumption.
Finally, if the data is too large to be kept in memory, the option
'hdf5' allows to consume the data directly from disk at the cost
of higher access time.


Caching
--------
Genomic datasets might take a significant amount of time to load. For example,
it might take hours to load the whole-genome coverage profile from a set of BAM files.
In order to avoid having to compute the coverage profile each time you want
to use the coverage track the genomic datasets can be cached and reloaded
later. Caching works with all storage types, 'ndarray', 'sparse' and 'hdf5'.
Caching can be activated by setting :code:`cache=True`.
In order to control the caching behavior, use the dataset options: :code:`cache=True`
to declare if the data should be cached and :code:`overwrite=True` if the
cached file should be overwritten.

A cache file will be constructed using the dataset name and
associated dataset tags.
For example, the entire human genome can be cached and later reloaded
with different region of interests

.. code:: python
   # load hg19 if the cache file does not exist yet, otherwise
   # reload it.
   Bioseq.create_from_refgenome('dna', refgenome, datatags=['hg19'],
                                order=1, cache=True)
   
