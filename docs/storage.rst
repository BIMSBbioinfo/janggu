.. _storage:

================
Genomic Datasets
================

One of the central features of Janggu are the genomic datasets `Cover` and
`Bioseq`. On the one hand, they allow
quick and flexible access to genomics data, including **FASTA**,
**BAM**, **BIGWIG**, **BED** and **GFF** file formats, which bridges the gap
between the data being present in raw file formats
and the numpy inputs required for python-based deep learning models (e.g. keras).
On the other hand, predictions from a deep learning library again are in numpy
format. Janggu facilitates a convertion between numpy arrays and `Cover` objects
in order to reassociate the predictions with their associated genomic coordinates.
Finally, coverage information may be exported to BIGWIG or inspected directly
via  genome browser-like plots.


General principle of the Genomics Datasets
------------------------------------------
Internally, the genomics datasets maintains coverage or
sequence type of information along with the associated genomic intervals.
Externally, the datasets behave similar to a numpy array. This
makes it possible to directly consume the datasets using keras, for instance.

In this section we briefly describe the internal organisation of these datasets.
The classes `Cover` and `Bioseq` maintain a `GenomicArray` and a `GenomicIndexer`.
The genomic array is a general data structure that holds numeric information about the
the genome. For instance, read count coverage. It can be accessed via
genomic coordinates (e.g. chromosome name, start and end) and returns the respective
data as numpy array.
The GenomicIndexer maintains a list of genomic coordinates, which might for instance
be traversed during training. The GenomicIndexer can be accessed by an integer-valued
index representing the region associated with that integer.

When querying the `i`th region from `Cover` or `Bioseq`, the index is passed
to the genomic indexer which in turn hands over the genomic coordinates to the
genomic array. The result is returned in numpy format. Similarly, the dataset objects
also support slicing and indexing via a list of indices, which is usually relevant
when using mini-batch learning.


Normalization
-------------
Upon creation of a `Cover` object, normalization of the raw data might be require.
For instance, to make coverage tracks comparable across replicates or experiments.
To this end, `create_from_bam`, `create_from_bigwig` and `create_from_bed` expose
a `normalizer` option. Janggu already implements various normalization methods,
including zscore-, zscore on log-transformed data and TPM (transcript per million)
normalization. These may be used via the argument names: 'zscore', 'zscorelog'
and 'tpm'.

It is also possible to provide a custom normalization procedure in terms of a
python function.
To this end, a function with the following signature should be used:

.. code-block:: python

   def custom_normalizer(genomicarray):

      # perform normalization genomicarray

      return genomicarray

Note that concrete implementation of the normalization procedure may depend on
how the data is stored in genomicarray, which may be as numpy arrays or hdf5 format.


Collapse across a region
------------------------

For different applications, one might demand different granularity of the
coverage information. For instance, one might be interested in reading out
nucleotide-resolution coverage or coverage in equally sized 50 base-pair bins
or one might want to collapse variable size bins to be represented by a single
summarizing value. For example, a TPM value describing the gene expression level
of a gene.

In order to control the granularity, the `resolution` parameter is used
for `create_from_bam`, `create_from_bigwig` and `create_from_bed`.
The resolution parameter may be integer valued, describing the coverage granularity
in equally sized bins across the genome. For example, `resolution=1` would create
nucleotide resolution coverage profiles, while `resolution=50` reduces the granularity
to a 50 base-pair bin resolution. The latter option also has the side effect
of reducing the required memory by roughly a factor of 50.

If resolution=None, each entry in the region of interest (ROI) file
will be treated as a separate
region and the signal will be collapsed across the entire interval.
As a result the sequence length dimension and strandedness dimension will be 1. This
option is relevant for working with variable size intervals.

Caching
--------

The construction, including loading and preprocessing,
of a genomic dataset might require a significant amount of time.
In order to avoid having to create the coverage profiles each time you want
to use them, the genomic datasets can be cached and quickly reloaded
later.
Caching can be activated via the options :code:`cache=True`.
In order to prevent naming conflicts with the cache files, a list of
 :code:`datatags` may be specified, describing the dataset.

For example, caching the data of hg19 may be done as follows:

.. code:: python

   # load hg19 if the cache file does not exist yet, otherwise
   # reload it.
   Bioseq.create_from_refgenome('dna', refgenome, datatags=['hg19'],
                                order=1, cache=True)

Finally, in order to force recreation of `Cover` or `Bioseq` from scratch,
:code:`overwrite=True` may be set which leads to preexisting cache files being overwritten.


Dataset storage
---------------

Storage option
==============
Depending on the structure of the dataset, the required memory to store the data
and the available memory on your machine, different storage options are available
for the genomic datasets, including **numpy array**, as **sparse array** or as **hdf5 dataset**.
To this end, `create_from_bam`, `create_from_bigwig`, `create_from_bed`, `create_from_seq`
and `create_from_refgenome` expose the `storage` option, which may be 'ndarray',
'sparse' or 'hdf5', respectively.

'ndarray' amounts to perhaps the fastest access time,
but also most memory demanding option for storing the data.
It might be useful for dense datasets, and relatively small datasets that conveniently
fit into memory.
If the data is sparse, the option `sparse` yields a good compromise between access time
and speed. In that case, the data is stored in its compressed sparse form and converted
to a dense representation when querying mini-batches.
This option may be used to store e.g. genome wide ChIP-seq peaks profiles, if peaks
occur relatively rarely.

Finally, if the data is too large to be kept in memory, the option
`hdf5` allows to consume the data directly from disk. While,
the access time for processing data from hdf5 files may be higher,
it allows to processing huge datasets with a small amount of RAM in your machine.

Whole and partial genome storage
================================

`Cover` and `Bioseq` further allow to maintain coverage and sequence information
from the entire genome or only the part that is actively consumed during training.
This option can be configured by `store_whole_genome=True/False`.

In most situations, the user may find it convenient to set `store_whole_genome=False`.
In that case, when loading `Cover` and `Bioseq` only information overlapping
the region of interest will be gathered. The advantage of this would be not to have
to store an overhead of information when only a small part of the genome is of interest
for you.

On the other hand, `store_whole_genome=True` might be an advantage for the following purposes:
1. If a large part of the genome is consumed for training/evaluation
2. If in addition the `stepsize` of traversing the genome is smaller than `binsize`, in
which case mutually overlapping intervals do not have to be stored redundantly.
3. It simplifies sharing of the same genomic array for different tasks.
For example, during training and testing different parts of the same genomic array may be consumed.



Converting Numpy to Cover
-------------------------

When performing predictions, e.g. with a keras model,
the output corresponds to an ordinary numpy array.
In order to reestablish the association of the predicted values
with the genomic coordinates **Cover** exposes the constructor: `create_from_array`.
Upon invocation, a new `Cover` object is composed that holds the predicted values.
These predictions may subsequently be illustrated via `plotGenomeTrack` or exported
to a BIGWIG file.


Evaluation features
----------------------------

`Cover` objects may be exported as BIGWIG files. Accordingly,
for each condition in the `Cover` a file will be created.

It is also possible to illustrate predictions in terms of
a genome browser-like plot using `plotGenomeTrack`, allowing to interactively explore
prediction scores (perhaps in comparison with the true labels) or
feature activities of the internal layers of a neural net.
`plotGenomeTrack` return a matplotlib figure that can be stored into a file
using native matplotlib functionality.


Rearranging channel dimensions
------------------------------

Depending on the deep learning library that is used, the dimensionality
of the tensors need to be set up in a specific order.
For example, tensorflow expects the channel to be represented by the last
dimension, while theano or pytorch expect the channel at the first dimension.
With the option `channel_last=True/False` it is possible to configure the output
dimensionality of `Cover` and `Bioseq`.


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
