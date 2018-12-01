.. _storage:

================
Genomic Datasets
================

One of the central features of Janggu are the genomic datasets :code:`Cover` and
:code:`Bioseq`. On the one hand, they allow
quick and flexible access to genomics data, including **FASTA**,
**BAM**, **BIGWIG**, **BED** and **GFF** file formats, which bridges the gap
between the data being present in raw file formats
and the numpy inputs required for python-based deep learning models (e.g. keras).
On the other hand, predictions from a deep learning library again are in numpy
format. Janggu facilitates a convertion between numpy arrays and :code:`Cover` objects
in order to associate the predictions with the respective genomic coordinates.
Finally, coverage information may be exported to BIGWIG or inspected directly
via  genome browser-like plots.


General principle of the Genomics Datasets
------------------------------------------
Internally, the genomics datasets maintains coverage or
sequence type of information along with the associated genomic intervals.
Externally, the datasets behave similar to a numpy array. This
makes it possible to directly consume the datasets using keras, for instance.

In this section we briefly describe the internal organisation of these datasets.
The classes :code:`Cover` and :code:`Bioseq` maintain a
:code:`GenomicArray` and a :code:`GenomicIndexer` object.
:code:`GenomicArray` is a general data structure that holds numeric
information about the genome. For instance, read count coverage.
It can be accessed via
genomic coordinates (e.g. chromosome name, start and end) and returns
the respective data as numpy array.
The :code:`GenomicIndexer` maintains a list of genomic coordinates,
which should be traversed during training/evaluation.
The GenomicIndexer can be accessed by an integer-valued
index which returns the associated genomic coordinates.

When querying the `i`th region from :code:`Cover` or :code:`Bioseq`, the index is passed
to the  :code:`GenomicIndexer` which yields a genomic coordinates
that is passed on to the :code:`GenomicArray`.
The result is returned in numpy format.
Similarly, the dataset objects
also support slicing and indexing via a list of indices, which is usually relevant
when using mini-batch learning.


Normalization
-------------
Upon creation of a :code:`Cover` object, normalization of the raw data might be require.
For instance, to make coverage tracks comparable across replicates or experiments.
To this end, :code:`create_from_bam`, :code:`create_from_bigwig`
and :code:`create_from_bed` expose a :code:`normalizer` option.
Janggu already implements various normalization methods,
including zscore-, zscore on log-transformed data and TPM (transcript per million)
normalization. These can be used via the argument names: 'zscore', 'zscorelog'
and 'tpm'.

Apart from using 'zscore' and 'zscorelog' by name, you may alternatively
want to use :code:`ZScore` or :code:`ZScoreLog` objects. These allow you
to retrieve the obtained means and standard deviations on one dataset and
apply the same ones to another dataset.

Furthermore, it is possible to provide a custom
normalization procedure in terms of a python function.
To this end, a function with the following signature should be used:

.. code-block:: python

   def custom_normalizer(genomicarray):

      # perform normalization genomicarray

      return genomicarray

Note that concrete implementation of the normalization procedure may depend on
how the data is stored in genomicarray (e.g. numpy arrays or hdf5 format).


Collapse across a region
------------------------

For different applications, one might demand different granularity of the
coverage information. For instance, one might be interested in reading out
nucleotide-resolution coverage or coverage in equally sized 50 base-pair bins
or one might want to collapse variable size bins to be represented by a single
summarizing value like a TPM value describing the gene expression level
of a gene.

In order to control the granularity, the :code:`resolution` parameter is used
for :code:`create_from_bam`, :code:`create_from_bigwig` and :code:`create_from_bed`.
The resolution parameter may be integer valued or None.
In case it is integer valued, it describes the coverage granularity
in equally sized bins across the region of interest.
For example, `resolution=1` would create
nucleotide resolution coverage profiles, while `resolution=50`
amounts to a 50 base-pair resolution.
The latter option also has the side effect
of reducing the required memory by roughly a factor of 50.

On the other hand, if resolution=None the coverage signal
will be collapsed across the entire interval
where the interval size may be determined by the option :code:`binsize`
or, in case variable size intervals shall be used, by setting :code:`binsize=None`.

In conjunction with the resolution parameter, it is necessary to specify
the collapse methods that should be performed (except for resolution=1).
This method might depend on the particular application.
:code:`create_from_bigwig` and :code:`create_from_bed`
support commonly used methods by
name via the :code:`collapser` argument, including 'sum', 'mean', 'max'.

Moreover, for more specific applications, it is possible to supply
a custom collapser function defined by a python function.
In this case, the function should adhere to the following signature:

.. code-block:: python

   def custom_collapser(numpyarray):

      # custom collapser expects a 3D numpy array
      # The first dimension corresponds to the bin
      # and the second to the nucleotide resolution signal
      # within the bin.
      # The second dimension is expected to be collapsed across
      # The third dimension denotes strandedness and is expected
      # to be kept unchanged.

      # Example: collapsing by summation
      # numpyarray = numpyarray.sum(axis=1)

      return numpyarray


Caching
--------

The construction, including loading and preprocessing,
of a genomic dataset might require a significant amount of time.
In order to avoid having to create the coverage profiles each time you want
to use them, they can be cached and quickly reloaded
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

Finally, in order to force recreation of :code:`Cover` or :code:`Bioseq` from scratch,
:code:`overwrite=True` may be set which leads to
preexisting cache files being overwritten.


Dataset storage
---------------

Storage option
==============
Depending on the structure of the dataset, the required memory to store the data
and the available memory on your machine, different storage options are available
for the genomic datasets, including **numpy array**, as **sparse array** or as **hdf5 dataset**.
To this end, :code:`create_from_bam`, :code:`create_from_bigwig`,
:code:`create_from_bed`, :code:`create_from_seq`
and :code:`create_from_refgenome` expose the `storage` option, which may be 'ndarray',
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

:code:`Cover` and :code:`Bioseq` further allow to maintain coverage and sequence information
from the entire genome or only the part that is actively consumed during training.
This option can be configured by :code:`store_whole_genome=True/False`.

In most situations, the user may find it convenient to set `store_whole_genome=False`.
In that case, when loading :code:`Cover` and :code:`Bioseq` only information overlapping
the region of interest will be gathered. The advantage of this would be not to have
to store an overhead of information when only a small part of the genome is of interest
for consumption.

On the other hand, `store_whole_genome=True` might be an advantage
for the following purposes:

1. If a large part of the genome is consumed for training/evaluation
2. If in addition the `stepsize` for traversing the genome is smaller than `binsize`, in which case mutually overlapping intervals do not have to be stored redundantly.
3. It simplifies sharing of the same genomic array for different tasks. For example, during training and testing different parts of the same genomic array may be consumed.



Converting Numpy to Cover
-------------------------

When performing predictions, e.g. with a keras model,
the output corresponds to an ordinary numpy array.
In order to reestablish the association of the predicted values
with the genomic coordinates **Cover** exposes the constructor: `create_from_array`.
Upon invocation, a new :code:`Cover` object is composed that holds the predicted values.
These predictions may subsequently be illustrated via `plotGenomeTrack` or exported
to a BIGWIG file.


Evaluation features
----------------------------

:code:`Cover` objects may be exported as BIGWIG files. Accordingly,
for each condition in the :code:`Cover` a file will be created.

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
dimensionality of :code:`Cover` and :code:`Bioseq`.


==============================
Output directory configuration
==============================

Optionally, janggu produces various kinds of output files, including cache files
for the datasets, log files for monitoring the training / evaluation procedure,
stored model parameters or summary output files about the evaluation performance.

The root directory specifying the janggu output location can be configured
via setting the environment variable :code:`JANGGU_OUTPUT`.
This might be done in the following ways:

Setting the directory globally::

   export JANGGU_OUTPUT='/output/dir'

on startup of the script::

  JANGGU_OUTPUT='/output/dir' python classify.py

or inside your model script using

.. code:: python

   import os
   os.environ['JANGGU_OUTPUT']='/output/dir'

If  :code:`JANGGU_OUTPUT` is not set, root directory will be set
to :code:`/home/user/janggu_results`.
