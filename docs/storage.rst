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

When querying the 'i'-th region from :code:`Cover` or :code:`Bioseq`, the index is passed
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
Janggu already implements various normalization methods which can be called by name,
TPM (transcript per million) normalization. For instance, using

.. code-block:: python

   Cover.create_from_bam('track', bamfiles=samplefile, roi=roi, normalizer='tpm')

Other preprocessing and normalization options are:  :code:`zscore`, :code:`zscorelog`, :code:`binsizenorm` and :code:`perctrim`.
The latter two apply normalization for read depth and trimming the signal intensities at the 99%-ile.

Normalizers can also be applied via callables and/or in combination with other transformations.
For instance, suppose we want to trim the outliers at the 95%-tile instead and
subsequently apply the z-score transformation then we could use

.. code-block:: python

   from janggu.data import PercentileTrimming
   from janggu.data import ZScore

   Cover.create_from_bam('track', bamfiles=samplefile, roi=roi,
                         normalizer=[PercentileTrimming(95), ZScore()])


It might be necessary to evaluate the normalization parameter on one dataset and apply the same
transformation on other datasets. For instance, in the case of the ZScore, we might want to keep
the mean and standard deviation that was obtained from, say the training set, and reuse the
to normalize the test set.
This is possible by just creating a zscore object that is used multiple times.
At the first invokation the mean and standard deviation are determine and the transformation
is applied. Subsequently, the zscore is determined using the predetermined mean and standard deviation.
For example:

.. code-block:: python

   from janggu.data import ZScore

   zscore = ZScore()

   # First, mean and std will be determined.
   # Then zscore transformation is applied.
   Cover.create_from_bam('track_train', bamfiles=samplefile, roi=roitrain,
                         normalizer=[zscore])

   # Subsequently, zscore transformation is applied with
   # the same mean and std determined from the training set.
   Cover.create_from_bam('track_test', bamfiles=samplefile, roi=roitest,
                         normalizer=[zscore])

In case a different normalization procedure is required that is not contained in janggu,
it is possible to define a custom_normalizer as follows:

.. code-block:: python

   def custom_normalizer(genomicarray):

      # perform normalization genomicarray

      return genomicarray

The currently implemented normalizers may be a good starting point
for this purpose.


Granularity of the coverage
----------------------------

Depending on the applications, different granularity of the
coverage data might be required. For instance, one might be interested in reading out
nucleotide-resolution coverage for one purpose or 50 base-pair resolution bins for another.
Furthermore, in some cases the signal of variable size regions might be of interest. For
example, the read counts across the gene bodies, to measure gene expression levels.

These adjustments can be made when invoking :code:`create_from_bam`,
:code:`create_from_bigwig` and :code:`create_from_bed`
using an appropriate region of interest ROI file in conjunction
with specifying the :code:`resolution` and  :code:`collapser` parameter.

First, we the resolution parameter allows to the coverage granularity.
For example, base-pair and 50-base-pair resolution would be possible using

.. code-block:: python

   Cover.create_from_bam('track', bamfiles=samplefile, roi=roi,
                         resolution=1)

   Cover.create_from_bam('track', bamfiles=samplefile, roi=roi,
                         resolution=50)

.. sidebar:: janggu-trim

  When using N-based pair resolution with :code:`n>1` in conjunction with the
  option :code:`store_whole_genome=True`, then the region of interest starts
  and ends must be divisible by the resolution. Otherwise, undesired rounding
  effect might occur. This can be achieved by using :code:`janggu-trim`.
  See Section command line tools.

In case the signal intensity should be summarized across the entire interval,
specify :code:`resolution=None`.
For instance, if the region of interest contains a set of variable length
gene bodies, the total read count per gene can be obtained using

.. code-block:: python

   Cover.create_from_bam('genes',
                         bamfiles=samplefile,
                         roi=geneannot,
                         resolution=None)

It is also possible to use :code:`resolution=None` in conjunction with e.g. :code:`binsize=200`
which would have the same effect as chosing :code:`binsize=resolution=200`.

Whenever we deal with :code:`resolution>1`, an aggregation operation needs to be performed
to summarize the signal intensity across the region. For instance, for
:code:`create_from_bam` the reads are summed within each interval.

For :code:`create_from_bigwig` and :code:`create_from_bed`,
it is possible to adjust the collapser. For example, 'mean' or 'sum' aggregation
can be applied by name or by handing over a callable according to

.. code-block:: python

   import numpy as np

   Cover.create_from_bigwig('bwtrack',
                            bigwigfiles=samplefile,
                            roi=roi,
                            resolution=50,
                            collapser='mean')

   Cover.create_from_bigwig('bwtrack',
                            bigwigfiles=samplefile,
                            roi=roi,
                            resolution=50,
                            collapser=np.sum)


Moreover, more specialized aggregations may
require a custom collaper function. In that case,
it is important to note that the function expects a 3D numpy array and
the aggragation should be performed across the second dimension.
For example

.. code-block:: python

   def custom_collapser(numpyarray):

      # Initially, the dimensions of numpyarray correspond to
      # (intervallength // resolution, resolution, strand)

      numpyarray = numpyarray.sum(axis=1)

      # Subsequently, we return the array of shape
      # (intervallength // resolution, strand)

      return numpyarray


Caching
--------

The construction, including loading and preprocessing,
of a genomic dataset might require a significant amount of time.
In order to avoid having to create the coverage profiles each time you want
to use them, they can be cached and quickly reloaded
later.
Caching can be activated via the options :code:`cache=True`.
When caching is required, janggu will check for changes in the
file content, file composition and various dataset specific argument
(e.g. binsize, resolution) by constructing a SHA256. The dataset will
be loaded or reloaded from scratch if the determined hash does not exist.

Example:

.. code:: python

   # load hg19 if the cache file does not exist yet, otherwise
   # reload it.
   Bioseq.create_from_refgenome('dna', refgenome, order=1, cache=True)


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
These predictions may subsequently be illustrated via `plotGenomeTrack` or exported to a BIGWIG file.


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

Wrapper Datasets
----------------

A Cover object is represents a 4D object. However, sometimes one or more
dimensions of Cover might be single dimensional (e.g. containing only one element).
These dimensions can be dropped using :code:`ReduceDim`.
For example :code:`ReduceDim(cover)`.


Different views datasets
------------------------

Suppose you already have loaded DNA sequence from a reference genome
and you want to use a different parts of it
for training and validating the model performance.
This is achieved by the view mechanism, which allows to
reuse the same dataset by instantiating views that reading out different subsets.

For example, a view constituting the training and test set, respectively.

.. code-block:: python

    # union ROI for training and test set.
    ROI_FILE = resource_filename('janggu', 'resources/roi.bed')
    ROI_TRAIN_FILE = resource_filename('janggu', 'resources/roi_train.bed')
    ROI_TEST_FILE = resource_filename('janggu', 'resources/roi_test.bed')

    DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                       roi=ROI_FILE,
                                       binsize=200,
                                       store_whole_genome=True)

    DNA_TRAIN = view(DNA, ROI_TRAIN_FILE)
    DNA_TEST = view(DNA, ROI_TEST_FILE)


Since underneath the actual dataset is just referenced rather than copied,
the memory footprint won't increase. It just allows to read out different parts
of the genome.

An example is illustrated in the `using view notebook <https://nbviewer.jupyter.org/github/BIMSBbioinfo/janggu/blob/master/src/examples/janggu_convnet_examples_with_view.ipynb>`_.


Randomized dataset
------------------

In order to achieve good predictive performances,
it is recommended to randomize the mini-batches  during model fitting.
This is usually achieved by specifying `shuffle=True` in the fit method.

However, when using HDF5 dataset, this approach may be prohibitively slow due
to the limitations that data from HDF5 files need to be accessed in chunks
rather than in random access fashion.

In order to overcome this issue, it is possible to randomize the dataset
already during loading time such that the data can be consumed later
by reading coherent chunks by setting  `shuffle=False`.

For example, randomization is induced by specifying an integer-valued
:code:`random_state` as in the example below

.. code-block:: python

    DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                       roi=ROI_TRAIN_FILE,
                                       binsize=200,
                                       storage='hdf5',
                                       cache=True,
                                       store_whole_genome=False,
                                       random_state=43)

For this option to be effective and correct, all datasets consumed during
e.g. training need to be provided with the same :code:`random_state` value.
Furthermore, the HDF5 file needs to be stored with :code:`store_whole_genome=False`,
since data storage is not affected by the random_state when the entire genome
is stored.
An example is illustrated in the `using hdf5 notebook <https://nbviewer.jupyter.org/github/BIMSBbioinfo/janggu/blob/master/src/examples/janggu_convnet_examples_with_hdf5.ipynb>`_.

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
