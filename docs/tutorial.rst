=========
Tutorial
=========

In this section we shall illustrate a plethora of ways in which
`janggo` can be used.

Building a neural network
-------------------------
Janggo allow you to instantiate neural network model in several ways:

1. Similar as with `keras.models.Model`, a `Janggo` object can be created from a set of Input and Output layers, respectively.
2. Janggo offers a `Janggo.create` constructor method which helps to reduce boiler plate code and redundant code when defining many rather similar models.


Example 1: Instantiate Janggo similar to keras.models.Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	from keras.layers import Input
	from keras.layers import Dense

	from janggo import Janggo

	# Define neural network layers using keras
	in_ = Input(shape=(10,), name='ip')
	layer = Dense(3)(in_)
	output = Dense(1, activation='sigmoid', name='out')(layer)

	# Instantiate model name.
	model = Janggo(inputs=in_, outputs=output)
	model.summary()


Example 2: Specify a model using a model definition function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As an alternative to the above stated variant, it is also possible to specify
a network via a python function as in the following example

.. code-block:: python

   def test_manual_model(inputs, inp, oup, params):
       inputs = Input(shape=(10,), name='ip')
       layer = Dense(params)(inputs)
       output = Dense(1, activation='sigmoid', name='out')(layer)
       return inputs, output

   # Defines the same model by invoking the definition function
   # and the create constructor.
   model = Janggo.create(template=test_manual_model, modelparams=3)


The advantage of this variant is that it allows you to define a model template
along with arbitrary parameters to fill in upon creation. For example,
`modelparams=3` is passed on to the `params` argument in `test_manual_model`.
Generally, rather than having to define many very similar networks,
the same template can be instantiated with different
model parameters which reduces redundant code.


Example 3: Automatic Input and Output layer extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A second benefit to invoke Janggo.create is that it can automatically
determine and append appropriate Input and Output layers to the network.
This means, only the network body remains to be defined which reduces
redundant boiler plate code.
This is illustrated in the following piece of code

.. code-block:: python

	 import numpy as np
	 from janggo import inputlayer, outputdense
	 from janggo.data import NumpyDataset

	 # Some random data which you would like to use as input for the
	 # model.
	 DATA = NumpyDataset('ip', np.random.random((1000, 10)))
	 LABELS = NumpyDataset('out', np.random.randint(2, size=(1000, 1)))

	 # inputlayer and outputdense automatically extract the layer shapes
	 # so that only the model body needs to be specified.
	 @inputlayer
	 @outputdense('sigmoid')
	 def test_inferred_model(inputs, inp, oup, params):
	 	with inputs.use('ip') as layer:
	 		# the with block allows for easy access of a specific named input.
	 		output = Dense(params)(layer)
	 	return inputs, output

	 # create the model.
	 model = Janggo.create(template=test_inferred_model, modelparams=3,
	 		inputs=DATA, outputs=LABELS)
	 model.summary()

Importantly, this example introduces the two ingredients
which are required for the automatic Input and Output layer determination:
First, decorators that append additional network layers and second,
the optional arguments inputs and outputs for the create method.


.. tip:: **Automatic name determination**

   Janggo objects hold a name property which are used to store model results. By default, Janggo automatically determines a unique model name, based on a md5-hash of the network configuration. This simplifies model comparison. However, the user may also provide a different name.

.. tip:: **Output directory**

   Janggo stores all outputs in `outputdir` which by default is located at `/home/user/janggo_results`.

Genomic Datasets
----------------------------------
The janggo package provides a number of useful data containers which
can conveniently fetch genomics data. The datasets can directly be used
e.g. for training or evaluating neural networks.

Two of the most useful containers are :class:`DnaDataset` and :class:`CoverageDataset`.

.. note:: Datasets are named

   All datasets defined by janggo are named.
	 When instantiating a network with Janggo, the network
   Input and Output layers must have the same name as the Datasets that are
   provided to the network.


DnaDataset
^^^^^^^^^^
The DnaDataset allows you to fetch sequence data directly from fasta sequences
or from a reference genome with genomic coordinates of interest.

Loading DNA sequences from fasta files can be achieved by `create_from_fasta`
as is illustrated in the following example

.. code:: python

   import os
   import pkg_resources
   from janggo.data import DnaDataset
	 data_path = pkg_resources.resource_filename('janggo', 'resources/')

	 fastafile = os.path.join(data_path, 'oct4.fa')
	 dna = DnaDataset.create_from_fasta('dna', fastafile=fastafile)

	 # the fasta file contains 4 sequences of length 200 bp each.
   dna.shape  # is (4, 200, 4, 1)
	 dna[0]  # one-hot encoding or region 0

Alternatively, sequences can be fetched from a reference genome with
genomic coordinates of interest from a bed or gff file.

.. code:: python

   data_path = pkg_resources.resource_filename('janggo', 'resources/')
   bed_file = os.path.join(data_path, 'regions.bed')

   refgenome = os.path.join(data_path, 'genome.fa')

   dna = DnaDataset.create_from_refgenome('dna', refgenome=refgenome,
				regions=bed_file)

   # the regions defined in the bed_file are by default split up in
	 # 200 bp bins with stepsize 50. Hence, there are 14344 intervals.
   dna.shape  # is (14344, 200, 4, 1)
	 dna[0]  # One-hot encoding of region 0

Furthermore, DnaDataset offers a number of options. For instance,
the sequence information can be loaded as numpy arrays into RAM or from
a hdf5 file from disk. The latter option is especially useful if only
limited resources are available.
The raw DNA sequence is transformed to one-hot encoding which can be
used as input for a convolutional neural network. Usually, the one-hot encoding
encodes each nucleotide in the sequence as a 4-dim vector where only one element
is *one* and the others are *zero*. `janggo` allow you to specify one-hot encodings
also for higher-order nucleotide composition. For example, di-nucleotide based
one-hot encoding is achieved by setting the argument `order=2`.


CoverageDataset
^^^^^^^^^^^^^^^
The CoverageDataset can be utilized to fetch data from commonly used data formats,
including BAM, BIGWIG, BED and GFF.

Get strand specific read coverage from a BAM file at the 5' end of the reads:

.. code:: python

   from janggo.data import CoverageDataset

   data_path = pkg_resources.resource_filename('janggo', 'resources/')
	 bamfile_ = os.path.join(data_path, "yeast_I_II_III.bam")
	 bed_file = os.path.join(data_path, "yeast.bed")

   cover = CoverageDataset.create_from_bam(
         'read_coverage',
         bamfiles=bamfile_,
         regions=bed_file,
         binsize=10, stepsize=10)

   # The regions in the bed_file are split into non-overlapping 10 bp bins
	 # which amounts to 4 regions of length 10 bp.
	 cover.shape  # is (4, 10, 2, 1)
	 cover[0]  # coverage of the first region


Get the coverage from a BIGWIG file:

.. code:: python

   bwfile_ = os.path.join(data_path, "yeast_I_II_III.bw")
   bed_file = os.path.join(data_path, "yeast.bed")

	 CoverageDataset.create_from_bigwig(
	 			'bigwig_coverage',
	 			bigwigfiles=bwfile_,
	 			regions=bed_file,
	 			binsize=10, stepsize=10,
	 			resolution=2)

   # The regions in the bed_file are split into non-overlapping 10 bp bins
   # which amounts to 4 regions of length 10 bp. Additionally, resolution
	 # computes the average signal in a given window.
	 # shape is (4, 5, 1, 1), because there are 5 x 2 bp
	 # resolution windows summing up to the binsize of 10 bp.
   cover.shape
   cover[0]  # coverage of the first region


Get the coverage from a BED file:

.. code:: python
   # bed_file contains the region of interest
   bed_file = os.path.join(data_path, 'regions.bed')

	 # score_bed_file contains the scores, labels or categories.
   score_bed_file = os.path.join(data_path, "indiv_regions.bed")

   CoverageDataset.create_from_bed(
		'bed_coverage',
		bedfiles=score_bed_files,
		regions=bed_file,
		binsize=200, stepsize=50,
		resolution=50)

   #
   cover.shape
   cover[0]  # coverage of the first region


Fit a neural network on DNA sequences
-------------------------------------
Now that we know how to acquire data and how to instantiate neural networks,
lets create a simple convolutional neural network that learn to predict
ChIP-seq labels from DNA sequences

.. code:: python

# 1. get data
DNA = DnaDataset()
LABELS = CoverageDataset

# 2. define a network
@inputlayer
@outputconv
def _conv_net(inputs, inp, oup, params):
   with inputs.use('dna') as layer:


# 3. instantiate the model
model = Janggo.create(template=)

# 4. fit the model
model.fit(DNA, LABELS)

Evaluation
----------

Predict TF binding from the DNA sequence
--------------------------------------------

Predict TF binding from both DNA strands
-----------------------------------------------

Predict TF binding using higher-order motifs
-----------------------------------------------
