=========
Tutorial
=========

In this section we shall illustrate a number of ways that allow
you to define, train and evaluate neural networks using `janggo`.

Building a neural network
-------------------------
A neural network can be created by instantiating a :class:`Janggo` object.
There are two ways of achieving this:

1. Similar as with `keras.models.Model`, a :class:`Janggo` object can be created from a set of native keras Input and Output layers, respectively.
2. Janggo offers a `Janggo.create` constructor method which helps to reduce redundant code when defining many rather similar models.


Example 1: Instantiate Janggo similar to keras.models.Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. sidebar:: **Output directory**

   Model results,
   e.g. trained parameters, are automatically stored with an associated model name in `outputdir` which is set to '`/home/user/janggo_results`' by default. Additionally, Janggo determines a unique model name, based on a md5-hash of the network configuration.


.. code-block:: python

  from keras.layers import Input
  from keras.layers import Dense

  from janggo import Janggo

  # Define neural network layers using keras
  in_ = Input(shape=(10,), name='ip')
  layer = Dense(3)(in_)
  output = Dense(1, activation='sigmoid',
                 name='out')(layer)

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
       output = Dense(1, activation='sigmoid',
                      name='out')(layer)
       return inputs, output

   # Defines the same model by invoking the definition function
   # and the create constructor.
   model = Janggo.create(template=test_manual_model,
                         modelparams=3)


The advantage of this variant is that it allows you to define a model template
along with arbitrary parameters to fill in upon creation. For example,
`modelparams=3` is passed on to the `params` argument in `test_manual_model`.
This is useful if one seeks to test numerous slightly different models,
e.g. with different numbers of neurons per layer or different activations.


Example 3: Automatic Input and Output layer extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A second benefit to invoke Janggo.create is that it can automatically
determine and append appropriate Input and Output layers to the network.
This means, only the network body remains to be defined:

.. code-block:: python

    import numpy as np
    from janggo import inputlayer, outputdense
    from janggo.data import NumpyWrapper

    # Some random data
    DATA = NumpyWrapper('ip', np.random.random((1000, 10)))
    LABELS = NumpyWrapper('out', np.random.randint(2, size=(1000, 1)))

    # inputlayer and outputdense automatically
    # extract dataset shapes and extend the
    # Input and Output layers appropriately.
    # That is, only the model body needs to be specified.
    @inputlayer
    @outputdense('sigmoid')
    def test_inferred_model(inputs, inp, oup, params):
        with inputs.use('ip') as layer:
            # the with block allows
            # for easy access of a specific named input.
            output = Dense(params)(layer)
        return inputs, output

    # create the model.
    model = Janggo.create(template=test_inferred_model,
                          modelparams=3,
                          inputs=DATA, outputs=LABELS)
    model.summary()

As is illustrated by the example, automatic Input and Output layer determination
can be achieved by using the decorators inputlayer and/or outputdense which extract
the layer dimensions from the provided inputs and outputs in the create constructor.


Genomic Datasets
----------------------------------
.. sidebar:: Datasets are named

   In :class:`Janggo`, a Dataset is linked to
   its Input and Output layers via corresponding Dataset and Layer names.


:mod:`janggo.data` provides a number of special genomics data containers which
to easily access and fetch genomics data. The datasets can directly be used
e.g. for training or evaluating neural networks.

Two of the most useful containers are :class:`Dna` and :class:`Cover`.



Dna
^^^^^^^^^^
The :class:`Dna` allows you to fetch raw sequence data from
fasta files or from a reference genome with genomic coordinates of interest
and translates the sequences into a *one-hot encoding*. Specifically,
the sequence is stored in a 4D array with dimensions corresponding
to :code:`(region, region_length, alphabet_size, 1)`.
The Dna offers a number of features:

1. Strand-specific sequence extraction
2. Higher-order one-hot encoding, e.g. di-nucleotide based
3. Dataset access from disk via the hdf5 option for large datasets.

Loading DNA sequences from fasta files can be achieved by invoking
`create_from_fasta`

.. code-block:: python

   from pkg_resources import resource_filename
   from janggo.data import Dna

   fasta_file = resource_filename('janggo', 'resources/sample.fa')

   dna = Dna.create_from_fasta('dna', fastafile=fastafile)

   len(dna)  # there are 4 sequences in the fastafile

   #
   dna.shape  # is (4, 200, 4, 1)
   dna[0]  # one-hot encoding or region 0

Alternatively, sequences can be fetched from a reference genome with
genomic coordinates of interest from a bed or gff file.

.. code-block:: python

   bed_file = resource_filename('janggo', 'resources/sample.bed')
   refgenome = resource_filename('janggo', 'resources/sample_genome.fa')

   dna = Dna.create_from_refgenome('dna',
                                   refgenome=refgenome,
                                   regions=bed_file)

   # the regions defined in the bed_file are by default split up in
   # 200 bp bins with stepsize 50. Hence, there are 14344 intervals.
   dna.shape  # is (14344, 200, 4, 1)
   dna[0]  # One-hot encoding of region 0



Cover
^^^^^^^^^^^^^^^
The :class:`Cover` can be utilized to fetch different kinds of
coverage data from commonly used data formats, including BAM, BIGWIG, BED and GFF.
Regardless of the input file format the coverage tracks are
stored as a 4D array with dimensions corresponding
to :code:`(region, region_length, strand, condition)`.

The :class:`Cover` offers the following feature:

1. Strand-specific sequence extraction, if strandedness information is available in the bed_file
2. :class:`Cover` can be loaded from one or more input files. Then the each condition dimension is associated with an input file.
3. Dataset access from disk via the hdf5 option for large datasets.

Moreover, additional features are available depending on the input file format (see References).

The following examples illustrate how to instantiate :class:`Cover`.

**Coverage from BAM files** is extracted by counting the 5' ends of the tags
in a strand specific manner.

.. code:: python

   from janggo.data import Cover

   bam_file = resource_filename('janggo', 'resources/sample.bam')
   bed_file = resource_filename('janggo', 'resources/sample.bed')

   cover = Cover.create_from_bam('read_coverage',
                                 bamfiles=bam_file,
                                 regions=bed_file)

   # The regions in the bed_file are split into non-overlapping 10 bp bins
   # which amounts to 4 regions of length 10 bp.
   cover.shape  # is (4, 10, 2, 1)
   cover[0]  # coverage of the first region

By default, the region of interest in :code:`bed_file` is split
into non-overlapping 200 bp windows. Different windowing options are available
by setting :code:`binsize`, :code:`stepsize` and :code:`flank`.

**Coverage from a BIGWIG files** is extracted as the average signal intensity
of a specified resolution (in base pairs):

.. code-block:: python

   bed_file = resource_filename('janggo', 'resources/sample.bed')
   bw_file = resource_filename('janggo', 'resources/sample.bw')

   cover = Cover.create_from_bigwig('bigwig_coverage',
                                    bigwigfiles=bw_file,
                                    regions=bed_file)

   # The regions in the bed_file are split into non-overlapping 10 bp bins
   # which amounts to 4 regions of length 10 bp. Additionally, resolution
   # computes the average signal in a given window.
   # shape is (4, 5, 1, 1), because there are 5 x 2 bp
   # resolution windows summing up to the binsize of 10 bp.
   cover.shape
   cover[0]  # coverage of the first region

By default, the region of interest in :code:`bed_file` is split
into non-overlapping 200 bp windows with a resolution of 200 bp.
Different windowing and signal resolution options are available
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and/or :code:`resolution`.


**Coverage from a BED files** (or GFF files alike) is extracted by

1. Extracting the **score** field value from the associated regions, if available.
2. Treating presence of a region as positive labels (*one*), while the absence of a region is treated as a negative label (*zero*).
3. Treating the scores as categories.

.. code-block:: python

   bed_file = resource_filename('janggo', 'resources/sample.bed')
   score_bed_file = resource_filename('janggo', 'resources/sample_scores.bed')

   cover = Cover.create_from_bed('bed_coverage',
                                 bedfiles=score_bed_files,
                                 regions=bed_file)

   cover.shape
   cover[0]  # coverage of the first region


By default, the region of interest in :code:`bed_file` is split
into non-overlapping 200 bp windows with a resolution of 200 bp.
Different windowing and signal resolution options are available
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and/or :code:`resolution`.


Fit a neural network on DNA sequences
-------------------------------------
In the previous sections, we learned how to acquire data and
how to instantiate neural networks. Now let's
create and fit a simple convolutional neural network based on DNA sequence
and labels from a BED file.

.. code:: python

   refgenome = resource_filename('janggo', 'resources/sample_refgenome.bed')
   bed_file = resource_filename('janggo', 'resources/sample.bed')
   score_bed_file = resource_filename('janggo', 'resources/sample_scores.bed')

   # 1. get data
   DNA = Dna.create_from_refgenome('dna',
                                   refgenome=refgenome,
                                   regions=bed_file)
   LABELS = Cover.create_from_bed('peaks',
                                  bedfiles=score_bed_file,
                                  regions=bed_file)

   # 2. define a simple conv net with 30 filters of length 15 bp
   # and relu activation
   @inputlayer
   @outputconv('sigmoid')
   def _conv_net(inputs, inp, oup, params):
      with inputs.use('dna') as layer:
         layer_ = Conv2D(params[0], (params[1], layer.shape.as_list()[2]),
                         activation=params[2])(layer)
         output = GlobalAveragePooling2D()(layer_)
      return inputs, output

   # 3. instantiate and compile the model
   model = Janggo.create(template=_conv_net
                         modelparams=(30, 15, 'relu'),
                         inputs=DNA, outputs=LABELS)
   model.compile(optimizer='adadelta', loss='binary_crossentropy')

   # 4. fit the model
   model.fit(DNA, LABELS)

Congratulations! You've finished the getting started janggo tutorial!
Next, you might be interested in how to evaluate the model performances
and delve into some more advanced examples.

Evaluation
----------

The evaluation of models are controlled by :class:`EvaluatorList`.
