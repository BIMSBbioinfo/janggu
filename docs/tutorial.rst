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
    from janggo.data import Array

    # Some random data
    DATA = Array('ip', np.random.random((1000, 10)))
    LABELS = Array('out', np.random.randint(2, size=(1000, 1)))

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


:mod:`janggo.data` provides Dataset classes that can be used for
training and evaluating neural networks.
Of particular importance are the Genomics-specific dataset,
:class:`Dna` and :class:`Cover` which
to easily access and fetch genomics data.
Additional Dataset classes are described in the Reference section of the
documentation.


Dna
^^^^^^^^^^
The :class:`Dna` allows to fetch raw sequence data from
fasta files or from a reference genome along with
genomic coordinates of interest
and translates the sequences into a *one-hot encoding*. Specifically,
the *one-hot encoding* is represented as a
4D array with dimensions corresponding
to :code:`(region, region_length, alphabet_size, 1)`.
The Dna offers a number of features:

1. Strand-specific sequence extraction
2. Higher-order one-hot encoding, e.g. di-nucleotide based
3. Dataset access from disk via the hdf5 option for large datasets.

A sequence can be loaded from a fasta file using
the :code:`create_from_fasta` constructor method. For example:

.. code-block:: python

   from pkg_resources import resource_filename
   from janggo.data import Dna

   fasta_file = resource_filename('janggo', 'resources/sample.fa')

   dna = Dna.create_from_fasta('dna', fastafile=fasta_file)

   len(dna)  # there are 3997 sequences in the in sample.fa

   # Each sequence is 200 bp of length
   dna.shape  # is (4, 200, 4, 1)

   # One-hot encoding for the first 10 bases of the first region
   dna[0][0, :10, :, 0]

Alternatively, sequences can be fetched from a reference genome using
genomic coordinates of interest that are provided by a bed or gff file.

.. code-block:: python

   bed_file = resource_filename('janggo', 'resources/sample.bed')
   refgenome = resource_filename('janggo', 'resources/sample_genome.fa')

   dna = Dna.create_from_refgenome('dna',
                                   refgenome=refgenome,
                                   regions=bed_file)

   dna.shape  # is (100, 200, 4, 1)
   dna[0]  # One-hot encoding of region 0


By default, when using :code:`create_from_genome`, the regions
in *bed_file* are split into non-overlapping bins of length 200 bp.
Different tiling procedures can be chosen by specifying
the arguments: :code:`binsize`, :code:`stepsize` and
:code:`flank`.


Cover
^^^^^^^^^^^^^^^
The :class:`Cover` can be utilized to fetch different kinds of
coverage data from commonly used data formats, including BAM, BIGWIG, BED and GFF.
Coverage data is stored as a 4D array with dimensions corresponding
to :code:`(region, region_length, strand, condition)`.

The :class:`Cover` offers the following feature:

1. Strand-specific sequence extraction.
2. :class:`Cover` can be loaded from one or more input files. Then the each condition dimension is associated with an input file.
3. Coverage data can be accessed from disk via the hdf5 option for large datasets.

Additional features are available depending on the input file format.

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

   cover.shape  # is (100, 200, 2, 1)
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

   cover.shape  # is (100, 1, 1, 1)
   cover[0]  # coverage of the first region

By default, the region of interest in :code:`bed_file` is split
into non-overlapping 200 bp windows with a resolution of 200 bp.
Different windowing and signal resolution options are available
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and :code:`resolution`.


**Coverage from a BED files** can be extracted in various ways:

1. Extracting the **score** field value from the associated regions, if available.
2. Extracting binary labels: Treating presence of a region as positive labels (*one*), while the absence of a region is treated as a negative label (*zero*).
3. Treating the scores as categories.

.. code-block:: python

   bed_file = resource_filename('janggo', 'resources/sample.bed')
   score_file = resource_filename('janggo', 'resources/scored_sample.bed')

   # load as binary labels
   cover = Cover.create_from_bed('bed_coverage',
                                 bedfiles=score_file,
                                 regions=bed_file)

   cover.shape  # is (100, 1, 1, 1)
   cover[4]  # contains one

   # load as binary labels
   cover = Cover.create_from_bed('bed_coverage',
                                 bedfiles=score_file,
                                 regions=bed_file,
                                 mode='score')

   cover.shape  # is (100, 1, 1, 1)
   cover[4]  # contains the score 5

   # load as binary labels
   cover = Cover.create_from_bed('bed_coverage',
                                 bedfiles=score_file,
                                 regions=bed_file,
                                 mode='categorical')

   cover.shape  # is (100, 1, 1, 6)
   cover[4]  # contains [0., 0., 0., 0., 0., 1.]

By default, the region of interest in :code:`bed_file` is split
into non-overlapping 200 bp windows with a resolution of 200 bp.
Different windowing and signal resolution options are available
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and :code:`resolution`.


Fit a neural network on DNA sequences
-------------------------------------
In the previous sections, we learned how to acquire data and
how to instantiate neural networks. Now let's
create and fit a simple convolutional neural network that predicts
labels derived from a BED file from the DNA sequence:

.. code:: python

   from keras.layers import Conv2D
   from keras.layers import AveragePooling2D
   from janggo import inputlayer
   from janggo import outputconv

   refgenome = resource_filename('janggo', 'resources/sample_genome.fa')
   bed_file = resource_filename('janggo', 'resources/sample.bed')
   score_file = resource_filename('janggo', 'resources/scored_sample.bed')

   # 1. get data
   DNA = Dna.create_from_refgenome('dna',
                                   refgenome=refgenome,
                                   regions=bed_file)
   LABELS = Cover.create_from_bed('peaks',
                                  bedfiles=score_file,
                                  regions=bed_file)

   # 2. define a simple conv net with 30 filters of length 15 bp
   # and relu activation
   @inputlayer
   @outputconv('sigmoid')
   def _conv_net(inputs, inp, oup, params):
      with inputs.use('dna') as layer:
         layer_ = Conv2D(params[0], (params[1], layer.shape.as_list()[2]),
                         activation=params[2])(layer)
         output = AveragePooling2D(pool_size=(layer_.shape.as_list()[1], 1))(layer_)
      return inputs, output

   # 3. instantiate and compile the model
   model = Janggo.create(template=_conv_net,
                         modelparams=(30, 15, 'relu'),
                         inputs=DNA, outputs=LABELS)
   model.compile(optimizer='adadelta', loss='binary_crossentropy')

   # 4. fit the model
   model.fit(DNA, LABELS)


The network takes as input a 200 bp nucleotide sequence. It uses
30 convolution kernels of length 21 bp, average pooling and another convolution
layer that combines the activities of the 30 kernels
to predict binary valued peaks.

Upon creation of the model a network depiction is
automatically produced in :code:`<results_root>/models` which is illustrated
below

.. image:: dna_peak.png
   :width: 70%
   :alt: Prediction from DNA to peaks
   :align: center

Logging information about the model fitting, model and dataset dimensions
are written to :code:`<results_root>/logs`.


Evaluation
----------

Finally, we would like to evaluate various aspects of the model performance
and investigate the predictions. This can be done by invoking

.. code-block:: python

   model.evaluate(DNA, LABELS)
   model.predict(DNA)

which resemble the familiar keras methods.
Janggo additinally offers features to simplify the
analysis of the results through callback objects that you can
attach when invoking
:code:`model.evaluate` and :code:`model.predict`.
This allows you to determine different performance metrics and/or
export the results in various ways, e.g. as tsv file, as plot or
as bigwig or bed file.

There are two callback classes :code:`InOutScorer` and :code:`InScorer`,
which can be used with :code:`evaluate` and :code:`predict`, respectively.

Both of them maintain a name, a scoring function and an exporter function.

An example of using :code:`InOutScorer` to
write the area under the ROC curve (auROC)
into a tsv file is illustrate in the following

.. code:: python

   from sklearn.metrics import roc_auc_score
   from janggo import InOutScorer
   from janggo.utils import export_tsv

   # create a scorer
   score_auroc = InOutScorer('auROC',
                             roc_auc_score,
                             exporter=export_tsv)

   # determine the auROC
   model.evaluate(DNA, LABELS, callbacks=[score_auroc])

After the evaluation, you will find the results
in :code:`<results-root>/evaluation/<modelname>/auROC.tsv`.

Similarly, you can use :code:`InScorer` to export the predictions
of the model into a json file

.. code:: python

   from janggo import InScorer

   # create scorer
   # in this case, the scoring function is optional.
   pred_scorer = InScorer('predict', exporter=export_json)

   # Evaluate predictions
   model.predict(DNA, datatags=['training_set'],
                 callbacks=[pred_scorer])

The Scorer objects support a number of scoring and exporter function
combinations that can be used to analyze the model results.
For example, you can :code:`InOutScorer` with other `sklearn.metrics`, including
`roc_curve` or `precision_recall_curve` and create a plot using :code:`export_score_plot`.
Or you can export prediction to a bigwig or bed file using :code:`export_bigwig`
and :code:`export_bed`, respectively.

Alternatively, you can supply custom scoring and exporter functions

.. code:: python

   # computes the per-data point loss
   score_loss = InOutScorer('loss', lambda t, p: -t * numpy.log(p),
                            exporter=export_json)

Further examples are illustrated in the reference section and
in the module :code:`janggo.utils`.
