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

   dna = Dna.create_from_fasta('dna', fastafile=fasta_file)

   len(dna)  # there are 3997 sequences in the in sample.fa

   # Each sequence is 200 bp of length
   dna.shape  # is (4, 200, 4, 1)

   # One-hot encoding for the first 10 bases of the first region
   dna[0][0, :10, :, 0]

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
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and :code:`resolution`.


**Coverage from a BED files** can be extracted in various ways:

1. Extracting the **score** field value from the associated regions, if available.
2. Extracting binary labels: Treating presence of a region as positive labels (*one*), while the absence of a region is treated as a negative label (*zero*).
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
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and :code:`resolution`.


Fit a neural network on DNA sequences
-------------------------------------
In the previous sections, we learned how to acquire data and
how to instantiate neural networks. Now let's
create and fit a simple convolutional neural network that predicts
labels derived from a BED file from the DNA sequence:

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



Evaluation
----------

Finally, we would like to evaluate various aspects of the model performance
and investigate the predictions. This can be done by invoking the
methods :code:`evaluate` and :code:`predict`.
While, this is also possible using a native keras model, janggo
also offers a number of useful functions to 1. export the prediction
and evaluation results in e.g. json, tsv, 2. plot the scoring metrics such as
AUC-ROC, and 3. allows to export predictions and model loss in BED or BIGWIG
format for further investigation of what the model has (or has not) trained
in a genome browser of your choice.

:code:`InOutScorer`
^^^^^^^^^^^^^^^^^^^
Evaluating the predictive performance in comparison with ground truth labels,
you need to instantiate one or more :code:`InOutScorer` object that
can be attached as callbacks to :code:`Janggo.evaluate`.
The following example shows how to compute the AUC-ROC, plot the ROC curve
and export the prediction loss to bigwig format

In order to compute
.. code:: python

   import numpy
   from sklearn.metrics import roc_auc_score, roc_curve

   # Instantiate several evaluation scorers
   score_auroc = InOutScorer('auROC', roc_auc_score, dumper=dump_tsv)
   score_roc = InOutScorer('ROC', roc_curve, dumper=plot_score)
   score_loss = InOutScorer('loss', lambda t, p: -t * numpy.log(p),
                            dumper=export_bigwig,
                            dump_args={'gindexer': DNA.gindexer})

   # Evaluate the results
   model.evaluate(DNA, LABELS, datatags=['training_set'],
                  callbacks=[score_auroc, score_roc, score_loss])

   # Until this point the evaluators have only collected the scores
   # Finally, we need to dump the evaluated information
   [ev.dump(model.outputdir) for ev in [score_auroc, score_roc, score_loss]]


:code:`InScorer`
^^^^^^^^^^^^^^^^^^^
Sometimes it is useful to evaluate the results based on input data only.
For example, when you want to have a look at the predicted regions
or if you want to investigate the feature activities of a specified layer.
In this case, you need to instantiate one or more :code:`InScorer` objects
which are attached as callbacks to :code:`Janggo.predict`.

For example to export the model predictions to BED format
you can invoke the following lines of code:

.. code:: python

   # Instantiate several evaluation scorers
   pred = InOutScorer('predict', dumper=export_bed,
                         dump_args={'gindexer': DNA.gindexer})

   # Evaluate predictions
   model.predict(DNA, datatags=['training_set'],
                 callbacks=[pred])
