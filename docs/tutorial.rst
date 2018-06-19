=========
Tutorial
=========

In this section we shall illustrate the genomics datasets,
how to construct and fit a neural network and finally
how to evaluate the results.


Genomic Datasets
----------------------------------
.. sidebar:: Datasets are named

   Dataset names must match with the Input and Output layers of the neural
   network.


:mod:`janggu.data` provides Dataset classes that can be used for
training and evaluating neural networks.
Of particular importance are the Genomics-specific dataset,
:class:`Bioseq` and :class:`Cover` which
to easily access and fetch genomics data.
Furthermore, the datasets :class:`Bioseq` and :class:`Cover` facilitate
caching so that you can reload the data later much faster.
Other Dataset classes, e.g. :class:`Array` are described in the Reference section.


Bioseq
^^^^^^^^^^
The :class:`Bioseq` fetches raw sequence data from
fasta files or from a reference genome along with a set of
genomic coordinates
and translates the sequences into a *one-hot encoding*. Specifically,
the *one-hot encoding* is represented as a
4D array with dimensions corresponding
to :code:`(region, region_length, 1, alphabet_size)`.
The Bioseq offers a number of features:

1. Strand-specific sequence extraction
2. Higher-order one-hot encoding, e.g. di-nucleotide based
3. Dataset access from disk via the hdf5 option for large datasets.

A sequence can be loaded from a fasta file using
the :code:`create_from_seq` constructor method:

.. code-block:: python

   from pkg_resources import resource_filename
   from janggu.data import Bioseq

   fasta_file = resource_filename('janggu', 'resources/sample.fa')

   dna = Bioseq.create_from_seq(name='dna', fastafile=fasta_file)

   len(dna)  # there are 3997 sequences in the in sample.fa

   # Each sequence is 200 bp of length
   dna.shape  # is (4, 200, 1, 4)

   # One-hot encoding for the first 10 bases of the first region
   dna[0][0, :10, :, 0]

Alternatively, sequences can be obtained from a reference genome along with
genomic coordinates of interest that are provided by a BED or GFF file.

.. code-block:: python

   bed_file = resource_filename('janggu', 'resources/sample.bed')
   refgenome = resource_filename('janggu', 'resources/sample_genome.fa')

   dna = Bioseq.create_from_refgenome(name='dna',
                                   refgenome=refgenome,
                                   regions=bed_file, datatags=['refgenome'])

   dna.shape  # is (100, 200, 4, 1)
   dna[0]  # One-hot encoding of region 0


By default, when using :code:`create_from_genome`, the regions
in *bed_file* are split into non-overlapping bins of length 200 bp.
Different tiling procedures can be chosen by specifying
the arguments: :code:`binsize`, :code:`stepsize` and
:code:`flank`.


Cover
^^^^^^^^^^^^^^^
:class:`Cover` can be utilized to fetch different kinds of
coverage data from commonly used data formats, including BAM, BIGWIG, BED and GFF.
Coverage data is stored as a 4D array with dimensions corresponding
to :code:`(region, region_length, strand, condition)`.

:class:`Cover` offers the following feature:

1. Strand-specific sequence extraction.
2. :class:`Cover` can be loaded from one or more input files in which case file is associated with a condition.
3. Coverage data can be accessed from disk.

Additional features are available depending on the input file format.

The following examples illustrate how to instantiate :class:`Cover`.

**Coverage from BAM files** is extracted by counting the 5' ends of the tags
in a strand specific manner.

.. code:: python

   from janggu.data import Cover

   bam_file = resource_filename('janggu', 'resources/sample.bam')
   bed_file = resource_filename('janggu', 'resources/sample.bed')

   cover = Cover.create_from_bam('read_coverage',
                                 bamfiles=bam_file,
                                 regions=bed_file)

   cover.shape  # is (100, 200, 2, 1)
   cover[0]  # coverage of the first region

By default, the region of interest in :code:`bed_file` is split
into non-overlapping 200 bp windows. Different windowing options are available
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and :code:`resolution`.

**Coverage from a BIGWIG files** is extracted as the average signal intensity
of a specified resolution (in base pairs):

.. code-block:: python

   bed_file = resource_filename('janggu', 'resources/sample.bed')
   bw_file = resource_filename('janggu', 'resources/sample.bw')

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

1. **Binary** or Presence/Absence mode.
2. **Score** mode reads out the score field value from the associated regions.
3. **Categorical** mode transforms the scores into one-hot representation.

Examples of loading data from a BED file are shown below

.. code-block:: python

   bed_file = resource_filename('janggu', 'resources/sample.bed')
   score_file = resource_filename('janggu', 'resources/scored_sample.bed')

   # binary mode (default)
   cover = Cover.create_from_bed('binary_coverage',
                                 bedfiles=score_file,
                                 regions=bed_file)

   cover.shape  # is (100, 1, 1, 1)
   cover[4]  # contains [[[[1.]]]]

   # score mode
   cover = Cover.create_from_bed('score_coverage',
                                 bedfiles=score_file,
                                 regions=bed_file,
                                 mode='score')

   cover.shape  # is (100, 1, 1, 1)
   cover[4]  # contains the score [[[[5.]]]]

   # categorical mode
   cover = Cover.create_from_bed('cat_coverage',
                                 bedfiles=score_file,
                                 regions=bed_file,
                                 mode='categorical')

   cover.shape  # is (100, 1, 1, 6)
   cover[4]  # contains [[[[0., 0., 0., 0., 0., 1.]]]]

By default, the region of interest in :code:`bed_file` is split
into non-overlapping 200 bp windows with a resolution of 200 bp.
Different windowing and signal resolution options are available
by setting :code:`binsize`, :code:`stepsize`, :code:`flank` and :code:`resolution`.



Building a neural network
-------------------------
A neural network can be created by instantiating a :class:`Janggu` object.
There are two ways of achieving this:

1. Similar as with `keras.models.Model`, a :class:`Janggu` object can be created from a set of native keras Input and Output layers, respectively.
2. Janggu offers a `Janggu.create` constructor method which helps to reduce redundant code when defining many rather similar models.


Example 1: Instantiate Janggu similar to keras.models.Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. sidebar:: **Model name**

   Model results,
   e.g. trained parameters, are automatically stored with the associated model name. To simplify the determination of a unique name for the model, Janggu automatically derives the model name based on a md5-hash of the network configuration. However, you can also specify a name yourself.


.. code-block:: python

  from keras.layers import Input
  from keras.layers import Dense

  from janggu import Janggu

  # Define neural network layers using keras
  in_ = Input(shape=(10,), name='ip')
  layer = Dense(3)(in_)
  output = Dense(1, activation='sigmoid',
                 name='out')(layer)

  # Instantiate model name.
  model = Janggu(inputs=in_, outputs=output)
  model.summary()



Example 2: Specify a model using a model template function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As an alternative to the above stated variant, it is also possible to specify
a network via a python function as in the following example

.. code-block:: python

   def model_template(inputs, inp, oup, params):
       inputs = Input(shape=(10,), name='ip')
       layer = Dense(params)(inputs)
       output = Dense(1, activation='sigmoid',
                      name='out')(layer)
       return inputs, output

   # Defines the same model by invoking the definition function
   # and the create constructor.
   model = Janggu.create(template=model_template,
                         modelparams=3)

The model template function must adhere to the
signature :code:`template(inputs, inp, oup, params)`.
Notice, that :code:`modelparams=3` gets passed on to :code:`params`
upon model creation. This allows to parametrize the network
and reduces code redundancy.


Example 3: Automatic Input and Output layer extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A second benefit to invoke :code:`Janggu.create` is that it can automatically
determine and append appropriate Input and Output layers to the network.
This means, only the network body remains to be defined.

.. code-block:: python

    import numpy as np
    from janggu import inputlayer, outputdense
    from janggu.data import Array

    # Some random data
    DATA = Array('ip', np.random.random((1000, 10)))
    LABELS = Array('out', np.random.randint(2, size=(1000, 1)))

    # inputlayer and outputdense automatically
    # extract dataset shapes and extend the
    # Input and Output layers appropriately.
    # That is, only the model body needs to be specified.
    @inputlayer
    @outputdense('sigmoid')
    def model_body_template(inputs, inp, oup, params):
        with inputs.use('ip') as layer:
            # the with block allows
            # for easy access of a specific named input.
            output = Dense(params)(layer)
        return inputs, output

    # create the model.
    model = Janggu.create(template=test_inferred_model,
                          modelparams=3,
                          inputs=DATA, outputs=LABELS)
    model.summary()

As is illustrated by the example, automatic Input and Output layer determination
can be achieved by using the decorators :code:`inputlayer` and/or
:code:`outputdense` which extract the layer dimensions from the
provided input and output Datasets in the create constructor.


Fit a neural network on DNA sequences
-------------------------------------
In the previous sections, we learned how to acquire data and
how to instantiate neural networks. Now let's
create and fit a simple convolutional neural network that learns
to discriminate between two classes of sequences. In the following example
the sample sequences are of length 200 bp each. `sample.fa` contains Oct4 CHip-seq
peaks and sample2.fa contains Mafk CHip-seq peaks. We shall use a simple
convolutional neural network with 30 filters of length 21 bp to learn
the sequence features that discriminate the two sets of sequences:

.. code:: python

   from keras.layers import Conv2D
   from keras.layers import AveragePooling2D
   from janggu import inputlayer
   from janggu import outputconv

   # load the dataset
   SAMPLE_1 = resource_filename('janggu', 'resources/', 'sample.fa')
   SAMPLE_2 = resource_filename('janggu', 'resources/', 'sample2.fa')

   DNA = Bioseq.create_from_seq('dna', fastafile=[SAMPLE_1, SAMPLE_2],
                               order=args.order)

   # helper function returns the number of sequences
   def nseqs(filename):
      return sum((1 for line in open(filename) if line[0] == '>'))

   Y = np.asarray([1 for line in range(nseqs(SAMPLE_1))] +
                  [0 for line in range(nseqs(SAMPLE_2))])
   LABELS = Array('y', Y, conditions=['TF-binding'])

   # 2. define a simple conv net with 30 filters of length 15 bp
   # and relu activation
   @inputlayer
   @outputconv('sigmoid')
   def _conv_net(inputs, inp, oup, params):
      with inputs.use('dna') as layer:
         layer_ = Conv2D(params[0], (params[1], 1), activation=params[2])(layer)
         output = AveragePooling2D(pool_size=(layer_.shape.as_list()[1], 1))(layer_)
      return inputs, output

   # 3. instantiate and compile the model
   model = Janggu.create(template=_conv_net,
                         modelparams=(30, 15, 'relu'),
                         inputs=DNA, outputs=LABELS)
   model.compile(optimizer='adadelta', loss='binary_crossentropy')

   # 4. fit the model
   model.fit(DNA, LABELS)

An illustration of the network architecture is depicted below.
Upon creation of the model a network depiction is
automatically produced in :code:`<results_root>/models` which is illustrated
below

.. image:: dna_peak.png
   :width: 70%
   :alt: Prediction from DNA to peaks
   :align: center

After the model has been trained, the model parameters and the
illustration of the architecture are stored in :code:`<results_root>/models`.
Furthermore, information about the model fitting, model and dataset dimensions
are written to :code:`<results_root>/logs`.


Evaluation through Scorer callbacks
------------------------------------

Finally, we would like to evaluate various aspects of the model performance
and investigate the predictions. This can be done by invoking

.. code-block:: python

   model.evaluate(DNA_TEST, LABELS_TEST)
   model.predict(DNA_TEST)

which resemble the familiar keras methods.
Janggu additinally offers a simple way to evaluate and export model results,
for example on independent test data.
To this end, objects of :code:`Scorer` can be created
and passed to
:code:`model.evaluate` and :code:`model.predict`.
This allows you to determine different performance metrics and/or
export the results in various ways, e.g. as tsv file, as plot or
as a BIGWIG file.

A :code:`Scorer` maintains a **name**, a **scoring function** and
an **exporter function**. The latter two dictate which score is evaluated
and how the results should be stored.

An example of using :code:`Scorer` to
evaluate the ROC curve and the area under the ROC curve (auROC)
and export it as plot and into a tsv file, respectively, is shown below

.. code:: python

   from sklearn.metrics import roc_auc_score
   from sklearn.metrics import roc_curve
   from janggu import Scorer
   from janggu.utils import export_tsv
   from janggu.utils import export_score_plot

   # create a scorer
   score_auroc = Scorer('auROC',
                        roc_auc_score,
                        exporter=export_tsv)
   score_roc = Scorer('ROC',
                        roc_curve,
                        exporter=export_score_plot)
   # determine the auROC
   model.evaluate(DNA, LABELS, callbacks=[score_auroc, score_roc])

After the evaluation, you will find :code:`auROC.tsv` and :code:`ROC.png`
in :code:`<results-root>/evaluation/<modelname>/`.

Similarly, you can use :code:`Scorer` to export the predictions
of the model. Below, the output predictions are exported in json format.

.. code:: python

   from janggu import Scorer
   from janggu import export_json

   # create scorer
   pred_scorer = Scorer('predict', exporter=export_json)

   # Evaluate predictions
   model.predict(DNA, callbacks=[pred_scorer])

Using the Scorer callback objects, a number of evaluations can
be run out of the box. For example, with different `sklearn.metrics`
and different exporter options. A list of available exporters
can be found in the Reference section.

Alternatively, you can also plug in custom functions

.. code:: python

   # computes the per-data point loss
   score_loss = Scorer('loss', lambda t, p: -t * numpy.log(p),
                            exporter=export_json)


Browse through the results
------------------------------------
Finally, after you have fitted and evaluated your results
you can browse through the results using the
the Dash-based Janggu web application.
To start the application server just run

::

   janggu -path <results-root>

Then you can inspect the outputs in a browser of your choice:

.. image:: janggu_example.png
   :width: 70%
   :alt: Prediction from DNA to peaks
   :align: center
