=========
Tutorial
=========

In this section we shall illustrate the Genomic datasets,
how to construct and fit a neural network and finally
how to evaluate the results.


Genomic Datasets
----------------------------------
.. sidebar:: Datasets are named

   Dataset names must match with the Input and Output layers of the neural
   network.


:mod:`janggu.data` provides Dataset classes
that can be used for
training and evaluating neural networks.
Of particular importance are the Genomics-specific dataset,
:class:`Bioseq` and :class:`Cover` which
allow easy access to genomics data,
including DNA sequences or coverage information.
The datasets can be cached and stored in different formats,
including as numpy array, as sparse matrix or on disk
as hdf5 file (see :ref:`storage`).
Other Dataset classes, e.g. :class:`Array` are described
in :ref:`reference-label`.
In the following sections, we shall illustrate the
usage of :class:`Bioseq` and :class:`Cover`.

.. sidebar:: Genomic Datasets

   Most of the parameters are consistent across
   :class:`Bioseq` and :class:`Cover`.


Bioseq
^^^^^^^^^^
The :class:`Bioseq` can be used to load nucleotide
or protein sequence data from
fasta files or from a reference genome
along with a set of genomic coordinates.
The class facilitates access the
*one-hot encoding* representation of the sequences.
Specifically,
the *one-hot encoding* is represented as a
4D array with dimensions corresponding
to :code:`(region, region_length, 1, alphabet_size)`.
The Bioseq offers a number of features:

1. Strand-specific sequence extraction (if DNA sequences are extracted from the reference genome)
2. Higher-order one-hot encoding, e.g. di-nucleotide based

Sequences can be loaded in two ways: using
:code:`Bioseq.create_from_seq` or
:code:`Bioseq.create_from_refgenome`.
The former constructor method can be used to load
DNA or protein sequences from fasta files directly
or from as list of `Bio.SeqRecord.SeqRecord` entries.
An example is shown below:

.. code-block:: python

   from pkg_resources import resource_filename
   from janggu.data import Bioseq

   fasta_file = resource_filename('janggu',
                                  'resources/sample.fa')

   dna = Bioseq.create_from_seq(name='dna',
                                fastafile=fasta_file)

   # there are 3897 sequences in the in sample.fa
   len(dna)

   # Each sequence is 200 bp of length
   dna.shape  # is (3897, 200, 1, 4)

   # One-hot encoding for the first 10 bases of the first region
   dna[0][0, :10, :, 0]

Alternatively, nucleotide sequences can be
obtained from a reference genome directly along with
a BED or GFF file that indicates the region of interest (ROI).
In the example below, the sequence lengths and stepsizes
are determined from the BED-file directly, which requires
all intervals to be of equal lengths:

.. code-block:: python

   roi = resource_filename('janggu',
                           'resources/sample_equalsize.bed')
   refgenome = resource_filename('janggu',
                                 'resources/sample_genome.fa')

   dna = Bioseq.create_from_refgenome(name='dna',
                                      refgenome=refgenome,
                                      regions=roi)

   dna.shape  # is (4, 200, 1, 4)
   # One-hot encoding of the first 10 nucleotides in region 0
   dna[0][0, :10, 0, :]


Sometimes it is more convenient to provide the ROI
as a set of variable-sized broad intervals
(e.g. chr1:10000-50000 and chr3:4000-8000)
which should be divided into sub-intervals
of equal length (e.g. of length 200 bp).
This can be achieved
by explicitly specifying a desired :code:`binsize`
and :code:`stepsize` as shown below:

.. code-block:: python

   roi = resource_filename('janggu',
                           'resources/sample.bed')
   refgenome = resource_filename('janggu',
                                 'resources/sample_genome.fa')

   dna = Bioseq.create_from_refgenome(name='dna',
                                      refgenome=refgenome,
                                      regions=roi,
                                      binsize=200,
                                      stepsize=200)

   dna.shape  # is (100, 200, 4, 1)
   # One-hot encoding of the first 10 nucleotides in region 0
   dna[0][0, :10, 0, :]


The argument :code:`flank` can be used to extend
the intervals up and downstream by a given length

.. code-block:: python

   dna = Bioseq.create_from_refgenome(name='dna',
                                      refgenome=refgenome,
                                      regions=bed_file,
                                      binsize=200,
                                      stepsize=200,
                                      flank=100)

   dna.shape  # is (100, 400, 1, 4)


Finally, sequences can be represented using **higher-order**
one-hot representation using the :code:`order` argument. An example
of a di-nucleotide-based one-hot representation is shown below


.. code-block:: python

   dna = Bioseq.create_from_refgenome(name='dna',
                                      refgenome=refgenome,
                                      regions=bed_file,
                                      binsize=200,
                                      stepsize=200,
                                      order=2)

   # is (100, 200, 1, 16)
   # that is the last dimension represents di-nucleotides
   dna.shape


Cover
^^^^^^^^^^^^^^^
:class:`Cover` can be utilized to fetch different kinds of
coverage data from commonly used data formats,
including BAM, BIGWIG, BED and GFF.
Coverage data is stored as a 4D array with dimensions corresponding
to :code:`(region, region_length, strand, condition)`.

The following examples illustrate how
to instantiate :class:`Cover`.
Additional features are described in the :ref:`reference-label`.

**Coverage from BAM files** can be obtained from
single- or paired-end alignments, which is automatically
determined from the respective SAM flag.
Single-end reads are always counted at the 5' end of
the read. Paired-end reads can optionally be counted
at the 5' end of the first mate
or at the template mid-point.

The following examples illustrate how the alignment coverage can be loaded
from single-end reads:

.. code:: python

   from janggu.data import Cover

   bam_file = resource_filename('janggu',
                                'resources/sample.bam')
   roi = resource_filename('janggu',
                           'resources/sample_equalsize.bed')

   # example with automatic binsize determination from ROI
   cover = Cover.create_from_bam('whole_genome',
                                 bamfiles=bam_file,
                                 regions=roi)
   cover.shape  # is (4, 200, 2, 1)

   # example with specified binsize and stepsize
   roi = resource_filename('janggu',
                           'resources/sample.bed')

   cover = Cover.create_from_bam('whole_genome',
                                 bamfiles=bam_file,
                                 binsize=200,
                                 stepsize=200,
                                 regions=roi)

   cover.shape  # is (100, 200, 2, 1)
   cover[0]  # coverage of the first region

   # example with stranded=False
   cover = Cover.create_from_bam('read_coverage',
                                 bamfiles=bam_file,
                                 binsize=200,
                                 stepsize=200,
                                 stranded=False,
                                 regions=roi)

   cover.shape  # is (100, 200, 1, 1)


By default, the coverage is obtained at nucleotide resolution.
However, often it is more adequate to determine the read
counts in small windows of e.g. 200 bp.
In this case, the :code:`resolution`
argument can be used to count reads in windows of length
resolution.
For example, consider

.. code:: python

   # example with resolution=200 bp
   cover = Cover.create_from_bam('read_coverage',
                                 bamfiles=bam_file,
                                 binsize=200,
                                 stepsize=200,
                                 resolution=200,
                                 regions=roi)

   cover.shape  # is (100, 1, 2, 1)


**Coverage from a BIGWIG files** can be extracted
analogously:

.. code-block:: python

   roi = resource_filename('janggu',
                           'resources/sample.bed')
   bw_file = resource_filename('janggu',
                               'resources/sample.bw')

   cover = Cover.create_from_bigwig('bigwig_coverage',
                                    bigwigfiles=bw_file,
                                    regions=roi,
                                    binsize=200,
                                    stepsize=200)

   cover.shape  # is (100, 1, 1, 1)
   cover[0]  # coverage of the first region


**Coverage from a BED files** can be
extracted in various ways:

1. **Binary** or Presence/Absence mode
2. **Score** mode reads out the score field value from the associated regions.
3. **Categorical** mode transforms the scores into one-hot representation.

Examples of loading data from a BED file are shown below

.. code-block:: python

   roi = resource_filename('janggu',
                           'resources/sample.bed')
   score_file = resource_filename('janggu',
                                  'resources/scored_sample.bed')

   # binary mode (default)
   cover = Cover.create_from_bed('binary_coverage',
                                 bedfiles=score_file,
                                 regions=roi,
                                 binsize=200,
                                 stepsize=200,
                                 resolution=200)

   cover.shape  # is (100, 1, 1, 1)
   cover[4]  # contains [[[[1.]]]]

   # score mode
   cover = Cover.create_from_bed('score_coverage',
                                 bedfiles=score_file,
                                 regions=roi,
                                 binsize=200,
                                 stepsize=200,
                                 resolution=200,
                                 mode='score')

   cover.shape  # is (100, 1, 1, 1)
   cover[4]  # contains the score [[[[5.]]]]

   # categorical mode
   cover = Cover.create_from_bed('cat_coverage',
                                 bedfiles=score_file,
                                 regions=roi,
                                 binsize=200,
                                 stepsize=200,
                                 resolution=200,
                                 mode='categorical')

   cover.shape  # is (100, 1, 1, 6)
   cover[4]  # contains [[[[0., 0., 0., 0., 0., 1.]]]]


Building a neural network
-------------------------
A neural network can be created by
instantiating a :class:`Janggu` object.
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
Janggu additionally offers a simple way to evaluate and export model results,
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
can be found in :ref:`reference-label`.

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
