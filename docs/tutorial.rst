=========
Tutorial
=========

In this section we shall illustrate a plethora of ways in which
`janggo` can be used.

Building a neural network
-------------------------
Janggo allow you to instantiate neural network model in several ways.

1. Similar as for `keras.models.Model`, a `Janggo` object can be created from Input and Output layers, respectively.
2. Using the `Janggo.create` constructor method.

.. tip:: **Automatic name determination**

   Janggo objects hold a name property which are used to store model results. By default, Janggo automatically determines a unique model name, based on a md5-hash of the network configuration. This simplifies model comparison. However, the user may also provide a different name.

.. tip:: **Output directory**

   Janggo stores all outputs in `outputdir` which by default is located at `/home/user/janggo_results`.



Variant 0: Use Janggo similar to keras.models.Model.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Variant 1: Specify a model using a model definition function.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As an alternative to the above stated variant, it is also possible to specify
a  network via a python function as in the following example

.. code-block:: python

   def test_manual_model(inputs, inp, oup, params):
       in_ = Input(shape=(10,), name='ip')
       layer = Dense(params)(in_)
       output = Dense(1, activation='sigmoid', name='out')(in_)
       return in_, output

   # Defines the same model by invoking the definition function
   # and the create constructor.
   model = Janggo.create(template=test_manual_model, modelparams=3)

The advantage of this variant is that it allows you to define a model template
along with arbitrary parameters to fill in upon creation. So rather than having
to define many very similar networks, the same template can be used with different
model parameters.

The second important advantage of the Janggo.create is that it can be
used to automatically determine the Input and Output shapes from a given dataset

.. code-block:: python

	from numpy as np
	from janggo import Janggo
	from janggo import inputlayer, outputdense
	from janggo.data import input_props, output_props
	from jangoo.data import NumpyDataset

	# Some random data which you would like to use as input for the
	# model.
	DATA = NumpyDataset('ip', np.random.random((1000, 10)))
	LABELS = NumpyDataset('out', np.random.randint(2, size=(1000, 1)))

	# inputlayer and outputdense automatically extract the layer shapes
	# so that only the model body needs to be specified.
	@inputlayer
	@outputdense
	def test_inferred_model(inputs, inp, oup, params):
			with inputs.use('ip') as in_:
					# the with block allows for easy access of a specific named input.
					output = Dense(params)(in_)
			return in_, output

	# create the model.
	inp = input_props(DATA)
	oup = output_props(LABELS)
	model = Janggo.create(template=test_inferred_model, modelparams=3
												inputp=inp, outputp=oup)

	# Compile the model
	model.compile(optimizer='adadelta', loss='binary_crossentropy')



Predict TF binding from the DNA sequence
--------------------------------------------

Predict TF binding from both DNA strands
-----------------------------------------------

Predict TF binding using higher-order motifs
-----------------------------------------------
