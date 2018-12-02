import os

import matplotlib
import numpy as np
import pandas as pd
import pkg_resources
import pytest
from keras.layers import Average
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Maximum
from keras.layers import MaxPooling2D

from janggu import Janggu
from janggu import inputlayer
from janggu import outputconv
from janggu import outputdense
from janggu.data import Array
from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data.data import JangguSequence
from janggu.layers import Complement
from janggu.layers import DnaConv2D
from janggu.layers import LocalAveragePooling2D
from janggu.layers import Reverse

matplotlib.use('AGG')


def test_localaveragepooling2D(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # some test data
    testin = np.ones((1, 10, 1, 3))
    testin[:, :, :, 1] += 1
    testin[:, :, :, 2] += 2

    # test local average pooling
    lin = Input((10, 1, 3))
    out = LocalAveragePooling2D(3)(lin)
    m = Janggu(lin, out)

    testout = m.predict(testin)
    np.testing.assert_equal(testout, testin[:, :8, :, :])

    # more tests
    testin = np.ones((1, 3, 1, 2))
    testin[:, 0, :, :] = 0
    testin[:, 2, :, :] = 2
    testin[:, :, :, 1] += 1

    # test local average pooling
    lin = Input((3, 1, 2))
    out = LocalAveragePooling2D(3)(lin)
    m = Janggu(lin, out)

    testout = m.predict(testin)
    np.testing.assert_equal(testout.shape, (1, 1, 1, 2))
    np.testing.assert_equal(testout[0, 0, 0, 0], 1)
    np.testing.assert_equal(testout[0, 0, 0, 1], 2)


def test_janggu_generate_name(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    def _cnn_model(inputs, inp, oup, params):
        inputs = Input((10, 1))
        layer = Flatten()(inputs)
        output = Dense(params[0])(layer)
        return inputs, output

    bwm = Janggu.create(_cnn_model, modelparams=(2,))
    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=bwm.outputdir)

    bwm.save()
    bwm.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name(bwm.name)


def test_janggu_instance_dense(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    csvfile = os.path.join(data_path, 'sample.csv')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file, order=1)

    df = pd.read_csv(csvfile, header=None)
    ctcf = Array('ctcf', df.values, conditions='peaks')

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs['.']
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output

    with pytest.raises(Exception):
        # due to No input name . defined
        bwm = Janggu.create(_cnn_model, modelparams=(2,),
                            inputs=dna,
                            outputs=ctcf,
                            name='dna_ctcf_HepG2-cnn')

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs[list()]
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output

    with pytest.raises(Exception):
        # due to Wrong type for indexing
        bwm = Janggu.create(_cnn_model, modelparams=(2,),
                            inputs=dna,
                            outputs=ctcf,
                            name='dna_ctcf_HepG2-cnn')

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs()[0]
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output

    with pytest.raises(Exception):
        # name with must be string
        bwm = Janggu.create(_cnn_model, modelparams=(2,),
                            inputs=dna,
                            outputs=ctcf,
                            name=12342134)

    # test with given model name
    bwm = Janggu.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn')
    # test with auto. generated modelname.
    bwm = Janggu.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn')

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs[0]
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output
    bwm = Janggu.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn')

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs['dna']
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output
    bwm = Janggu.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn')
    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()
    bwm.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name('dna_ctcf_HepG2-cnn')


def test_janggu_instance_conv(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    posfile = os.path.join(data_path, 'scored_sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       roi=bed_file, order=1,
                                       binsize=200,
                                       stepsize=50)

    ctcf = Cover.create_from_bed(
        "positives",
        bedfiles=posfile,
        roi=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        store_whole_genome=False,
        flank=0,
        collapser=None,
        storage='ndarray')

    ctcf = Cover.create_from_bed(
        "positives",
        bedfiles=posfile,
        roi=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        store_whole_genome=True,
        flank=0,
        collapser=None,
        storage='ndarray')

    @inputlayer
    @outputconv('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
        layer = Complement()(layer)
        layer = Reverse()(layer)
        return inputs, layer

    bwm = Janggu.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()
    bwm.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name('dna_ctcf_HepG2-cnn')


def test_janggu_use_dnaconv_none(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    posfile = os.path.join(data_path, 'scored_sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file, order=1)

    @inputlayer
    def _cnn_model1(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            layer = DnaConv2D(Conv2D(5, (3, 1), name='fconv1'),
                              merge_mode=None, name='bothstrands')(layer)
        return inputs, layer

    bwm1 = Janggu.create(_cnn_model1, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn1')

    p1 = bwm1.predict(dna[1:2])
    w = bwm1.kerasmodel.get_layer('bothstrands').get_weights()

    @inputlayer
    def _cnn_model2(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            conv = Conv2D(5, (3, 1), name='singlestrand')
            fl = conv(layer)
            rl = Reverse()(conv(Complement()(Reverse()(inlayer))))
        return inputs, [fl, rl]

    bwm2 = Janggu.create(_cnn_model2, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn2')

    bwm2.kerasmodel.get_layer('singlestrand').set_weights(w)

    p2 = bwm2.predict(dna[1:2])
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-3)

    bwm1.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm1._storage_path(bwm1.name, outputdir=tmpdir.strpath)

    bwm1.save()
    bwm1.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name('dna_ctcf_HepG2-cnn1')

def test_janggu_use_dnaconv_concat(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    posfile = os.path.join(data_path, 'positive.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file, order=1)

    @inputlayer
    def _cnn_model1(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            layer = DnaConv2D(Conv2D(5, (3, 1), name='fconv1'),
                              merge_mode='concat', name='bothstrands')(layer)
        return inputs, layer

    bwm1 = Janggu.create(_cnn_model1, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn1')

    p1 = bwm1.predict(dna[1:2])
    w = bwm1.kerasmodel.get_layer('bothstrands').get_weights()

    @inputlayer
    def _cnn_model2(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            conv = Conv2D(5, (3, 1), name='singlestrand')
            fl = conv(layer)
            rl = Reverse()(conv(Complement()(Reverse()(inlayer))))
            layer = Concatenate()([fl, rl])
        return inputs, layer

    bwm2 = Janggu.create(_cnn_model2, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn2')

    bwm2.kerasmodel.get_layer('singlestrand').set_weights(w)

    p2 = bwm2.predict(dna[1:2])
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-3)

    bwm1.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm1._storage_path(bwm1.name, outputdir=tmpdir.strpath)

    bwm1.save()
    bwm1.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name('dna_ctcf_HepG2-cnn1')


def test_janggu_use_dnaconv_ave(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    posfile = os.path.join(data_path, 'positive.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file, order=1)

    @inputlayer
    def _cnn_model1(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            layer = DnaConv2D(Conv2D(5, (3, 1), name='fconv1'),
                              merge_mode='ave', name='bothstrands')(layer)
        return inputs, layer

    bwm1 = Janggu.create(_cnn_model1, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn1')

    p1 = bwm1.predict(dna[1:2])
    w = bwm1.kerasmodel.get_layer('bothstrands').get_weights()

    @inputlayer
    def _cnn_model2(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            conv = Conv2D(5, (3, 1), name='singlestrand')
            fl = conv(layer)
            rl = Reverse()(conv(Complement()(Reverse()(inlayer))))
            layer = Average()([fl, rl])
        return inputs, layer

    bwm2 = Janggu.create(_cnn_model2, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn2')

    bwm2.kerasmodel.get_layer('singlestrand').set_weights(w)

    p2 = bwm2.predict(dna[1:2])
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-3)

    bwm1.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm1._storage_path(bwm1.name, outputdir=tmpdir.strpath)

    bwm1.save()
    bwm1.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name('dna_ctcf_HepG2-cnn1')


def test_janggu_use_dnaconv_max(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    posfile = os.path.join(data_path, 'positive.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file, order=1)

    @inputlayer
    def _cnn_model1(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            layer = DnaConv2D(Conv2D(5, (3, 1), name='fconv1'),
                              merge_mode='max', name='bothstrands')(layer)
        return inputs, layer

    bwm1 = Janggu.create(_cnn_model1, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn1')

    p1 = bwm1.predict(dna[1:2])
    w = bwm1.kerasmodel.get_layer('bothstrands').get_weights()

    @inputlayer
    def _cnn_model2(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            conv = Conv2D(5, (3, 1), name='singlestrand')
            fl = conv(layer)
            rl = Reverse()(conv(Complement()(Reverse()(inlayer))))
            layer = Maximum()([fl, rl])
        return inputs, layer

    bwm2 = Janggu.create(_cnn_model2, modelparams=(2,),
                        inputs=dna,
                        name='dna_ctcf_HepG2-cnn2')

    bwm2.kerasmodel.get_layer('singlestrand').set_weights(w)

    p2 = bwm2.predict(dna[1:2])
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-3)

    bwm1.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm1._storage_path(bwm1.name, outputdir=tmpdir.strpath)

    bwm1.save()
    bwm1.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name('dna_ctcf_HepG2-cnn1')



def test_janggu_chr2_validation(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    posfile = os.path.join(data_path, 'scored_sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    binsize=200, stepsize=50,
                                    roi=bed_file, order=1)

    ctcf = Cover.create_from_bed(
        "positives",
        bedfiles=posfile,
        roi=bed_file,
        binsize=200, stepsize=50,
        resolution=None,
        flank=0,
        collapser='max',
        storage='ndarray')

    @inputlayer
    @outputconv('sigmoid')
    def _cnn_model1(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
            layer = DnaConv2D(Conv2D(5, (3, 1), name='fconv1'),
                              merge_mode='max', name='bothstrands')(layer)
            layer = MaxPooling2D((198, 1))(layer)
        return inputs, layer

    bwm1 = Janggu.create(_cnn_model1, modelparams=(2,),
                        inputs=dna, outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn1')

    bwm1.compile(optimizer='adadelta', loss='binary_crossentropy')
    p1 = bwm1.fit(dna, ctcf, validation_data=['chr2'])


def test_janggu_train_predict_option1(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Train, predict and evaluate on dummy data.

    create: by_shape
    Input args: Dataset
    """

    inputs = Array("X", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    @inputlayer
    @outputdense('sigmoid')
    def test_model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = Janggu.create(test_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs)


def test_janggu_train_predict_option2(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Train, predict and evaluate on dummy data.

    create: NO
    Input args: list(Dataset)
    """

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    def _model():
        inputs = Input((10,), name='x')
        output = Dense(1, activation='sigmoid', name='y')(inputs)
        model = Janggu(inputs=inputs, outputs=output, name='test')
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model()

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict([inputs])
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs])


def test_janggu_train_predict_option3(tmpdir):
    """Train, predict and evaluate on dummy data.

    Only works without generators and without evaluators.

    create: NO
    Input args: list(np.array)
    """

    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    inputs = np.random.random((100, 10))
    outputs = np.random.randint(2, size=(100, 1))

    def _model():
        inputs = Input((10,), name='x')
        output = Dense(1, activation='sigmoid')(inputs)
        model = Janggu(inputs=inputs, outputs=output, name='test')
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model()

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32)
    assert os.path.exists(storage)

    pred = bwm.predict([inputs])

    bwm.predict([inputs], batch_size=32)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs])

    bwm.evaluate([inputs], [outputs], batch_size=32)


def test_janggu_train_predict_option4(tmpdir):
    """Train, predict and evaluate on dummy data.

    Only works without generators and without evaluators.

    create: NO
    Input args: np.array
    """
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    inputs = np.random.random((100, 10))
    outputs = np.random.randint(2, size=(100, 1))

    def _model(path):
        inputs = Input((10,), name='x')
        output = Dense(1, activation='sigmoid')(inputs)
        model = Janggu(inputs=inputs, outputs=output, name='test')
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32)

    # This used to not work with normal numpy arrays,
    # but now the numpy arrays are matched automatically
    # with the layer names.
    bwm.fit(inputs, outputs, epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs)

    bwm.predict(inputs, batch_size=32)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs)

    bwm.evaluate(inputs, outputs, batch_size=32)


def test_janggu_train_predict_option5(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: NO
    Input args: list(Dataset)
    """

    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    def _model():
        inputs = Input((10,), name='x')
        output = Dense(1, name='y', activation='sigmoid')(inputs)
        model = Janggu(inputs=inputs, outputs=output, name='test_model')
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model()

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32,
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict([inputs],
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs],
                 use_multiprocessing=False)


def test_janggu_train_predict_option6(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: YES
    Input args: Dataset
    """
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    @inputlayer
    @outputdense('sigmoid')
    def _model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = Janggu.create(_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32,
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs,
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs,
                 use_multiprocessing=False)


def test_janggu_train_predict_option7(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: YES
    Input args: Dataset
    validation_set: YES
    batch_size: None
    """
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    @inputlayer
    @outputdense('sigmoid')
    def _model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = Janggu.create(_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    print('storage', storage)
    print('env', os.environ['JANGGU_OUTPUT'])
    print('name', bwm.name)
    print('outputdir', bwm.outputdir)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2,
            validation_data=(inputs, outputs),
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs,
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs,
                 use_multiprocessing=False)


def test_janggu_train_predict_sequence(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: YES
    Input args: Dataset
    validation_set: YES
    batch_size: None
    """
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    inputs = {'x': Array("x", np.random.random((100, 10)))}
    outputs = {'y': Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])}

    jseq = JangguSequence(10, inputs, outputs)

    @inputlayer
    @outputdense('sigmoid')
    def _model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = Janggu.create(_model,
                        inputs=jseq.inputs['x'],
                        outputs=jseq.outputs['y'],
                        name='nptest')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    print('storage', storage)
    print('env', os.environ['JANGGU_OUTPUT'])
    print('name', bwm.name)
    print('outputdir', bwm.outputdir)
    assert not os.path.exists(storage)

    bwm.fit(jseq, epochs=2,
            validation_data=jseq,
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict(jseq, use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs['x']))
    np.testing.assert_equal(pred.shape, outputs['y'].shape)
    bwm.evaluate(jseq, use_multiprocessing=False)
