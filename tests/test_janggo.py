import os

import h5py
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
from keras import Model

from janggu import Janggu
from janggu import input_attribution
from janggu import inputlayer
from janggu import model_from_json
from janggu import model_from_yaml
from janggu import outputconv
from janggu import outputdense
from janggu.data import Array
from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data import GenomicIndexer
from janggu.data import ReduceDim
from janggu.data.data import JangguSequence
from janggu.layers import Complement
from janggu.layers import DnaConv2D
from janggu.layers import LocalAveragePooling2D
from janggu.layers import Reverse

matplotlib.use('AGG')


@pytest.mark.filterwarnings("ignore:inspect")
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


@pytest.mark.filterwarnings("ignore:inspect")
@pytest.mark.filterwarnings("ignore:The truth value")
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


def test_dnaconv():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file, order=1)

    xin = Input(dna.shape[1:])
    l1 = DnaConv2D(Conv2D(30, (21, 1), activation='relu'))(xin)
    m1 = Model(xin, l1)
    res1 =m1.predict(dna[0])[0,0,0,:]

    clayer = m1.layers[1].forward_layer
    # forward only
    l1 = clayer(xin)
    m2 = Model(xin, l1)
    res2 = m2.predict(dna[0])[0,0, 0,:]

    rxin = Reverse()(Complement()(xin))
    l1 = clayer(rxin)
    l1 = Reverse()(l1)
    m3 = Model(xin, l1)
    res3 = m3.predict(dna[0])[0,0, 0,:]

    res4 = np.maximum(res3,res2)
    np.testing.assert_allclose(res1, res4, rtol=1e-4)


def test_dnaconv2():
    # this checks if DnaConv2D layer is instantiated correctly if
    # the conv2d layer has been instantiated beforehand.
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file, order=1)

    xin = Input(dna.shape[1:])
    clayer = Conv2D(30, (21, 1), activation='relu')

    clayer(xin)

    l1 = DnaConv2D(clayer)(xin)
    m1 = Model(xin, l1)
    res1 =m1.predict(dna[0])[0,0,0,:]

    np.testing.assert_allclose(clayer.get_weights()[0], m1.layers[1].forward_layer.get_weights()[0])
    assert len(clayer.weights) == 2


@pytest.mark.filterwarnings("ignore:The truth value")
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
    ctcf = Array('ctcf', df.values, conditions=['peaks'])

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
    kbwm2 = model_from_json(bwm.kerasmodel.to_json())
    kbwm3 = model_from_yaml(bwm.kerasmodel.to_yaml())

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()
    bwm.summary()

    assert os.path.exists(storage)

    Janggu.create_by_name('dna_ctcf_HepG2-cnn')


@pytest.mark.filterwarnings("ignore:The truth value")
def test_janggu_influence_genomic(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    csvfile = os.path.join(data_path, 'sample.csv')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       binsize=50,
                                       roi=bed_file, order=1)

    df = pd.read_csv(csvfile, header=None)
    ctcf = Array('ctcf', df.values, conditions=['peaks'])

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs['dna']
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output
    model = Janggu.create(_cnn_model, modelparams=(2,),
                          inputs=dna,
                          outputs=ctcf,
                          name='dna_ctcf_HepG2-cnn')

    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    # check with some nice offset
    iv = dna.gindexer[0]
    chrom, start, end = iv.chrom, iv.start, iv.end
    influence = input_attribution(model, dna, chrom=chrom, start=start, end=end)

    # check with an odd offset

#    chrom, start, end =
    influence2 = input_attribution(model, dna, chrom=chrom, start=start-1, end=end+1)
    np.testing.assert_equal(influence[0][:], influence2[0][:][:,1:-1])



@pytest.mark.filterwarnings("ignore:The truth value")
def test_janggu_variant_prediction(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    for order in [1, 2, 3]:
        refgenome = os.path.join(data_path, 'sample_genome.fa')
        vcffile = os.path.join(data_path, 'sample.vcf')

        dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                           storage='ndarray',
                                           binsize=50,
                                           store_whole_genome=True,
                                           order=order)

        def _cnn_model(inputs, inp, oup, params):
            inputs = Input((50 - params['order'] + 1, 1, pow(4, params['order'])))
            layer = Flatten()(inputs)
            layer = Dense(params['hiddenunits'])(layer)
            output = Dense(4, activation='sigmoid')(layer)
            return inputs, output

        model = Janggu.create(_cnn_model, modelparams={'hiddenunits':2, 'order':order},
                              name='dna_ctcf_HepG2-cnn')

        model.predict_variant_effect(dna, vcffile, conditions=['m'+str(i) for i in range(4)],
                                     output_folder=os.path.join(os.environ['JANGGU_OUTPUT']))
        assert os.path.exists(os.path.join(os.environ['JANGGU_OUTPUT'], 'scores.hdf5'))
        assert os.path.exists(os.path.join(os.environ['JANGGU_OUTPUT'], 'snps.bed.gz'))

        f = h5py.File(os.path.join(os.environ['JANGGU_OUTPUT'], 'scores.hdf5'), 'r')

        gindexer = GenomicIndexer.create_from_file(os.path.join(os.environ['JANGGU_OUTPUT'],
                                                                'snps.bed.gz'), None, None)

        cov = Cover.create_from_array('snps', f['diffscore'],
                                      gindexer,
                                      store_whole_genome=True)

        print(cov['chr2', 55, 65].shape)
        print(cov['chr2', 55, 65])

        assert np.abs(cov['chr2', 59, 60]).sum() > 0.0
        assert np.abs(cov['chr2', 54, 55]).sum() == 0.0
        f.close()


@pytest.mark.filterwarnings("ignore:The truth value")
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


@pytest.mark.filterwarnings("ignore:The truth value")
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


@pytest.mark.filterwarnings("ignore:The truth value")
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


@pytest.mark.filterwarnings("ignore:The truth value")
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


@pytest.mark.filterwarnings("ignore:The truth value")
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



@pytest.mark.filterwarnings("ignore:inspect")
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


@pytest.mark.filterwarnings("ignore:inspect")
def test_janggu_bedfile_validation(tmpdir):
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
    p1 = bwm1.fit(dna, ctcf, validation_data=posfile)


@pytest.mark.filterwarnings("ignore:inspect")
def test_janggu_train_predict_option0(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Train, predict and evaluate on dummy data.

    create: by_shape
    Input args: Dataset
    """

    inputs = Array("X", np.random.random((100, 10)))
    outputs = ReduceDim(Array('y', np.random.randint(2, size=(100, 1))[:,None],
                    conditions=['random']), axis=(1,))

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

    # test if the condition name is correctly used in the output table
    bwm.evaluate(inputs, outputs, callbacks=['auc'])

    outputauc = os.path.join(tmpdir.strpath, 'evaluation', 'nptest', 'auc.tsv')
    assert os.path.exists(outputauc)
    assert pd.read_csv(outputauc).columns[0] == 'random'


@pytest.mark.filterwarnings("ignore:inspect")
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

    # test if the condition name is correctly used in the output table
    bwm.evaluate(inputs, outputs, callbacks=['auc'])

    outputauc = os.path.join(tmpdir.strpath, 'evaluation', 'nptest', 'auc.tsv')
    assert os.path.exists(outputauc)
    assert pd.read_csv(outputauc).columns[0] == 'random'


@pytest.mark.filterwarnings("ignore:inspect")
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


@pytest.mark.filterwarnings("ignore:inspect")
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


@pytest.mark.filterwarnings("ignore:inspect")
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


@pytest.mark.filterwarnings("ignore:inspect")
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


@pytest.mark.filterwarnings("ignore:inspect")
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


@pytest.mark.filterwarnings("ignore:inspect")
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

@pytest.mark.filterwarnings("ignore:inspect")
def test_sequence_config():
    """Train, predict and evaluate on dummy data.

    create: YES
    Input args: Dataset
    validation_set: YES
    batch_size: None
    """

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    jseq = JangguSequence(inputs.data, outputs.data, batch_size=10, as_dict=False)
    assert len(jseq) == 10
    for x, y, _ in jseq:
        assert x[0].shape == (10, 10)
        assert y[0].shape == (10, 1)
        break

    jseq = JangguSequence(inputs, outputs, batch_size=10, as_dict=False)
    assert len(jseq) == 10
    for x, y, _ in jseq:
        assert x[0].shape == (10, 10)
        assert y[0].shape == (10, 1)
        break

    jseq = JangguSequence(inputs, outputs, batch_size=10, as_dict=True)
    assert len(jseq) == 10
    for x, y, _ in jseq:
        assert x['x'].shape == (10, 10)
        assert y['y'].shape == (10, 1)
        break


@pytest.mark.filterwarnings("ignore:inspect")
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

    jseq = JangguSequence(inputs, outputs, batch_size=10)

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
