import argparse
import os

import numpy as np
import pandas as pd
import pkg_resources
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Maximum
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from janggo import InOutScorer
from janggo import InScorer
from janggo import Janggo
from janggo import inputlayer
from janggo import outputdense
from janggo.data import Array
from janggo.data import Dna
from janggo.layers import Complement
from janggo.layers import Reverse
from janggo.utils import export_clustermap
from janggo.utils import export_json
from janggo.utils import export_plotly
from janggo.utils import export_score_plot
from janggo.utils import export_tsne
from janggo.utils import export_tsv

np.random.seed(1234)

# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')
PARSER.add_argument('model', choices=['single', 'double'],
                    help="Single or double stranded model.")
PARSER.add_argument('-path', dest='path',
                    default='tf_results',
                    help="Output directory for the examples.")
PARSER.add_argument('-order', dest='order', type=int,
                    default=1,
                    help="One-hot order.")

args = PARSER.parse_args()


# load the dataset
DATA_PATH = pkg_resources.resource_filename('janggo', 'resources/')
SAMPLE_1 = os.path.join(DATA_PATH, 'sample.fa')
SAMPLE_2 = os.path.join(DATA_PATH, 'sample2.fa')
X1 = Dna.create_from_fasta('dna', fastafile=SAMPLE_1,
                           order=args.order)

DNA = Dna.create_from_fasta('dna', fastafile=[SAMPLE_1, SAMPLE_2],
                            order=args.order)

Y = np.zeros((len(DNA), 1))
Y[:len(X1)] = 1
LABELS = Array('y', Y, conditions=['TF-binding'])
annot = pd.DataFrame(Y[:,0], columns=LABELS.conditions).applymap(
    lambda x: 'Oct4' if x==1 else 'Mafk').to_dict(orient='list')

# evaluation metrics from sklearn.metrics
def wrap_roc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    aux = str('({:.2%})'.format(roc_auc_score(y_true, y_pred)))
    print('roc', aux)
    return fpr, tpr, aux


def wrap_prc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aux = str('({:.2%})'.format(average_precision_score(y_true, y_pred)))
    print('prc', aux)
    return recall, precision, aux


# instantiate various evaluation callback objects
# score metrics
auc_eval = InOutScorer('auROC', roc_auc_score, exporter=export_tsv)
prc_eval = InOutScorer('PRC', wrap_prc, exporter=export_score_plot)
roc_eval = InOutScorer('ROC', wrap_roc, exporter=export_score_plot)
auprc_eval = InOutScorer('auPRC', average_precision_score, exporter=export_tsv)

# clustering plots based on hidden features
heatmap_eval = InScorer('heatmap', exporter=export_clustermap,
                        exporter_args={'annot': annot,
                                       'z_score': 1})
tsne_eval = InScorer('tsne', exporter=export_tsne, exporter_args={'alpha': .1,
                                                                  'annot': annot})

# output the predictions as tables or json files
pred_tsv = InScorer('pred', exporter=export_tsv, exporter_args={'annot': annot})
pred_json = InScorer('pred', exporter=export_json, exporter_args={'annot': annot})

# plotly will export a special table that is used for interactive browsing
# of the results
pred_plotly = InScorer('pred', exporter=export_plotly,
                       exporter_args={'annot': annot,
                                      'row_names': DNA.gindexer.chrs})

# Option 3:
# Instantiate an ordinary keras model
@inputlayer
@outputdense('sigmoid')
def single_stranded_model(inputs, inp, oup, params):
    with inputs.use('dna') as layer:
        layer = Conv2D(params[0], (params[1], layer.shape.as_list()[2]),
                       activation=params[2])(layer)
    output = GlobalAveragePooling2D(name='motif')(layer)
    return inputs, output


@inputlayer
@outputdense('sigmoid')
def double_stranded_model(inputs, inp, oup, params):
    with inputs.use('dna') as layer:
        forward = layer
    convlayer = Conv2D(params[0], (params[1], layer.shape.as_list()[2]),
                       activation=params[2])
    revcomp = Reverse()(forward)
    revcomp = Complement()(revcomp)

    forward = convlayer(forward)
    revcomp = convlayer(revcomp)
    revcomp = Reverse()(revcomp)
    layer = Maximum()([forward, revcomp])
    output = GlobalAveragePooling2D(name='motif')(layer)
    return inputs, output

modeltemplate = single_stranded_model if args.model == 'single' \
                else double_stranded_model
K.clear_session()
# create a new model object
model = Janggo.create(template=modeltemplate,
                      modelparams=(30, 21, 'relu'),
                      inputs=DNA,
                      outputs=LABELS,
                      outputdir=args.path)

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])

hist = model.fit(DNA, LABELS, epochs=150)

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

model.evaluate(DNA, LABELS, datatags=['train'],
               callbacks=[auc_eval, prc_eval, roc_eval, auprc_eval])
# model.predict(DNA, datatags=['train'],
#               callbacks=[heatmap_eval, tsne_eval],
#               layername='motif')
model.predict(DNA, datatags=['train'],
              callbacks=[pred_tsv, pred_json, pred_plotly,
              heatmap_eval, tsne_eval],
              layername='motif')

# model.predict(DNA, datatags=['train', 'output'],
#               callbacks=[pred_eval])
