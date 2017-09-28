'''
This module contains functions to create the Relevance network and load the weights
as well as some helper functions, e.g. for running the demos.
For more information on the method, see
 Arun Balajee Vasudevan*, Michael Gygli*, Anna Volokitin, Luc Van Gool(* denotes equal contribution)
    "Query-adaptive Video Summarization via Quality-aware Relevance Estimation", ACM Multimedia 2017
'''

__author__ = 'Arun Balajee Vasudevan'
import ConfigParser
import numpy as np
import os
import model
import gensim
import theano.tensor as T
import cv2

# Load the configuration
if not 'QVSUM_DATA_DIR' in os.environ:
    os.environ['QVSUM_DATA_DIR']='./data'
config = ConfigParser.SafeConfigParser(os.environ)
print('Loaded config file from %s' % config.read('%s/config.ini' % os.path.dirname(__file__))[0])

import shells

try:
    import lasagne
    import theano
except (ImportError, AssertionError) as e:
    print(e.message)

def get_QAR_function():
    '''
    Get Relevance function (CNN-LSTM Relevance model)
    @return: theano function that scores video frames
    '''
    # Set LSTM model
    print('Load weights and compile Relevance model...')
    input_var_lstm = theano.tensor.tensor3('inputs_lstm')
    input_var_mask = theano.tensor.bmatrix('inputs_mask')
    l_out = model.LSTMmodel(input_var_lstm, input_var_mask)
    model.set_lstmweights(l_out, config.get('paths', 'LSTM_weight_file'))
    network_output = lasagne.layers.get_output(l_out, deterministic=True)

    # Set CNN model
    network = model.build_vggmodel(batch_size=1)
    model.set_vggweights(network['fc7'], config.get('paths', 'vgg_weight_file'), 36)
    inter_layer = lasagne.layers.get_output(network['fc7'], deterministic=True)
    netpool = model.build_custom_mlp(inter_layer, depth=0, width=4096, drop_input=0.5, drop_hidden=0.5)
    model.set_cnnweights(netpool, config.get('paths', 'cnn_weight_file'))
    mlp_output = lasagne.layers.get_output(netpool, deterministic=True)

    test_similarity = model.relevance_score(mlp_output, network_output)
    test_quality = model.interestingness_score(mlp_output)
    val_fn = theano.function([network['input_layer'].input_var, input_var_lstm, input_var_mask],
                             [test_similarity, test_quality], on_unused_input='warn')

    return val_fn


def get_word2vec_function():
    '''
    Get word2vec function
    @param feature_layer: a layer name (see model.py). If provided, pred_fn returns (score, and the activations at feature_layer)
    @return: theano function that scores video frames
    '''
    # Set word2vec model
    print('Load word2vec model...')
    if (os.path.isfile(config.get('paths', 'word2vec_file'))):
        w2vmodel = gensim.models.Word2Vec.load(config.get('paths', 'word2vec_file'))
    else:
        w2vmodel = gensim.models.Word2Vec.load(config.get('paths', 'word2vec_smallfile'))
    return w2vmodel


def get_rel_Q_scores(val_fn, w2vmodel, query, frames):
    '''
    Predict similarity and quality scores for frames
    @param val_fn: prediction function
    @param w2vmodel: word2vec model
    @param query: given text query as string
    @param frames: list of paths of video frames
    @return: list of scores
    '''

    def load_dataset(image_names):
        i = 0
        image_size = 224
        data = np.zeros((len(image_names), 3, image_size, image_size), dtype=np.float32)
        MEAN_PIXEL = [103.939, 116.779, 123.68]
        p = 0
        for i in range(len(image_names)):
            image = image_names[i]
            im = cv2.imread(image)
            # im = im[:,:,::-1]
            im = cv2.resize(im, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            im = im - MEAN_PIXEL
            data[i, :, :, :] = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        X_train = data
        return X_train

    def load_query(query):
        query = query.lower()
        query = ' '.join(word for word in query.split(' ') if word in w2vmodel.vocab)
        words = query.split()
        SEQ_LENGTH = 14
        num_features = 300
        BATCH_SIZE = 1
        qdata = np.zeros((BATCH_SIZE, SEQ_LENGTH, num_features), dtype=np.float32)
        mask = np.ones((BATCH_SIZE, SEQ_LENGTH), dtype=np.bool)
        for j in range(SEQ_LENGTH):
            if j < len(words):
                qdata[0, j, :] = np.array(w2vmodel[str(words[j])])
            else:
                mask[0, j] = 0
        return qdata, mask

    valid_array_sim = [];
    valid_array_q = []
    qdata, mask = load_query(query)
    print "Scoring frames... "
    for m, p in enumerate(frames):
        path = []
        path.append(p)
        X = load_dataset(path)
        sim, quality = val_fn(X, qdata, mask)
        valid_array_sim.append(sim[0])
        valid_array_q.append(quality[0])
    return valid_array_sim, valid_array_q
