import lasagne
import theano.tensor as T
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
import numpy as np
import pickle

num_features = 300

# Sequence Length
SEQ_LENGTH = 14

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 300

# All gradients above this will be clipped
GRAD_CLIP = 5


def build_vggmodel(input_var=None, batch_size=1):
    net = {}
    net['input_layer'] = InputLayer((batch_size, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = ConvLayer(net['input_layer'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=300, nonlinearity=None)
    net['prob'] = DenseLayer(net['fc7_dropout'], num_units=300, nonlinearity=None)
    return net


def build_vggpool5model(input_var=None):
    net = {}
    net['input_layer'] = InputLayer((None, 512, 7, 7), input_var=input_var)
    net['fc6'] = DenseLayer(net['input_layer'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['prob'] = DenseLayer(net['fc7_dropout'], num_units=300, nonlinearity=None)
    return net


def LSTMmodel(input_var_lstm=None, input_var_mask=None, batch_size=1):
    # print "Building LSTM network"
    l_in = lasagne.layers.InputLayer(shape=(batch_size, SEQ_LENGTH, num_features), input_var=input_var_lstm)
    mask_input = lasagne.layers.InputLayer(shape=(batch_size, SEQ_LENGTH), input_var=input_var_mask)
    l_forward_1 = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=mask_input,
                                           nonlinearity=lasagne.nonlinearities.rectify)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)
    l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=num_features, W=lasagne.init.Normal(),
                                      nonlinearity=None)
    return l_out


def build_custom_mlp(input_var=None, depth=1, width=300, drop_input=0, drop_hidden=0.5):
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 4096), input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
            network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    tanh = lasagne.nonlinearities.tanh
    network = lasagne.layers.DenseLayer(network, 301, nonlinearity=None)
    return network


def set_lstmweights(net,
                    LSTM_weight_file):
    '''
    set the weights of the given model.
    @param net: a lasagne network
    @param LSTM_weight_file:
    @return:
    '''
    # Get LSTM weights
    with np.load(LSTM_weight_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net, param_values)
    print('Set LSTM learned weights...')


def set_vggweights(net,
                   vgg_weight_file, k):
    '''
    set the weights of the given model.
    @param net: a lasagne network
    @param vgg_weight_file:
    @return:
    '''
    # Get VGG weights
    # print('Set vgg19 weights...')
    vggmodel = pickle.load(open(vgg_weight_file))
    lasagne.layers.set_all_param_values(net, vggmodel['param values'][0:k])


def set_cnnweights(net,
                   cnn_weight_file):
    '''
    set the weights of the given model.
    @param net: a lasagne network
    @param cnn_weight_file:
    @return:
    '''
    # Get LSTM weights
    print('Set CNN learned weights...')
    with np.load(cnn_weight_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values([net], param_values)


def relevance_score(I_out, Q_out):
    I_out = I_out[:, 0:300]
    I_out = I_out / I_out.norm(L=2, axis=1).reshape((I_out.shape[0], 1))
    Q_out = Q_out / Q_out.norm(L=2, axis=1).reshape((Q_out.shape[0], 1))
    value = T.diagonal(T.dot(I_out, Q_out.T))
    return value


def interestingness_score(I_out):
    I_out = I_out[:, 300]
    return I_out
